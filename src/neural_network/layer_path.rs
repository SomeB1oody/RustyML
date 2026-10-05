//! The address of 1 layer inside a model, and the walks over a tree of layers
//!
//! A layer can hold other layers. [`LayerBase::sublayers`] names the layers that a layer holds,
//! and each of those can hold more. The layers of 1 model position therefore form a tree. The
//! model position is the root, and every other node is a sublayer.
//!
//! A [`LayerPath`] addresses 1 node of that tree. It holds the model position of the root, and
//! the sublayer names from the root down to the node. The path `2.forward` is the sublayer
//! `forward` of the layer at position 2. The path `2` is the layer at position 2.
//!
//! Every channel that keys on a layer keys on its path:
//!
//! 1. The gradient store and the optimizer state key on [`ParamId`], which is a path and the
//!    name of 1 array.
//! 2. The state channel of [`Ctx`] keys on a path and the name of 1 value.
//! 3. The cache channel of [`Ctx`] keys on the call of the root and on the sublayer frames.
//! 4. A checkpoint records the tree of each model position, and a checkpoint path is
//!    `<path>.<name>`.
//!
//! A layer that holds no sublayer is a tree of 1 node. Its path is its model position, so
//! every address of such a layer has the form `<scope>.<name>`.
//!
//! The walks in this module visit a tree in pre-order: a node first, and then each sublayer in
//! the order that [`LayerBase::sublayers`] gives. This order is the canonical order of a model.
//! The optimizer update, the global gradient norm, the checkpoint, and the list of weight
//! paths all use it.
//!
//! [`Ctx`]: crate::neural_network::Ctx
//! [`LayerBase::sublayers`]: crate::neural_network::traits::LayerBase::sublayers
//! [`ParamId`]: crate::neural_network::traits::ParamId

use crate::error::Error;
use crate::math::reduction::det_reduce;
use crate::neural_network::ctx::{Ctx, Grads};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Layer, LayerBase, Optimizer, ParamId};
use crate::parallel_gates::sq_sum_f32_parallel_min_elems;
use ndarray::ArrayViewD;
use std::borrow::Cow;
use std::fmt;

/// The name that a layer gives 1 of its sublayers
///
/// A name is a fixed string such as `"forward"` for most layers. A layer that builds its
/// sublayers at run time can give an owned string. Both forms compare as text, so `"cell"`
/// and `String::from("cell")` are the same name
pub type SublayerName = Cow<'static, str>;

/// The address of 1 layer inside a model
///
/// See the [module documentation](self) for the tree that the path walks
///
/// The text form is the model position and then each sublayer name, joined by `.`. A
/// checkpoint path appends the name of 1 array to this text
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::LayerPath;
///
/// let root = LayerPath::root(2);
/// assert_eq!(root.to_string(), "2");
///
/// let cell = root.child("forward").child("cell");
/// assert_eq!(cell.to_string(), "2.forward.cell");
/// assert_eq!(cell.scope(), 2);
/// assert_eq!(cell.sublayers().len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerPath {
    /// Position of the root layer in the model that drives it
    scope: usize,
    /// The sublayer names from the root down to the node. Empty for the root itself
    sublayers: Vec<SublayerName>,
}

impl LayerPath {
    /// The path of the layer at a model position
    ///
    /// # Parameters
    ///
    /// - `scope` - Position of the layer in the model
    ///
    /// # Returns
    ///
    /// - `LayerPath` - The path, with no sublayer name
    #[inline]
    pub const fn root(scope: usize) -> Self {
        Self {
            scope,
            sublayers: Vec::new(),
        }
    }

    /// The path of 1 sublayer of the layer at this path
    ///
    /// # Parameters
    ///
    /// - `name` - The name that the layer at this path gives the sublayer
    ///
    /// # Returns
    ///
    /// - `LayerPath` - This path, with the name appended
    pub fn child(&self, name: impl Into<SublayerName>) -> Self {
        let mut sublayers = Vec::with_capacity(self.sublayers.len() + 1);
        sublayers.extend(self.sublayers.iter().cloned());
        sublayers.push(name.into());
        Self {
            scope: self.scope,
            sublayers,
        }
    }

    /// Position of the root layer in the model
    ///
    /// # Returns
    ///
    /// - `usize` - The model position
    #[inline]
    pub fn scope(&self) -> usize {
        self.scope
    }

    /// The sublayer names from the root down to the node
    ///
    /// # Returns
    ///
    /// - `&[SublayerName]` - The names, root side first. Empty for a root path
    #[inline]
    pub fn sublayers(&self) -> &[SublayerName] {
        &self.sublayers
    }

    /// Whether the path addresses a model position and not a sublayer
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when the path holds no sublayer name
    #[inline]
    pub fn is_root(&self) -> bool {
        self.sublayers.is_empty()
    }

    /// The address of 1 named array of the layer at this path
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the array, such as `"kernel"`
    ///
    /// # Returns
    ///
    /// - `ParamId` - The address of the array
    #[inline]
    pub fn param(&self, name: &'static str) -> ParamId {
        ParamId::at(self.clone(), name)
    }

    /// Builds a path from the model position and the sublayer names
    pub(crate) fn from_parts(scope: usize, sublayers: Vec<SublayerName>) -> Self {
        Self { scope, sublayers }
    }
}

impl fmt::Display for LayerPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.scope)?;
        for name in &self.sublayers {
            write!(f, ".{name}")?;
        }
        Ok(())
    }
}

/// 1 sublayer of a layer, borrowed for reading
///
/// [`LayerBase::sublayers`] gives 1 of these per layer that the layer holds
pub struct Sublayer<'a> {
    /// The name that the holding layer gives the sublayer
    pub name: SublayerName,
    /// The sublayer
    pub layer: &'a dyn LayerBase,
}

impl<'a> Sublayer<'a> {
    /// A read view of 1 sublayer under 1 name
    ///
    /// # Parameters
    ///
    /// - `name` - The name that the holding layer gives the sublayer
    /// - `layer` - The sublayer
    ///
    /// # Returns
    ///
    /// - `Sublayer` - The entry
    #[inline]
    pub fn new(name: impl Into<SublayerName>, layer: &'a dyn LayerBase) -> Self {
        Self {
            name: name.into(),
            layer,
        }
    }
}

/// 1 sublayer of a layer, borrowed for writing
///
/// [`LayerBase::sublayers_mut`] gives 1 of these per layer that the layer holds, under the
/// same names and in the same order as [`LayerBase::sublayers`]
pub struct SublayerMut<'a> {
    /// The name that the holding layer gives the sublayer
    pub name: SublayerName,
    /// The sublayer
    pub layer: &'a mut dyn LayerBase,
}

impl<'a> SublayerMut<'a> {
    /// A write view of 1 sublayer under 1 name
    ///
    /// # Parameters
    ///
    /// - `name` - The name that the holding layer gives the sublayer
    /// - `layer` - The sublayer
    ///
    /// # Returns
    ///
    /// - `SublayerMut` - The entry
    #[inline]
    pub fn new(name: impl Into<SublayerName>, layer: &'a mut dyn LayerBase) -> Self {
        Self {
            name: name.into(),
            layer,
        }
    }
}

/// Visits every node of 1 tree for reading, in the canonical pre-order
///
/// # Parameters
///
/// - `layer` - The root of the tree
/// - `path` - The path of the root
/// - `visit` - Called once per node, with the path of the node and the node
pub(crate) fn walk(
    layer: &dyn LayerBase,
    path: &LayerPath,
    visit: &mut dyn FnMut(&LayerPath, &dyn LayerBase),
) {
    visit(path, layer);
    for sub in layer.sublayers() {
        walk(sub.layer, &path.child(sub.name), visit);
    }
}

/// Visits every node of 1 tree for writing, in the canonical pre-order
///
/// # Parameters
///
/// - `layer` - The root of the tree
/// - `path` - The path of the root
/// - `visit` - Called once per node, with the path of the node and the node
pub(crate) fn walk_mut(
    layer: &mut dyn LayerBase,
    path: &LayerPath,
    visit: &mut dyn FnMut(&LayerPath, &mut dyn LayerBase),
) {
    visit(path, &mut *layer);
    for sub in layer.sublayers_mut() {
        walk_mut(sub.layer, &path.child(sub.name), visit);
    }
}

/// Visits every node of every tree of a model for writing, in the canonical order
///
/// The canonical order is the model position order, and the pre-order inside each position
///
/// # Parameters
///
/// - `layers` - The layers of the model, by position
/// - `visit` - Called once per node, with the path of the node and the node
pub(crate) fn walk_model_mut(
    layers: &mut [Box<dyn Layer>],
    visit: &mut dyn FnMut(&LayerPath, &mut dyn LayerBase),
) {
    for (scope, layer) in layers.iter_mut().enumerate() {
        walk_mut(&mut **layer, &LayerPath::root(scope), visit);
    }
}

/// How many parameter elements a layer and all of its sublayers hold
///
/// [`LayerBase::param_count`] counts the arrays of 1 node. This sum is what a model summary
/// shows for 1 model position
///
/// # Parameters
///
/// - `layer` - The root of the tree
///
/// # Returns
///
/// - `ParamCounts` - The sum over every node of the tree
pub fn total_param_count(layer: &dyn LayerBase) -> ParamCounts {
    let mut total = ParamCounts::none();
    walk(layer, &LayerPath::root(0), &mut |_, node| {
        let counts = node.param_count();
        total.trainable += counts.trainable;
        total.non_trainable += counts.non_trainable;
    });
    total
}

/// The checkpoint path of every array of a model, in the canonical order
///
/// # Parameters
///
/// - `layers` - The layers of the model, by position
///
/// # Returns
///
/// - `Vec<String>` - 1 path per array
pub(crate) fn model_weight_paths(layers: &[Box<dyn Layer>]) -> Vec<String> {
    let mut paths = Vec::new();
    for (scope, layer) in layers.iter().enumerate() {
        walk(&**layer, &LayerPath::root(scope), &mut |path, node| {
            paths.extend(
                node.weights()
                    .iter()
                    .map(|entry| weight_path(path, entry.name)),
            );
        });
    }
    paths
}

/// Builds the checkpoint path of 1 array
///
/// # Parameters
///
/// - `path` - The path of the layer that holds the array
/// - `name` - The name the layer gives the array
///
/// # Returns
///
/// - `String` - The dotted path, such as `2.kernel` or `2.forward.kernel`
#[inline]
pub fn weight_path(path: &LayerPath, name: &str) -> String {
    format!("{path}.{name}")
}

/// 1 array of a model, by its checkpoint path
///
/// The text before the first `.` is the model position. The text after the last `.` is the
/// name of the array. Each part between them is 1 sublayer name. A model build refuses every
/// name that holds a `.`, so 1 text gives 1 address
///
/// # Parameters
///
/// - `layers` - The layers of the model, by position
/// - `path` - The checkpoint path, such as `"0.kernel"` or `"1.forward.kernel"`
///
/// # Returns
///
/// - `Option<ArrayViewD<'_, f32>>` - A read view of the array, or `None` when the model holds
///   no array at that path
pub(crate) fn find_weight<'a>(
    layers: &'a [Box<dyn Layer>],
    path: &str,
) -> Option<ArrayViewD<'a, f32>> {
    let mut parts: Vec<&str> = path.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let name = parts.pop()?;
    let scope: usize = parts[0].parse().ok()?;
    let mut node: &'a dyn LayerBase = &**layers.get(scope)?;
    for step in &parts[1..] {
        node = node
            .sublayers()
            .into_iter()
            .find(|sub| sub.name == *step)?
            .layer;
    }
    node.weight(name)
}

/// Moves the state that a pass proposed into every node of 1 tree
///
/// Each node takes the values at its own path. The caller checks afterwards that no value
/// stayed behind, with [`check_state_taken`]
///
/// # Parameters
///
/// - `layer` - The root of the tree
/// - `path` - The path of the root
/// - `ctx` - The context of the pass
pub(crate) fn apply_state_tree(layer: &mut dyn LayerBase, path: &LayerPath, ctx: &mut Ctx) {
    walk_mut(layer, path, &mut |node_path, node| {
        if ctx.has_state(node_path) {
            node.apply_state(&mut ctx.state_slot(node_path));
        }
    });
}

/// Refuses a pass that left a state value of a model position in the context
///
/// A value that stays behind is a value that no node took back. The running statistics or the
/// random stream that it holds would then never move, and no other check reports it
///
/// # Parameters
///
/// - `ctx` - The context of the pass, after the state of the position was applied
/// - `scope` - The model position
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok` when the context holds no state of the position
///
/// # Errors
///
/// - `Error::Computation` - If a value of the position stayed in the context. The message
///   names the path and the name of each value
pub(crate) fn check_state_taken(ctx: &Ctx, scope: usize) -> Result<(), Error> {
    let left = ctx.state_left_under(scope);
    if left.is_empty() {
        return Ok(());
    }
    Err(Error::computation(format!(
        "the forward pass proposed the state value(s) {} and no layer took them back. A layer \
         must take every value that it writes with `Ctx::set_state` in its own \
         `LayerBase::apply_state`, and a layer that holds sublayers must list each of them in \
         `LayerBase::sublayers_mut`",
        left.join(", ")
    )))
}

/// Updates every parameter of every node of a model, in the canonical order
///
/// # Parameters
///
/// - `optimizer` - The optimizer that updates the parameters
/// - `layers` - The layers of the model, by position
/// - `grads` - Every gradient the backward pass produced
/// - `grad_scale` - The uniform factor of the global-norm clip, or `1.0`
pub(crate) fn update_model(
    optimizer: &mut dyn Optimizer,
    layers: &mut [Box<dyn Layer>],
    grads: &Grads,
    grad_scale: f32,
) {
    walk_model_mut(layers, &mut |path, node| {
        optimizer.update(path, node, grads, grad_scale);
    });
}

/// The global L2 norm of every gradient of a model, for a global-norm clip
///
/// The walk is the canonical order of the model, and the parameter order of each node. The
/// gradient store sorts by address instead. A sum of `f64` squares is not associative, so a
/// reduction in store order would move the last bit of the norm. A tensor folds in
/// deterministic blocks, and the rayon path above the square-sum gate gives the same result as
/// the serial path. A node with no gradient contributes nothing, and a pass with no gradient
/// at all gives a norm of 0.0
///
/// This is the same walk order that [`update_model`] uses
///
/// # Parameters
///
/// - `layers` - The layers of the model, by position
/// - `grads` - Every gradient the backward pass produced
///
/// # Returns
///
/// - `f32` - The global norm
pub(crate) fn global_grad_norm(layers: &mut [Box<dyn Layer>], grads: &Grads) -> f32 {
    let mut sum_sq = 0.0_f64;
    walk_model_mut(layers, &mut |path, node| {
        for param in node.parameters_mut() {
            let Some(grad) = grads.get(&path.param(param.name)) else {
                continue;
            };
            let grad = grad
                .as_slice()
                .expect("a stored gradient is in the standard memory order");
            sum_sq += det_reduce(
                grad,
                grad.len() >= sq_sum_f32_parallel_min_elems(),
                |block| block.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>(),
                |a, b| a + b,
                0.0,
            );
        }
    });
    sum_sq.sqrt() as f32
}
