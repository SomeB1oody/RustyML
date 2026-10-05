//! Core traits for the neural network module: layers, losses, and optimizers, plus the
//! address and view types that connect them.
//!
//! [`LayerBase`](crate::neural_network::traits::LayerBase) holds what every layer has regardless of
//! its input count: its type name, its arrays, its sublayers, and its build state.
//! [`UnaryLayer`](crate::neural_network::traits::UnaryLayer) adds the forward and backward pass for
//! a layer with 1 input, and a blanket implementation gives it the general
//! [`Layer`](crate::neural_network::traits::Layer) interface that a model holds every layer
//! through. Implement [`Layer`](crate::neural_network::traits::Layer) directly only for a layer
//! with several inputs, such as a merge layer.
//!
//! [`ParamId`](crate::neural_network::traits::ParamId) names the address of 1 parameter tensor:
//! the [`LayerPath`](crate::neural_network::LayerPath) of the layer that holds it, plus the name
//! the layer gives the tensor.
//! [`ParamRef`](crate::neural_network::traits::ParamRef),
//! [`WeightRef`](crate::neural_network::traits::WeightRef), and
//! [`WeightMut`](crate::neural_network::traits::WeightMut) are the borrowed views that a layer
//! exposes under that name, for an optimizer to update or a checkpoint to read and write.
//! `check_addresses` and `check_every_gradient_is_claimed` are the build-time and pass-time
//! checks that keep every address unique and every gradient reachable.
//!
//! [`Loss`](crate::neural_network::traits::Loss) computes a scalar loss and its gradient.
//! [`Optimizer`](crate::neural_network::traits::Optimizer) reads the gradient store and updates the
//! parameters of a layer, keyed on [`ParamId`](crate::neural_network::traits::ParamId).

use crate::error::Error;
use crate::neural_network::Shape;
use crate::neural_network::Tensor;
use crate::neural_network::ctx::{Ctx, Grads, StateSlot};
use crate::neural_network::layer_path::{
    LayerPath, Sublayer, SublayerMut, apply_state_tree, walk, walk_model_mut, walk_mut,
};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::checkpoint::BuildConfig;
use crate::{Deserialize, Serialize};
use ndarray::{ArrayViewD, ArrayViewMutD};
use std::any::{Any, TypeId};
use std::fmt;

/// The stable address of 1 parameter tensor inside a model
///
/// A parameter is identified by the layer that holds it and by the name that the layer gives
/// it. Neither half moves while the model trains. An optimizer can therefore key its
/// per-parameter state on the pair and reach the same buffer on every step
///
/// The layer half is a [`LayerPath`]: the position of the layer in the model that drives the
/// update, and the sublayer names down to the layer that holds the tensor.
/// [`Sequential`](crate::neural_network::sequential::Sequential) gives each layer its index,
/// counted from the input. A caller that drives 1 layer directly takes position 0
///
/// The name is the `&'static str` that [`LayerBase::parameters_mut`] puts in the
/// [`ParamRef`]. It follows the layer. A layer that stops yielding 1 of its
/// tensors, or that starts yielding a new one, moves no other tensor's address
///
/// The text form is the checkpoint path of the tensor, such as `2.kernel` or
/// `2.forward.kernel`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::LayerPath;
/// use rustyml::neural_network::traits::ParamId;
///
/// assert_eq!(ParamId::new(2, "kernel").to_string(), "2.kernel");
///
/// let nested = LayerPath::root(2).child("forward").param("kernel");
/// assert_eq!(nested.to_string(), "2.forward.kernel");
/// assert_ne!(nested, ParamId::new(2, "kernel"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId {
    /// The path of the layer that holds the tensor
    pub layer: LayerPath,
    /// Name the layer gives the tensor, such as `"kernel"`, `"bias"`, or `"gamma"`
    pub name: &'static str,
}

impl ParamId {
    /// Builds the address of the named parameter of the layer at a model position
    ///
    /// The address reaches the layer at the position itself, and no sublayer of it. Use
    /// [`ParamId::at`] or [`LayerPath::param`] for a sublayer
    ///
    /// # Parameters
    ///
    /// - `scope` - Position of the owning layer in the model, counted from the input
    /// - `name` - Name the layer gives the tensor
    ///
    /// # Returns
    ///
    /// - `ParamId` - The parameter address
    #[inline]
    pub const fn new(scope: usize, name: &'static str) -> Self {
        Self {
            layer: LayerPath::root(scope),
            name,
        }
    }

    /// Builds the address of the named parameter of the layer at a path
    ///
    /// # Parameters
    ///
    /// - `layer` - The path of the layer that holds the tensor
    /// - `name` - Name the layer gives the tensor
    ///
    /// # Returns
    ///
    /// - `ParamId` - The parameter address
    #[inline]
    pub fn at(layer: LayerPath, name: &'static str) -> Self {
        Self { layer, name }
    }
}

impl fmt::Display for ParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.layer, self.name)
    }
}

/// Refuses a model position whose tree of layers gives 2 things 1 address
///
/// The name of an array is its address inside the layer. The gradient store, the per-parameter
/// state of the optimizer, and the path of the checkpoint all key on it. 2 arrays under 1 name
/// therefore share 1 gradient, 1 momentum buffer, and 1 checkpoint path. Each of those is a
/// wrong number that no later check reports.
///
/// A sublayer name is the address of a sublayer in the same way, and the check holds it to the
/// same rule. The check also holds the rules that keep the text form of an address readable
/// in 1 way only, and the rules that keep the 2 sublayer rosters in agreement.
///
/// A model build calls this once per model position, before the model computes anything, so a
/// layer that breaks a rule never reaches a training step. The check visits every node of the
/// tree, and it holds each node to these rules:
///
/// 1. No 2 arrays of [`LayerBase::weights`] share a name. The same holds for
///    [`LayerBase::parameters_mut`].
/// 2. No array name and no sublayer name is empty or holds a `.`.
/// 3. No 2 sublayers share a name.
/// 4. [`LayerBase::sublayers`] and [`LayerBase::sublayers_mut`] give the same names, in the same
///    order, and reach the same layers.
/// 5. No layer that occupies memory appears at 2 nodes of the tree. 2 nodes that reach 1
///    storage would update it twice per step.
///
/// # Parameters
///
/// - `scope` - Position of the layer in the model, which the message names
/// - `layer` - The root of the tree to check
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok(())` when every node of the tree keeps every rule
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If a node breaks a rule. The message names the path of the node
pub(crate) fn check_addresses(scope: usize, layer: &mut dyn LayerBase) -> Result<(), Error> {
    // A layer is identified by its address and its concrete type. The address alone is not
    // enough: a struct and its first field share 1 address, and both can be nodes of 1 tree.
    // A zero-sized layer holds no storage, so 2 nodes of it share nothing
    let mut storages: Vec<((*const (), TypeId), LayerPath)> = Vec::new();
    let mut failure: Option<Error> = None;
    walk(layer, &LayerPath::root(scope), &mut |path, node| {
        if failure.is_some() {
            return;
        }
        if std::mem::size_of_val(node) > 0 {
            let storage = (
                node as *const dyn LayerBase as *const (),
                (node as &dyn Any).type_id(),
            );
            if let Some((_, first)) = storages.iter().find(|(seen, _)| *seen == storage) {
                failure = Some(Error::invalid_input(format!(
                    "layer `{path}` (`{}`) is the same storage as layer `{first}`. The \
                     optimizer would update that storage once per node. Give each node of a \
                     layer tree its own layer",
                    node.layer_type()
                )));
                return;
            }
            storages.push((storage, path.clone()));
        }
        if let Err(error) = check_node(path, node) {
            failure = Some(error);
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }

    let mut failure: Option<Error> = None;
    walk_mut(layer, &LayerPath::root(scope), &mut |path, node| {
        if failure.is_none()
            && let Err(error) = check_node_mut(path, node)
        {
            failure = Some(error);
        }
    });
    failure.map_or(Ok(()), Err)
}

/// Checks the read roster of 1 node: array names and sublayer names
fn check_node(path: &LayerPath, node: &dyn LayerBase) -> Result<(), Error> {
    let layer_type = node.layer_type();
    let weights: Vec<&str> = node.weights().iter().map(|entry| entry.name).collect();
    check_names(path, layer_type, &weights, "weights", "array")?;
    let sublayers: Vec<String> = node
        .sublayers()
        .iter()
        .map(|sub| sub.name.to_string())
        .collect();
    let sublayer_refs: Vec<&str> = sublayers.iter().map(String::as_str).collect();
    check_names(path, layer_type, &sublayer_refs, "sublayers", "sublayer")
}

/// Checks the write rosters of 1 node: parameter names, and the agreement of the 2 sublayer
/// rosters
fn check_node_mut(path: &LayerPath, node: &mut dyn LayerBase) -> Result<(), Error> {
    let layer_type = node.layer_type().to_string();
    let parameters: Vec<&'static str> = node
        .parameters_mut()
        .iter()
        .map(|entry| entry.name)
        .collect();
    check_names(path, &layer_type, &parameters, "parameters_mut", "array")?;

    let read: Vec<RosterEntry> = node
        .sublayers()
        .iter()
        .map(|sub| roster_entry(&sub.name, sub.layer))
        .collect();
    let write: Vec<RosterEntry> = node
        .sublayers_mut()
        .iter()
        .map(|sub| roster_entry(&sub.name, &*sub.layer))
        .collect();
    if read != write {
        let names = |roster: &[RosterEntry]| {
            roster
                .iter()
                .map(|entry| format!("{} (`{}`)", entry.name, entry.layer_type))
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(Error::invalid_input(format!(
            "layer `{path}` (`{layer_type}`) gives the sublayers [{}] through \
             `LayerBase::sublayers`, and [{}] through `LayerBase::sublayers_mut`. The 2 rosters \
             must give the same names, in the same order, and reach the same layers",
            names(&read),
            names(&write)
        )));
    }
    Ok(())
}

/// 1 entry of a sublayer roster, as the agreement check compares it
#[derive(PartialEq)]
struct RosterEntry {
    /// The name of the sublayer
    name: String,
    /// The address of the sublayer
    address: *const (),
    /// The concrete type of the sublayer. A struct and its first field share 1 address, so the
    /// address alone does not identify a layer
    type_id: TypeId,
    /// The type name of the sublayer, for the message
    layer_type: String,
}

/// Reads 1 sublayer into the form that the agreement check compares
fn roster_entry(name: &str, layer: &dyn LayerBase) -> RosterEntry {
    RosterEntry {
        name: name.to_string(),
        address: layer as *const dyn LayerBase as *const (),
        type_id: (layer as &dyn Any).type_id(),
        layer_type: layer.layer_type().to_string(),
    }
}

/// Refuses an empty name, a name that holds a `.`, and a name that the list holds twice
fn check_names(
    path: &LayerPath,
    layer_type: &str,
    names: &[&str],
    roster: &str,
    what: &str,
) -> Result<(), Error> {
    for (index, name) in names.iter().enumerate() {
        if name.is_empty() || name.contains('.') {
            return Err(Error::invalid_input(format!(
                "layer `{path}` (`{layer_type}`) gives 1 {what} the name `{name}`, through \
                 `LayerBase::{roster}`. A name must not be empty and must not hold a `.`, \
                 because a checkpoint path joins the names with `.`"
            )));
        }
        if names[..index].contains(name) {
            return Err(Error::invalid_input(format!(
                "layer `{path}` (`{layer_type}`) gives 2 of its {what}s the name `{name}`, \
                 through `LayerBase::{roster}`. The name is the address, and the gradient \
                 store, the state of the optimizer, and the path of the checkpoint all key on \
                 it. Give every {what} of 1 layer its own name. A layer that holds other layers \
                 lists them in `LayerBase::sublayers` and does not pass their arrays on as its \
                 own"
            )));
        }
    }
    Ok(())
}

/// Refuses a pass that parked a gradient at an address no parameter of the model reads
///
/// The optimizer walk is a pull: it enumerates the parameters and looks each address up, and it
/// skips an address that holds no gradient. A gradient parked at any other address is therefore
/// dropped in silence. That is what turns a layer that misspells 1 of its own names into a
/// parameter that never trains. It reports no error of its own.
///
/// The walk below is the same walk that the optimizer makes. It visits every node of every
/// layer tree, so the count it reaches is the number of addresses the optimizer will read. A
/// store that holds more than that holds an address that nothing claims.
///
/// # Parameters
///
/// - `layers` - Every layer of the model, in the order that gives the model position of an
///   address
/// - `grads` - The gradient store of the pass
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok(())` when every gradient of the store reaches a parameter
///
/// # Errors
///
/// - [`Error::Computation`] - If the store holds an address that no parameter reads
pub(crate) fn check_every_gradient_is_claimed(
    layers: &mut [Box<dyn Layer>],
    grads: &Grads,
) -> Result<(), Error> {
    let mut reachable: Vec<ParamId> = Vec::new();
    walk_model_mut(layers, &mut |path, node| {
        for param in node.parameters_mut() {
            reachable.push(path.param(param.name));
        }
    });
    let claimed = reachable
        .iter()
        .filter(|id| grads.get(id).is_some())
        .count();
    if claimed == grads.len() {
        return Ok(());
    }

    let orphans: Vec<String> = grads
        .iter()
        .filter(|(id, _)| !reachable.contains(id))
        .map(|(id, _)| id.to_string())
        .collect();
    Err(Error::computation(format!(
        "the backward pass parked a gradient at {} address(es) that no parameter of the model \
         reads: {}. An optimizer reads a gradient at the address that \
         `LayerBase::parameters_mut` gives the parameter, so a gradient at any other address \
         updates nothing and the parameter it was meant for keeps its value. A layer must add \
         every gradient under a name that its own roster holds. A layer that holds sublayers \
         must call each of them inside `Ctx::sublayer` with the name that \
         `LayerBase::sublayers` gives it",
        orphans.len(),
        orphans.join(", ")
    )))
}

/// A single trainable parameter tensor of a layer, exposed as a flat slice
///
/// Layers yield their trainable tensors (weights, biases, kernels, gamma/beta, ...) as
/// `ParamRef`s. This lets optimizers update any parameter shape with 1 flat-slice kernel,
/// instead of every layer/optimizer pair re-implementing the update
///
/// The entry holds no gradient. A backward pass puts every gradient in the
/// [`Grads`] store of the context. The optimizer reads
/// it back with the [`ParamId`] that this name and the path of the layer build
///
/// Construct one with [`ParamRef::weight`] for a tensor that decoupled weight decay applies to
/// (weight matrices, conv/recurrent kernels). Use [`ParamRef::no_decay`] for a tensor it skips
/// (biases and normalization scale/shift `gamma`/`beta`). The `decays` flag tells the optimizer
/// which rule applies, so it never has to guess
///
/// The `name` is the half of the parameter address that the layer owns: `kernel`,
/// `recurrent_kernel`, `depthwise_kernel`, `bias`, `embeddings`, `alpha`, `gamma`, or `beta`. A
/// layer must give the same name to the same storage on every call, and must give 2 different
/// tensors 2 different names
pub struct ParamRef<'a> {
    /// Name the layer gives this tensor. See [`ParamId`]
    pub name: &'static str,
    /// Mutable view of the parameter's contiguous data that the optimizer updates in place
    pub value: &'a mut [f32],
    /// Whether decoupled (AdamW/SGDW-style) weight decay applies to this tensor. `true` for
    /// weight matrices and conv/recurrent kernels, `false` for biases and normalization
    /// scale/shift (`gamma`/`beta`)
    pub decays: bool,
}

impl<'a> ParamRef<'a> {
    /// A weight tensor that decoupled weight decay applies to (dense/conv/recurrent kernels)
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the tensor, such as `"kernel"`
    /// - `value` - Mutable view of the parameter's contiguous data
    ///
    /// # Returns
    ///
    /// - `ParamRef` - The entry, with `decays` set to `true`
    #[inline]
    pub fn weight(name: &'static str, value: &'a mut [f32]) -> Self {
        Self {
            name,
            value,
            decays: true,
        }
    }

    /// A bias or normalization scale/shift (`gamma`/`beta`) tensor that weight decay skips
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the tensor, such as `"bias"`
    /// - `value` - Mutable view of the parameter's contiguous data
    ///
    /// # Returns
    ///
    /// - `ParamRef` - The entry, with `decays` set to `false`
    #[inline]
    pub fn no_decay(name: &'static str, value: &'a mut [f32]) -> Self {
        Self {
            name,
            value,
            decays: false,
        }
    }
}

/// Whether training updates a named array of a layer
///
/// The 2 kinds are what [`ParamCounts`] counts, seen 1 array at a time.
/// A checkpoint holds the kind next to every array. A load can therefore refuse a file that
/// offers a trainable array where the layer keeps state, and the other way round
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightKind {
    /// An optimizer updates this array. It is a kernel, a bias, or a normalization
    /// scale/shift
    Trainable,
    /// The layer keeps this array and no optimizer writes it. The running statistics of
    /// [`BatchNormalization`](crate::neural_network::layers::BatchNormalization)
    /// are the 1 example today
    NonTrainable,
}

/// 1 named array of a layer, borrowed for reading
///
/// [`LayerBase::weights`] gives 1 of these per array that the layer holds. The name is the layer
/// half of the address of the array, and the checkpoint format writes it. See
/// [`checkpoint`](crate::neural_network::layers::checkpoint)
///
/// The view borrows the live array, so nothing is copied
pub struct WeightRef<'a> {
    /// Name the layer gives this array, such as `"kernel"` or `"moving_mean"`
    pub name: &'static str,
    /// Whether an optimizer updates the array
    pub kind: WeightKind,
    /// Read-only view of the array, at the rank the layer holds it
    pub value: ArrayViewD<'a, f32>,
}

impl<'a> WeightRef<'a> {
    /// A read view of an array that an optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightRef` - The entry, with the kind set to [`WeightKind::Trainable`]
    #[inline]
    pub fn trainable(name: &'static str, value: ArrayViewD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::Trainable,
            value,
        }
    }

    /// A read view of an array that no optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightRef` - The entry, with the kind set to [`WeightKind::NonTrainable`]
    #[inline]
    pub fn non_trainable(name: &'static str, value: ArrayViewD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::NonTrainable,
            value,
        }
    }
}

/// 1 named array of a layer, borrowed for writing
///
/// [`LayerBase::weights_mut`] gives 1 of these per array that the layer holds, under the same
/// names and in the same order as [`LayerBase::weights`]. A checkpoint load writes through these
/// views, so it reaches the storage of the layer itself and keeps the memory order of that
/// storage
pub struct WeightMut<'a> {
    /// Name the layer gives this array, such as `"kernel"` or `"moving_mean"`
    pub name: &'static str,
    /// Whether an optimizer updates the array
    pub kind: WeightKind,
    /// Writable view of the array, at the rank the layer holds it
    pub value: ArrayViewMutD<'a, f32>,
}

impl<'a> WeightMut<'a> {
    /// A write view of an array that an optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightMut` - The entry, with the kind set to [`WeightKind::Trainable`]
    #[inline]
    pub fn trainable(name: &'static str, value: ArrayViewMutD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::Trainable,
            value,
        }
    }

    /// A write view of an array that no optimizer updates
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the array
    /// - `value` - View of the array
    ///
    /// # Returns
    ///
    /// - `WeightMut` - The entry, with the kind set to [`WeightKind::NonTrainable`]
    #[inline]
    pub fn non_trainable(name: &'static str, value: ArrayViewMutD<'a, f32>) -> Self {
        Self {
            name,
            kind: WeightKind::NonTrainable,
            value,
        }
    }
}

/// How many inputs a layer takes
///
/// Almost every layer takes 1. A merge layer takes 1 or more, and gives 1 output. A model
/// reads this before it wires a layer. A node with the wrong fan-in is therefore refused at
/// build time and not in the middle of a pass
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// The layer takes exactly this many inputs
    Exactly(usize),
    /// The layer takes this many inputs or more
    AtLeast(usize),
}

impl Arity {
    /// Whether the layer accepts that many inputs
    ///
    /// # Parameters
    ///
    /// - `count` - How many inputs a caller offers
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when the layer accepts the count
    #[inline]
    pub const fn accepts(self, count: usize) -> bool {
        match self {
            Self::Exactly(n) => count == n,
            Self::AtLeast(n) => count >= n,
        }
    }

    /// Refuses a count of inputs that the layer does not accept
    ///
    /// # Parameters
    ///
    /// - `layer` - The type name of the layer, for the message
    /// - `count` - How many inputs a caller offers
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the layer accepts the count
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer takes another count
    pub fn check(self, layer: &str, count: usize) -> Result<(), Error> {
        if self.accepts(count) {
            return Ok(());
        }
        let wanted = match self {
            Self::Exactly(n) => format!("exactly {n}"),
            Self::AtLeast(n) => format!("{n} or more"),
        };
        Err(Error::invalid_input(format!(
            "layer `{layer}` takes {wanted} inputs, and it received {count}"
        )))
    }
}

/// What every layer holds, whatever number of inputs it takes
///
/// The trait covers the 4 things that do not depend on the arity of a layer. It covers what the
/// layer is, what arrays it owns, what sublayers it holds, and what shape it was built for. The
/// computation itself lives in [`UnaryLayer`] for a layer with 1 input and in [`Layer`] for a
/// layer with several
///
/// A layer holds no gradient and no cache. See [`Ctx`]
///
/// # A layer that holds other layers
///
/// A layer can hold other layers, which are its sublayers. Each sublayer is a full layer with
/// its own arrays, its own state, and its own sublayers. The rosters of this trait, such as
/// [`weights`](LayerBase::weights) and [`parameters_mut`](LayerBase::parameters_mut), give the
/// arrays of the layer itself and never the arrays of a sublayer. The model visits every
/// sublayer through [`sublayers`](LayerBase::sublayers) and
/// [`sublayers_mut`](LayerBase::sublayers_mut), and it gives each 1 its own
/// [`LayerPath`]. An optimizer, a checkpoint, and the state channel therefore reach every
/// sublayer with no help from the holding layer.
///
/// A layer that holds sublayers does these 4 things:
///
/// 1. List each sublayer in [`sublayers`](LayerBase::sublayers) and in
///    [`sublayers_mut`](LayerBase::sublayers_mut), under 1 fixed name.
/// 2. Call each sublayer inside [`Ctx::sublayer`], with the name of step 1. Do this in the
///    forward pass and in the backward pass.
/// 3. Build each sublayer in its own build.
/// 4. Report itself as built only when every sublayer is built.
///
/// A model build refuses a layer whose 2 rosters disagree. A training step refuses a gradient
/// at a path that no roster gives, and a state value that no node takes back. A mistake in
/// step 1 or step 2 therefore stops the model and does not train it wrong
///
/// # Examples
///
/// A layer that runs 2 [`Dense`](crate::neural_network::layers::Dense) layers in a chain:
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::error::Error;
/// use rustyml::neural_network::layers::{Activation, Dense, ParamCounts};
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::traits::{LayerBase, UnaryLayer, WeightMut, WeightRef};
/// use rustyml::neural_network::{Ctx, Shape, Sublayer, SublayerMut, Tensor};
///
/// struct TwoDense {
///     first: Dense,
///     second: Dense,
/// }
///
/// impl LayerBase for TwoDense {
///     fn layer_type(&self) -> &str {
///         "TwoDense"
///     }
///     fn param_count(&self) -> ParamCounts {
///         ParamCounts::none()
///     }
///     fn weights(&self) -> Vec<WeightRef<'_>> {
///         Vec::new()
///     }
///     fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
///         Vec::new()
///     }
///     fn is_built(&self) -> bool {
///         self.first.is_built() && self.second.is_built()
///     }
///     fn sublayers(&self) -> Vec<Sublayer<'_>> {
///         vec![Sublayer::new("first", &self.first), Sublayer::new("second", &self.second)]
///     }
///     fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
///         vec![
///             SublayerMut::new("first", &mut self.first),
///             SublayerMut::new("second", &mut self.second),
///         ]
///     }
/// }
///
/// impl UnaryLayer for TwoDense {
///     fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
///         let hidden = ctx.sublayer("first", |ctx| self.first.forward(input, ctx))?;
///         ctx.sublayer("second", |ctx| self.second.forward(&hidden, ctx))
///     }
///     fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
///         let grad_hidden = ctx.sublayer("second", |ctx| self.second.backward(grad_output, ctx))?;
///         ctx.sublayer("first", |ctx| self.first.backward(&grad_hidden, ctx))
///     }
///     fn build(&mut self, input: &Shape) -> Result<(), Error> {
///         self.first.build(input)?;
///         let hidden = self.first.compute_output_shape(input)?;
///         self.second.build(&hidden)
///     }
///     fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
///         self.second.compute_output_shape(&self.first.compute_output_shape(input)?)
///     }
/// }
///
/// let layer = TwoDense {
///     first: Dense::new(3, Activation::ReLU).unwrap(),
///     second: Dense::new(1, Activation::Linear).unwrap(),
/// };
/// let model = SequentialBuilder::new()
///     .add(layer)
///     .build(&Shape::known(&[2, 4]))
///     .unwrap();
///
/// // Each sublayer has its own addresses, so the 2 kernels never share 1 gradient
/// assert_eq!(
///     model.weight_paths(),
///     vec!["0.first.kernel", "0.first.bias", "0.second.kernel", "0.second.bias"]
/// );
/// let prediction = model.predict(&Array::ones((2, 4)).into_dyn()).unwrap();
/// assert_eq!(prediction.shape(), &[2, 1]);
/// ```
pub trait LayerBase: std::any::Any + Send + Sync {
    /// Returns the type name of the layer (e.g. "Dense")
    ///
    /// # Returns
    ///
    /// - `&str` - A string slice representing the layer type
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Returns how many parameters the layer holds, split by whether training updates them
    ///
    /// The count covers the arrays of the layer itself, and no array of a sublayer.
    /// [`total_param_count`](crate::neural_network::layer_path::total_param_count) adds the
    /// sublayers
    ///
    /// # Returns
    ///
    /// - `ParamCounts` - The trainable and the non-trainable element counts
    fn param_count(&self) -> ParamCounts;

    /// Exposes the layer's trainable parameters to the optimizer, by name
    ///
    /// Each returned [`ParamRef`] gives a parameter tensor's flat data and names the tensor.
    /// Layers without trainable parameters return the empty vector that the default gives.
    /// A layer yields every trainable tensor it holds on every call, whether a backward pass
    /// gave that tensor a gradient or not. The optimizer looks the gradient up in the store of
    /// the context and skips a tensor that holds none
    ///
    /// The name is the identity of the tensor, and the optimizer keys its per-parameter state
    /// on it (see [`ParamId`]). The same storage must therefore always come back under the same
    /// name, and 2 tensors of 1 layer must never share a name. A model build refuses a layer
    /// that breaks that rule, so the mistake never reaches a training step. The order is free,
    /// and it is the order that a global gradient norm reduces in, so a layer must keep it
    /// stable
    ///
    /// The roster holds the tensors of the layer itself, and no tensor of a sublayer. The model
    /// reaches each sublayer through [`sublayers_mut`](LayerBase::sublayers_mut)
    ///
    /// # Returns
    ///
    /// - `Vec<ParamRef<'_>>` - 1 named entry per trainable tensor
    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        Vec::new()
    }

    /// Every array the layer holds, by name, borrowed for reading
    ///
    /// The set covers the trainable arrays and the non-trainable state alike, which is exactly
    /// what a checkpoint holds. It is therefore a superset of what
    /// [`parameters_mut`](LayerBase::parameters_mut) yields:
    /// [`BatchNormalization`](crate::neural_network::layers::BatchNormalization)
    /// adds its running statistics here
    ///
    /// The name of an array is its address inside the layer: `kernel`, `recurrent_kernel`,
    /// `depthwise_kernel`, `pointwise_kernel`, `bias`, `embeddings`, `alpha`, `gamma`, `beta`,
    /// `moving_mean`, `moving_variance`. A layer must give 1 array the same name on every call,
    /// and must never give 2 arrays the same name.
    /// A model build refuses a layer that gives 2 arrays 1 name, over this roster and over
    /// [`parameters_mut`](LayerBase::parameters_mut) alike. The 2 lists are separate, and
    /// nothing else binds them. A name that `parameters_mut` also uses must reach the same
    /// storage
    ///
    /// The order is free, and it is the order a checkpoint records. Layers without any array
    /// return the empty vector
    ///
    /// The roster holds the arrays of the layer itself, and no array of a sublayer. A
    /// checkpoint reaches each sublayer through [`sublayers`](LayerBase::sublayers)
    ///
    /// # Returns
    ///
    /// - `Vec<WeightRef<'_>>` - 1 named view per array the layer holds
    fn weights(&self) -> Vec<WeightRef<'_>>;

    /// Every array the layer holds, by name, borrowed for writing
    ///
    /// The roster, the names, the kinds, and the order repeat [`weights`](LayerBase::weights)
    /// exactly. A checkpoint load looks a name up here and writes into the view. The values then
    /// reach the storage of the layer and take the memory order that the storage already has
    ///
    /// # Returns
    ///
    /// - `Vec<WeightMut<'_>>` - 1 named view per array the layer holds
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>>;

    /// 1 named array of the layer, or `None` when the layer holds no array of that name
    ///
    /// # Parameters
    ///
    /// - `name` - The name the layer gives the array, such as `"kernel"`
    ///
    /// # Returns
    ///
    /// - `Option<ArrayViewD<'_, f32>>` - A read view of the array
    fn weight(&self, name: &str) -> Option<ArrayViewD<'_, f32>> {
        self.weights()
            .into_iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.value)
    }

    /// The input shapes the layer holds, or `None` while it holds none
    ///
    /// A layer that [`build`](UnaryLayer::build) has run on reports the shapes it was built
    /// for, with the batch axis freed. The batch axis is freed because 1 layer serves every
    /// batch size. A layer built from a tensor of 2 samples therefore still describes itself
    /// for every batch. A layer that needs no build at all, such as
    /// [`Rescaling`](crate::neural_network::layers::rescaling::Rescaling), always reports
    /// `None`. The vector holds 1 shape per input of the layer, so a merge layer reports
    /// several
    ///
    /// [`Layer::output_shape`] runs
    /// [`Layer::compute_output_shape_many`] against these shapes. The method is therefore the
    /// 1 place where the display value reads state, and the shape algebra stays pure
    ///
    /// # Returns
    ///
    /// - `Option<Vec<Shape>>` - The input shapes the layer holds
    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        None
    }

    /// Whether the layer already holds every array it needs
    ///
    /// [`UnaryLayer::forward_mut`] reads this, and builds the layer from the tensor it
    /// received when the answer is `false`. A layer that owns no array and reads no extent of
    /// its input keeps the default `true`, because there is nothing left to allocate
    ///
    /// # Returns
    ///
    /// - `bool` - `true` when no build is outstanding
    fn is_built(&self) -> bool {
        true
    }

    /// The shapes that the layer was built for, when the layer knows them
    ///
    /// A layer that [`build`](UnaryLayer::build) has run on reports the shapes it was built
    /// for, with the batch axis freed. A layer that owns no array and reads no extent of its
    /// input keeps the default `None`, and so does any layer before its build. A checkpoint
    /// records this value, and a load compares it. See [`BuildConfig`]
    ///
    /// The batch axis is freed because 1 layer serves every batch size. A model built for 32
    /// samples and a model built for 1 sample therefore report the same build shape, and a
    /// checkpoint moves between them
    ///
    /// # Returns
    ///
    /// - `Option<BuildConfig>` - The build shapes, or `None` while the layer holds none
    fn build_config(&self) -> Option<BuildConfig> {
        None
    }

    /// Moves the non-trainable state that a forward pass proposed into the layer
    ///
    /// A forward pass takes `&self`, so a layer that changes non-trainable state writes the
    /// new value into the state channel of the context instead. The model calls this after the
    /// forward pass, and the layer takes back every value it recognizes.
    /// [`BatchNormalization`](crate::neural_network::layers::BatchNormalization)
    /// takes its running statistics here, and a dropout layer takes its random stream
    ///
    /// The model calls this once for each layer and each sublayer that has a value. The slot
    /// reaches the values of this layer alone, and never a value of a sublayer. If a value
    /// stays in the context after the walk, the training step of the model returns an error
    ///
    /// The default does nothing, which is right for every layer whose forward pass changes
    /// nothing outside the context
    ///
    /// # Parameters
    ///
    /// - `state` - The state channel of this layer, for the pass that just ran
    fn apply_state(&mut self, state: &mut StateSlot<'_>) {
        let _ = state;
    }

    /// Every layer that this layer holds, by name, borrowed for reading
    ///
    /// The name is the address of the sublayer inside this layer. A [`LayerPath`] appends it to
    /// the path of this layer, so the sublayer `forward` of the layer at position 2 has the
    /// path `2.forward`. A name must not be empty, must not hold a `.`, and must be unique
    /// among the sublayers of 1 layer. A layer must give 1 sublayer the same name on every
    /// call
    ///
    /// The order is the canonical order of the sublayers. The optimizer update, the global
    /// gradient norm, and the checkpoint all visit the sublayers in this order, so a layer must
    /// keep it stable
    ///
    /// The default is the empty roster, which is right for every layer that holds no other
    /// layer. See the [trait documentation](LayerBase) for what a layer that holds sublayers
    /// must do
    ///
    /// # Returns
    ///
    /// - `Vec<Sublayer<'_>>` - 1 named entry per sublayer
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        Vec::new()
    }

    /// Every layer that this layer holds, by name, borrowed for writing
    ///
    /// The roster, the names, and the order repeat [`sublayers`](LayerBase::sublayers)
    /// exactly, and each entry reaches the same layer. A model build refuses a layer whose 2
    /// rosters disagree
    ///
    /// # Returns
    ///
    /// - `Vec<SublayerMut<'_>>` - 1 named entry per sublayer
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        Vec::new()
    }
}

/// A layer that takes 1 input and gives 1 output
///
/// Almost every layer of this crate is one. Implement this trait, and the blanket
/// `impl<T: UnaryLayer> Layer for T` gives the layer the general [`Layer`] interface that a
/// model holds it through. Implement [`Layer`] itself only for a layer that takes several
/// inputs, such as a merge layer
///
/// The forward pass takes `&self`. Everything it needs to remember for its backward pass goes
/// into the [`Ctx`], and so does every gradient it produces.
/// A layer that changes non-trainable state writes it into the state channel of the context.
/// The layer itself is therefore read-only during a pass. A built
/// [`Sequential`](crate::neural_network::sequential::Sequential) is `Send` and `Sync` for the
/// same reason, so several threads can run inference against 1 model
pub trait UnaryLayer: LayerBase {
    /// Runs the forward pass through the layer
    ///
    /// A training pass parks in `ctx` whatever the backward pass needs. An inference pass
    /// parks nothing at all and takes the inference behavior of a mode-dependent layer. Read
    /// [`Ctx::is_training`](crate::neural_network::ctx::Ctx::is_training) to tell the 2 apart
    ///
    /// The method refuses an unbuilt layer. Use [`forward_mut`](UnaryLayer::forward_mut) to
    /// build from the tensor itself, or [`build`](UnaryLayer::build) first
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Tensor` - The layer's output, in the standard memory order
    ///
    /// # Errors
    ///
    /// - `Error` - If the forward pass fails (e.g. shape mismatch)
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer holds no build
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error>;

    /// Runs the backward pass through the layer
    ///
    /// The method takes back what the matching forward pass parked in `ctx`. It adds the
    /// gradient of every parameter it owns to the gradient store of `ctx`
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    /// - `ctx` - The context of the pass, holding what the forward pass parked
    ///
    /// # Returns
    ///
    /// - `Tensor` - The gradient to pass to the previous layer, in the standard memory order
    ///
    /// # Notes
    ///
    /// Backward is pure math: it does **not** sanitize NaN/Inf (no zeroing, no element-wise
    /// clamping). The backward pass propagates such values instead of masking them. The forward
    /// pass masks nothing either: it validates the rank, the shape, and the layer parameters. It
    /// never reads the input values to reject them. A NaN or an infinity therefore stays in the
    /// tensor and moves on through every later layer. It shows itself in the output and in a
    /// non-finite loss. [`Embedding`](crate::neural_network::layers::Embedding) is the 1
    /// exception, because it reads its input as a table of row indices and rejects a non-finite
    /// index with `Error::InvalidInput`. To tame large-but-finite gradients, enable
    /// clip-by-global-norm on the optimizer
    /// ([`Optimizer::global_clipnorm`]) instead of clamping inside a layer. Global-norm scaling
    /// preserves gradient direction, unlike per-element clamping
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer, because no forward pass of this layer wrote one
    /// - `Error` - If the layer encountered an error during processing (e.g. shape mismatch)
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error>;

    /// Allocates every array the layer owns, from the shape of the input it will receive
    ///
    /// A constructor takes the configuration of a layer and nothing else. It draws no weight,
    /// because a kernel extent comes from the input and the input is not there yet. `build` is
    /// where the layer learns that shape, checks it against its own configuration, and
    /// allocates. A layer that owns no array still records the shape, so it can check every
    /// later input against it and report an honest output shape
    ///
    /// [`SequentialBuilder::build`](crate::neural_network::sequential::SequentialBuilder::build)
    /// calls this once per layer, threading the output shape of each layer into the next
    /// through [`compute_output_shape`](UnaryLayer::compute_output_shape).
    /// [`forward_mut`](UnaryLayer::forward_mut) builds a layer that a caller drives directly,
    /// from the shape of the tensor it receives
    ///
    /// A second call with the same shape does nothing, and no array is drawn twice. A second
    /// call with another shape is an error, because it would silently replace every weight the
    /// layer holds
    ///
    /// The default does nothing. It is right for every layer that owns no array and reads no
    /// extent of its input, such as an activation
    ///
    /// # Parameters
    ///
    /// - `input` - Shape of the tensor that enters the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the layer holds every array it needs
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept an input of that shape, or if the
    ///   layer is already built for another shape
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let _ = input;
        Ok(())
    }

    /// The output shape the layer gives for an input of the given shape
    ///
    /// The answer is a pure function of the layer configuration and of `input`. The method
    /// reads no cache that a forward pass wrote. It therefore gives the same answer before any
    /// tensor reaches the layer and after any number of passes. A free axis of the input stays
    /// free in the output wherever the layer passes it through. That is how 1 layer describes
    /// itself for every batch size
    ///
    /// The method refuses an input the layer cannot accept, and the message names the layer
    /// and the axis at fault. A pure shape function lets a model walk its layers at build time
    /// and thread each output shape into the next layer. It also lets the model reject a bad
    /// stack before any data arrives, naming the position and the type of the layer at fault
    ///
    /// The default passes the input through unchanged, which is right for every layer that
    /// changes values and not extents
    ///
    /// # Parameters
    ///
    /// - `input` - Shape of the tensor that enters the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<Shape, Error>` - Shape of the tensor the layer gives back
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept an input of that shape
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        Ok(input.clone())
    }

    /// Builds the layer from the tensor when it holds no build, then runs the forward pass
    ///
    /// This is the entry point for a caller that drives 1 layer by hand and never calls
    /// [`build`](UnaryLayer::build). The whole shape of the tensor goes into the build, batch
    /// extent included, so the layer reports the shape it was really given. A model never uses
    /// this method, because
    /// [`SequentialBuilder::build`](crate::neural_network::sequential::SequentialBuilder::build)
    /// has already built every layer it holds
    ///
    /// The method also moves the non-trainable state that the pass proposed into the layer and
    /// into each of its sublayers, with [`LayerBase::apply_state`]. A model does that step
    /// itself, and a caller that drives 1 layer by hand has no other place for it. Without the
    /// step the random stream of a dropout layer never advances, and 2 calls draw the same mask
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The layer's output
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept an input of that shape
    /// - `Error` - If the forward pass fails
    fn forward_mut(&mut self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            self.build(&Shape::known(input.shape()))?;
        }
        let output = self.forward(input, ctx)?;
        // A forward pass cannot write state itself, so this entry point applies it here
        apply_own_state(self, ctx);
        Ok(output)
    }
}

/// Moves the state that a pass proposed into a layer and into each of its sublayers
///
/// The layer is the root of the tree at the current path of `ctx`. The 2 entry points that let
/// a caller drive 1 layer by hand call this. A model calls
/// [`apply_state_tree`] itself
fn apply_own_state<L: LayerBase + ?Sized>(layer: &mut L, ctx: &mut Ctx) {
    let path = ctx.layer_path();
    if ctx.has_state(&path) {
        layer.apply_state(&mut ctx.state_slot(&path));
    }
    for sub in layer.sublayers_mut() {
        apply_state_tree(sub.layer, &path.child(sub.name), ctx);
    }
}

/// A layer that takes 1 input or several, and gives 1 output
///
/// This is the general interface, and it is the one a model holds a layer through. A layer
/// with 1 input implements [`UnaryLayer`] instead, and the blanket
/// `impl<T: UnaryLayer> Layer for T` gives it this interface for free. Implement this trait by
/// hand only for a layer that takes several inputs
///
/// Every method that ends in `_many` is the same operation as the [`UnaryLayer`] method of the
/// same stem, over every input of the layer
pub trait Layer: LayerBase {
    /// How many inputs the layer takes
    ///
    /// # Returns
    ///
    /// - `Arity` - The accepted input count
    fn arity(&self) -> Arity {
        Arity::Exactly(1)
    }

    /// Runs the forward pass through the layer, over every input
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 tensor per input of the layer, in the order the model wired them
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Tensor` - The layer's output, in the standard memory order
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the count of `inputs` does not match [`arity`](Layer::arity)
    /// - `Error` - If the forward pass fails
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error>;

    /// Runs the backward pass through the layer, giving 1 gradient per input
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    /// - `ctx` - The context of the pass, holding what the forward pass parked
    ///
    /// # Returns
    ///
    /// - `Vec<Tensor>` - 1 gradient per input, in the order the forward pass took them
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer
    /// - `Error` - If the backward pass fails
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error>;

    /// Allocates every array the layer owns, from the shapes of the inputs it will receive
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 shape per input of the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the layer holds every array it needs
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept inputs of those shapes
    fn build_many(&mut self, inputs: &[Shape]) -> Result<(), Error>;

    /// The output shape the layer gives for inputs of the given shapes
    ///
    /// The answer is a pure function of the layer configuration and of `inputs`
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 shape per input of the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<Shape, Error>` - Shape of the tensor the layer gives back
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept inputs of those shapes
    fn compute_output_shape_many(&self, inputs: &[Shape]) -> Result<Shape, Error>;

    /// Builds the layer from the tensors when it holds no build, then runs the forward pass
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 tensor per input of the layer
    /// - `ctx` - The context of the pass
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The layer's output
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer cannot accept inputs of those shapes
    /// - `Error` - If the forward pass fails
    fn forward_many_mut(&mut self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            let shapes: Vec<Shape> = inputs.iter().map(|t| Shape::known(t.shape())).collect();
            self.build_many(&shapes)?;
        }
        let output = self.forward_many(inputs, ctx)?;
        // See [`UnaryLayer::forward_mut`]: this entry point completes the pass of a layer that
        // a caller drives by hand, by moving the proposed state into the layer
        apply_own_state(self, ctx);
        Ok(output)
    }

    /// Returns a description of the output shape of the layer
    ///
    /// The value is [`compute_output_shape_many`](Layer::compute_output_shape_many) run
    /// against [`known_input_shapes`](LayerBase::known_input_shapes). A layer that holds no
    /// input shape, and a layer whose held shapes the shape algebra refuses, both report
    /// `"Unknown"`
    ///
    /// # Returns
    ///
    /// - `String` - A string describing the output dimensions
    fn output_shape(&self) -> String {
        match self.known_input_shapes() {
            Some(inputs) => match self.compute_output_shape_many(&inputs) {
                Ok(output) => output.to_string(),
                Err(_) => "Unknown".to_string(),
            },
            None => "Unknown".to_string(),
        }
    }
}

impl<T: UnaryLayer> Layer for T {
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::Exactly(1).check(self.layer_type(), inputs.len())?;
        UnaryLayer::forward(self, inputs[0], ctx)
    }

    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        Ok(vec![UnaryLayer::backward(self, grad_output, ctx)?])
    }

    fn build_many(&mut self, inputs: &[Shape]) -> Result<(), Error> {
        Arity::Exactly(1).check(self.layer_type(), inputs.len())?;
        UnaryLayer::build(self, &inputs[0])
    }

    fn compute_output_shape_many(&self, inputs: &[Shape]) -> Result<Shape, Error> {
        Arity::Exactly(1).check(self.layer_type(), inputs.len())?;
        UnaryLayer::compute_output_shape(self, &inputs[0])
    }
}

/// Defines the interface for loss functions used in neural network training
///
/// An implementation computes both the loss value and its gradient with respect to
/// the predicted values
///
/// # Notes
///
/// Each loss normalizes by what is natural for its family, so the conventions differ on
/// purpose. `compute_grad` is always exactly the gradient of `compute_loss`. Switching loss
/// families rescales the gradient magnitude, and thus the effective learning rate:
///
/// - [`MeanSquaredError`](crate::neural_network::losses::MeanSquaredError),
///   [`MeanAbsoluteError`](crate::neural_network::losses::MeanAbsoluteError) and
///   [`BinaryCrossEntropy`](crate::neural_network::losses::BinaryCrossEntropy) average over
///   **every element** (`y.len()`), treating each output as an independent target
/// - [`CategoricalCrossEntropy`](crate::neural_network::losses::CategoricalCrossEntropy) sums
///   over the trailing **class** axis and averages over every **prediction site** (the product
///   of all leading axes). For a `[batch, classes]` target, that divisor is the batch. For the
///   `[batch, height, width, classes]` output of a channels-last softmax conv head, the divisor
///   is `batch * height * width`, 1 site per pixel
/// - [`SparseCategoricalCrossEntropy`][scce] is the same per-sample categorical cross-entropy,
///   but accepts only rank-2 `[batch, classes]` predictions, so its divisor is always the batch
///
/// The 2 categorical losses also renormalize `y_pred` along the class axis before
/// clipping when `from_logits` is off. That leaves the loss value alone for an
/// already-normalized head but contributes a row-constant term to the gradient, which a softmax
/// backward annihilates
///
/// [scce]: crate::neural_network::losses::SparseCategoricalCrossEntropy
pub trait Loss: Send + Sync {
    /// Computes the loss between true and predicted values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// - `f32` - The scalar loss value
    ///
    /// # Errors
    ///
    /// - `Error` - If the inputs are inconsistent (e.g. mismatched shapes or, for the
    ///   sparse loss, out-of-range labels)
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error>;

    /// Computes the gradient of the loss with respect to the predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// - `Tensor` - Tensor containing the gradient of the loss with respect to predictions
    ///
    /// # Errors
    ///
    /// - `Error` - If the inputs are inconsistent (see [`compute_loss`](Loss::compute_loss))
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error>;
}

/// Defines the interface for optimization algorithms
///
/// An implementation updates layer parameters during training
pub trait Optimizer: Send + Sync {
    /// Advances the optimizer's global training step
    ///
    /// Called exactly once per batch, before the per-layer [`update`](Optimizer::update) calls.
    /// A step-dependent optimizer advances the counter that its own math reads.
    /// [`Adam`](crate::neural_network::optimizers::Adam) and
    /// [`AdamW`](crate::neural_network::optimizers::AdamW) advance the bias-correction timestep
    /// here, so the correction moves once per batch rather than once per layer. SGD, RMSprop,
    /// and AdaGrad hold no such counter and keep the no-op default
    ///
    /// Per-parameter state is keyed by [`ParamId`], the path of the layer plus the name the
    /// layer gives the tensor. No cursor or call order matters to that address
    fn step(&mut self) {}

    /// The global gradient-norm clip threshold, or `None` (the default) to disable clipping
    ///
    /// When `Some(max_norm)`, the training loop computes the global L2 norm across **all** of
    /// the model's gradients. If the norm exceeds `max_norm`, it scales every gradient by
    /// `max_norm / global_norm` before [`update`](Optimizer::update). This single uniform factor
    /// preserves gradient direction (unlike per-element clamping). When the global norm is
    /// non-finite, the clip leaves gradients unscaled, so divergence still surfaces instead of
    /// being masked
    ///
    /// Only the global form of clipping exists here. A clip that renormalized each parameter's
    /// gradient independently against the threshold would give a different direction whenever
    /// more than 1 tensor is over it. A threshold tuned for 1 form would therefore not suit
    /// the other
    ///
    /// # Returns
    ///
    /// - `Option<f32>` - The clip threshold, or `None` when clipping is off
    fn global_clipnorm(&self) -> Option<f32> {
        None
    }

    /// Updates the parameters of 1 layer according to the optimization algorithm
    ///
    /// The optimizer builds a [`ParamId`] from `path` and the name of each [`ParamRef`], and
    /// keys its per-parameter state on that address. It reads the gradient of that address out
    /// of `grads`, and it skips a parameter that holds none. The caller must therefore give the
    /// same layer the same `path` on every step
    ///
    /// The call covers the parameters of `layer` itself, and no parameter of a sublayer. A model
    /// calls this once for every node of every layer tree, each with its own path
    ///
    /// # Parameters
    ///
    /// - `path` - The path of this layer in the model. It is the layer half of the parameter
    ///   address
    /// - `layer` - The layer whose parameters should be updated
    /// - `grads` - Every gradient the backward pass produced
    /// - `grad_scale` - Uniform factor that the training loop applies to every gradient before
    ///   the update, to implement clip-by-global-norm. Pass `1.0` for an unscaled update
    fn update(
        &mut self,
        path: &LayerPath,
        layer: &mut dyn LayerBase,
        grads: &Grads,
        grad_scale: f32,
    );

    /// The current learning rate
    ///
    /// The read half of the scheduling pair. A schedule (exponential decay, cosine annealing,
    /// warmup restarts) derives its next step size from the current one. It needs this method
    /// for that, rather than a copy of the rate kept alongside the model. A separate copy would
    /// drift the moment anything else retunes the optimizer. Reports whatever was last set.
    /// Unlike the constructors, [`set_learning_rate`](Optimizer::set_learning_rate) does not
    /// validate, so a rate set to 0 or to a negative value comes back unchanged
    ///
    /// # Returns
    ///
    /// - `f32` - The current learning rate
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate, the hook for external learning-rate scheduling
    ///
    /// Call this between batches or epochs, for step decay or warmup, to retune the step size
    /// without rebuilding the optimizer. The optimizer keeps its accumulated state
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - The new learning rate to use for subsequent updates
    fn set_learning_rate(&mut self, learning_rate: f32);
}
