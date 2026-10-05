//! The checkpoint and the build checks of a layer that holds other layers
//!
//! A composite layer lists its children in `LayerBase::sublayers` and in
//! `LayerBase::sublayers_mut` under fixed names. It calls each child inside `Ctx::sublayer`. The
//! checkpoint path of an array is `<scope>.<sub>.<sub>.<name>`.
//!
//! The tests below hold these rules:
//!
//! 1. `weight_paths` lists every array in the canonical pre-order. A node gives its own arrays
//!    first and then the arrays of each sublayer. `weight` reads every listed path, and it
//!    gives `None` for a path that reaches no array.
//! 2. A save and a load move every array of the tree, the running statistics included.
//! 3. A strict load refuses a file whose tree disagrees with the model. The message names the
//!    full path, and the model keeps every bit of every array.
//! 4. A lenient load reports each path of the tree in the right list.
//! 5. A file of format version 3 is refused.
//! 6. A model build refuses a tree that breaks an address rule, and the message names the path.
//! 7. `total_param_count` sums the tree, and `LayerBase::param_count` counts 1 node.

use ndarray::{Array, ArrayD, ArrayViewD, Axis, IxDyn};
use rustyml::error::{Error, IoError};
use rustyml::neural_network::graph::{Graph, GraphBuilder};
use rustyml::neural_network::layer_path::total_param_count;
use rustyml::neural_network::layers::checkpoint::{
    BuildConfig, LayerCheckpoint, LoadReport, MODEL_FORMAT_VERSION, ModelCheckpoint,
    SublayerCheckpoint,
};
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::{Activation, Concatenate, Dense, ParamCounts};
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::{Sequential, SequentialBuilder};
use rustyml::neural_network::traits::{
    LayerBase, ParamRef, UnaryLayer, WeightKind, WeightMut, WeightRef,
};
use rustyml::neural_network::{Ctx, Shape, Sublayer, SublayerMut, Tensor};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

// The composite layers of the tests

/// 2 Dense layers in a chain, under the names `first` and `second`
struct TwoDense {
    /// The first link of the chain
    first: Dense,
    /// The second link of the chain
    second: Dense,
}

impl TwoDense {
    /// A chain of a Tanh layer of `hidden` units and a linear layer of `units` units
    fn new(hidden: usize, units: usize, seed: u64) -> Self {
        Self {
            first: Dense::new(hidden, Activation::Tanh)
                .unwrap()
                .with_random_state(seed),
            second: Dense::new(units, Activation::Linear)
                .unwrap()
                .with_random_state(seed + 1),
        }
    }
}

impl LayerBase for TwoDense {
    fn layer_type(&self) -> &str {
        "TwoDense"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.first.is_built() && self.second.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("first", &self.first),
            Sublayer::new("second", &self.second),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("first", &mut self.first),
            SublayerMut::new("second", &mut self.second),
        ]
    }
}

impl UnaryLayer for TwoDense {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("first", &self.first, |ctx| self.first.forward(input, ctx))?;
        ctx.sublayer("second", &self.second, |ctx| {
            self.second.forward(&hidden, ctx)
        })
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("second", &self.second, |ctx| {
            self.second.backward(grad, ctx)
        })?;
        ctx.sublayer("first", &self.first, |ctx| self.first.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.first.build(input)?;
        let hidden = self.first.compute_output_shape(input)?;
        self.second.build(&hidden)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.second
            .compute_output_shape(&self.first.compute_output_shape(input)?)
    }
}

/// A Dense layer and a BatchNormalization layer, under the names `dense` and `norm`
///
/// The `norm` sublayer holds non-trainable running statistics
struct DenseNorm {
    /// The projection
    dense: Dense,
    /// The normalization after the projection
    norm: BatchNormalization,
}

impl DenseNorm {
    /// A projection to `units` units, and a normalization of the result
    fn new(units: usize, seed: u64) -> Self {
        Self {
            dense: Dense::new(units, Activation::Linear)
                .unwrap()
                .with_random_state(seed),
            norm: BatchNormalization::new(0.5, 1e-5).unwrap(),
        }
    }
}

impl LayerBase for DenseNorm {
    fn layer_type(&self) -> &str {
        "DenseNorm"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.dense.is_built() && self.norm.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("dense", &self.dense),
            Sublayer::new("norm", &self.norm),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("dense", &mut self.dense),
            SublayerMut::new("norm", &mut self.norm),
        ]
    }
}

impl UnaryLayer for DenseNorm {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("dense", &self.dense, |ctx| self.dense.forward(input, ctx))?;
        ctx.sublayer("norm", &self.norm, |ctx| self.norm.forward(&hidden, ctx))
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("norm", &self.norm, |ctx| self.norm.backward(grad, ctx))?;
        ctx.sublayer("dense", &self.dense, |ctx| self.dense.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.dense.build(input)?;
        let hidden = self.dense.compute_output_shape(input)?;
        self.norm.build(&hidden)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.norm
            .compute_output_shape(&self.dense.compute_output_shape(input)?)
    }
}

/// A layer with its own array `scale` and 1 sublayer `inner`
///
/// The output is the output of `inner`, times `scale` per feature
struct ScaledDense {
    /// The per-feature factor, of shape `[1, units]`
    scale: ArrayD<f32>,
    /// The projection before the factor
    inner: Dense,
}

impl ScaledDense {
    /// A projection to `units` units, and a factor per unit that depends on `seed`
    fn new(units: usize, seed: u64) -> Self {
        let scale = Array::from_shape_fn(IxDyn(&[1, units]), |index| {
            1.0 + 0.01 * (seed as f32) + 0.1 * (index[1] as f32)
        });
        Self {
            scale,
            inner: Dense::new(units, Activation::Linear)
                .unwrap()
                .with_random_state(seed),
        }
    }
}

impl LayerBase for ScaledDense {
    fn layer_type(&self) -> &str {
        "ScaledDense"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(self.scale.len())
    }
    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        vec![ParamRef::weight(
            "scale",
            self.scale.as_slice_mut().expect("scale is contiguous"),
        )]
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        vec![WeightRef::trainable("scale", self.scale.view())]
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        vec![WeightMut::trainable("scale", self.scale.view_mut())]
    }
    fn is_built(&self) -> bool {
        self.inner.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![Sublayer::new("inner", &self.inner)]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![SublayerMut::new("inner", &mut self.inner)]
    }
}

impl UnaryLayer for ScaledDense {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("inner", &self.inner, |ctx| self.inner.forward(input, ctx))?;
        let output = &hidden * &self.scale;
        if ctx.is_training() {
            ctx.push_cache("ScaledDense", hidden);
        }
        Ok(output)
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden: Tensor = ctx.pop_cache("ScaledDense")?;
        let grad_scale = (grad * &hidden).sum_axis(Axis(0)).insert_axis(Axis(0));
        ctx.add_grad("scale", grad_scale)?;
        let grad_hidden = grad * &self.scale;
        ctx.sublayer("inner", &self.inner, |ctx| {
            self.inner.backward(&grad_hidden, ctx)
        })
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.inner.build(input)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.inner.compute_output_shape(input)
    }
}

/// A tree of depth 2: a `TwoDense` under the name `pair`, and a Dense under the name `tail`
struct Deep {
    /// The composite child
    pair: TwoDense,
    /// The plain child
    tail: Dense,
}

impl Deep {
    /// A pair of 3 then 2 units, and a tail of `units` units
    fn new(units: usize, seed: u64) -> Self {
        Self {
            pair: TwoDense::new(3, 2, seed),
            tail: Dense::new(units, Activation::Linear)
                .unwrap()
                .with_random_state(seed + 7),
        }
    }
}

impl LayerBase for Deep {
    fn layer_type(&self) -> &str {
        "Deep"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.pair.is_built() && self.tail.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("pair", &self.pair),
            Sublayer::new("tail", &self.tail),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("pair", &mut self.pair),
            SublayerMut::new("tail", &mut self.tail),
        ]
    }
}

impl UnaryLayer for Deep {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("pair", &self.pair, |ctx| self.pair.forward(input, ctx))?;
        ctx.sublayer("tail", &self.tail, |ctx| self.tail.forward(&hidden, ctx))
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("tail", &self.tail, |ctx| self.tail.backward(grad, ctx))?;
        ctx.sublayer("pair", &self.pair, |ctx| self.pair.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.pair.build(input)?;
        let hidden = self.pair.compute_output_shape(input)?;
        self.tail.build(&hidden)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.tail
            .compute_output_shape(&self.pair.compute_output_shape(input)?)
    }
}

/// A chain of Dense layers whose names the caller gives at run time
///
/// Each name is an owned `String`, so each roster entry holds a `Cow::Owned`
struct Stack {
    /// The name of each cell, in chain order
    names: Vec<String>,
    /// The cells, in chain order
    cells: Vec<Dense>,
}

impl Stack {
    /// 1 cell of `units` units per name
    fn new(names: &[&str], units: usize, seed: u64) -> Self {
        Self {
            names: names.iter().map(|name| name.to_string()).collect(),
            cells: (0..names.len() as u64)
                .map(|index| {
                    Dense::new(units, Activation::Tanh)
                        .unwrap()
                        .with_random_state(seed + 10 * index)
                })
                .collect(),
        }
    }
}

impl LayerBase for Stack {
    fn layer_type(&self) -> &str {
        "Stack"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.cells.iter().all(Dense::is_built)
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        self.names
            .iter()
            .zip(&self.cells)
            .map(|(name, cell)| Sublayer::new(Cow::Owned(name.clone()), cell))
            .collect()
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        self.names
            .iter()
            .zip(self.cells.iter_mut())
            .map(|(name, cell)| SublayerMut::new(Cow::Owned(name.clone()), cell))
            .collect()
    }
}

impl UnaryLayer for Stack {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let mut value = input.clone();
        for (name, cell) in self.names.iter().zip(&self.cells) {
            value = ctx.sublayer(name.clone(), cell, |ctx| cell.forward(&value, ctx))?;
        }
        Ok(value)
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let mut grad = grad.clone();
        for (name, cell) in self.names.iter().zip(&self.cells).rev() {
            grad = ctx.sublayer(name.clone(), cell, |ctx| cell.backward(&grad, ctx))?;
        }
        Ok(grad)
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let mut shape = input.clone();
        for cell in &mut self.cells {
            cell.build(&shape)?;
            shape = cell.compute_output_shape(&shape)?;
        }
        Ok(())
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        let mut shape = input.clone();
        for cell in &self.cells {
            shape = cell.compute_output_shape(&shape)?;
        }
        Ok(shape)
    }
}

/// A layer that holds 1 other layer under the name `inner`
///
/// The build tests put a faulty layer inside it, so each message names a path of depth 1 or
/// more. The struct has 1 field, so it shares its address with `inner`
struct Holder<L> {
    /// The held layer
    inner: L,
}

impl<L: UnaryLayer> LayerBase for Holder<L> {
    fn layer_type(&self) -> &str {
        "Holder"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.inner.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![Sublayer::new("inner", &self.inner)]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![SublayerMut::new("inner", &mut self.inner)]
    }
}

impl<L: UnaryLayer> UnaryLayer for Holder<L> {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        ctx.sublayer("inner", &self.inner, |ctx| self.inner.forward(input, ctx))
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        ctx.sublayer("inner", &self.inner, |ctx| self.inner.backward(grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.inner.build(input)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.inner.compute_output_shape(input)
    }
}

/// The fault that a `Faulty` layer puts in its 2 sublayer rosters
#[derive(Debug, Clone, Copy)]
enum Fault {
    /// Both rosters give the 2 sublayers the name `x`
    DuplicateName,
    /// Both rosters give the first sublayer an empty name
    EmptyName,
    /// Both rosters give the first sublayer the name `a.b`
    DottedName,
    /// The write roster gives the 2 sublayers in the other order
    SwappedOrder,
    /// The write roster calls the second sublayer `c` and not `b`
    RenamedInMut,
    /// The write roster leaves out the second sublayer
    ShorterInMut,
    /// The write roster gives the names in order, and points each name at the other layer
    CrossedTargetsInMut,
    /// The read roster lists the first sublayer under 2 names
    SameChildTwice,
}

/// A layer with 2 Dense sublayers whose rosters hold 1 fault
struct Faulty {
    /// The fault in the rosters
    fault: Fault,
    /// The sublayer that a correct roster calls `a`
    a: Dense,
    /// The sublayer that a correct roster calls `b`
    b: Dense,
}

impl Faulty {
    /// 2 Dense layers of 4 units, with the given fault
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            a: Dense::new(4, Activation::Linear)
                .unwrap()
                .with_random_state(1),
            b: Dense::new(4, Activation::Linear)
                .unwrap()
                .with_random_state(2),
        }
    }
}

impl LayerBase for Faulty {
    fn layer_type(&self) -> &str {
        "Faulty"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.a.is_built() && self.b.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        let (a, b) = (&self.a, &self.b);
        match self.fault {
            Fault::DuplicateName => vec![Sublayer::new("x", a), Sublayer::new("x", b)],
            Fault::EmptyName => vec![Sublayer::new("", a), Sublayer::new("b", b)],
            Fault::DottedName => vec![Sublayer::new("a.b", a), Sublayer::new("b", b)],
            Fault::SameChildTwice => vec![Sublayer::new("a", a), Sublayer::new("b", a)],
            Fault::SwappedOrder
            | Fault::RenamedInMut
            | Fault::ShorterInMut
            | Fault::CrossedTargetsInMut => {
                vec![Sublayer::new("a", a), Sublayer::new("b", b)]
            }
        }
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        let (a, b) = (&mut self.a, &mut self.b);
        match self.fault {
            Fault::DuplicateName => vec![SublayerMut::new("x", a), SublayerMut::new("x", b)],
            Fault::EmptyName => vec![SublayerMut::new("", a), SublayerMut::new("b", b)],
            Fault::DottedName => vec![SublayerMut::new("a.b", a), SublayerMut::new("b", b)],
            Fault::SwappedOrder => vec![SublayerMut::new("b", b), SublayerMut::new("a", a)],
            Fault::RenamedInMut => vec![SublayerMut::new("a", a), SublayerMut::new("c", b)],
            Fault::ShorterInMut => vec![SublayerMut::new("a", a)],
            Fault::CrossedTargetsInMut => {
                vec![SublayerMut::new("a", b), SublayerMut::new("b", a)]
            }
            Fault::SameChildTwice => vec![SublayerMut::new("a", a), SublayerMut::new("b", b)],
        }
    }
}

impl UnaryLayer for Faulty {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("a", &self.a, |ctx| self.a.forward(input, ctx))?;
        ctx.sublayer("b", &self.b, |ctx| self.b.forward(&hidden, ctx))
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("b", &self.b, |ctx| self.b.backward(grad, ctx))?;
        ctx.sublayer("a", &self.a, |ctx| self.a.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.a.build(input)?;
        self.b.build(input)
    }
}

/// A layer that lists 1 Dense at depth 2, under `pair.first`, and again at depth 1, under
/// `alias`
struct Aliased {
    /// The composite child, whose `first` sublayer the read roster lists a second time
    pair: TwoDense,
    /// The layer that the write roster gives under `alias`
    other: Dense,
}

impl LayerBase for Aliased {
    fn layer_type(&self) -> &str {
        "Aliased"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.pair.is_built() && self.other.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("pair", &self.pair),
            Sublayer::new("alias", &self.pair.first),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("pair", &mut self.pair),
            SublayerMut::new("alias", &mut self.other),
        ]
    }
}

impl UnaryLayer for Aliased {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("pair", &self.pair, |ctx| self.pair.forward(input, ctx))?;
        ctx.sublayer("alias", &self.other, |ctx| self.other.forward(&hidden, ctx))
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("alias", &self.other, |ctx| self.other.backward(grad, ctx))?;
        ctx.sublayer("pair", &self.pair, |ctx| self.pair.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.pair.build(input)?;
        let hidden = self.pair.compute_output_shape(input)?;
        self.other.build(&hidden)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.other
            .compute_output_shape(&self.pair.compute_output_shape(input)?)
    }
}

/// A layer that gives its 1 array the name `w.x`
struct DottedArray {
    /// The array
    w: ArrayD<f32>,
}

impl LayerBase for DottedArray {
    fn layer_type(&self) -> &str {
        "DottedArray"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(self.w.len())
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        vec![WeightRef::trainable("w.x", self.w.view())]
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        vec![WeightMut::trainable("w.x", self.w.view_mut())]
    }
}

impl UnaryLayer for DottedArray {
    fn forward(&self, input: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(input.clone())
    }
    fn backward(&self, grad: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(grad.clone())
    }
}

/// A zero-sized layer that passes its input through
struct Pass;

impl LayerBase for Pass {
    fn layer_type(&self) -> &str {
        "Pass"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
}

impl UnaryLayer for Pass {
    fn forward(&self, input: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(input.clone())
    }
    fn backward(&self, grad: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(grad.clone())
    }
}

/// A layer with 2 zero-sized sublayers and 1 Dense
///
/// `repr(C)` puts every field at offset 0, because the 2 `Pass` fields take no space. The 2
/// `Pass` sublayers therefore share 1 address and 1 type
#[repr(C)]
struct TwinPass {
    /// The zero-sized sublayer before the projection
    left: Pass,
    /// The zero-sized sublayer after the projection
    right: Pass,
    /// The projection
    dense: Dense,
}

impl LayerBase for TwinPass {
    fn layer_type(&self) -> &str {
        "TwinPass"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.dense.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("left", &self.left),
            Sublayer::new("dense", &self.dense),
            Sublayer::new("right", &self.right),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("left", &mut self.left),
            SublayerMut::new("dense", &mut self.dense),
            SublayerMut::new("right", &mut self.right),
        ]
    }
}

impl UnaryLayer for TwinPass {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let value = ctx.sublayer("left", &self.left, |ctx| self.left.forward(input, ctx))?;
        let value = ctx.sublayer("dense", &self.dense, |ctx| self.dense.forward(&value, ctx))?;
        ctx.sublayer("right", &self.right, |ctx| self.right.forward(&value, ctx))
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("right", &self.right, |ctx| self.right.backward(grad, ctx))?;
        let grad = ctx.sublayer("dense", &self.dense, |ctx| self.dense.backward(&grad, ctx))?;
        ctx.sublayer("left", &self.left, |ctx| self.left.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.dense.build(input)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.dense.compute_output_shape(input)
    }
}

/// A layer whose first field is itself a layer
///
/// `repr(C)` puts `lead` at offset 0, so the struct and `lead` share 1 address
#[repr(C)]
struct Lead {
    /// The first sublayer, at the address of the struct
    lead: Dense,
    /// The second sublayer
    follow: Dense,
}

impl LayerBase for Lead {
    fn layer_type(&self) -> &str {
        "Lead"
    }
    fn param_count(&self) -> ParamCounts {
        ParamCounts::none()
    }
    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }
    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
    fn is_built(&self) -> bool {
        self.lead.is_built() && self.follow.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("lead", &self.lead),
            Sublayer::new("follow", &self.follow),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("lead", &mut self.lead),
            SublayerMut::new("follow", &mut self.follow),
        ]
    }
}

impl UnaryLayer for Lead {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("lead", &self.lead, |ctx| self.lead.forward(input, ctx))?;
        ctx.sublayer("follow", &self.follow, |ctx| {
            self.follow.forward(&hidden, ctx)
        })
    }
    fn backward(&self, grad: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("follow", &self.follow, |ctx| {
            self.follow.backward(grad, ctx)
        })?;
        ctx.sublayer("lead", &self.lead, |ctx| self.lead.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.lead.build(input)?;
        let hidden = self.lead.compute_output_shape(input)?;
        self.follow.build(&hidden)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.follow
            .compute_output_shape(&self.lead.compute_output_shape(input)?)
    }
}

// Helpers

/// The batch and the feature count of every model input
const INPUT: [usize; 2] = [4, 4];

/// Temporary file that deletes itself when dropped
struct TempFile(PathBuf);

impl TempFile {
    /// A path for a new temporary file, under a name unique to the test and the process
    fn new(name: &str) -> Self {
        TempFile(std::env::temp_dir().join(format!(
            "rustyml_sublayers_test_{}_{name}.bin",
            std::process::id()
        )))
    }

    /// The path of the temporary file
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The checkpoint surface that `Sequential` and `Graph` share
trait Checkpointed {
    fn paths(&self) -> Vec<String>;
    fn array(&self, path: &str) -> Option<ArrayViewD<'_, f32>>;
    fn save(&self, path: &Path);
    fn load(&mut self, path: &Path) -> Result<(), Error>;
    fn load_partial(&mut self, path: &Path) -> Result<LoadReport, Error>;
}

impl Checkpointed for Sequential {
    fn paths(&self) -> Vec<String> {
        self.weight_paths()
    }
    fn array(&self, path: &str) -> Option<ArrayViewD<'_, f32>> {
        self.weight(path)
    }
    fn save(&self, path: &Path) {
        self.save_to_path(path).unwrap();
    }
    fn load(&mut self, path: &Path) -> Result<(), Error> {
        self.load_from_path(path)
    }
    fn load_partial(&mut self, path: &Path) -> Result<LoadReport, Error> {
        self.load_partial_from_path(path)
    }
}

impl Checkpointed for Graph {
    fn paths(&self) -> Vec<String> {
        self.weight_paths()
    }
    fn array(&self, path: &str) -> Option<ArrayViewD<'_, f32>> {
        self.weight(path)
    }
    fn save(&self, path: &Path) {
        self.save_to_path(path).unwrap();
    }
    fn load(&mut self, path: &Path) -> Result<(), Error> {
        self.load_from_path(path)
    }
    fn load_partial(&mut self, path: &Path) -> Result<LoadReport, Error> {
        self.load_partial_from_path(path)
    }
}

/// Every array of a model, as a path and the raw f32 bits of its elements
fn all_bits(model: &dyn Checkpointed) -> Vec<(String, Vec<u32>)> {
    model
        .paths()
        .into_iter()
        .map(|path| {
            let bits = model
                .array(&path)
                .unwrap_or_else(|| panic!("the model holds no array at `{path}`"))
                .iter()
                .map(|value| value.to_bits())
                .collect();
            (path, bits)
        })
        .collect()
}

/// The raw f32 bits of 1 array of a model
fn bits_at(model: &dyn Checkpointed, path: &str) -> Vec<u32> {
    model
        .array(path)
        .unwrap_or_else(|| panic!("the model holds no array at `{path}`"))
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// The raw f32 bits of 1 tensor
fn tensor_bits(tensor: &Tensor) -> Vec<u32> {
    tensor.iter().map(|value| value.to_bits()).collect()
}

/// A deterministic tensor of the given shape
fn data(shape: &[usize], phase: f32) -> Tensor {
    let count: usize = shape.iter().product();
    let values = (0..count)
        .map(|flat| (flat as f32 * 0.37 + phase).sin())
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).unwrap()
}

/// Reads a checkpoint file, applies `edit` to it, and writes it back
fn rewrite(path: &Path, edit: impl FnOnce(&mut ModelCheckpoint<'static>)) {
    let bytes = std::fs::read(path).unwrap();
    let mut file: ModelCheckpoint<'static> = postcard::from_bytes(&bytes).unwrap();
    edit(&mut file);
    std::fs::write(path, postcard::to_allocvec(&file).unwrap()).unwrap();
}

/// A sequential model with 1 composite of each kind
///
/// Position 0 is a `DenseNorm`, position 1 a `Stack` with run-time names, position 2 a
/// `ScaledDense`, and position 3 a `Deep`
fn rich_sequential(seed: u64) -> Sequential {
    SequentialBuilder::new()
        .add(DenseNorm::new(4, seed))
        .add(Stack::new(&["cell0", "cell1"], 4, seed + 100))
        .add(ScaledDense::new(3, seed + 200))
        .add(Deep::new(1, seed + 300))
        .build(&Shape::known(&INPUT))
        .unwrap()
}

/// The canonical path list of [`rich_sequential`]
const RICH_SEQUENTIAL_PATHS: [&str; 19] = [
    "0.dense.kernel",
    "0.dense.bias",
    "0.norm.gamma",
    "0.norm.beta",
    "0.norm.moving_mean",
    "0.norm.moving_variance",
    "1.cell0.kernel",
    "1.cell0.bias",
    "1.cell1.kernel",
    "1.cell1.bias",
    "2.scale",
    "2.inner.kernel",
    "2.inner.bias",
    "3.pair.first.kernel",
    "3.pair.first.bias",
    "3.pair.second.kernel",
    "3.pair.second.bias",
    "3.tail.kernel",
    "3.tail.bias",
];

/// A graph with 2 inlets, a `TwoDense` that 2 nodes share, and 3 more composites
///
/// The arena holds the shared `TwoDense` at 0, a `Concatenate` at 1, a `DenseNorm` at 2, a
/// `ScaledDense` at 3, and a `Deep` at 4
fn rich_graph(seed: u64) -> Graph {
    let mut builder = GraphBuilder::new();
    let left = builder.input(Shape::known(&INPUT));
    let right = builder.input(Shape::known(&INPUT));
    let shared = builder.layer(TwoDense::new(3, 2, seed));
    let a = builder.apply(shared, &[left]);
    let b = builder.apply(shared, &[right]);
    let joined = builder.add(Concatenate::new(-1), &[a, b]);
    let normed = builder.add(DenseNorm::new(3, seed + 100), &[joined]);
    let scaled = builder.add(ScaledDense::new(3, seed + 200), &[normed]);
    let out = builder.add(Deep::new(1, seed + 300), &[scaled]);
    builder.build(&[out]).unwrap()
}

/// The canonical path list of [`rich_graph`]
const RICH_GRAPH_PATHS: [&str; 19] = [
    "0.first.kernel",
    "0.first.bias",
    "0.second.kernel",
    "0.second.bias",
    "2.dense.kernel",
    "2.dense.bias",
    "2.norm.gamma",
    "2.norm.beta",
    "2.norm.moving_mean",
    "2.norm.moving_variance",
    "3.scale",
    "3.inner.kernel",
    "3.inner.bias",
    "4.pair.first.kernel",
    "4.pair.first.bias",
    "4.pair.second.kernel",
    "4.pair.second.bias",
    "4.tail.kernel",
    "4.tail.bias",
];

/// The 2 model kinds that every checkpoint test runs on
#[derive(Debug, Clone, Copy)]
enum Kind {
    Sequential,
    Graph,
}

const KINDS: [Kind; 2] = [Kind::Sequential, Kind::Graph];

/// A model that holds 1 `TwoDense` at position 0, of the given kind
///
/// The paths are `0.first.kernel`, `0.first.bias`, `0.second.kernel`, and `0.second.bias`
fn pair_model(kind: Kind, seed: u64) -> Box<dyn Checkpointed> {
    match kind {
        Kind::Sequential => Box::new(
            SequentialBuilder::new()
                .add(TwoDense::new(3, 2, seed))
                .build(&Shape::known(&INPUT))
                .unwrap(),
        ),
        Kind::Graph => {
            let mut builder = GraphBuilder::new();
            let input = builder.input(Shape::known(&INPUT));
            let out = builder.add(TwoDense::new(3, 2, seed), &[input]);
            Box::new(builder.build(&[out]).unwrap())
        }
    }
}

/// Saves a pair model, edits the file, and checks that a strict load refuses it
///
/// The file carries other values in every record, so a write would show. Each needle must be
/// in the message, and the model must keep every bit of every array
fn assert_strict_refusal(
    name: &str,
    edit: impl Fn(&mut ModelCheckpoint<'static>),
    needles: &[&str],
) {
    for kind in KINDS {
        let tmp = TempFile::new(&format!("{name}_{kind:?}"));
        let source = pair_model(kind, 100);
        source.save(tmp.path());
        // Every record moves away from the model, so a write would show in every array
        rewrite(tmp.path(), |file| {
            for layer in &mut file.layers {
                shift_records(layer);
            }
        });
        rewrite(tmp.path(), &edit);

        let mut model = pair_model(kind, 200);
        let before = all_bits(&*model);

        match model.load(tmp.path()) {
            Err(Error::Io(IoError::ModelStructureMismatch(message))) => {
                for needle in needles {
                    assert!(
                        message.contains(needle),
                        "{kind:?}: the refusal must hold {needle:?}, got {message:?}"
                    );
                }
            }
            other => panic!("{kind:?}: expected ModelStructureMismatch, got {other:?}"),
        }
        assert_eq!(
            all_bits(&*model),
            before,
            "{kind:?}: the refused load changed an array"
        );
    }
}

/// Saves a pair model, edits the file, and checks the report of a lenient load
///
/// The file carries other values in every record. Each path in `applied` must then hold the
/// value of the file. Each path in `missing` must keep its own value
fn assert_partial_report(
    name: &str,
    edit: impl Fn(&mut ModelCheckpoint<'static>),
    applied: &[&str],
    missing: &[&str],
    unused: &[&str],
) {
    for kind in KINDS {
        let tmp = TempFile::new(&format!("{name}_{kind:?}"));
        let source = pair_model(kind, 100);
        source.save(tmp.path());
        // Every record moves away from the model, so a write would show in every array
        rewrite(tmp.path(), |file| {
            for layer in &mut file.layers {
                shift_records(layer);
            }
        });
        rewrite(tmp.path(), &edit);

        let mut model = pair_model(kind, 200);
        let before = all_bits(&*model);
        let report = model.load_partial(tmp.path()).unwrap();
        assert_eq!(report.applied, applied, "{kind:?}: applied");
        assert_eq!(report.missing, missing, "{kind:?}: missing");
        assert_eq!(report.unused, unused, "{kind:?}: unused");

        for path in applied {
            let shifted: Vec<u32> = source
                .array(path)
                .unwrap()
                .iter()
                .map(|value| (value + 0.5).to_bits())
                .collect();
            assert_eq!(
                bits_at(&*model, path),
                shifted,
                "{kind:?}: `{path}` must hold the value of the file"
            );
        }
        for path in missing {
            let own = &before.iter().find(|(p, _)| p == path).unwrap().1;
            assert_eq!(
                &bits_at(&*model, path),
                own,
                "{kind:?}: `{path}` must keep its own value"
            );
        }
    }
}

/// Adds 0.5 to every element of every record of a layer and of its sublayers
fn shift_records(layer: &mut LayerCheckpoint<'static>) {
    for record in &mut layer.weights {
        for value in record.data.to_mut() {
            *value += 0.5;
        }
    }
    for sub in &mut layer.sublayers {
        shift_records(&mut sub.layer);
    }
}

/// The second sublayer record of position 0 of a file
fn second<'a>(file: &'a mut ModelCheckpoint<'static>) -> &'a mut SublayerCheckpoint<'static> {
    let record = &mut file.layers[0].sublayers[1];
    assert_eq!(record.name, "second");
    record
}

/// Builds a sequential model and a graph model that hold `make()` at position 1 under a
/// `Holder`, and checks that each build refuses with every needle in the message
fn assert_build_refused<L: UnaryLayer + 'static>(make: impl Fn() -> L, needles: &[&str]) {
    let sequential = SequentialBuilder::new()
        .add(Dense::new(4, Activation::Linear).unwrap())
        .add(Holder { inner: make() })
        .build(&Shape::known(&INPUT));
    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&INPUT));
    let hidden = builder.add(Dense::new(4, Activation::Linear).unwrap(), &[input]);
    let out = builder.add(Holder { inner: make() }, &[hidden]);
    let graph = builder.build(&[out]);

    for (kind, result) in [("Sequential", sequential.err()), ("Graph", graph.err())] {
        match result {
            Some(Error::InvalidInput(message)) => {
                for needle in needles {
                    assert!(
                        message.contains(needle),
                        "{kind}: the refusal must hold {needle:?}, got {message:?}"
                    );
                }
            }
            other => panic!("{kind}: expected an InvalidInput refusal, got {other:?}"),
        }
    }
}

/// Builds a sequential model and a graph model that hold `make()` at position 1 under a
/// `Holder`, and gives both models
fn build_both<L: UnaryLayer + 'static>(make: impl Fn() -> L) -> (Sequential, Graph) {
    let sequential = SequentialBuilder::new()
        .add(Dense::new(4, Activation::Linear).unwrap())
        .add(Holder { inner: make() })
        .build(&Shape::known(&INPUT))
        .unwrap_or_else(|error| panic!("Sequential: the build must pass, got {error:?}"));
    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&INPUT));
    let hidden = builder.add(Dense::new(4, Activation::Linear).unwrap(), &[input]);
    let out = builder.add(Holder { inner: make() }, &[hidden]);
    let graph = builder
        .build(&[out])
        .unwrap_or_else(|error| panic!("Graph: the build must pass, got {error:?}"));
    (sequential, graph)
}

/// The sum of `param_count` over every node of a tree, by a walk of the test itself
fn tree_sum(layer: &dyn LayerBase) -> ParamCounts {
    let mut total = layer.param_count();
    for sub in layer.sublayers() {
        let counts = tree_sum(sub.layer);
        total.trainable += counts.trainable;
        total.non_trainable += counts.non_trainable;
    }
    total
}

/// Builds a layer on `shape`, moves `shape` to the output shape, and gives the tree sum
fn built_total<L: UnaryLayer>(mut layer: L, shape: &mut Shape) -> ParamCounts {
    layer.build(shape).unwrap();
    *shape = layer.compute_output_shape(shape).unwrap();
    total_param_count(&layer)
}

// A. The weight paths

/// A sequential model lists every array of every tree in the canonical pre-order
///
/// `ScaledDense` at position 2 gives its own array `scale` before the arrays of `inner`.
/// `Deep` at position 3 reaches depth 2. `Stack` at position 1 gives run-time names
#[test]
fn sequential_weight_paths_are_the_canonical_pre_order() {
    let model = rich_sequential(1);
    assert_eq!(model.weight_paths(), RICH_SEQUENTIAL_PATHS);
}

/// A graph lists every array in arena order, and a layer that 2 nodes call gives 1 set of paths
#[test]
fn graph_weight_paths_are_the_canonical_pre_order() {
    let model = rich_graph(1);
    assert_eq!(model.weight_paths(), RICH_GRAPH_PATHS);
}

/// `weight` reads the array of every path that `weight_paths` lists
///
/// The shape of each array is the shape that the tree gives it
#[test]
fn weight_reads_every_listed_path() {
    let sequential = rich_sequential(1);
    let graph = rich_graph(1);
    let models: [&dyn Checkpointed; 2] = [&sequential, &graph];
    for model in models {
        for path in model.paths() {
            assert!(model.array(&path).is_some(), "no array at `{path}`");
        }
    }

    // The arrays are the right ones, by shape
    assert_eq!(
        sequential.weight("0.dense.kernel").unwrap().shape(),
        &[4, 4]
    );
    assert_eq!(
        sequential.weight("0.norm.moving_mean").unwrap().shape(),
        &[4]
    );
    assert_eq!(sequential.weight("2.scale").unwrap().shape(), &[1, 3]);
    assert_eq!(
        sequential.weight("2.inner.kernel").unwrap().shape(),
        &[4, 3]
    );
    assert_eq!(
        sequential.weight("3.pair.first.kernel").unwrap().shape(),
        &[3, 3]
    );
    assert_eq!(
        sequential.weight("3.pair.second.kernel").unwrap().shape(),
        &[3, 2]
    );
    assert_eq!(sequential.weight("3.tail.kernel").unwrap().shape(), &[2, 1]);
    assert_eq!(graph.weight("0.first.kernel").unwrap().shape(), &[4, 3]);
    assert_eq!(graph.weight("2.dense.kernel").unwrap().shape(), &[4, 3]);

    // 2 sublayers of 1 type at 1 depth give 2 different arrays
    assert_ne!(
        sequential.weight("1.cell0.kernel").unwrap(),
        sequential.weight("1.cell1.kernel").unwrap()
    );
    // `2.scale` is the own array of the node, with the values its constructor gave
    assert_eq!(
        sequential.weight("2.scale").unwrap(),
        ScaledDense::new(3, 201).scale.view()
    );
}

/// `weight` gives `None` for every path that reaches no array
#[test]
fn weight_gives_none_for_a_path_that_reaches_no_array() {
    let sequential = rich_sequential(1);
    let graph = rich_graph(1);
    let cases = [
        // An unknown sublayer
        ("3.pair.third.kernel", "4.pair.third.kernel"),
        ("1.cell2.kernel", "0.third.kernel"),
        // Too few parts
        ("3", "4"),
        ("kernel", "kernel"),
        ("", ""),
        // A sublayer and no array
        ("3.pair", "4.pair"),
        ("3.pair.first", "0.first"),
        ("2.inner", "3.inner"),
        // An array name used as a sublayer name
        ("2.scale.kernel", "3.scale.kernel"),
        // A scope that is not a number, or that the model does not hold
        ("x.pair.first.kernel", "x.first.kernel"),
        ("-1.dense.kernel", "-1.first.kernel"),
        ("9.kernel", "9.kernel"),
        // A trailing dot
        ("3.pair.first.kernel.", "0.first.kernel."),
        ("3.pair.first.", "0.first."),
        // An empty part
        ("3..pair.first.kernel", "0..first.kernel"),
    ];
    for (in_sequential, in_graph) in cases {
        assert!(
            sequential.weight(in_sequential).is_none(),
            "Sequential: `{in_sequential}` must reach no array"
        );
        assert!(
            graph.weight(in_graph).is_none(),
            "Graph: `{in_graph}` must reach no array"
        );
    }
}

/// `weight` gives `None` for a scope text that `weight_paths` never writes
///
/// `weight_paths` writes a scope as a plain decimal number. A text with a sign or a leading
/// zero names no checkpoint path of the model
#[test]
fn weight_gives_none_for_a_scope_that_is_not_canonical() {
    let model = rich_graph(1);
    assert!(model.weight("0.first.kernel").is_some());
    for path in ["+0.first.kernel", "00.first.kernel", "+3.scale", "03.scale"] {
        assert!(
            model.weight(path).is_none(),
            "`{path}` is not in `weight_paths`, and it must reach no array"
        );
    }
}

// B. The round trip

/// A trained sequential model reloads into a model of other seeds, bit for bit
///
/// The load writes every array of every tree, the running statistics of the nested
/// BatchNormalization included. The 2 models then predict the same bits
#[test]
fn sequential_round_trip_moves_every_array_of_every_tree() {
    let tmp = TempFile::new("round_trip_sequential");
    let x = data(&INPUT, 0.0);
    let y = data(&[4, 1], 1.0);

    let mut trained = rich_sequential(1);
    trained.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    trained.fit(&x, &y, 3).unwrap();
    // The running statistics moved away from their start values
    assert!(
        trained
            .weight("0.norm.moving_mean")
            .unwrap()
            .iter()
            .any(|value| *value != 0.0)
    );
    trained.save_to_path(tmp.path()).unwrap();

    let mut fresh = rich_sequential(2);
    for (path, bits) in all_bits(&fresh) {
        assert_ne!(
            bits,
            bits_at(&trained, &path),
            "`{path}` must differ before the load, or the test proves nothing"
        );
    }

    fresh.load_from_path(tmp.path()).unwrap();
    assert_eq!(all_bits(&fresh), all_bits(&trained));
    assert_eq!(
        tensor_bits(&fresh.predict(&x).unwrap()),
        tensor_bits(&trained.predict(&x).unwrap())
    );
}

/// A trained graph reloads into a graph of other seeds, bit for bit
///
/// The shared `TwoDense` holds 1 set of arrays in the file. The nested BatchNormalization
/// carries its running statistics
#[test]
fn graph_round_trip_moves_every_array_of_every_tree() {
    let tmp = TempFile::new("round_trip_graph");
    let left = data(&INPUT, 0.0);
    let right = data(&INPUT, 2.0);
    let y = data(&[4, 1], 1.0);

    let mut trained = rich_graph(1);
    trained.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    trained.fit(&[&left, &right], &[&y], 3).unwrap();
    assert!(
        trained
            .weight("2.norm.moving_mean")
            .unwrap()
            .iter()
            .any(|value| *value != 0.0)
    );
    trained.save_to_path(tmp.path()).unwrap();

    let mut fresh = rich_graph(2);
    for (path, bits) in all_bits(&fresh) {
        assert_ne!(
            bits,
            bits_at(&trained, &path),
            "`{path}` must differ before the load, or the test proves nothing"
        );
    }

    fresh.load_from_path(tmp.path()).unwrap();
    assert_eq!(all_bits(&fresh), all_bits(&trained));
    let before = trained.predict(&[&left, &right]).unwrap();
    let after = fresh.predict(&[&left, &right]).unwrap();
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(tensor_bits(a), tensor_bits(b));
    }
}

/// The file holds the tree of each position, with the names and the kinds of the model
#[test]
fn a_saved_file_holds_the_tree_of_each_position() {
    let tmp = TempFile::new("file_tree");
    let model = rich_sequential(1);
    model.save_to_path(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    let file: ModelCheckpoint<'static> = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(file.format_version, MODEL_FORMAT_VERSION);
    assert_eq!(file.layers.len(), 4);

    let norm = &file.layers[0];
    assert_eq!(norm.layer_type, "DenseNorm");
    assert!(norm.weights.is_empty());
    let names: Vec<&str> = norm.sublayers.iter().map(|sub| &*sub.name).collect();
    assert_eq!(names, ["dense", "norm"]);
    let kinds: Vec<(&str, WeightKind)> = norm.sublayers[1]
        .layer
        .weights
        .iter()
        .map(|record| (&*record.name, record.kind))
        .collect();
    assert_eq!(
        kinds,
        [
            ("gamma", WeightKind::Trainable),
            ("beta", WeightKind::Trainable),
            ("moving_mean", WeightKind::NonTrainable),
            ("moving_variance", WeightKind::NonTrainable),
        ]
    );

    let scaled = &file.layers[2];
    assert_eq!(scaled.weights.len(), 1);
    assert_eq!(scaled.weights[0].name, "scale");
    assert_eq!(scaled.sublayers[0].name, "inner");

    let deep = &file.layers[3];
    assert_eq!(deep.sublayers[0].layer.layer_type, "TwoDense");
    assert_eq!(deep.sublayers[0].layer.sublayers[1].name, "second");
}

// C. The strict refusals

/// A strict load refuses a sublayer of another type, and names its full path
#[test]
fn strict_load_refuses_a_sublayer_of_another_type() {
    assert_strict_refusal(
        "strict_type",
        |file| second(file).layer.layer_type = Cow::Borrowed("Conv1D"),
        &["layer `0.second` type mismatch", "`Dense`", "`Conv1D`"],
    );
}

/// A strict load refuses a sublayer of another name, and names the layer that holds it
#[test]
fn strict_load_refuses_a_sublayer_of_another_name() {
    assert_strict_refusal(
        "strict_name",
        |file| second(file).name = Cow::Borrowed("third"),
        &[
            "layer `0` (`TwoDense`)",
            r#"["first", "second"]"#,
            r#"["first", "third"]"#,
        ],
    );
}

/// A strict load refuses a file with fewer sublayers, and a file with more
#[test]
fn strict_load_refuses_another_sublayer_count() {
    assert_strict_refusal(
        "strict_fewer",
        |file| {
            file.layers[0].sublayers.pop();
        },
        &[
            "layer `0` (`TwoDense`)",
            r#"["first", "second"]"#,
            r#"["first"]"#,
        ],
    );
    assert_strict_refusal(
        "strict_more",
        |file| {
            let extra = SublayerCheckpoint {
                name: Cow::Borrowed("extra"),
                layer: second(file).layer.clone(),
            };
            file.layers[0].sublayers.push(extra);
        },
        &["layer `0` (`TwoDense`)", r#"["first", "second", "extra"]"#],
    );
}

/// A strict load refuses a sublayer that was built for another input shape
#[test]
fn strict_load_refuses_a_sublayer_of_another_build_shape() {
    assert_strict_refusal(
        "strict_build",
        |file| {
            second(file).layer.build = Some(BuildConfig::unary(&Shape::known(&[4, 7])));
        },
        &["layer `0.second` (`Dense`) was built for input shape"],
    );
}

/// A strict load refuses a nested array of another shape, and names its full path
#[test]
fn strict_load_refuses_a_nested_array_of_another_shape() {
    assert_strict_refusal(
        "strict_shape",
        |file| {
            let kernel = &mut second(file).layer.weights[0];
            assert_eq!(kernel.name, "kernel");
            kernel.shape = vec![3, 5];
            kernel.data = Cow::Owned(vec![0.5; 15]);
        },
        &["`0.second.kernel` has shape [3, 2] in the model, and [3, 5] in the file"],
    );
}

/// A strict load refuses a nested array of another kind, and names its full path
#[test]
fn strict_load_refuses_a_nested_array_of_another_kind() {
    assert_strict_refusal(
        "strict_kind",
        |file| {
            let bias = &mut second(file).layer.weights[1];
            assert_eq!(bias.name, "bias");
            bias.kind = WeightKind::NonTrainable;
        },
        &["`0.second.bias` is trainable in the model, and non-trainable in the file"],
    );
}

// D. The lenient load

/// A file without the sublayer `second` loads `first`, and reports `second` as missing
#[test]
fn partial_load_reports_a_missing_sublayer() {
    assert_partial_report(
        "partial_missing",
        |file| {
            file.layers[0].sublayers.pop();
        },
        &["0.first.kernel", "0.first.bias"],
        &["0.second.kernel", "0.second.bias"],
        &[],
    );
}

/// A file that renames `second` to `third` loads `first`, and reports each side of the rename
#[test]
fn partial_load_reports_a_renamed_sublayer() {
    assert_partial_report(
        "partial_renamed",
        |file| second(file).name = Cow::Borrowed("third"),
        &["0.first.kernel", "0.first.bias"],
        &["0.second.kernel", "0.second.bias"],
        &["0.third.kernel", "0.third.bias"],
    );
}

/// A file whose `second` is of another type loads `first`, and reports `second` on both sides
#[test]
fn partial_load_reports_a_sublayer_of_another_type() {
    assert_partial_report(
        "partial_type",
        |file| second(file).layer.layer_type = Cow::Borrowed("Conv1D"),
        &["0.first.kernel", "0.first.bias"],
        &["0.second.kernel", "0.second.bias"],
        &["0.second.kernel", "0.second.bias"],
    );
}

/// A file whose `second` was built for another shape loads `first`, and reports `second` on
/// both sides
#[test]
fn partial_load_reports_a_sublayer_of_another_build_shape() {
    assert_partial_report(
        "partial_build",
        |file| {
            second(file).layer.build = Some(BuildConfig::unary(&Shape::known(&[4, 7])));
        },
        &["0.first.kernel", "0.first.bias"],
        &["0.second.kernel", "0.second.bias"],
        &["0.second.kernel", "0.second.bias"],
    );
}

/// A file with an extra sublayer loads every array of the model, and reports the extra paths
/// as unused
#[test]
fn partial_load_reports_an_extra_sublayer_as_unused() {
    assert_partial_report(
        "partial_extra",
        |file| {
            let extra = SublayerCheckpoint {
                name: Cow::Borrowed("extra"),
                layer: second(file).layer.clone(),
            };
            file.layers[0].sublayers.push(extra);
        },
        &[
            "0.first.kernel",
            "0.first.bias",
            "0.second.kernel",
            "0.second.bias",
        ],
        &[],
        &["0.extra.kernel", "0.extra.bias"],
    );
}

/// A lenient load of the same model reports every path of every tree as applied
#[test]
fn partial_load_of_a_matching_file_applies_every_path() {
    let tmp = TempFile::new("partial_full");
    let source = rich_sequential(1);
    source.save_to_path(tmp.path()).unwrap();
    let mut model = rich_sequential(2);
    let report = model.load_partial_from_path(tmp.path()).unwrap();
    assert_eq!(report.applied, RICH_SEQUENTIAL_PATHS);
    assert!(report.missing.is_empty());
    assert!(report.unused.is_empty());
    assert_eq!(all_bits(&model), all_bits(&source));
}

// E. The format version

/// A file of format version 3 is refused by the strict and the lenient load, and the message
/// names both versions
#[test]
fn a_file_of_format_version_3_is_refused() {
    assert_eq!(MODEL_FORMAT_VERSION, 4);
    for kind in KINDS {
        let tmp = TempFile::new(&format!("version_3_{kind:?}"));
        let source = pair_model(kind, 100);
        source.save(tmp.path());
        rewrite(tmp.path(), |file| file.format_version = 3);

        let mut model = pair_model(kind, 200);
        let before = all_bits(&*model);
        let strict = model.load(tmp.path()).err();
        let lenient = model.load_partial(tmp.path()).err();
        for result in [strict, lenient] {
            match result {
                Some(Error::Io(IoError::UnsupportedModelFormat(message))) => {
                    assert!(
                        message.contains("format version 3") && message.contains("reads version 4"),
                        "{kind:?}: the refusal must name both versions, got {message:?}"
                    );
                }
                other => panic!("{kind:?}: expected UnsupportedModelFormat, got {other:?}"),
            }
        }
        assert_eq!(all_bits(&*model), before);
    }
}

// F. The build checks

/// A build refuses 2 sublayers of 1 name
#[test]
fn build_refuses_2_sublayers_of_1_name() {
    assert_build_refused(
        || Faulty::new(Fault::DuplicateName),
        &[
            "layer `1.inner` (`Faulty`)",
            "gives 2 of its sublayers the name `x`",
        ],
    );
}

/// A build refuses an empty sublayer name
#[test]
fn build_refuses_an_empty_sublayer_name() {
    assert_build_refused(
        || Faulty::new(Fault::EmptyName),
        &["layer `1.inner` (`Faulty`)", "gives 1 sublayer the name ``"],
    );
}

/// A build refuses a sublayer name that holds a dot
#[test]
fn build_refuses_a_sublayer_name_with_a_dot() {
    assert_build_refused(
        || Faulty::new(Fault::DottedName),
        &[
            "layer `1.inner` (`Faulty`)",
            "gives 1 sublayer the name `a.b`",
        ],
    );
}

/// A build refuses an array name that holds a dot, in a node below the root
#[test]
fn build_refuses_an_array_name_with_a_dot() {
    assert_build_refused(
        || DottedArray {
            w: ArrayD::zeros(IxDyn(&[2])),
        },
        &[
            "layer `1.inner` (`DottedArray`)",
            "gives 1 array the name `w.x`",
        ],
    );
}

/// A build refuses 2 rosters that give the sublayers in another order
#[test]
fn build_refuses_rosters_in_another_order() {
    assert_build_refused(
        || Faulty::new(Fault::SwappedOrder),
        &[
            "layer `1.inner` (`Faulty`)",
            "[a (`Dense`), b (`Dense`)]",
            "[b (`Dense`), a (`Dense`)]",
        ],
    );
}

/// A build refuses 2 rosters that give 1 sublayer another name
#[test]
fn build_refuses_rosters_with_another_name() {
    assert_build_refused(
        || Faulty::new(Fault::RenamedInMut),
        &[
            "layer `1.inner` (`Faulty`)",
            "[a (`Dense`), b (`Dense`)]",
            "[a (`Dense`), c (`Dense`)]",
        ],
    );
}

/// A build refuses 2 rosters of different lengths
#[test]
fn build_refuses_rosters_of_another_count() {
    assert_build_refused(
        || Faulty::new(Fault::ShorterInMut),
        &[
            "layer `1.inner` (`Faulty`)",
            "[a (`Dense`), b (`Dense`)]",
            "and [a (`Dense`)] through `LayerBase::sublayers_mut`",
        ],
    );
}

/// A build refuses 2 rosters that give the same names and point them at other layers
///
/// The names and the types agree, so only the address tells the rosters apart
#[test]
fn build_refuses_rosters_that_reach_other_layers() {
    assert_build_refused(
        || Faulty::new(Fault::CrossedTargetsInMut),
        &[
            "layer `1.inner` (`Faulty`)",
            "must give the same names, in the same order, and reach the same layers",
        ],
    );
}

/// A build refuses 1 child listed under 2 names
#[test]
fn build_refuses_1_child_under_2_names() {
    assert_build_refused(
        || Faulty::new(Fault::SameChildTwice),
        &["layer `1.inner.b` (`Dense`) is the same storage as layer `1.inner.a`"],
    );
}

/// A build refuses 1 child listed at depth 2 and again at depth 1
#[test]
fn build_refuses_1_child_at_2_depths() {
    assert_build_refused(
        || Aliased {
            pair: TwoDense::new(4, 4, 1),
            other: Dense::new(4, Activation::Linear).unwrap(),
        },
        &["layer `1.inner.alias` (`Dense`) is the same storage as layer `1.inner.pair.first`"],
    );
}

/// A layer whose first field is a layer builds, trains, and round trips
///
/// The struct and its first field share 1 address. The check tells them apart by type, so it
/// does not take them for 1 storage
#[test]
fn a_layer_that_shares_its_address_with_its_first_sublayer_builds() {
    let lead = Lead {
        lead: Dense::new(3, Activation::Tanh).unwrap(),
        follow: Dense::new(2, Activation::Linear).unwrap(),
    };
    assert!(std::ptr::eq(
        &lead as *const Lead as *const (),
        &lead.lead as *const Dense as *const ()
    ));

    let make = || Lead {
        lead: Dense::new(3, Activation::Tanh).unwrap(),
        follow: Dense::new(2, Activation::Linear).unwrap(),
    };
    let (mut sequential, graph) = build_both(make);
    let expected = [
        "0.kernel",
        "0.bias",
        "1.inner.lead.kernel",
        "1.inner.lead.bias",
        "1.inner.follow.kernel",
        "1.inner.follow.bias",
    ];
    assert_eq!(sequential.weight_paths(), expected);
    assert_eq!(graph.weight_paths(), expected);

    // The model trains, so every gradient reaches a parameter of the tree
    sequential.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let before = bits_at(&sequential, "1.inner.lead.kernel");
    sequential
        .fit(&data(&INPUT, 0.0), &data(&[4, 2], 1.0), 2)
        .unwrap();
    assert_ne!(bits_at(&sequential, "1.inner.lead.kernel"), before);
}

/// A zero-sized layer type that the tree holds twice is not refused
///
/// The 2 `Pass` fields take no space, so they share 1 address and 1 type. A zero-sized layer
/// holds no storage, so the 2 nodes share no array, no gradient, and no optimizer state. The
/// storage check therefore skips a node of size 0
#[test]
fn a_zero_sized_layer_at_2_nodes_is_not_refused() {
    let twin = TwinPass {
        left: Pass,
        right: Pass,
        dense: Dense::new(4, Activation::Linear).unwrap(),
    };
    assert_eq!(std::mem::size_of::<Pass>(), 0);
    assert!(std::ptr::eq(
        &twin.left as *const Pass as *const (),
        &twin.right as *const Pass as *const ()
    ));

    let make = || TwinPass {
        left: Pass,
        right: Pass,
        dense: Dense::new(4, Activation::Linear).unwrap(),
    };
    let (mut sequential, graph) = build_both(make);
    let expected = [
        "0.kernel",
        "0.bias",
        "1.inner.dense.kernel",
        "1.inner.dense.bias",
    ];
    assert_eq!(sequential.weight_paths(), expected);
    assert_eq!(graph.weight_paths(), expected);

    sequential.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    sequential
        .fit(&data(&INPUT, 0.0), &data(&INPUT, 1.0), 2)
        .unwrap();
}

// G. The parameter counts

/// `total_param_count` sums every node of a tree, and `param_count` counts the node alone
#[test]
fn total_param_count_sums_the_tree_and_param_count_counts_1_node() {
    let shape = Shape::known(&INPUT);

    let mut dense_norm = DenseNorm::new(3, 1);
    dense_norm.build(&shape).unwrap();
    // Dense 4 x 3 + 3, and BatchNormalization 3 + 3 trainable and 3 + 3 non-trainable
    assert_eq!(dense_norm.param_count(), ParamCounts::none());
    assert_eq!(total_param_count(&dense_norm), ParamCounts::new(21, 6));
    assert_eq!(total_param_count(&dense_norm), tree_sum(&dense_norm));

    let mut scaled = ScaledDense::new(3, 1);
    scaled.build(&shape).unwrap();
    // The node holds only `scale`. The tree adds Dense 4 x 3 + 3
    assert_eq!(scaled.param_count(), ParamCounts::trainable(3));
    assert_eq!(total_param_count(&scaled), ParamCounts::new(18, 0));
    assert_eq!(total_param_count(&scaled), tree_sum(&scaled));

    let mut deep = Deep::new(1, 1);
    deep.build(&shape).unwrap();
    // Dense 4 x 3 + 3, Dense 3 x 2 + 2, and Dense 2 x 1 + 1
    assert_eq!(deep.param_count(), ParamCounts::none());
    assert_eq!(deep.pair.param_count(), ParamCounts::none());
    assert_eq!(total_param_count(&deep.pair), ParamCounts::new(23, 0));
    assert_eq!(total_param_count(&deep), ParamCounts::new(26, 0));
    assert_eq!(total_param_count(&deep), tree_sum(&deep));

    let mut stack = Stack::new(&["cell0", "cell1"], 4, 1);
    stack.build(&shape).unwrap();
    assert_eq!(total_param_count(&stack), ParamCounts::new(40, 0));
    assert_eq!(total_param_count(&stack), tree_sum(&stack));

    // A layer with no sublayer gives its own count
    let mut plain = Dense::new(2, Activation::Linear).unwrap();
    plain.build(&shape).unwrap();
    assert_eq!(total_param_count(&plain), plain.param_count());
}

/// The counts of a whole model match the element count of every listed array
#[test]
fn total_param_count_of_every_position_covers_every_listed_array() {
    let model = rich_sequential(1);
    let elements: usize = model
        .weight_paths()
        .iter()
        .map(|path| model.weight(path).unwrap().len())
        .sum();

    // Build each layer on the shape that reaches it, and sum the tree of each
    let mut shape = Shape::known(&INPUT);
    let counts = [
        built_total(DenseNorm::new(4, 1), &mut shape),
        built_total(Stack::new(&["cell0", "cell1"], 4, 101), &mut shape),
        built_total(ScaledDense::new(3, 201), &mut shape),
        built_total(Deep::new(1, 301), &mut shape),
    ];
    let mut total = ParamCounts::none();
    for count in counts {
        total.trainable += count.trainable;
        total.non_trainable += count.non_trainable;
    }
    assert_eq!(total.trainable + total.non_trainable, elements);
    assert_eq!(total.non_trainable, 8);
}
