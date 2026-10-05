//! Guard tests for layers that hold other layers
//!
//! These tests hold the checks that stop a defective composite before it trains wrong:
//!
//! 1. A sublayer roster that lists its own layer stops the model build with an error. It does
//!    not overflow the stack.
//! 2. A panic inside `Ctx::sublayer` leaves the context at the path it had before the call.
//! 3. A checkpoint path has 1 spelling for 1 model position.
//! 4. A caller that drives a composite by hand can update and apply the state of every
//!    sublayer, and a state value that no layer takes back is an error.

use ndarray::{Array, IxDyn};
use rustyml::error::Error;
use rustyml::neural_network::layer_path::{apply_state_tree, update_tree};
use rustyml::neural_network::layers::{Activation, Dense, Dropout, ParamCounts};
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::{LayerBase, UnaryLayer, WeightMut, WeightRef};
use rustyml::neural_network::{Ctx, LayerPath, Shape, Sublayer, SublayerMut, Tensor};

/// A layer that lists itself as its own sublayer, through 1 roster or both
struct SelfLoop {
    /// A Dense layer, so the layer holds storage
    inner: Dense,
    /// Whether the read roster lists the layer itself
    loop_read: bool,
}

impl LayerBase for SelfLoop {
    fn layer_type(&self) -> &str {
        "SelfLoop"
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
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        if self.loop_read {
            vec![Sublayer::new("me", self)]
        } else {
            vec![Sublayer::new("me", &self.inner)]
        }
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![SublayerMut::new("me", self)]
    }
}

impl UnaryLayer for SelfLoop {
    fn forward(&self, input: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(input.clone())
    }
    fn backward(&self, grad_output: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(grad_output.clone())
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        Ok(input.clone())
    }
}

/// A zero-sized layer that lists itself as its own sublayer
struct EmptyLoop;

impl LayerBase for EmptyLoop {
    fn layer_type(&self) -> &str {
        "EmptyLoop"
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
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![Sublayer::new("me", self)]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![SublayerMut::new("me", self)]
    }
}

impl UnaryLayer for EmptyLoop {
    fn forward(&self, input: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(input.clone())
    }
    fn backward(&self, grad_output: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(grad_output.clone())
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        Ok(input.clone())
    }
}

/// A composite of a Dense layer and a Dropout layer, in a chain
struct DenseDrop {
    /// The first sublayer
    dense: Dense,
    /// The second sublayer
    drop: Dropout,
}

impl LayerBase for DenseDrop {
    fn layer_type(&self) -> &str {
        "DenseDrop"
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
        self.dense.is_built() && self.drop.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![
            Sublayer::new("dense", &self.dense),
            Sublayer::new("drop", &self.drop),
        ]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![
            SublayerMut::new("dense", &mut self.dense),
            SublayerMut::new("drop", &mut self.drop),
        ]
    }
}

impl UnaryLayer for DenseDrop {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("dense", &self.dense, |ctx| self.dense.forward(input, ctx))?;
        ctx.sublayer("drop", &self.drop, |ctx| self.drop.forward(&hidden, ctx))
    }
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer("drop", &self.drop, |ctx| {
            self.drop.backward(grad_output, ctx)
        })?;
        ctx.sublayer("dense", &self.dense, |ctx| self.dense.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.dense.build(input)?;
        let hidden = self.dense.compute_output_shape(input)?;
        self.drop.build(&hidden)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.drop
            .compute_output_shape(&self.dense.compute_output_shape(input)?)
    }
}

/// A layer that writes a state value and never takes it back
struct Leaky;

impl LayerBase for Leaky {
    fn layer_type(&self) -> &str {
        "Leaky"
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

impl UnaryLayer for Leaky {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if ctx.is_training() {
            ctx.set_state("count", 1_usize);
        }
        Ok(input.clone())
    }
    fn backward(&self, grad_output: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        Ok(grad_output.clone())
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        Ok(input.clone())
    }
}

/// A tensor from a fixed formula, so the data never moves between runs
fn ramp(shape: &[usize]) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) / 40.0)
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).unwrap()
}

/// A new DenseDrop composite with fixed seeds
fn dense_drop() -> DenseDrop {
    DenseDrop {
        dense: Dense::new(3, Activation::Tanh)
            .unwrap()
            .with_random_state(5),
        drop: Dropout::new(0.5).unwrap().with_random_state(9),
    }
}

/// The text of the error that a model build gives back
fn build_refusal(layer: impl UnaryLayer + 'static) -> String {
    match SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(&[2, 4]))
    {
        Ok(_) => panic!("the build must refuse the layer"),
        Err(error) => error.to_string(),
    }
}

/// A read roster that lists its own layer is a duplicate storage. The build refuses it before
/// the walk goes down a level, so the stack does not overflow
#[test]
fn a_read_roster_that_lists_its_own_layer_is_refused() {
    let message = build_refusal(SelfLoop {
        inner: Dense::new(2, Activation::Linear).unwrap(),
        loop_read: true,
    });
    assert!(message.contains("same storage"), "{message}");
    assert!(message.contains("`0.me`"), "{message}");
}

/// A write roster that lists its own layer disagrees with the read roster. The build refuses
/// it before the walk goes down the write roster
#[test]
fn a_write_roster_that_lists_its_own_layer_is_refused() {
    let message = build_refusal(SelfLoop {
        inner: Dense::new(2, Activation::Linear).unwrap(),
        loop_read: false,
    });
    assert!(message.contains("LayerBase::sublayers_mut"), "{message}");
}

/// A zero-sized layer that lists itself passes the storage check. The depth limit stops it
#[test]
fn a_zero_sized_layer_that_lists_itself_is_refused() {
    let message = build_refusal(EmptyLoop);
    assert!(message.contains("sublayers deep"), "{message}");
}

/// A panic inside a sublayer call removes the frame of that call. The model can then point
/// the context at another position
#[test]
fn a_panic_inside_a_sublayer_call_restores_the_path() {
    let layer = Dense::new(2, Activation::Linear).unwrap();
    let mut ctx = Ctx::training();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.sublayer("inner", &layer, |_| panic!("the body fails"));
    }));
    assert!(caught.is_err());
    assert_eq!(ctx.layer_path(), LayerPath::root(0));
    ctx.set_owner(1);
    assert_eq!(ctx.layer_path(), LayerPath::root(1));
}

/// An error inside a sublayer call also removes the frame of that call
#[test]
fn an_error_inside_a_sublayer_call_restores_the_path() {
    let layer = Dense::new(2, Activation::Linear).unwrap();
    let mut ctx = Ctx::training();
    let result: Result<(), Error> = ctx.sublayer("inner", &layer, |ctx| {
        assert_eq!(ctx.layer_path().to_string(), "0.inner");
        Err(Error::computation("the body fails"))
    });
    assert!(result.is_err());
    assert_eq!(ctx.layer_path(), LayerPath::root(0));
}

/// A position has 1 spelling. A leading 0 or a sign reaches no array
#[test]
fn a_checkpoint_path_has_1_spelling_per_position() {
    let model = SequentialBuilder::new()
        .add(dense_drop())
        .build(&Shape::known(&[2, 4]))
        .unwrap();
    assert!(model.weight("0.dense.kernel").is_some());
    assert!(model.weight("00.dense.kernel").is_none());
    assert!(model.weight("+0.dense.kernel").is_none());
    assert!(model.weight("0.dense").is_none());
    assert!(model.weight("0.dense.kernel.").is_none());
}

/// `update_tree` reaches every sublayer of a composite that a caller drives by hand, and
/// gives the same result as a model step
#[test]
fn a_hand_driven_step_updates_every_sublayer() {
    let x = ramp(&[2, 4]);
    let y = ramp(&[2, 3]);

    let mut model = SequentialBuilder::new()
        .add(dense_drop())
        .build(&Shape::known(&[2, 4]))
        .unwrap();
    model.compile(
        SGD::new(0.1, 0.0, false, 0.0).unwrap(),
        rustyml::neural_network::losses::MeanSquaredError::new(),
    );
    model.train_batch(&x, &y).unwrap();

    let mut layer = dense_drop();
    layer.build(&Shape::known(&[2, 4])).unwrap();
    let before = layer.dense.weight("kernel").unwrap().to_owned();
    let mut ctx = Ctx::training();
    let output = layer.forward_mut(&x, &mut ctx).unwrap();
    // The gradient of the mean squared error over every element
    let grad = (&output - &y) * (2.0 / output.len() as f32);
    layer.backward(&grad, &mut ctx).unwrap();
    let mut optimizer = SGD::new(0.1, 0.0, false, 0.0).unwrap();
    update_tree(
        &mut optimizer,
        &mut layer,
        &LayerPath::root(0),
        ctx.grads(),
        1.0,
    );

    let after = layer.dense.weight("kernel").unwrap();
    assert_ne!(after, before.view(), "the sublayer kernel must move");
    assert_eq!(after, model.weight("0.dense.kernel").unwrap());
    assert_eq!(
        layer.dense.weight("bias").unwrap(),
        model.weight("0.dense.bias").unwrap()
    );
}

/// `apply_state_tree` moves the state of every sublayer, so a hand-driven pass leaves no
/// value behind and the random stream of the Dropout sublayer advances
#[test]
fn a_hand_driven_state_apply_reaches_every_sublayer() {
    let x = ramp(&[2, 4]);
    let mut layer = dense_drop();
    layer.build(&Shape::known(&[2, 4])).unwrap();

    let mut first = Ctx::training();
    let a = layer.forward(&x, &mut first).unwrap();
    apply_state_tree(&mut layer, &LayerPath::root(0), &mut first);
    assert_eq!(first.pending_states(), 0);

    let mut second = Ctx::training();
    let b = layer.forward(&x, &mut second).unwrap();
    apply_state_tree(&mut layer, &LayerPath::root(0), &mut second);
    assert_ne!(a, b, "the random stream of the sublayer must advance");
}

/// `forward_mut` refuses a state value that the layer does not take back
#[test]
fn forward_mut_refuses_a_state_value_that_no_layer_takes_back() {
    let mut ctx = Ctx::training();
    let message = match Leaky.forward_mut(&ramp(&[2, 4]), &mut ctx) {
        Ok(_) => panic!("a leftover state value must stop the pass"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("0.count"), "{message}");
}

/// A model step refuses a state value that the layer does not take back, and the message
/// names the path
#[test]
fn a_model_step_refuses_a_state_value_that_no_layer_takes_back() {
    let mut model = SequentialBuilder::new()
        .add(Leaky)
        .build(&Shape::known(&[2, 4]))
        .unwrap();
    model.compile(
        SGD::new(0.1, 0.0, false, 0.0).unwrap(),
        rustyml::neural_network::losses::MeanSquaredError::new(),
    );
    let message = match model.train_batch(&ramp(&[2, 4]), &ramp(&[2, 4])) {
        Ok(_) => panic!("a leftover state value must stop the step"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("0.count"), "{message}");
}
