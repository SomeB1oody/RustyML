//! Tests for layers that hold other layers
//!
//! The sublayer mechanism does not depend on the type of a sublayer. 1 generic composite,
//! [`Pair`], holds 2 common layers and covers the mechanism. The tests compare a model that holds
//! a composite against the flat model of the same layers, bit for bit.

use ndarray::{Array, IxDyn};
use rustyml::error::Error;
use rustyml::neural_network::graph::{Graph, GraphBuilder};
use rustyml::neural_network::layers::{
    Activation, BatchNormalization, Dense, Dropout, ParamCounts,
};
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::Adam;
use rustyml::neural_network::sequential::{Sequential, SequentialBuilder};
use rustyml::neural_network::traits::{LayerBase, UnaryLayer, WeightMut, WeightRef};
use rustyml::neural_network::{Ctx, LayerPath, Shape, Sublayer, SublayerMut, Tensor};

/// How a [`Pair`] names its second sublayer when it calls it
#[derive(Clone, Copy)]
enum Wiring {
    /// Under its own name, `second`
    Scoped,
    /// Under the name of the first sublayer, which is a defect
    Crossed,
}

/// A composite that runs 2 sublayers in a chain
struct Pair<A, B> {
    /// The first sublayer
    first: A,
    /// The second sublayer
    second: B,
    /// The name under which the pair calls `second`
    wiring: Wiring,
}

impl<A, B> Pair<A, B> {
    /// A pair that calls each sublayer under its own name
    fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            wiring: Wiring::Scoped,
        }
    }

    /// The name under which the pair calls `second`
    fn second_name(&self) -> &'static str {
        match self.wiring {
            Wiring::Scoped => "second",
            Wiring::Crossed => "first",
        }
    }
}

impl<A: UnaryLayer, B: UnaryLayer> LayerBase for Pair<A, B> {
    fn layer_type(&self) -> &str {
        "Pair"
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

impl<A: UnaryLayer, B: UnaryLayer> UnaryLayer for Pair<A, B> {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let hidden = ctx.sublayer("first", &self.first, |ctx| self.first.forward(input, ctx))?;
        ctx.sublayer(self.second_name(), &self.second, |ctx| {
            self.second.forward(&hidden, ctx)
        })
    }
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad = ctx.sublayer(self.second_name(), &self.second, |ctx| {
            self.second.backward(grad_output, ctx)
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

/// A composite that calls 1 Dense sublayer twice per pass: `y = cell(cell(x))`
struct Tied {
    /// The sublayer of both calls
    cell: Dense,
}

impl LayerBase for Tied {
    fn layer_type(&self) -> &str {
        "Tied"
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
        self.cell.is_built()
    }
    fn sublayers(&self) -> Vec<Sublayer<'_>> {
        vec![Sublayer::new("cell", &self.cell)]
    }
    fn sublayers_mut(&mut self) -> Vec<SublayerMut<'_>> {
        vec![SublayerMut::new("cell", &mut self.cell)]
    }
}

impl UnaryLayer for Tied {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cell = &self.cell;
        let hidden = ctx.sublayer_call("cell", 0, cell, |ctx| cell.forward(input, ctx))?;
        ctx.sublayer_call("cell", 1, cell, |ctx| cell.forward(&hidden, ctx))
    }
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cell = &self.cell;
        let grad = ctx.sublayer_call("cell", 1, cell, |ctx| cell.backward(grad_output, ctx))?;
        ctx.sublayer_call("cell", 0, cell, |ctx| cell.backward(&grad, ctx))
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.cell.build(input)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.cell.compute_output_shape(input)
    }
}

/// A defective layer whose roster lists the layer itself
struct SelfLoop {
    /// Gives the layer a size, so the storage check sees it
    _size: u8,
}

impl LayerBase for SelfLoop {
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

/// A Dense layer with a fixed seed
fn dense(units: usize, seed: u64) -> Dense {
    Dense::new(units, Activation::Tanh)
        .unwrap()
        .with_random_state(seed)
}

/// A tensor from a fixed formula, so the data never moves between runs
fn ramp(shape: &[usize], offset: usize) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| ((((i + offset) * 37) % 101) as f32 - 50.0) / 40.0)
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).unwrap()
}

/// Adam with a global norm clip that the test data exceeds
fn adam() -> Adam {
    Adam::new(0.01, 0.9, 0.999, 1e-7, 0.0)
        .unwrap()
        .with_global_clipnorm(0.05)
        .unwrap()
}

/// Builds a model for 4 input features, and compiles it
fn compiled(builder: SequentialBuilder) -> Sequential {
    let mut model = builder.build(&Shape::new(vec![None, Some(4)])).unwrap();
    model.compile(adam(), MeanSquaredError::new());
    model
}

/// Asserts that 2 tensors hold the same bits at every position
fn assert_same_bits(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{what}");
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.to_bits(), e.to_bits(), "{what}: {a:e} against {e:e}");
    }
}

/// A model that holds `Pair(Dense, Dense)` trains to the same bits as the 2 Dense layers in a
/// flat model. The 2 kernels have the same shape, so the result also holds that each sublayer
/// has its own gradient, its own Adam moments, and its place in the global gradient norm
#[test]
fn a_composite_trains_like_the_flat_model() {
    let x = ramp(&[6, 4], 0);
    let y = ramp(&[6, 4], 50);
    let mut nested = compiled(SequentialBuilder::new().add(Pair::new(dense(4, 1), dense(4, 2))));
    let mut flat = compiled(SequentialBuilder::new().add(dense(4, 1)).add(dense(4, 2)));

    assert_eq!(
        nested.weight_paths(),
        vec![
            "0.first.kernel",
            "0.first.bias",
            "0.second.kernel",
            "0.second.bias"
        ]
    );
    for _ in 0..5 {
        let a = nested.train_batch(&x, &y).unwrap();
        let b = flat.train_batch(&x, &y).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (nested_path, flat_path) in nested.weight_paths().iter().zip(flat.weight_paths()) {
        assert_same_bits(
            &nested.weight(nested_path).unwrap().to_owned(),
            &flat.weight(&flat_path).unwrap().to_owned(),
            nested_path,
        );
    }
}

/// The state of a sublayer moves into that sublayer. The moving statistics of a
/// BatchNormalization and the random stream of a Dropout inside a composite match the flat
/// model after training, and so does the prediction
#[test]
fn the_state_of_a_sublayer_moves_into_that_sublayer() {
    let x = ramp(&[6, 4], 0);
    let y = ramp(&[6, 4], 50);
    let mut nested = compiled(
        SequentialBuilder::new()
            .add(Pair::new(
                BatchNormalization::new(0.9, 1e-3).unwrap(),
                Dropout::new(0.5).unwrap().with_random_state(7),
            ))
            .add(dense(4, 3)),
    );
    let mut flat = compiled(
        SequentialBuilder::new()
            .add(BatchNormalization::new(0.9, 1e-3).unwrap())
            .add(Dropout::new(0.5).unwrap().with_random_state(7))
            .add(dense(4, 3)),
    );

    for _ in 0..3 {
        let a = nested.train_batch(&x, &y).unwrap();
        let b = flat.train_batch(&x, &y).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let moving_mean = nested.weight("0.first.moving_mean").unwrap().to_owned();
    assert!(moving_mean.iter().any(|v| *v != 0.0), "the statistics move");
    assert_same_bits(
        &moving_mean,
        &flat.weight("0.moving_mean").unwrap().to_owned(),
        "moving_mean",
    );
    assert_same_bits(
        &nested.predict(&x).unwrap(),
        &flat.predict(&x).unwrap(),
        "prediction",
    );
}

/// A sublayer that 1 pass calls twice gets the sum of the gradients of both calls. The tied
/// composite trains like a graph that calls 1 Dense layer at 2 nodes
#[test]
fn a_sublayer_called_twice_trains_like_a_shared_graph_layer() {
    let x = ramp(&[6, 4], 0);
    let y = ramp(&[6, 4], 50);
    let mut tied = compiled(SequentialBuilder::new().add(Tied { cell: dense(4, 1) }));

    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::new(vec![None, Some(4)]));
    let cell = builder.layer(dense(4, 1));
    let hidden = builder.apply(cell, &[input]);
    let output = builder.apply(cell, &[hidden]);
    let mut graph: Graph = builder.build(&[output]).unwrap();
    graph.compile(adam(), MeanSquaredError::new());

    for _ in 0..3 {
        let a = tied.train_batch(&x, &y).unwrap();
        let b = graph.train_batch(&[&x], &[&y]).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_same_bits(
        &tied.weight("0.cell.kernel").unwrap().to_owned(),
        &graph.weight("0.kernel").unwrap().to_owned(),
        "kernel",
    );
}

/// A checkpoint holds the tree of each position. A load into a fresh model gives back every
/// array. A load into a model whose sublayer has another type is refused, names the path of
/// the sublayer, and leaves the model as it was
#[test]
fn a_checkpoint_holds_the_tree_of_each_position() {
    let path = std::env::temp_dir().join(format!("rustyml_sublayers_{}.bin", std::process::id()));
    let mut source = compiled(
        SequentialBuilder::new().add(Pair::new(dense(4, 1), Pair::new(dense(4, 2), dense(4, 3)))),
    );
    source
        .train_batch(&ramp(&[6, 4], 0), &ramp(&[6, 4], 50))
        .unwrap();
    source.save_to_path(&path).unwrap();

    let mut target = compiled(
        SequentialBuilder::new().add(Pair::new(dense(4, 4), Pair::new(dense(4, 5), dense(4, 6)))),
    );
    target.load_from_path(&path).unwrap();
    assert!(
        target
            .weight_paths()
            .contains(&"0.second.second.kernel".to_string())
    );
    for name in source.weight_paths() {
        assert_same_bits(
            &target.weight(&name).unwrap().to_owned(),
            &source.weight(&name).unwrap().to_owned(),
            &name,
        );
    }

    let mut other = compiled(SequentialBuilder::new().add(Pair::new(
        dense(4, 4),
        Pair::new(dense(4, 5), BatchNormalization::new(0.9, 1e-3).unwrap()),
    )));
    let before = other.weight("0.first.kernel").unwrap().to_owned();
    let message = other.load_from_path(&path).unwrap_err().to_string();
    std::fs::remove_file(&path).unwrap();
    assert!(message.contains("`0.second.second`"), "{message}");
    assert_same_bits(
        &other.weight("0.first.kernel").unwrap().to_owned(),
        &before,
        "the refusal writes nothing",
    );
}

/// A composite that calls its second Dense under the name of its first Dense stops the step.
/// The 2 sublayers have the same type and shape, so no other check would see the defect
#[test]
fn a_sublayer_call_under_the_wrong_name_is_refused() {
    let mut pair = Pair::new(dense(4, 1), dense(4, 2));
    pair.wiring = Wiring::Crossed;
    let mut model = compiled(SequentialBuilder::new().add(pair));
    let before = model.weight("0.second.kernel").unwrap().to_owned();
    let message = model
        .train_batch(&ramp(&[6, 4], 0), &ramp(&[6, 4], 50))
        .unwrap_err()
        .to_string();
    assert!(message.contains("`0.first`"), "{message}");
    assert_same_bits(
        &model.weight("0.second.kernel").unwrap().to_owned(),
        &before,
        "the refusal writes nothing",
    );
}

/// A roster that lists its own layer stops the build with an error, not a stack overflow
#[test]
fn a_roster_that_lists_its_own_layer_is_refused() {
    let built = SequentialBuilder::new()
        .add(SelfLoop { _size: 0 })
        .build(&Shape::new(vec![None, Some(4)]));
    let Err(error) = built else {
        panic!("the build must refuse the layer");
    };
    let message = error.to_string();
    assert!(message.contains("same storage"), "{message}");
}

/// A panic inside a sublayer call removes the frame of that call from the context
#[test]
fn a_panic_inside_a_sublayer_call_restores_the_path() {
    let layer = dense(4, 1);
    let mut ctx = Ctx::training();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.sublayer("inner", &layer, |_| panic!("the body fails"));
    }));
    assert!(caught.is_err());
    assert_eq!(ctx.layer_path(), LayerPath::root(0));
}
