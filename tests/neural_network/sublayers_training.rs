//! Training tests for layers that hold other layers
//!
//! A composite layer lists its sublayers under fixed names, and it calls each of them inside
//! `Ctx::sublayer`. The tests here hold a composite to these rules:
//!
//! 1. A model that holds a composite trains to the same bits as a flat model of the same
//!    layers. The optimizer state, the global gradient norm, and the canonical order must all
//!    agree for that to be true.
//! 2. Every gradient of a sublayer lands at the path of that sublayer.
//! 3. The analytic gradients of a composite agree with finite differences.
//! 4. The state of a sublayer moves into that sublayer, and nowhere else.
//! 5. A composite that calls a sublayer outside its scope stops the step with an error.
//! 6. A sublayer that 1 pass calls twice gets the sum of the gradients of both calls.
//! 7. A graph that calls 1 composite at 2 nodes trains like a graph that shares plain layers.

use ndarray::{Array, ArrayViewD, IxDyn};
use rustyml::error::Error;
use rustyml::neural_network::graph::{Graph, GraphBuilder};
use rustyml::neural_network::layer_path::total_param_count;
use rustyml::neural_network::layers::{
    Activation, BatchNormalization, Dense, Dropout, ParamCounts,
};
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::{Adam, SGD};
use rustyml::neural_network::sequential::{Sequential, SequentialBuilder};
use rustyml::neural_network::traits::{
    LayerBase, Optimizer, ParamId, UnaryLayer, WeightMut, WeightRef,
};
use rustyml::neural_network::{Ctx, LayerPath, Shape, Sublayer, SublayerMut, SublayerName, Tensor};

// Test composites

/// How a [`Pair`] calls its second sublayer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wiring {
    /// Inside `Ctx::sublayer`, under the roster name `second`
    Scoped,
    /// Directly, with no `Ctx::sublayer` around the call
    Unscoped,
    /// Inside `Ctx::sublayer`, under the name `other`, which no roster gives
    Misnamed,
    /// Inside `Ctx::sublayer`, under the name `first`, which the roster gives the other
    /// sublayer
    Crossed,
}

/// A composite that runs 2 sublayers in a chain
///
/// The roster names are `first` and `second`. [`Wiring`] selects how the forward pass and the
/// backward pass call the second sublayer. Only [`Wiring::Scoped`] keeps the contract
struct Pair<A, B> {
    first: A,
    second: B,
    wiring: Wiring,
}

impl<A: UnaryLayer, B: UnaryLayer> Pair<A, B> {
    /// A pair that keeps the contract
    fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            wiring: Wiring::Scoped,
        }
    }

    /// A pair that calls its second sublayer as `wiring` says
    fn wired(first: A, second: B, wiring: Wiring) -> Self {
        Self {
            first,
            second,
            wiring,
        }
    }

    /// Runs 1 call of the second sublayer with the wiring of the pair
    fn run_second<R>(&self, ctx: &mut Ctx, body: impl FnOnce(&mut Ctx) -> R) -> R {
        match self.wiring {
            Wiring::Scoped => ctx.sublayer("second", &self.second, body),
            Wiring::Unscoped => body(ctx),
            Wiring::Misnamed => ctx.sublayer("other", &self.second, body),
            Wiring::Crossed => ctx.sublayer("first", &self.second, body),
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
        self.run_second(ctx, |ctx| self.second.forward(&hidden, ctx))
    }
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let grad_hidden = self.run_second(ctx, |ctx| self.second.backward(grad_output, ctx))?;
        ctx.sublayer("first", &self.first, |ctx| {
            self.first.backward(&grad_hidden, ctx)
        })
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

/// 2 Dense layers in a chain, as 1 composite
type TwoDense = Pair<Dense, Dense>;

/// How a [`Tied`] composite calls its 1 sublayer twice in 1 pass
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TiedPattern {
    /// `y = cell(cell(x))`, with call 0 and call 1 of `sublayer_call`
    ChainNumbered,
    /// `y = cell(cell(x))`, with `sublayer` for both calls. The backward pass runs the calls in
    /// reverse order, so each call takes the newest cache of the 1 stack
    ChainSameCall,
    /// `y = cell(x) + cell(x / 2)`, with call 0 and call 1. The backward pass runs call 0 first
    SumForwardOrder,
    /// `y = cell(x) + cell(x / 2)`, with call 0 and call 1. The backward pass runs call 1 first
    SumReverseOrder,
}

/// A composite that calls 1 sublayer, `cell`, twice per pass
struct Tied {
    cell: Dense,
    pattern: TiedPattern,
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
        match self.pattern {
            TiedPattern::ChainNumbered => {
                let hidden = ctx.sublayer_call("cell", 0, cell, |ctx| cell.forward(input, ctx))?;
                ctx.sublayer_call("cell", 1, cell, |ctx| cell.forward(&hidden, ctx))
            }
            TiedPattern::ChainSameCall => {
                let hidden = ctx.sublayer("cell", cell, |ctx| cell.forward(input, ctx))?;
                ctx.sublayer("cell", cell, |ctx| cell.forward(&hidden, ctx))
            }
            TiedPattern::SumForwardOrder | TiedPattern::SumReverseOrder => {
                let half = input * 0.5;
                let whole = ctx.sublayer_call("cell", 0, cell, |ctx| cell.forward(input, ctx))?;
                let halved = ctx.sublayer_call("cell", 1, cell, |ctx| cell.forward(&half, ctx))?;
                Ok(whole + halved)
            }
        }
    }
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cell = &self.cell;
        match self.pattern {
            TiedPattern::ChainNumbered => {
                let grad_hidden =
                    ctx.sublayer_call("cell", 1, cell, |ctx| cell.backward(grad_output, ctx))?;
                ctx.sublayer_call("cell", 0, cell, |ctx| cell.backward(&grad_hidden, ctx))
            }
            TiedPattern::ChainSameCall => {
                let grad_hidden =
                    ctx.sublayer("cell", cell, |ctx| cell.backward(grad_output, ctx))?;
                ctx.sublayer("cell", cell, |ctx| cell.backward(&grad_hidden, ctx))
            }
            TiedPattern::SumForwardOrder => {
                let whole =
                    ctx.sublayer_call("cell", 0, cell, |ctx| cell.backward(grad_output, ctx))?;
                let halved =
                    ctx.sublayer_call("cell", 1, cell, |ctx| cell.backward(grad_output, ctx))?;
                Ok(whole + halved * 0.5)
            }
            TiedPattern::SumReverseOrder => {
                let halved =
                    ctx.sublayer_call("cell", 1, cell, |ctx| cell.backward(grad_output, ctx))?;
                let whole =
                    ctx.sublayer_call("cell", 0, cell, |ctx| cell.backward(grad_output, ctx))?;
                Ok(whole + halved * 0.5)
            }
        }
    }
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.cell.build(input)
    }
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.cell.compute_output_shape(input)
    }
}

// Helpers

/// A Dense layer with a tanh activation and a seeded kernel
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

/// Asserts that 2 arrays hold the same bits at every position
fn assert_bitwise_eq(actual: ArrayViewD<'_, f32>, expected: ArrayViewD<'_, f32>, what: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{what}: the shapes differ"
    );
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{what}: element {index} differs, {a:e} against {e:e}"
        );
    }
}

/// Asserts that 2 models hold the same bits in every array, in the canonical order
///
/// `expected_paths` is the path list of `actual`, so the test also holds the canonical order
fn assert_models_agree(actual: &Sequential, expected: &Sequential, expected_paths: &[&str]) {
    let actual_paths = actual.weight_paths();
    assert_eq!(actual_paths, expected_paths, "the canonical order is wrong");
    let flat_paths = expected.weight_paths();
    assert_eq!(actual_paths.len(), flat_paths.len());
    for (path, flat) in actual_paths.iter().zip(flat_paths.iter()) {
        assert_bitwise_eq(
            actual.weight(path).unwrap(),
            expected.weight(flat).unwrap(),
            &format!("`{path}` against `{flat}`"),
        );
    }
}

/// Trains 2 models on the same batches, and asserts that every loss holds the same bits
fn train_in_step<O: 'static + Optimizer>(
    nested: &mut Sequential,
    flat: &mut Sequential,
    optimizer: impl Fn() -> O,
    steps: usize,
) -> Vec<f32> {
    nested.compile(optimizer(), MeanSquaredError::new());
    flat.compile(optimizer(), MeanSquaredError::new());
    let input_shape: Vec<usize> = nested
        .input_shape()
        .axes()
        .iter()
        .map(|extent| extent.unwrap_or(8))
        .collect();
    let output_shape: Vec<usize> = {
        let shape = nested.output_shape().unwrap();
        let mut axes: Vec<usize> = shape.axes().iter().map(|e| e.unwrap_or(8)).collect();
        axes[0] = input_shape[0];
        axes
    };
    let mut losses = Vec::with_capacity(steps);
    for step in 0..steps {
        let x = ramp(&input_shape, step * 3);
        let y = ramp(&output_shape, step * 7 + 1).mapv(|v| v * 0.5);
        let nested_loss = nested.train_batch(&x, &y).unwrap();
        let flat_loss = flat.train_batch(&x, &y).unwrap();
        assert_eq!(
            nested_loss.to_bits(),
            flat_loss.to_bits(),
            "step {step}: the loss differs, {nested_loss:e} against {flat_loss:e}"
        );
        losses.push(nested_loss);
    }
    losses
}

/// SGD with Nesterov momentum and a global norm clip that the test data exceeds
fn sgd_momentum() -> SGD {
    SGD::new(0.05, 0.9, true, 0.0)
        .unwrap()
        .with_global_clipnorm(0.05)
        .unwrap()
}

/// Adam with a global norm clip that the test data exceeds
fn adam_clipped() -> Adam {
    Adam::new(0.01, 0.9, 0.999, 1e-7, 0.0)
        .unwrap()
        .with_global_clipnorm(0.05)
        .unwrap()
}

/// A 4-layer stack in which positions 1 and 2 of the flat form are 1 composite
fn nested_two_dense() -> Sequential {
    SequentialBuilder::new()
        .add(dense(5, 1))
        .add(TwoDense::new(dense(4, 2), dense(3, 3)))
        .add(dense(2, 4))
        .build(&Shape::new(vec![None, Some(6)]))
        .unwrap()
}

/// The flat form of [`nested_two_dense`], with the same seeds
fn flat_two_dense() -> Sequential {
    SequentialBuilder::new()
        .add(dense(5, 1))
        .add(dense(4, 2))
        .add(dense(3, 3))
        .add(dense(2, 4))
        .build(&Shape::new(vec![None, Some(6)]))
        .unwrap()
}

/// The canonical path list of [`nested_two_dense`]
const TWO_DENSE_PATHS: &[&str] = &[
    "0.kernel",
    "0.bias",
    "1.first.kernel",
    "1.first.bias",
    "1.second.kernel",
    "1.second.bias",
    "2.kernel",
    "2.bias",
];

/// The address of 1 array below the model position 0
fn param_at(sublayers: &[&'static str], name: &'static str) -> ParamId {
    let mut path = LayerPath::root(0);
    for sub in sublayers {
        path = path.child(*sub);
    }
    path.param(name)
}

// 1. Training parity

/// A composite of 2 Dense layers trains to the same bits as the 2 Dense layers in a flat
/// stack. The test uses SGD with Nesterov momentum and a global norm clip
#[test]
fn two_dense_composite_trains_like_flat_stack_with_sgd_momentum() {
    let mut nested = nested_two_dense();
    let mut flat = flat_two_dense();
    assert_models_agree(&nested, &flat, TWO_DENSE_PATHS);

    let losses = train_in_step(&mut nested, &mut flat, sgd_momentum, 6);
    assert_models_agree(&nested, &flat, TWO_DENSE_PATHS);
    assert!(losses.iter().all(|loss| loss.is_finite()));

    // The clip must engage, or the test does not cover the global gradient norm
    let mut unclipped = nested_two_dense();
    let mut unclipped_flat = flat_two_dense();
    train_in_step(
        &mut unclipped,
        &mut unclipped_flat,
        || SGD::new(0.05, 0.9, true, 0.0).unwrap(),
        6,
    );
    assert_ne!(
        unclipped.weight("1.first.kernel").unwrap(),
        nested.weight("1.first.kernel").unwrap(),
        "the clip did not engage"
    );
}

/// A composite of 2 Dense layers trains to the same bits as the 2 Dense layers in a flat
/// stack. The test uses Adam with a global norm clip. Adam holds 2 moments per array, so the
/// test fails when 2 sublayers share 1 optimizer state
#[test]
fn two_dense_composite_trains_like_flat_stack_with_adam() {
    let mut nested = nested_two_dense();
    let mut flat = flat_two_dense();

    train_in_step(&mut nested, &mut flat, adam_clipped, 6);
    assert_models_agree(&nested, &flat, TWO_DENSE_PATHS);

    let start = nested_two_dense();
    assert_ne!(
        start.weight("1.second.kernel").unwrap(),
        nested.weight("1.second.kernel").unwrap(),
        "the sublayer did not train"
    );
}

// 2. Gradient addresses

/// A pass through a composite puts each gradient at the path of its sublayer. The values are
/// the bits that the 2 Dense layers give when a caller drives them at 2 model positions
#[test]
fn gradients_land_at_sublayer_paths() {
    let x = ramp(&[4, 6], 0);
    let mut composite = TwoDense::new(dense(4, 2), dense(3, 3));
    composite.build(&Shape::known(&[4, 6])).unwrap();

    let mut ctx = Ctx::training();
    let output = composite.forward(&x, &mut ctx).unwrap();
    let upstream = ramp(output.shape(), 5);
    let grad_input = composite.backward(&upstream, &mut ctx).unwrap();
    assert_eq!(ctx.pending_caches(), 0);
    assert_eq!(ctx.pending_states(), 0);

    let mut held: Vec<String> = ctx.grads().iter().map(|(id, _)| id.to_string()).collect();
    held.sort();
    assert_eq!(
        held,
        vec![
            "0.first.bias",
            "0.first.kernel",
            "0.second.bias",
            "0.second.kernel"
        ]
    );
    assert_eq!(
        param_at(&["first"], "kernel"),
        LayerPath::root(0).child("first").param("kernel")
    );
    assert!(ctx.grads().get(&ParamId::new(0, "kernel")).is_none());
    assert!(ctx.grads().get(&ParamId::new(0, "bias")).is_none());

    // The same 2 layers, driven at model positions 0 and 1
    let mut first = dense(4, 2);
    let mut second = dense(3, 3);
    first.build(&Shape::known(&[4, 6])).unwrap();
    second.build(&Shape::known(&[4, 4])).unwrap();
    let mut flat = Ctx::training();
    flat.set_owner(0);
    let hidden = first.forward(&x, &mut flat).unwrap();
    flat.set_owner(1);
    let flat_output = second.forward(&hidden, &mut flat).unwrap();
    let grad_hidden = second.backward(&upstream, &mut flat).unwrap();
    flat.set_owner(0);
    let flat_grad_input = first.backward(&grad_hidden, &mut flat).unwrap();

    assert_bitwise_eq(output.view(), flat_output.view(), "output");
    assert_bitwise_eq(grad_input.view(), flat_grad_input.view(), "input gradient");
    for (nested, plain) in [
        (param_at(&["first"], "kernel"), ParamId::new(0, "kernel")),
        (param_at(&["first"], "bias"), ParamId::new(0, "bias")),
        (param_at(&["second"], "kernel"), ParamId::new(1, "kernel")),
        (param_at(&["second"], "bias"), ParamId::new(1, "bias")),
    ] {
        assert_bitwise_eq(
            ctx.grads().get(&nested).unwrap().view(),
            flat.grads().get(&plain).unwrap().view(),
            &nested.to_string(),
        );
    }
}

// 3. Finite differences

/// Every array of a layer tree: the path of the node, the name of the array, and its length
fn param_nodes(
    layer: &mut dyn LayerBase,
    path: LayerPath,
    out: &mut Vec<(LayerPath, &'static str, usize)>,
) {
    for param in layer.parameters_mut() {
        out.push((path.clone(), param.name, param.value.len()));
    }
    for sub in layer.sublayers_mut() {
        param_nodes(sub.layer, path.child(sub.name), out);
    }
}

/// Runs `edit` on the flat data of 1 array of a layer tree
fn with_param<R>(
    layer: &mut dyn LayerBase,
    sublayers: &[SublayerName],
    name: &str,
    edit: impl FnOnce(&mut [f32]) -> R,
) -> R {
    match sublayers.split_first() {
        None => {
            let param = layer
                .parameters_mut()
                .into_iter()
                .find(|param| param.name == name)
                .expect("the roster gives the array");
            edit(param.value)
        }
        Some((head, rest)) => {
            let sub = layer
                .sublayers_mut()
                .into_iter()
                .find(|sub| sub.name == *head)
                .expect("the roster gives the sublayer");
            with_param(sub.layer, rest, name, edit)
        }
    }
}

/// A fixed weight tensor for the loss `L = sum(W * output)`
fn loss_weights(shape: &[usize]) -> Tensor {
    ramp(shape, 11).mapv(|v| v + 0.3)
}

/// The loss `L = sum(W * output)` of 1 training pass, summed in f64
fn weighted_loss<L: UnaryLayer>(layer: &L, x: &Tensor, weights: &Tensor) -> f64 {
    let output = layer.forward(x, &mut Ctx::training()).unwrap();
    output
        .iter()
        .zip(weights.iter())
        .map(|(&o, &w)| o as f64 * w as f64)
        .sum()
}

/// Compares every analytic gradient of a layer tree with a central finite difference
///
/// The check covers the input gradient and every array of every node. It also asserts that
/// the store holds no gradient outside the tree. It gives back the number of arrays it checked
fn check_tree_gradients<L: UnaryLayer>(layer: &mut L, x: &Tensor, eps: f32, tol: f64) -> usize {
    layer.build(&Shape::known(x.shape())).unwrap();
    let mut ctx = Ctx::training();
    let output = layer.forward(x, &mut ctx).unwrap();
    let weights = loss_weights(output.shape());
    let grad_input = layer.backward(&weights, &mut ctx).unwrap();
    assert_eq!(ctx.pending_caches(), 0, "a cache stayed behind");

    let mut probe = x.clone();
    for i in 0..x.len() {
        let original = probe.as_slice().unwrap()[i];
        probe.as_slice_mut().unwrap()[i] = original + eps;
        let plus = weighted_loss(layer, &probe, &weights);
        probe.as_slice_mut().unwrap()[i] = original - eps;
        let minus = weighted_loss(layer, &probe, &weights);
        probe.as_slice_mut().unwrap()[i] = original;
        let numeric = (plus - minus) / (2.0 * eps as f64);
        let analytic = grad_input.as_slice().unwrap()[i] as f64;
        assert!(
            (analytic - numeric).abs() <= tol,
            "input element {i}: analytic {analytic:e}, numeric {numeric:e}"
        );
    }

    let mut nodes = Vec::new();
    param_nodes(layer, LayerPath::root(0), &mut nodes);
    assert_eq!(
        ctx.grads().len(),
        nodes.len(),
        "the store holds a gradient outside the tree"
    );
    for (path, name, len) in &nodes {
        let id = path.param(name);
        let analytic = ctx
            .grads()
            .get(&id)
            .unwrap_or_else(|| panic!("no gradient at `{id}`"))
            .clone();
        assert_eq!(analytic.len(), *len, "`{id}` has the wrong length");
        for i in 0..*len {
            let original = with_param(layer, path.sublayers(), name, |value| {
                let original = value[i];
                value[i] = original + eps;
                original
            });
            let plus = weighted_loss(layer, x, &weights);
            with_param(layer, path.sublayers(), name, |value| {
                value[i] = original - eps
            });
            let minus = weighted_loss(layer, x, &weights);
            with_param(layer, path.sublayers(), name, |value| value[i] = original);
            let numeric = (plus - minus) / (2.0 * eps as f64);
            let analytic = analytic.as_slice().unwrap()[i] as f64;
            assert!(
                (analytic - numeric).abs() <= tol,
                "`{id}` element {i}: analytic {analytic:e}, numeric {numeric:e}"
            );
        }
    }
    nodes.len()
}

/// The analytic gradients of a composite of 2 Dense layers agree with finite differences
#[test]
fn two_dense_composite_gradients_match_finite_difference() {
    let mut layer = TwoDense::new(dense(4, 2), dense(3, 3));
    let checked = check_tree_gradients(&mut layer, &ramp(&[3, 5], 0), 1e-2, 1e-3);
    assert_eq!(checked, 4);
}

/// The analytic gradients of a composite that holds a composite agree with finite differences
#[test]
fn depth_two_composite_gradients_match_finite_difference() {
    let mut layer = Pair::new(TwoDense::new(dense(4, 2), dense(3, 3)), dense(2, 4));
    let checked = check_tree_gradients(&mut layer, &ramp(&[3, 5], 0), 1e-2, 1e-3);
    assert_eq!(checked, 6);
}

// 4. Depth 2

/// A composite that holds a composite trains to the same bits as the flat stack. The paths
/// hold 2 sublayer names
#[test]
fn depth_two_composite_trains_like_flat_stack() {
    let nested_model = || {
        SequentialBuilder::new()
            .add(dense(5, 1))
            .add(Pair::new(
                TwoDense::new(dense(4, 2), dense(3, 3)),
                dense(2, 4),
            ))
            .build(&Shape::new(vec![None, Some(6)]))
            .unwrap()
    };
    let flat_model = || {
        SequentialBuilder::new()
            .add(dense(5, 1))
            .add(dense(4, 2))
            .add(dense(3, 3))
            .add(dense(2, 4))
            .build(&Shape::new(vec![None, Some(6)]))
            .unwrap()
    };
    let paths = [
        "0.kernel",
        "0.bias",
        "1.first.first.kernel",
        "1.first.first.bias",
        "1.first.second.kernel",
        "1.first.second.bias",
        "1.second.kernel",
        "1.second.bias",
    ];

    let mut nested = nested_model();
    let mut flat = flat_model();
    train_in_step(&mut nested, &mut flat, sgd_momentum, 5);
    assert_models_agree(&nested, &flat, &paths);

    let mut nested = nested_model();
    let mut flat = flat_model();
    train_in_step(&mut nested, &mut flat, adam_clipped, 5);
    assert_models_agree(&nested, &flat, &paths);
}

// 5. State in sublayers

/// The BatchNormalization and Dropout of a composite keep their state at their own paths
///
/// The moving statistics, the dropout masks, every loss, every trained array, and the
/// inference output all hold the same bits as the flat stack
#[test]
fn normalization_and_dropout_state_inside_composite_matches_flat_stack() {
    let bn = || BatchNormalization::new(0.9, 1e-3).unwrap();
    let drop = || Dropout::new(0.3).unwrap().with_random_state(17);
    let mut nested = SequentialBuilder::new()
        .add(dense(6, 1))
        .add(Pair::new(bn(), drop()))
        .add(dense(2, 2))
        .build(&Shape::new(vec![None, Some(5)]))
        .unwrap();
    let mut flat = SequentialBuilder::new()
        .add(dense(6, 1))
        .add(bn())
        .add(drop())
        .add(dense(2, 2))
        .build(&Shape::new(vec![None, Some(5)]))
        .unwrap();
    let paths = [
        "0.kernel",
        "0.bias",
        "1.first.gamma",
        "1.first.beta",
        "1.first.moving_mean",
        "1.first.moving_variance",
        "2.kernel",
        "2.bias",
    ];

    train_in_step(&mut nested, &mut flat, adam_clipped, 5);
    assert_models_agree(&nested, &flat, &paths);

    let moving_mean = nested.weight("1.first.moving_mean").unwrap();
    assert!(
        moving_mean.iter().any(|&v| v != 0.0),
        "the moving mean did not move"
    );
    assert_bitwise_eq(
        moving_mean,
        flat.weight("1.moving_mean").unwrap(),
        "moving mean",
    );
    assert_bitwise_eq(
        nested.weight("1.first.moving_variance").unwrap(),
        flat.weight("1.moving_variance").unwrap(),
        "moving variance",
    );

    let x = ramp(&[8, 5], 23);
    assert_bitwise_eq(
        nested.predict(&x).unwrap().view(),
        flat.predict(&x).unwrap().view(),
        "inference output",
    );
}

/// The random stream of each Dropout in a composite advances once per training step
///
/// The model holds no trainable array, so only the masks change the loss. 2 steps on 1 batch
/// therefore give 2 losses. The 2 Dropout layers have 2 seeds, and each keeps its own stream
#[test]
fn dropout_stream_inside_composite_advances_per_step() {
    let mut nested = SequentialBuilder::new()
        .add(Pair::new(
            Dropout::new(0.5).unwrap().with_random_state(3),
            Dropout::new(0.5).unwrap().with_random_state(4),
        ))
        .build(&Shape::new(vec![None, Some(16)]))
        .unwrap();
    let mut flat = SequentialBuilder::new()
        .add(Dropout::new(0.5).unwrap().with_random_state(3))
        .add(Dropout::new(0.5).unwrap().with_random_state(4))
        .build(&Shape::new(vec![None, Some(16)]))
        .unwrap();
    nested.compile(sgd_momentum(), MeanSquaredError::new());
    flat.compile(sgd_momentum(), MeanSquaredError::new());

    let x = ramp(&[8, 16], 0);
    let y = Tensor::zeros(IxDyn(&[8, 16]));
    let mut losses = Vec::new();
    for step in 0..4 {
        let nested_loss = nested.train_batch(&x, &y).unwrap();
        let flat_loss = flat.train_batch(&x, &y).unwrap();
        assert_eq!(
            nested_loss.to_bits(),
            flat_loss.to_bits(),
            "step {step}: the masks differ from the flat stack"
        );
        losses.push(nested_loss);
    }
    for pair in losses.windows(2) {
        assert_ne!(pair[0], pair[1], "2 steps drew the same masks");
    }

    // Inference draws no mask, so the output is the input
    assert_bitwise_eq(nested.predict(&x).unwrap().view(), x.view(), "inference");
}

// 6 and 7. Composites that break the contract

/// Asserts that a training step fails, names every address, and changes no array
fn assert_step_refused(model: &mut Sequential, x: &Tensor, y: &Tensor, addresses: &[&str]) {
    let before: Vec<Tensor> = model
        .weight_paths()
        .iter()
        .map(|path| model.weight(path).unwrap().to_owned())
        .collect();
    let message = match model.train_batch(x, y) {
        Ok(loss) => panic!("the step trained, with the loss {loss}"),
        Err(error) => error.to_string(),
    };
    for address in addresses {
        assert!(
            message.contains(address),
            "the message does not name `{address}`: {message}"
        );
    }
    for (path, old) in model.weight_paths().iter().zip(before.iter()) {
        assert_bitwise_eq(model.weight(path).unwrap(), old.view(), path);
    }
}

/// A composite that calls a sublayer outside `Ctx::sublayer` stops the step. The gradients
/// of that sublayer land at the path of the composite, and no parameter claims them
#[test]
fn unscoped_sublayer_call_is_refused() {
    let mut model = SequentialBuilder::new()
        .add(Pair::wired(dense(4, 2), dense(3, 3), Wiring::Unscoped))
        .build(&Shape::new(vec![None, Some(5)]))
        .unwrap();
    model.compile(sgd_momentum(), MeanSquaredError::new());
    assert_step_refused(
        &mut model,
        &ramp(&[4, 5], 0),
        &ramp(&[4, 3], 1),
        &["0.kernel", "0.bias"],
    );
}

/// A composite that calls a Dropout outside `Ctx::sublayer` stops the step. The random
/// stream lands at the path of the composite, and no node takes it back
#[test]
fn unscoped_sublayer_state_is_refused() {
    let mut model = SequentialBuilder::new()
        .add(Pair::wired(
            dense(4, 2),
            Dropout::new(0.5).unwrap().with_random_state(1),
            Wiring::Unscoped,
        ))
        .build(&Shape::new(vec![None, Some(5)]))
        .unwrap();
    model.compile(sgd_momentum(), MeanSquaredError::new());
    assert_step_refused(&mut model, &ramp(&[4, 5], 0), &ramp(&[4, 4], 1), &["0.rng"]);
}

/// A composite that calls a sublayer under a name that its roster does not give stops the
/// step
#[test]
fn misnamed_sublayer_call_is_refused() {
    let mut model = SequentialBuilder::new()
        .add(dense(5, 1))
        .add(Pair::wired(dense(4, 2), dense(3, 3), Wiring::Misnamed))
        .build(&Shape::new(vec![None, Some(6)]))
        .unwrap();
    model.compile(adam_clipped(), MeanSquaredError::new());
    assert_step_refused(
        &mut model,
        &ramp(&[4, 6], 0),
        &ramp(&[4, 3], 1),
        &["1.other"],
    );
}

/// A composite that calls a Dropout under a name that its roster does not give stops the step
#[test]
fn misnamed_sublayer_state_is_refused() {
    let mut model = SequentialBuilder::new()
        .add(Pair::wired(
            dense(4, 2),
            Dropout::new(0.5).unwrap().with_random_state(1),
            Wiring::Misnamed,
        ))
        .build(&Shape::new(vec![None, Some(5)]))
        .unwrap();
    model.compile(sgd_momentum(), MeanSquaredError::new());
    assert_step_refused(
        &mut model,
        &ramp(&[4, 5], 0),
        &ramp(&[4, 4], 1),
        &["0.other"],
    );
}

/// A composite that calls its second Dense under the name of its first Dense stops the step.
/// The 2 sublayers have the same type and the same shapes, so the gradients of the second
/// would otherwise sum into the first, and the second would never train
#[test]
fn crossed_sublayer_call_is_refused() {
    let mut model = SequentialBuilder::new()
        .add(Pair::wired(dense(4, 2), dense(4, 3), Wiring::Crossed))
        .build(&Shape::new(vec![None, Some(4)]))
        .unwrap();
    model.compile(sgd_momentum(), MeanSquaredError::new());
    assert_step_refused(
        &mut model,
        &ramp(&[4, 4], 0),
        &ramp(&[4, 4], 1),
        &["0.first", "Dense"],
    );
}

/// A composite that calls its second Dropout under the name of its first Dropout stops the
/// step before any random stream moves
#[test]
fn crossed_sublayer_state_is_refused() {
    let mut model = SequentialBuilder::new()
        .add(Pair::wired(
            Dropout::new(0.5).unwrap().with_random_state(1),
            Dropout::new(0.5).unwrap().with_random_state(2),
            Wiring::Crossed,
        ))
        .build(&Shape::new(vec![None, Some(4)]))
        .unwrap();
    model.compile(sgd_momentum(), MeanSquaredError::new());
    assert_step_refused(
        &mut model,
        &ramp(&[4, 4], 0),
        &ramp(&[4, 4], 1),
        &["0.first"],
    );
}

// 8. 1 sublayer, 2 calls per pass

/// A tied composite with a square cell, so the chain patterns compose
fn tied(pattern: TiedPattern) -> Tied {
    Tied {
        cell: dense(3, 9),
        pattern,
    }
}

/// `y = cell(cell(x))` with 2 call numbers: the gradient of the cell is the sum over both
/// calls, and it agrees with finite differences
#[test]
fn tied_chain_with_call_numbers_matches_finite_difference() {
    let mut layer = tied(TiedPattern::ChainNumbered);
    let checked = check_tree_gradients(&mut layer, &ramp(&[4, 3], 0), 1e-2, 1e-3);
    assert_eq!(checked, 1 + 1, "the cell holds a kernel and a bias");
}

/// `y = cell(x) + cell(x / 2)` agrees with finite differences, in both backward orders
#[test]
fn tied_sum_matches_finite_difference_in_both_backward_orders() {
    for pattern in [TiedPattern::SumForwardOrder, TiedPattern::SumReverseOrder] {
        let mut layer = tied(pattern);
        check_tree_gradients(&mut layer, &ramp(&[4, 3], 0), 1e-2, 1e-3);
    }
}

/// 1 forward and backward pass of a tied composite, with the input gradient and the store
fn tied_pass(pattern: TiedPattern) -> (Tensor, Tensor, Ctx) {
    let x = ramp(&[4, 3], 2);
    let mut layer = tied(pattern);
    layer.build(&Shape::known(&[4, 3])).unwrap();
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let grad_input = layer
        .backward(&loss_weights(output.shape()), &mut ctx)
        .unwrap();
    assert_eq!(
        ctx.pending_caches(),
        0,
        "{pattern:?}: a cache stayed behind"
    );
    (output, grad_input, ctx)
}

/// The 2 backward orders of `y = cell(x) + cell(x / 2)` give the same bits. Each call number
/// has its own cache stack, so the order of the backward calls does not matter
#[test]
fn tied_sum_backward_order_does_not_change_the_bits() {
    let (out_a, grad_a, ctx_a) = tied_pass(TiedPattern::SumForwardOrder);
    let (out_b, grad_b, ctx_b) = tied_pass(TiedPattern::SumReverseOrder);
    assert_bitwise_eq(out_a.view(), out_b.view(), "output");
    assert_bitwise_eq(grad_a.view(), grad_b.view(), "input gradient");
    assert_eq!(ctx_a.grads().len(), 2);
    for name in ["kernel", "bias"] {
        let id = param_at(&["cell"], name);
        assert_bitwise_eq(
            ctx_a.grads().get(&id).unwrap().view(),
            ctx_b.grads().get(&id).unwrap().view(),
            &id.to_string(),
        );
    }
}

/// `y = cell(cell(x))` with `sublayer` for both calls gives the same bits as with 2 call
/// numbers, when the backward pass runs the calls in reverse order
#[test]
fn tied_chain_with_one_call_number_works_in_reverse_order() {
    let (out_a, grad_a, ctx_a) = tied_pass(TiedPattern::ChainNumbered);
    let (out_b, grad_b, ctx_b) = tied_pass(TiedPattern::ChainSameCall);
    assert_bitwise_eq(out_a.view(), out_b.view(), "output");
    assert_bitwise_eq(grad_a.view(), grad_b.view(), "input gradient");
    for name in ["kernel", "bias"] {
        let id = param_at(&["cell"], name);
        assert_bitwise_eq(
            ctx_a.grads().get(&id).unwrap().view(),
            ctx_b.grads().get(&id).unwrap().view(),
            &id.to_string(),
        );
    }
}

/// A tied composite trains in a model. The gradient of the cell is the sum of the 2 calls, so
/// the step must not fail and the cell must move
#[test]
fn tied_composite_trains_in_a_model() {
    for pattern in [
        TiedPattern::ChainNumbered,
        TiedPattern::ChainSameCall,
        TiedPattern::SumForwardOrder,
        TiedPattern::SumReverseOrder,
    ] {
        let mut model = SequentialBuilder::new()
            .add(tied(pattern))
            .build(&Shape::new(vec![None, Some(3)]))
            .unwrap();
        assert_eq!(model.weight_paths(), vec!["0.cell.kernel", "0.cell.bias"]);
        let before = model.weight("0.cell.kernel").unwrap().to_owned();
        model.compile(adam_clipped(), MeanSquaredError::new());
        model
            .train_batch(&ramp(&[4, 3], 0), &ramp(&[4, 3], 1))
            .unwrap();
        assert_ne!(
            model.weight("0.cell.kernel").unwrap(),
            before,
            "{pattern:?}: the cell did not train"
        );
    }
}

// 9. Graph

/// A graph that calls 1 composite at 2 nodes
fn graph_with_shared_composite() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input(Shape::new(vec![None, Some(4)]));
    let block = builder.layer(TwoDense::new(dense(3, 21), dense(4, 22)));
    let once = builder.apply(block, &[x]);
    let twice = builder.apply(block, &[once]);
    builder.build(&[twice]).unwrap()
}

/// A graph that calls 2 plain Dense layers at 2 nodes each, with the same seeds
fn graph_with_shared_dense() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input(Shape::new(vec![None, Some(4)]));
    let first = builder.layer(dense(3, 21));
    let second = builder.layer(dense(4, 22));
    let a = builder.apply(first, &[x]);
    let b = builder.apply(second, &[a]);
    let c = builder.apply(first, &[b]);
    let d = builder.apply(second, &[c]);
    builder.build(&[d]).unwrap()
}

/// Trains a graph that shares 1 composite and a graph that shares 2 Dense layers. Asserts that
/// every loss, every array, and the inference output hold the same bits
fn assert_shared_graphs_train_alike<O: 'static + Optimizer>(optimizer: impl Fn() -> O) {
    let nested_paths = [
        "0.first.kernel",
        "0.first.bias",
        "0.second.kernel",
        "0.second.bias",
    ];
    let flat_paths = ["0.kernel", "0.bias", "1.kernel", "1.bias"];
    let mut nested = graph_with_shared_composite();
    let mut flat = graph_with_shared_dense();
    assert_eq!(nested.weight_paths(), nested_paths);
    assert_eq!(flat.weight_paths(), flat_paths);
    nested.compile(optimizer(), MeanSquaredError::new());
    flat.compile(optimizer(), MeanSquaredError::new());
    for step in 0..5 {
        let x = ramp(&[6, 4], step);
        let y = ramp(&[6, 4], step + 40).mapv(|v| v * 0.5);
        let nested_loss = nested.train_batch(&[&x], &[&y]).unwrap();
        let flat_loss = flat.train_batch(&[&x], &[&y]).unwrap();
        assert_eq!(
            nested_loss.to_bits(),
            flat_loss.to_bits(),
            "step {step}: the loss differs"
        );
    }
    for (path, flat_path) in nested_paths.iter().zip(flat_paths.iter()) {
        assert_bitwise_eq(
            nested.weight(path).unwrap(),
            flat.weight(flat_path).unwrap(),
            path,
        );
    }
    let x = ramp(&[3, 4], 77);
    assert_bitwise_eq(
        nested.predict(&[&x]).unwrap()[0].view(),
        flat.predict(&[&x]).unwrap()[0].view(),
        "inference output",
    );
}

/// 1 composite at 2 nodes of a graph trains to the same bits as 2 plain Dense layers that
/// the graph calls twice each. The gradient of each sublayer is the sum over both nodes
#[test]
fn graph_shared_composite_trains_like_shared_dense_layers() {
    assert_shared_graphs_train_alike(sgd_momentum);
    assert_shared_graphs_train_alike(adam_clipped);
}

/// The gradient of a composite at 2 graph nodes is the sum of the gradients of the 2 calls
///
/// The test drives the 2 calls by hand at 2 call positions, with 1 owner. A plain SGD step
/// then moves each array by exactly the learning rate times that sum
#[test]
fn graph_shared_composite_gradient_is_the_sum_over_nodes() {
    let x = ramp(&[6, 4], 0);
    let y = ramp(&[6, 4], 40).mapv(|v| v * 0.5);

    let mut block = TwoDense::new(dense(3, 21), dense(4, 22));
    block.build(&Shape::new(vec![None, Some(4)])).unwrap();
    let mut ctx = Ctx::training();
    ctx.set_position(0, 1);
    let once = block.forward(&x, &mut ctx).unwrap();
    ctx.set_position(0, 2);
    let twice = block.forward(&once, &mut ctx).unwrap();
    let loss = MeanSquaredError::new();
    let seed = rustyml::neural_network::traits::Loss::compute_grad(&loss, &y, &twice).unwrap();
    let grad_once = block.backward(&seed, &mut ctx).unwrap();
    ctx.set_position(0, 1);
    block.backward(&grad_once, &mut ctx).unwrap();
    assert_eq!(ctx.pending_caches(), 0);

    let learning_rate = 0.125_f32;
    let mut model = graph_with_shared_composite();
    model.compile(
        SGD::new(learning_rate, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let before: Vec<Tensor> = model
        .weight_paths()
        .iter()
        .map(|path| model.weight(path).unwrap().to_owned())
        .collect();
    model.train_batch(&[&x], &[&y]).unwrap();

    for (path, old) in model.weight_paths().iter().zip(before.iter()) {
        let mut parts = path.split('.').skip(1);
        let sub = parts.next().unwrap();
        let name = parts.next().unwrap();
        let name: &'static str = if name == "kernel" { "kernel" } else { "bias" };
        let grad = ctx
            .grads()
            .get(&LayerPath::root(0).child(sub.to_string()).param(name))
            .unwrap();
        let expected = old - &(grad * learning_rate);
        assert_bitwise_eq(model.weight(path).unwrap(), expected.view(), path);
    }
}

// 10. Summary and parameter counts

/// The summary of a model that holds a composite does not panic. The parameter count of a
/// composite is the sum over its sublayers
#[test]
fn summary_and_param_count_cover_sublayers() {
    let composite = Pair::new(
        TwoDense::new(dense(4, 2), dense(3, 3)),
        BatchNormalization::new(0.9, 1e-3).unwrap(),
    );
    let model = SequentialBuilder::new()
        .add(dense(5, 1))
        .add(composite)
        .build(&Shape::new(vec![None, Some(6)]))
        .unwrap();
    model.summary();
    graph_with_shared_composite().summary();

    let mut composite = Pair::new(
        TwoDense::new(dense(4, 2), dense(3, 3)),
        BatchNormalization::new(0.9, 1e-3).unwrap(),
    );
    composite.build(&Shape::new(vec![None, Some(5)])).unwrap();
    let counts = total_param_count(&composite);
    let parts = [
        composite.first.first.param_count(),
        composite.first.second.param_count(),
        composite.second.param_count(),
    ];
    let trainable: usize = parts.iter().map(|count| count.trainable).sum();
    let non_trainable: usize = parts.iter().map(|count| count.non_trainable).sum();
    assert_eq!(trainable, (5 * 4 + 4) + (4 * 3 + 3) + (3 + 3));
    assert_eq!(non_trainable, 3 + 3);
    assert_eq!(counts.trainable, trainable);
    assert_eq!(counts.non_trainable, non_trainable);
}

// 11. Threads

/// A built model that holds a composite is `Send` and `Sync`. 4 threads that run inference
/// on 1 shared model give the same bits as 1 thread
#[test]
fn inference_on_shared_model_matches_across_threads() {
    fn assert_send_sync<T: Send + Sync>(_: &T) {}

    let mut model = SequentialBuilder::new()
        .add(dense(6, 1))
        .add(Pair::new(
            BatchNormalization::new(0.9, 1e-3).unwrap(),
            Dropout::new(0.3).unwrap().with_random_state(5),
        ))
        .add(TwoDense::new(dense(4, 2), dense(2, 3)))
        .build(&Shape::new(vec![None, Some(5)]))
        .unwrap();
    model.compile(adam_clipped(), MeanSquaredError::new());
    for step in 0..3 {
        model
            .train_batch(&ramp(&[8, 5], step), &ramp(&[8, 2], step + 9))
            .unwrap();
    }
    assert_send_sync(&model);

    let inputs: Vec<Tensor> = (0..4).map(|i| ramp(&[16, 5], i * 13)).collect();
    let serial: Vec<Tensor> = inputs.iter().map(|x| model.predict(x).unwrap()).collect();
    let parallel: Vec<Tensor> = std::thread::scope(|scope| {
        let handles: Vec<_> = inputs
            .iter()
            .map(|x| {
                let model = &model;
                scope.spawn(move || model.predict(x).unwrap())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (index, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
        assert_bitwise_eq(b.view(), a.view(), &format!("input {index}"));
    }
}
