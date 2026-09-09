//! Integration tests for the Identity layer.
//!
//! Covers the pass-through at every rank, memory layout, the error paths, and that a model
//! carrying the layer trains exactly as one without it.

use ndarray::{Array2, IxDyn};
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::identity::Identity;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::{Layer, LayerBase, UnaryLayer};
use rustyml::{error::Error, neural_network::NnError};

use super::common::{GlobalSeedGuard, assert_allclose};

/// A ramp of distinct values, in the given shape
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|k| k as f32 * 0.25 - 3.0).collect();
    Tensor::from_shape_vec(IxDyn(shape), data).expect("product of the shape")
}

/// Every value returns untouched, at every rank from 1 through 5
#[test]
fn identity_passes_every_rank_through_unchanged() {
    for shape in [
        vec![4],
        vec![2, 3],
        vec![2, 3, 4],
        vec![2, 3, 4, 5],
        vec![2, 2, 2, 2, 2],
    ] {
        let x = ramp(&shape);
        let mut layer = Identity::new();

        let out = layer.forward_mut(&x, &mut Ctx::training()).unwrap();
        assert_eq!(out, x, "rank {} changed a value", shape.len());

        let inferred = layer.forward(&x, &mut Ctx::inference()).unwrap();
        assert_eq!(
            inferred,
            x,
            "rank {} changed a value in predict",
            shape.len()
        );
    }
}

/// The gradient goes back exactly as it arrived
#[test]
fn identity_passes_its_gradient_through_unchanged() {
    let x = ramp(&[3, 4]);
    let mut layer = Identity::new();
    let mut ctx = Ctx::training();
    layer.forward_mut(&x, &mut ctx).unwrap();

    let upstream = ramp(&[3, 4]).mapv(|v| v * 2.0 + 1.0);
    let grad = layer.backward(&upstream, &mut ctx).unwrap();
    assert_eq!(grad, upstream);
}

#[test]
fn identity_default_matches_new() {
    let x = ramp(&[2, 3]);
    let from_new = Identity::new().forward(&x, &mut Ctx::inference()).unwrap();
    let from_default = Identity::default()
        .forward(&x, &mut Ctx::inference())
        .unwrap();
    assert_eq!(from_new, from_default);
}

/// Both passes emit C order, so a consumer can read either as 1 contiguous slice
#[test]
fn identity_output_is_in_c_order() {
    let base = Array2::from_shape_vec((3, 5), (0..15).map(|k| k as f32).collect())
        .unwrap()
        .into_dyn();
    let transposed = base.t().to_owned();
    assert!(
        !transposed.is_standard_layout(),
        "the test needs a transposed input that kept its strides"
    );

    let mut layer = Identity::new();
    let mut ctx = Ctx::training();
    let out = layer.forward_mut(&transposed, &mut ctx).unwrap();
    assert!(out.is_standard_layout(), "forward is not in C order");
    assert_eq!(out, transposed, "settling the layout changed a value");

    let grad = layer.backward(&transposed, &mut ctx).unwrap();
    assert!(grad.is_standard_layout(), "backward is not in C order");
    assert_eq!(grad, transposed);
}

/// The layer holds nothing an optimizer can update
#[test]
fn identity_holds_no_parameter() {
    let mut layer = Identity::new();
    assert_eq!(layer.layer_type(), "Identity");
    assert_eq!(layer.param_count(), ParamCounts::none());
    assert!(layer.weights().is_empty());
    assert!(layer.parameters_mut().is_empty());
}

/// The summary reads "Unknown" until a tensor has passed through the layer, and afterward it
/// reports the shape that the first forward pass built for
///
/// `known_input_shapes` reports the build shape, and the build reads the full shape of the
/// tensor that triggered it, batch axis included. A second forward pass through the same
/// instance still runs, at whatever rank it receives, because `Identity::forward` reads no
/// build state. The summary does not move, because the layer already holds its build
#[test]
fn identity_output_shape_needs_a_forward_pass() {
    let mut layer = Identity::new();
    assert_eq!(layer.output_shape(), "Unknown");

    layer
        .forward_mut(&ramp(&[2, 3, 4]), &mut Ctx::training())
        .unwrap();
    assert_eq!(layer.output_shape(), "(None, 3, 4)");

    // A rank-1 input still runs, and the summary keeps reporting the first build
    layer.forward(&ramp(&[4]), &mut Ctx::training()).unwrap();
    assert_eq!(layer.output_shape(), "(None, 3, 4)");
}

/// A model carrying the layer trains to the same weights as one without it
#[test]
fn identity_does_not_change_what_a_model_learns() {
    let x = Array2::from_shape_vec((4, 3), (0..12).map(|k| k as f32 * 0.1 - 0.5).collect())
        .unwrap()
        .into_dyn();
    let y = Array2::from_shape_vec((4, 2), (0..8).map(|k| k as f32 * 0.2 - 0.7).collect())
        .unwrap()
        .into_dyn();

    // The guard clears the seed on drop. Re-seeding before each build gives both models the
    // same Dense weights, so any difference in the loss came from the extra layers
    let _guard = GlobalSeedGuard::set(7);
    let build = |with_identity: bool| {
        rustyml::set_global_seed(7);
        let mut builder = SequentialBuilder::new();
        if with_identity {
            builder = builder.add(Identity::new());
        }
        builder = builder.add(Dense::new(2, Linear::new()).unwrap());
        if with_identity {
            builder = builder.add(Identity::new());
        }
        let mut model = builder.build(&Shape::known(x.shape())).unwrap();
        model.compile(
            SGD::new(0.05, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );
        model
    };

    let mut plain = build(false);
    let mut padded = build(true);
    let plain_history = plain.fit(&x, &y, 5).unwrap();
    let padded_history = padded.fit(&x, &y, 5).unwrap();

    for (a, b) in plain_history
        .loss()
        .iter()
        .zip(padded_history.loss().iter())
    {
        assert!(
            (a - b).abs() < 1e-6,
            "the extra layers moved the loss: {a} versus {b}"
        );
    }
    assert_allclose(
        &plain.predict(&x).unwrap(),
        &padded.predict(&x).unwrap(),
        1e-6,
    );
}

/// A forward pass with an inference context writes no cache, so it cannot serve a later
/// backward pass
#[test]
fn identity_predict_caches_nothing() {
    let x = ramp(&[2, 3]);
    let layer = Identity::new();
    let mut ctx = Ctx::inference();
    layer.forward(&x, &mut ctx).unwrap();

    assert!(
        matches!(
            layer.backward(&x, &mut ctx),
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "predict must not leave a cache behind"
    );
}

/// A tensor with no batch axis is rejected
#[test]
fn identity_rejects_a_rank_0_input() {
    let scalar = Tensor::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    assert!(
        matches!(
            Identity::new().forward(&scalar, &mut Ctx::inference()),
            Err(Error::InvalidInput(_))
        ),
        "a rank-0 input has no batch axis"
    );
}

/// An axis with no elements is rejected, as it is everywhere else in this crate
#[test]
fn identity_rejects_empty_input() {
    let empty = Tensor::zeros(IxDyn(&[0, 3]));
    assert!(
        matches!(
            Identity::new().forward_mut(&empty, &mut Ctx::training()),
            Err(Error::EmptyInput(_))
        ),
        "Identity must reject an empty input"
    );
}

/// A backward pass before any forward pass has no cache to read
#[test]
fn identity_backward_needs_a_forward_pass() {
    let layer = Identity::new();
    assert!(
        matches!(
            layer.backward(&ramp(&[2, 3]), &mut Ctx::training()),
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "backward must reject a missing cache"
    );
}

/// A gradient that does not have the shape of the forward input is rejected
#[test]
fn identity_backward_rejects_a_wrong_shape() {
    let mut layer = Identity::new();
    let mut ctx = Ctx::training();
    layer.forward_mut(&ramp(&[2, 3]), &mut ctx).unwrap();

    let result = layer.backward(&ramp(&[2, 4]), &mut ctx);
    assert!(
        matches!(
            result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == vec![2, 3] && found == vec![2, 4]
        ),
        "backward must report the shape it expected"
    );
}
