//! Integration tests for the Identity layer.
//!
//! Covers the pass-through at every rank, memory layout, the error paths, and that a model
//! carrying the layer trains exactly as one without it.

use ndarray::{Array2, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::identity::Identity;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::{GlobalSeedGuard, assert_allclose};

/// A ramp of distinct values, in the given shape
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|k| k as f32 * 0.25 - 3.0).collect();
    Tensor::from_shape_vec(IxDyn(shape), data).expect("product of the shape")
}

/// Every value comes back untouched, at every rank from 1 through 5
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

        let out = layer.forward(&x).unwrap();
        assert_eq!(out, x, "rank {} changed a value", shape.len());

        let inferred = layer.predict(&x).unwrap();
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
    layer.forward(&x).unwrap();

    let upstream = ramp(&[3, 4]).mapv(|v| v * 2.0 + 1.0);
    let grad = layer.backward(&upstream).unwrap();
    assert_eq!(grad, upstream);
}

/// `new` and `Default` build the same layer
#[test]
fn identity_default_matches_new() {
    let x = ramp(&[2, 3]);
    let from_new = Identity::new().predict(&x).unwrap();
    let from_default = Identity::default().predict(&x).unwrap();
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
    let out = layer.forward(&transposed).unwrap();
    assert!(out.is_standard_layout(), "forward is not in C order");
    assert_eq!(out, transposed, "settling the layout changed a value");

    let grad = layer.backward(&transposed).unwrap();
    assert!(grad.is_standard_layout(), "backward is not in C order");
    assert_eq!(grad, transposed);
}

/// The layer holds nothing an optimizer can update
#[test]
fn identity_holds_no_parameter() {
    let mut layer = Identity::new();
    assert_eq!(layer.layer_type(), "Identity");
    assert!(matches!(
        layer.param_count(),
        TrainingParameters::NoTrainable
    ));
    assert!(matches!(layer.get_weights(), LayerWeight::Empty));
    assert!(layer.parameters().is_empty());
}

/// The summary reads "Unknown" until a tensor has been through the layer
#[test]
fn identity_output_shape_needs_a_forward_pass() {
    let mut layer = Identity::new();
    assert_eq!(layer.output_shape(), "Unknown");

    layer.forward(&ramp(&[2, 3, 4])).unwrap();
    assert_eq!(layer.output_shape(), "(None, 3, 4)");

    // A rank-1 input has nothing after its batch axis
    layer.forward(&ramp(&[4])).unwrap();
    assert_eq!(layer.output_shape(), "(None,)");
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
        let mut model = Sequential::new();
        if with_identity {
            model.add(Identity::new());
        }
        model.add(Dense::new(3, 2, Linear::new()).unwrap());
        if with_identity {
            model.add(Identity::new());
        }
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

/// `predict` writes no cache, so it cannot serve a later backward pass
#[test]
fn identity_predict_caches_nothing() {
    let x = ramp(&[2, 3]);
    let mut layer = Identity::new();
    layer.predict(&x).unwrap();

    assert!(
        matches!(
            layer.backward(&x),
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
            Identity::new().predict(&scalar),
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
        matches!(Identity::new().forward(&empty), Err(Error::EmptyInput(_))),
        "Identity must reject an empty input"
    );
}

/// A backward pass before any forward pass has no cache to read
#[test]
fn identity_backward_needs_a_forward_pass() {
    let mut layer = Identity::new();
    assert!(
        matches!(
            layer.backward(&ramp(&[2, 3])),
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "backward must reject a missing cache"
    );
}

/// A gradient that does not have the shape of the forward input is rejected
#[test]
fn identity_backward_rejects_a_wrong_shape() {
    let mut layer = Identity::new();
    layer.forward(&ramp(&[2, 3])).unwrap();

    let result = layer.backward(&ramp(&[2, 4]));
    assert!(
        matches!(
            result,
            Err(Error::ShapeMismatch { expected, found })
                if expected == vec![2, 3] && found == vec![2, 4]
        ),
        "backward must report the shape it expected"
    );
}
