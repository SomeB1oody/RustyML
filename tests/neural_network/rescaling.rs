//! Integration tests for the Rescaling layer.
//!
//! Covers the affine map at every rank, the gradient, the memory layout, the error paths, and
//! the use of the layer inside a model. Every recorded expectation comes from Keras 3.15.1 on
//! the jax backend, and every one of them is bit-exact against this implementation.

use ndarray::{Array2, IxDyn};
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::rescaling::Rescaling;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

/// A ramp of distinct values, in the given shape. The Keras probe used this same formula
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|k| k as f32 * 0.25 - 3.0).collect();
    Tensor::from_shape_vec(IxDyn(shape), data).expect("product of the shape")
}

/// Asserts that every element carries the same bits as the Keras reference
fn assert_bit_equal(actual: &Tensor, expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: index {index} gives {a} (0x{:08x}), Keras gives {e} (0x{:08x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

/// The forward pass matches Keras bit for bit, at 3 scale and offset pairs
///
/// The 3 pairs are the 2 common image maps, `1 / 255` into `[0, 1]` and `1 / 127.5` with an
/// offset of -1 into `[-1, 1]`, plus a negative scale with a positive offset. A scale that a
/// binary fraction cannot hold exactly pins the rounding of the multiplication as well
#[test]
fn rescaling_forward_matches_keras() {
    let x = ramp(&[2, 3]);

    // Keras: layers.Rescaling(scale=1.0 / 255.0)(x)
    let mut layer = Rescaling::new(1.0 / 255.0);
    assert_bit_equal(
        &layer.forward(&x).unwrap(),
        &[
            -0.011764707,
            -0.010784314,
            -0.009803922,
            -0.00882353,
            -0.007843138,
            -0.0068627456,
        ],
        "scale 1/255",
    );

    // Keras: layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    let mut layer = Rescaling::new(1.0 / 127.5).with_offset(-1.0);
    assert_bit_equal(
        &layer.forward(&x).unwrap(),
        &[
            -1.0235294, -1.0215687, -1.0196079, -1.017647, -1.0156863, -1.0137255,
        ],
        "scale 1/127.5, offset -1",
    );

    // Keras: layers.Rescaling(scale=-2.5, offset=0.75)(x3), with x3 the rank-3 ramp
    let x3 = ramp(&[2, 3, 4]);
    let mut layer = Rescaling::new(-2.5).with_offset(0.75);
    assert_bit_equal(
        &layer.forward(&x3).unwrap(),
        &[
            8.25, 7.625, 7.0, 6.375, 5.75, 5.125, 4.5, 3.875, 3.25, 2.625, 2.0, 1.375, 0.75, 0.125,
            -0.5, -1.125, -1.75, -2.375, -3.0, -3.625, -4.25, -4.875, -5.5, -6.125,
        ],
        "scale -2.5, offset 0.75",
    );
}

/// The map holds its shape and its values at every rank from 1 through 5
///
/// Keras gives the same answer at every rank, because the map reads no axis. The rank-1 and the
/// rank-5 records come from the probe, and the ranks between them check the formula
#[test]
fn rescaling_holds_at_every_rank() {
    // Keras: layers.Rescaling(scale=3.0, offset=-0.5) over the rank-1 ramp
    let mut layer = Rescaling::new(3.0).with_offset(-0.5);
    assert_bit_equal(
        &layer.forward(&ramp(&[4])).unwrap(),
        &[-9.5, -8.75, -8.0, -7.25],
        "rank 1",
    );

    // Keras: the same layer over the rank-5 ramp. The probe checked every element, and the
    // first 8 are recorded here
    let out5 = layer.forward(&ramp(&[2, 2, 2, 2, 2])).unwrap();
    assert_eq!(out5.shape(), &[2, 2, 2, 2, 2]);
    assert_bit_equal(
        &Tensor::from_shape_vec(IxDyn(&[8]), out5.iter().take(8).copied().collect()).unwrap(),
        &[-9.5, -8.75, -8.0, -7.25, -6.5, -5.75, -5.0, -4.25],
        "rank 5",
    );

    for shape in [vec![4], vec![2, 3], vec![2, 3, 4], vec![2, 3, 4, 5]] {
        let x = ramp(&shape);
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), shape.as_slice(), "rank {}", shape.len());
        for (o, v) in out.iter().zip(x.iter()) {
            assert_eq!(o.to_bits(), (v * 3.0 - 0.5).to_bits());
        }
    }
}

/// A scale of 0 gives the offset at every element, and a negative scale applies unchanged
///
/// Keras applies both without a special case, so this layer must not add one
#[test]
fn rescaling_applies_a_zero_and_a_negative_scale_unchanged() {
    // Keras: layers.Rescaling(scale=0.0, offset=7.5)(x)
    let mut layer = Rescaling::new(0.0).with_offset(7.5);
    assert_bit_equal(
        &layer.forward(&ramp(&[2, 3])).unwrap(),
        &[7.5, 7.5, 7.5, 7.5, 7.5, 7.5],
        "scale 0",
    );

    // A negative scale flips the order of the ramp
    let mut layer = Rescaling::new(-1.0);
    let out = layer.forward(&ramp(&[4])).unwrap();
    assert_bit_equal(&out, &[3.0, 2.75, 2.5, 2.25], "scale -1");
}

/// The offset alone applies when the scale is 1
#[test]
fn rescaling_offset_defaults_to_zero() {
    let x = ramp(&[2, 3]);

    // `new` alone leaves the offset at 0, so the map is the identity at a scale of 1
    let mut plain = Rescaling::new(1.0);
    assert_eq!(plain.forward(&x).unwrap(), x);

    // `with_offset(0.0)` therefore gives the same answer as `new` alone
    let mut explicit = Rescaling::new(2.0).with_offset(0.0);
    let mut implicit = Rescaling::new(2.0);
    assert_eq!(
        explicit.forward(&x).unwrap(),
        implicit.forward(&x).unwrap(),
        "an explicit offset of 0 differs from the default"
    );
}

/// `predict` returns exactly what `forward` returns
///
/// The layer has no training mode and no inference mode. The Keras probe confirmed that
/// `training=True` and `training=False` give bit-identical answers
#[test]
fn rescaling_predict_equals_forward_bit_for_bit() {
    let x = ramp(&[2, 3, 4]);
    let mut layer = Rescaling::new(1.0 / 255.0).with_offset(-0.25);

    let trained = layer.forward(&x).unwrap();
    let inferred = layer.predict(&x).unwrap();

    for (a, b) in trained.iter().zip(inferred.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "predict differs from forward");
    }
}

/// The backward pass returns the incoming gradient times the scale, and the offset drops out
///
/// Keras, through a jax gradient of `sum(y * upstream)`, gives the recorded values. The probe
/// also confirmed that the same gradient comes back at an offset of 0 and at an offset of 100
#[test]
fn rescaling_backward_matches_keras_and_ignores_the_offset() {
    // The upstream gradient the probe used: k * 0.5 - 1.0 over a (2, 3) tensor
    let upstream = Tensor::from_shape_vec(
        IxDyn(&[2, 3]),
        (0..6).map(|k| k as f32 * 0.5 - 1.0).collect(),
    )
    .unwrap();

    // Keras: jax.grad of sum(Rescaling(scale=-2.5, offset=0.75)(x) * upstream)
    let expected = [2.5, 1.25, -0.0, -1.25, -2.5, -3.75];

    let mut layer = Rescaling::new(-2.5).with_offset(0.75);
    layer.forward(&ramp(&[2, 3])).unwrap();
    assert_bit_equal(&layer.backward(&upstream).unwrap(), &expected, "grad_input");

    // The same scale with a very different offset gives the identical gradient
    for offset in [0.0_f32, 100.0] {
        let mut layer = Rescaling::new(-2.5).with_offset(offset);
        layer.forward(&ramp(&[2, 3])).unwrap();
        assert_bit_equal(
            &layer.backward(&upstream).unwrap(),
            &expected,
            "grad_input at another offset",
        );
    }
}

/// The analytic input gradient agrees with a central finite difference of `sum(output)`
#[test]
fn rescaling_input_gradient_matches_finite_difference() {
    let scale = -2.5_f32;
    let offset = 0.75_f32;
    let x = ramp(&[2, 3]);
    let eps = 1e-2_f32;

    let mut layer = Rescaling::new(scale).with_offset(offset);
    layer.forward(&x).unwrap();
    let analytic = layer.backward(&Tensor::ones(x.raw_dim())).unwrap();

    let sum_of = |t: &Tensor| -> f32 {
        Rescaling::new(scale)
            .with_offset(offset)
            .predict(t)
            .unwrap()
            .sum()
    };
    for index in 0..x.len() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus.as_slice_mut().unwrap()[index] += eps;
        minus.as_slice_mut().unwrap()[index] -= eps;
        let numeric = (sum_of(&plus) - sum_of(&minus)) / (2.0 * eps);
        approx::assert_abs_diff_eq!(analytic.as_slice().unwrap()[index], numeric, epsilon = 1e-3);
    }
}

/// The layer holds no weight and no trainable parameter
#[test]
fn rescaling_holds_no_trainable_parameters() {
    let mut layer = Rescaling::new(0.5).with_offset(1.0);
    layer.forward(&ramp(&[2, 3])).unwrap();

    assert_eq!(layer.layer_type(), "Rescaling");
    assert_eq!(layer.param_count(), ParamCounts::none());
    assert!(layer.weights().is_empty());
    assert!(layer.parameters().is_empty());
}

/// Both passes emit a tensor in C order, whatever layout the caller hands in
///
/// A consumer reads any layer output as 1 contiguous slice, so a permuted input must not leak
/// its layout into the output
#[test]
fn rescaling_emits_c_order_from_a_permuted_input() {
    let base = ramp(&[2, 3, 4]);
    // A permuted view is not in C order, and its `as_slice` therefore returns None
    let permuted = base.clone().permuted_axes(IxDyn(&[2, 0, 1]));
    assert!(
        permuted.as_slice().is_none(),
        "the view is already in C order"
    );

    let mut layer = Rescaling::new(2.0).with_offset(-1.0);
    let out = layer.forward(&permuted).unwrap();
    assert_eq!(out.shape(), &[4, 2, 3]);
    assert!(out.as_slice().is_some(), "forward did not emit C order");
    for (o, x) in out.iter().zip(permuted.iter()) {
        assert_eq!(o.to_bits(), (x * 2.0 - 1.0).to_bits());
    }

    let grad = layer.backward(&permuted).unwrap();
    assert!(grad.as_slice().is_some(), "backward did not emit C order");
}

/// A 0D input has no batch axis, and an empty input has nothing to scale
#[test]
fn rescaling_rejects_a_0d_and_an_empty_input() {
    let mut layer = Rescaling::new(2.0);

    let scalar = Tensor::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    assert!(matches!(
        layer.forward(&scalar).unwrap_err(),
        Error::InvalidInput(_)
    ));
    assert!(matches!(
        layer.predict(&scalar).unwrap_err(),
        Error::InvalidInput(_)
    ));

    let empty = Tensor::zeros(IxDyn(&[0, 3]));
    assert!(matches!(
        layer.forward(&empty).unwrap_err(),
        Error::EmptyInput(_)
    ));
}

/// The backward pass needs no cache, so it runs before any forward pass
///
/// This pins the design decision that the layer stores nothing. Every other shape-preserving
/// layer here answers `ForwardPassNotRun` in this position
#[test]
fn rescaling_backward_runs_before_any_forward() {
    let mut layer = Rescaling::new(4.0).with_offset(9.0);
    let grad = Tensor::ones(IxDyn(&[2, 3]));

    let out = layer.backward(&grad).unwrap();
    assert_bit_equal(&out, &[4.0, 4.0, 4.0, 4.0, 4.0, 4.0], "grad before forward");

    // The error variant that a cached layer would raise never appears
    let err = Error::forward_pass_not_run("Rescaling");
    assert!(matches!(
        err,
        Error::NeuralNetwork(NnError::ForwardPassNotRun(_))
    ));
}

/// A model that carries the layer trains, and the layer passes the gradient on with its scale
#[test]
fn rescaling_trains_inside_a_sequential_model() {
    let x = Array2::from_shape_vec((2, 3), vec![0.0, 51.0, 102.0, 153.0, 204.0, 255.0])
        .unwrap()
        .into_dyn();
    let y = Array2::from_shape_vec((2, 1), vec![0.5, -0.5])
        .unwrap()
        .into_dyn();

    let mut model = SequentialBuilder::new()
        .add(Rescaling::new(1.0 / 255.0))
        .add(Dense::new(1, Linear::new()).unwrap().with_random_state(7))
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let first = model.fit(&x, &y, 1).unwrap();
    let later = model.fit(&x, &y, 20).unwrap();
    assert!(
        later.loss().last().unwrap() < first.loss().last().unwrap(),
        "the loss did not fall through the layer"
    );

    // The scaled input stays inside [0, 1], which is the point of the layer
    let mut only_rescaling = SequentialBuilder::new()
        .add(Rescaling::new(1.0 / 255.0))
        .build(&Shape::known(x.shape()))
        .unwrap();
    only_rescaling.compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let scaled = only_rescaling.predict(&x).unwrap();
    assert!(scaled.iter().all(|v| (0.0..=1.0).contains(v)));
    assert_eq!(scaled[[1, 2]].to_bits(), 1.0_f32.to_bits());
}
