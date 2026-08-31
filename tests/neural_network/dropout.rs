//! Integration tests for Dropout, SpatialDropout1D, SpatialDropout2D, and SpatialDropout3D
//!
//! Focus areas:
//! - Forward values (not just shapes): inverted-dropout scaling, rate=0 identity, rate=1 zeros
//! - Eval mode: output == input (identity), predict() == forward() in eval mode
//! - SpatialDropout channel-consistency: whole channel 0 or fully kept and scaled
//! - Constructor rejects invalid rates
//! - backward before forward -> Err(ForwardPassNotRun)
//! - backward() in eval mode passes gradient through unchanged
//! - Shape / ndim validation error paths
//! - Dropout noise_shape: the small draw, the shared axis, the right-aligned short form, and
//!   the entry validation. The recorded semantics come from Keras 3.15.1 on the jax backend

use ndarray::Array;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::dropout::dropout::Dropout;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_1d::SpatialDropout1D;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_2d::SpatialDropout2D;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_3d::SpatialDropout3D;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// Helpers

/// Tensor of all-ones with the given flat shape
fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(shape)
}

/// Flat tensor filled with the given constant value
fn filled(shape: &[usize], value: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = vec![value; n];
    Tensor::from_shape_vec(ndarray::IxDyn(shape), data).unwrap()
}

// Dropout - basic forward values

#[test]
fn dropout_rate_zero_is_identity_in_training_mode() {
    // rate=0 takes the dedicated early-return path, so no units drop
    let mut layer = Dropout::new(0.0, vec![3, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = Array::from_shape_vec(
        (3, 4),
        vec![
            1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -0.5, 2.5, 0.0, 1.0, -1.0, 0.25,
        ],
    )
    .unwrap()
    .into_dyn();

    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn dropout_rate_one_yields_zeros_in_training_mode() {
    // rate=1 takes the dedicated early-return path, giving all zeros
    let mut layer = Dropout::new(1.0, vec![2, 5]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 5], 3.0);
    let output = layer.forward(&input).unwrap();
    let expected = filled(&[2, 5], 0.0);
    assert_allclose(&output, &expected, 1e-6_f32);
}

#[test]
fn dropout_eval_mode_is_exact_identity() {
    // Inverted dropout: inference passes the input through unchanged
    let mut layer = Dropout::new(0.5, vec![2, 3]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = Array::from_shape_vec((2, 3), vec![1.0, -1.0, 2.0, 0.5, -0.5, 3.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn dropout_predict_equals_forward_in_eval_mode() {
    let mut layer = Dropout::new(0.3, vec![2, 4]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = Array::from_shape_vec((2, 4), (0..8).map(|v| v as f32 * 0.5).collect())
        .unwrap()
        .into_dyn();

    let out_forward = layer.forward(&input).unwrap();
    let out_predict = layer.predict(&input).unwrap();
    assert_allclose(&out_forward, &out_predict, 1e-6_f32);
    assert_allclose(&out_forward, &input, 1e-6_f32);
}

#[test]
fn dropout_predict_is_identity_in_training_mode() {
    let mut layer = Dropout::new(0.9, vec![3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = Array::from_shape_vec((3,), vec![1.0, -2.0, 3.0])
        .unwrap()
        .into_dyn();
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn dropout_training_inverted_scaling_on_kept_units() {
    // Inverted dropout scales kept units by 1/(1-rate). With rate=0.5, scale = 2.0.
    let mut layer = Dropout::new(0.5, vec![200]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[200]);
    let output = layer.forward(&input).unwrap();

    let mut zero_count = 0usize;
    for &v in output.iter() {
        if v == 0.0 {
            zero_count += 1;
        } else {
            // Kept element equals input * 1/(1-0.5) = 2.0
            approx::assert_abs_diff_eq!(v, 2.0_f32, epsilon = 1e-5);
        }
    }
    // The wide [20%, 80%] window tolerates randomness around the ~50% expected zero fraction
    let zero_frac = zero_count as f32 / 200.0;
    assert!(
        (0.20..=0.80).contains(&zero_frac),
        "Expected ~50% zeros, got {:.1}%",
        zero_frac * 100.0
    );
}

#[test]
fn dropout_rate_one_backward_returns_zeros() {
    // rate=1 zeroes the forward output. Backward returns zeros too, from its own early-return path
    let mut layer = Dropout::new(1.0, vec![2, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 3], 1.0);
    let _ = layer.forward(&input).unwrap();

    let grad = filled(&[2, 3], 1.0);
    let grad_in = layer.backward(&grad).unwrap();
    let expected = filled(&[2, 3], 0.0);
    assert_allclose(&grad_in, &expected, 1e-6_f32);
}

#[test]
fn dropout_rate_zero_backward_passes_gradient_through() {
    // rate=0 -> forward identity, backward passes gradient through unchanged
    let mut layer = Dropout::new(0.0, vec![2, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 3], 1.0);
    let _ = layer.forward(&input).unwrap();

    let grad = Array::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad).unwrap();
    assert_allclose(&grad_in, &grad, 1e-6_f32);
}

#[test]
fn dropout_eval_backward_passes_gradient_through() {
    // Inference mode: backward() passes gradient through unchanged
    let mut layer = Dropout::new(0.5, vec![2, 3]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[2, 3], 1.0);
    let _ = layer.forward(&input).unwrap();

    let grad = Array::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad).unwrap();
    assert_allclose(&grad_in, &grad, 1e-6_f32);
}

#[test]
fn dropout_backward_kept_units_scaled_correctly() {
    // Backward multiplies grad_output by 1/(1-rate). With rate=0.5 and all-ones input,
    // kept output and kept gradient both equal 2.0, and dropped units give 0.0
    let mut layer = Dropout::new(0.5, vec![50]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[50]);
    let output = layer.forward(&input).unwrap();
    let grad_upstream = ones(&[50]);
    let grad_in = layer.backward(&grad_upstream).unwrap();

    for (i, (&out_v, &grad_v)) in output.iter().zip(grad_in.iter()).enumerate() {
        if out_v == 0.0 {
            approx::assert_abs_diff_eq!(grad_v, 0.0_f32, epsilon = 1e-5);
        } else {
            // Kept unit: gradient = grad_upstream * scale = 1.0 * 2.0
            assert!(
                (grad_v - 2.0_f32).abs() < 1e-5,
                "index {i}: kept unit gradient should be 2.0, got {grad_v}"
            );
        }
    }
}

// Dropout - error paths

#[test]
fn dropout_constructor_rejects_invalid_rate() {
    // Same constructor and error for a negative rate and a rate above 1
    for rate in [-0.1_f32, 1.5_f32] {
        let err = Dropout::new(rate, vec![10]).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "rate {rate}: expected InvalidParameter, got {:?}",
            err
        );
    }
}

#[test]
fn dropout_constructor_accepts_boundary_rates_zero_and_one() {
    assert!(Dropout::new(0.0, vec![5]).is_ok());
    assert!(Dropout::new(1.0, vec![5]).is_ok());
}

#[test]
fn dropout_backward_before_forward_returns_forward_pass_not_run() {
    // mask is None -> backward before forward -> ForwardPassNotRun error
    let mut layer = Dropout::new(0.5, vec![4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[4], 1.0);
    let err = layer.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

#[test]
fn spatial_dropout_backward_before_forward_reports_concrete_layer_name() {
    // Regression: shared dropout_backward must name the concrete SpatialDropout layer
    // in ForwardPassNotRun, not a hardcoded "Dropout"
    // Channels-last: (batch=2, length=8, channels=4)
    let mut d1 = SpatialDropout1D::new(0.5, vec![2, 8, 4]).unwrap();
    d1.set_training_if_mode_dependent(true);
    let err1 = d1.backward(&filled(&[2, 8, 4], 1.0)).unwrap_err();
    assert!(
        matches!(
            err1,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SpatialDropout1D"))
        ),
        "expected ForwardPassNotRun(\"SpatialDropout1D\"), got {:?}",
        err1
    );

    // Channels-last: (batch=2, height=4, width=4, channels=3)
    let mut d2 = SpatialDropout2D::new(0.5, vec![2, 4, 4, 3]).unwrap();
    d2.set_training_if_mode_dependent(true);
    let err2 = d2.backward(&filled(&[2, 4, 4, 3], 1.0)).unwrap_err();
    assert!(
        matches!(
            err2,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SpatialDropout2D"))
        ),
        "expected ForwardPassNotRun(\"SpatialDropout2D\"), got {:?}",
        err2
    );

    // Channels-last: (batch=1, depth=3, height=3, width=3, channels=2)
    let mut d3 = SpatialDropout3D::new(0.5, vec![1, 3, 3, 3, 2]).unwrap();
    d3.set_training_if_mode_dependent(true);
    let err3 = d3.backward(&filled(&[1, 3, 3, 3, 2], 1.0)).unwrap_err();
    assert!(
        matches!(
            err3,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SpatialDropout3D"))
        ),
        "expected ForwardPassNotRun(\"SpatialDropout3D\"), got {:?}",
        err3
    );
}

#[test]
fn dropout_forward_rejects_shape_mismatch() {
    let mut layer = Dropout::new(0.5, vec![2, 4]).unwrap();
    // input has wrong feature dim
    let input = Array::ones((2, 5)).into_dyn();
    let err = layer.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {:?}",
        err
    );
}

/// The declared shape's leading axis is fixed at construction from the whole dataset.
/// Enforcing it on each call would reject every mini-batch, so both paths accept any batch size.
#[test]
fn dropout_accepts_any_batch_size() {
    let mut layer = Dropout::new(0.5, vec![8, 4]).unwrap();
    for batch in [1usize, 3, 8, 16] {
        let input = Array::ones((batch, 4)).into_dyn();
        let out = layer
            .forward(&input)
            .unwrap_or_else(|e| panic!("forward rejected batch {batch}: {e:?}"));
        assert_eq!(out.shape(), &[batch, 4]);
        let out = layer
            .predict(&input)
            .unwrap_or_else(|e| panic!("predict rejected batch {batch}: {e:?}"));
        assert_eq!(out.shape(), &[batch, 4]);
    }
}

#[test]
fn dropout_predict_rejects_shape_mismatch() {
    let layer = Dropout::new(0.5, vec![2, 4]).unwrap();
    let input = Array::ones((2, 5)).into_dyn();
    let err = layer.predict(&input).unwrap_err();
    assert!(
        matches!(err, Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {:?}",
        err
    );
}

#[test]
fn dropout_empty_input_shape_accepts_any_shape() {
    // validate_input_shape skips when expected_shape is empty
    let mut layer = Dropout::new(0.0, vec![]).unwrap();
    let a = Array::ones((2, 3)).into_dyn();
    let b = Array::ones((5, 7, 2)).into_dyn();
    assert!(layer.forward(&a).is_ok());
    assert!(layer.forward(&b).is_ok());
}

// SpatialDropout1D - channel-consistency and values

#[test]
fn spatial_dropout_1d_rate_zero_is_identity() {
    // rate=0 -> identity in training mode
    // Channels-last: (batch=2, length=8, channels=4)
    let mut layer = SpatialDropout1D::new(0.0, vec![2, 8, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 8, 4], 1.5_f32);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_rate_one_yields_zeros() {
    // rate=1 -> all zeros
    // Channels-last: (batch=1, length=5, channels=3)
    let mut layer = SpatialDropout1D::new(1.0, vec![1, 5, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 5, 3], 2.0);
    let output = layer.forward(&input).unwrap();
    let expected = filled(&[1, 5, 3], 0.0);
    assert_allclose(&output, &expected, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_eval_is_identity() {
    // Channels-last: (batch=2, length=6, channels=4)
    let mut layer = SpatialDropout1D::new(0.5, vec![2, 6, 4]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[2, 6, 4], 3.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_predict_is_identity() {
    // Channels-last: (batch=1, length=6, channels=4)
    let layer = SpatialDropout1D::new(0.8, vec![1, 6, 4]).unwrap();
    let input = filled(&[1, 6, 4], 2.0);
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_channel_consistency() {
    // For each (batch, channel), all length positions share a single value, either 0 (dropped)
    // or input * scale (kept). All-ones input gives kept value 1/(1-rate) and dropped value 0.
    // Channels-last layout: (batch, length, channels)
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate); // = 2.0

    let mut layer = SpatialDropout1D::new(rate, vec![1, 10, 8]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 10, 8]);
    let output = layer.forward(&input).unwrap();

    let batch_size = 1;
    let length = 10;
    let channels = 8;

    for b in 0..batch_size {
        for c in 0..channels {
            let first = output[[b, 0, c]];
            assert!(
                first == 0.0 || (first - scale).abs() < 1e-5,
                "channel ({b},{c}) first element {first} is neither 0 nor {scale}"
            );
            for l in 1..length {
                let v = output[[b, l, c]];
                assert!(
                    (v - first).abs() < 1e-5,
                    "channel ({b},{c}) position {l}: {v} != first={first}"
                );
            }
        }
    }
}

#[test]
fn spatial_dropout_1d_kept_channel_exact_scale() {
    // Kept channel: output = input * 1/(1-rate). Distinct per-position values catch
    // partial masking.
    let rate = 0.4_f32;
    let scale = 1.0 / (1.0 - rate); // ~=1.6667

    // Channels-last layout: (batch=1, length=4, channels=10)
    let mut layer = SpatialDropout1D::new(rate, vec![1, 4, 10]).unwrap();
    layer.set_training_if_mode_dependent(true);

    // Give each spatial position a distinct value
    let data: Vec<f32> = (0..40).map(|i| i as f32 + 1.0).collect();
    let input = Tensor::from_shape_vec(ndarray::IxDyn(&[1, 4, 10]), data).unwrap();

    let output = layer.forward(&input).unwrap();

    let mut found_kept = false;
    for c in 0..10 {
        let first = output[[0, 0, c]];
        if first != 0.0 {
            // Kept channel: every position == input * scale
            found_kept = true;
            for l in 0..4 {
                let expected = input[[0, l, c]] * scale;
                assert!(
                    (output[[0, l, c]] - expected).abs() < 1e-4,
                    "channel {c} position {l}: {} != {expected}",
                    output[[0, l, c]]
                );
            }
        }
    }
    // With rate=0.4 and 10 channels, the chance of all channels being dropped is
    // 0.4^10 ~= 0.0001. At least 1 kept channel is expected, so a failure signals a bug.
    assert!(
        found_kept,
        "all channels were dropped, likely a bug (rate=0.4, 10 channels)"
    );
}

#[test]
fn spatial_dropout_1d_backward_channel_consistency() {
    // Backward: gradient zeroed for dropped channels, scaled for kept channels
    // Channels-last layout: (batch=1, length=4, channels=6)
    let rate = 0.5_f32;
    let mut layer = SpatialDropout1D::new(rate, vec![1, 4, 6]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 4, 6]);
    let output = layer.forward(&input).unwrap();
    let grad_up = ones(&[1, 4, 6]);
    let grad_in = layer.backward(&grad_up).unwrap();

    let scale = 1.0 / (1.0 - rate);
    for c in 0..6 {
        let out_first = output[[0, 0, c]];
        let grad_first = grad_in[[0, 0, c]];
        if out_first == 0.0 {
            for l in 0..4 {
                approx::assert_abs_diff_eq!(grad_in[[0, l, c]], 0.0_f32, epsilon = 1e-5);
            }
        } else {
            approx::assert_abs_diff_eq!(grad_first, scale, epsilon = 1e-4);
            for l in 1..4 {
                approx::assert_abs_diff_eq!(grad_in[[0, l, c]], grad_first, epsilon = 1e-5);
            }
        }
    }
}

// SpatialDropout1D - error paths

#[test]
fn spatial_dropout_1d_rejects_invalid_rate() {
    // Channels-last: (batch=1, length=8, channels=4)
    assert!(matches!(
        SpatialDropout1D::new(-0.1, vec![1, 8, 4]).unwrap_err(),
        Error::InvalidParameter { .. }
    ));
    assert!(matches!(
        SpatialDropout1D::new(1.5, vec![1, 8, 4]).unwrap_err(),
        Error::InvalidParameter { .. }
    ));
}

#[test]
fn spatial_dropout_1d_accepts_boundary_rates() {
    assert!(SpatialDropout1D::new(0.0, vec![1, 8, 4]).is_ok());
    assert!(SpatialDropout1D::new(1.0, vec![1, 8, 4]).is_ok());
}

#[test]
fn spatial_dropout_1d_rejects_wrong_ndim_forward() {
    // SpatialDropout1D requires 3D input
    // Channels-last: (batch=2, length=8, channels=4)
    let mut layer = SpatialDropout1D::new(0.5, vec![2, 8, 4]).unwrap();
    let input_2d = Array::ones((2, 8)).into_dyn();
    assert!(layer.forward(&input_2d).is_err());

    let input_4d = Array::ones((2, 8, 4, 3)).into_dyn();
    assert!(layer.forward(&input_4d).is_err());
}

#[test]
fn spatial_dropout_1d_rejects_wrong_ndim_predict() {
    // predict() also enforces ndim
    let layer = SpatialDropout1D::new(0.5, vec![2, 8, 4]).unwrap();
    let input_2d = Array::ones((2, 8)).into_dyn();
    assert!(layer.predict(&input_2d).is_err());
}

#[test]
fn spatial_dropout_1d_backward_before_forward_returns_error() {
    let mut layer = SpatialDropout1D::new(0.5, vec![1, 8, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[1, 8, 4], 1.0);
    let err = layer.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

// SpatialDropout2D - channel-consistency and values

#[test]
fn spatial_dropout_2d_rate_zero_is_identity() {
    // Channels-last: (batch=2, height=4, width=4, channels=3)
    let mut layer = SpatialDropout2D::new(0.0, vec![2, 4, 4, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 4, 4, 3], 1.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_rate_one_yields_zeros() {
    // Channels-last: (batch=1, height=3, width=3, channels=2)
    let mut layer = SpatialDropout2D::new(1.0, vec![1, 3, 3, 2]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 3, 3, 2], 5.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &filled(&[1, 3, 3, 2], 0.0), 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_eval_is_identity() {
    // Channels-last: (batch=1, height=4, width=4, channels=2)
    let mut layer = SpatialDropout2D::new(0.5, vec![1, 4, 4, 2]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[1, 4, 4, 2], 2.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_predict_is_identity() {
    // Channels-last: (batch=1, height=3, width=3, channels=2)
    let layer = SpatialDropout2D::new(0.7, vec![1, 3, 3, 2]).unwrap();
    let input = filled(&[1, 3, 3, 2], 2.5);
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_channel_consistency() {
    // For each (batch, channel): all (height, width) positions share the same value
    // Channels-last layout: (batch=1, height=4, width=4, channels=8)
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout2D::new(rate, vec![1, 4, 4, 8]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 4, 4, 8]);
    let output = layer.forward(&input).unwrap();

    for c in 0..8 {
        let first = output[[0, 0, 0, c]];
        assert!(
            first == 0.0 || (first - scale).abs() < 1e-5,
            "channel {c} first = {first}, expected 0 or {scale}"
        );
        for h in 0..4 {
            for w in 0..4 {
                assert!(
                    (output[[0, h, w, c]] - first).abs() < 1e-5,
                    "channel {c} ({h},{w}): {} != first={first}",
                    output[[0, h, w, c]]
                );
            }
        }
    }
}

#[test]
fn spatial_dropout_2d_rejects_wrong_ndim() {
    // Channels-last: (batch=1, height=4, width=4, channels=2)
    let mut layer = SpatialDropout2D::new(0.5, vec![1, 4, 4, 2]).unwrap();
    let input_3d = Array::ones((1, 4, 4)).into_dyn();
    assert!(layer.forward(&input_3d).is_err());
}

#[test]
fn spatial_dropout_2d_predict_rejects_wrong_ndim() {
    let layer = SpatialDropout2D::new(0.5, vec![1, 4, 4, 2]).unwrap();
    let input_3d = Array::ones((1, 4, 4)).into_dyn();
    assert!(layer.predict(&input_3d).is_err());
}

#[test]
fn spatial_dropout_2d_rejects_invalid_rate() {
    assert!(SpatialDropout2D::new(-0.5, vec![1, 4, 4, 2]).is_err());
    assert!(SpatialDropout2D::new(1.1, vec![1, 4, 4, 2]).is_err());
}

#[test]
fn spatial_dropout_2d_backward_before_forward_returns_error() {
    let mut layer = SpatialDropout2D::new(0.5, vec![1, 4, 4, 2]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[1, 4, 4, 2], 1.0);
    let err = layer.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

#[test]
fn spatial_dropout_2d_backward_channel_consistency() {
    // Backward: dropped-channel gradient is 0 at every (h,w). Kept-channel gradient is
    // upstream * scale at every (h,w).
    // Channels-last layout: (batch=1, height=3, width=3, channels=6)
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout2D::new(rate, vec![1, 3, 3, 6]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 3, 3, 6]);
    let output = layer.forward(&input).unwrap();
    let grad_up = ones(&[1, 3, 3, 6]);
    let grad_in = layer.backward(&grad_up).unwrap();

    for c in 0..6 {
        let out_first = output[[0, 0, 0, c]];
        let expected_grad = if out_first == 0.0 { 0.0 } else { scale };
        for h in 0..3 {
            for w in 0..3 {
                assert!(
                    (grad_in[[0, h, w, c]] - expected_grad).abs() < 1e-4,
                    "channel {c} ({h},{w}): grad {} != expected {expected_grad}",
                    grad_in[[0, h, w, c]]
                );
            }
        }
    }
}

// SpatialDropout3D - channel-consistency and values

#[test]
fn spatial_dropout_3d_rate_zero_is_identity() {
    // Channels-last: (batch=1, depth=2, height=3, width=3, channels=3)
    let mut layer = SpatialDropout3D::new(0.0, vec![1, 2, 3, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 2, 3, 3, 3], 1.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_rate_one_yields_zeros() {
    // Channels-last: (batch=1, depth=2, height=2, width=2, channels=2)
    let mut layer = SpatialDropout3D::new(1.0, vec![1, 2, 2, 2, 2]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 2, 2, 2, 2], 3.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &filled(&[1, 2, 2, 2, 2], 0.0), 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_eval_is_identity() {
    // Channels-last: (batch=1, depth=2, height=3, width=3, channels=2)
    let mut layer = SpatialDropout3D::new(0.5, vec![1, 2, 3, 3, 2]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[1, 2, 3, 3, 2], 4.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_predict_is_identity() {
    // Channels-last: (batch=1, depth=2, height=3, width=3, channels=2)
    let layer = SpatialDropout3D::new(0.6, vec![1, 2, 3, 3, 2]).unwrap();
    let input = filled(&[1, 2, 3, 3, 2], 1.5);
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_channel_consistency() {
    // For each (batch, channel): all (depth, height, width) voxels share the same value
    // Channels-last layout: (batch=1, depth=2, height=3, width=3, channels=8)
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout3D::new(rate, vec![1, 2, 3, 3, 8]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 2, 3, 3, 8]);
    let output = layer.forward(&input).unwrap();

    for c in 0..8 {
        let first = output[[0, 0, 0, 0, c]];
        assert!(
            first == 0.0 || (first - scale).abs() < 1e-5,
            "channel {c} first = {first}"
        );
        for d in 0..2 {
            for h in 0..3 {
                for w in 0..3 {
                    assert!(
                        (output[[0, d, h, w, c]] - first).abs() < 1e-5,
                        "channel {c} ({d},{h},{w}): {} != first={first}",
                        output[[0, d, h, w, c]]
                    );
                }
            }
        }
    }
}

#[test]
fn spatial_dropout_3d_rejects_wrong_ndim() {
    // Channels-last: (batch=1, depth=2, height=3, width=3, channels=2)
    let mut layer = SpatialDropout3D::new(0.5, vec![1, 2, 3, 3, 2]).unwrap();
    let input_4d = Array::ones((1, 2, 3, 3)).into_dyn();
    assert!(layer.forward(&input_4d).is_err());
}

#[test]
fn spatial_dropout_3d_predict_rejects_wrong_ndim() {
    let layer = SpatialDropout3D::new(0.5, vec![1, 2, 3, 3, 2]).unwrap();
    let input_4d = Array::ones((1, 2, 3, 3)).into_dyn();
    assert!(layer.predict(&input_4d).is_err());
}

#[test]
fn spatial_dropout_3d_rejects_invalid_rate() {
    assert!(SpatialDropout3D::new(-0.1, vec![1, 2, 3, 3, 2]).is_err());
    assert!(SpatialDropout3D::new(2.0, vec![1, 2, 3, 3, 2]).is_err());
}

#[test]
fn spatial_dropout_3d_backward_before_forward_returns_error() {
    let mut layer = SpatialDropout3D::new(0.5, vec![1, 2, 3, 3, 2]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[1, 2, 3, 3, 2], 1.0);
    let err = layer.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

#[test]
fn spatial_dropout_3d_backward_channel_consistency() {
    // Backward: a whole channel's gradient must be uniform (all zero or all scaled)
    // Channels-last layout: (batch=1, depth=2, height=2, width=2, channels=4)
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout3D::new(rate, vec![1, 2, 2, 2, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 2, 2, 2, 4]);
    let output = layer.forward(&input).unwrap();
    let grad_up = ones(&[1, 2, 2, 2, 4]);
    let grad_in = layer.backward(&grad_up).unwrap();

    for c in 0..4 {
        let out_first = output[[0, 0, 0, 0, c]];
        let expected_grad = if out_first == 0.0 { 0.0 } else { scale };
        for d in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    assert!(
                        (grad_in[[0, d, h, w, c]] - expected_grad).abs() < 1e-4,
                        "channel {c} ({d},{h},{w}): grad {} != expected {expected_grad}",
                        grad_in[[0, d, h, w, c]]
                    );
                }
            }
        }
    }
}

// Cross-type: predict() never caches a mask. Backward after predict uses the mask
// from the previous forward() call, not the predict call.

#[test]
fn dropout_predict_does_not_overwrite_mask_from_forward() {
    // Sequence forward() -> predict() -> backward(): predict() must not touch the stored
    // mask, so backward() still uses the forward() mask
    let mut layer = Dropout::new(0.5, vec![10]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[10]);
    let fwd_output = layer.forward(&input).unwrap();

    // Calling predict (in training mode) does not cache a mask
    let _ = layer.predict(&input).unwrap();

    // backward() still succeeds using the mask from forward()
    let grad = ones(&[10]);
    let grad_in = layer.backward(&grad).unwrap();

    // Zero where output was zero, 2.0 where output was 2.0
    for (o, g) in fwd_output.iter().zip(grad_in.iter()) {
        if *o == 0.0 {
            approx::assert_abs_diff_eq!(*g, 0.0_f32, epsilon = 1e-5);
        } else {
            approx::assert_abs_diff_eq!(*g, 2.0_f32, epsilon = 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Dropout noise_shape
//
// `noise_shape` sets the shape of the random mask, which then broadcasts up to the input. An
// entry of 1 gives that axis 1 SHARED draw, so the same units drop at every position of the
// axis. The point is a correlated draw, not independent draws whose average happens to agree.
//
// Every expectation below comes from a Keras 3.15.1 probe on the jax backend. The probe
// recovered the mask over 40 seeds per case and derived the resolved mask shape from the axes
// that stay constant in every seed:
//
//   input (2,3,4)  noise_shape None        -> mask (2,3,4)
//   input (2,3,4)  noise_shape (2,1,4)     -> mask (2,1,4)
//   input (2,3,4)  noise_shape (2,3,1)     -> mask (2,3,1)
//   input (2,3,4)  noise_shape (1,4)       -> mask (1,1,4)
//   input (2,3,4)  noise_shape (3,4)       -> mask (1,3,4)
//   input (2,3,4)  noise_shape (4,)        -> mask (1,1,4)
//   input (2,3,4)  noise_shape (1,)        -> mask (1,1,1)
//   input (2,3,4)  noise_shape (None,1,4)  -> mask (2,1,4)
//   input (2,4,3,5) noise_shape (1,3,1)    -> mask (1,1,3,1)
// ---------------------------------------------------------------------------------------

/// The dropout rate every noise_shape test uses, and the kept-element scale that goes with it
const NS_RATE: f32 = 0.5;
/// The kept-element scale of `NS_RATE`, which is `1 / (1 - rate)`
const NS_SCALE: f32 = 2.0;

/// Recovers the binary keep mask from the output of an all-ones input
fn keep_mask(output: &Tensor) -> Vec<u8> {
    output
        .iter()
        .map(|&v| {
            if v == 0.0 {
                0
            } else {
                assert_eq!(v, NS_SCALE, "an all-ones input gives 0 or the kept scale");
                1
            }
        })
        .collect()
}

/// Runs 1 training forward over an all-ones input and returns the recovered keep mask
fn masked_ones(shape: &[usize], noise_shape: Option<Vec<Option<usize>>>, seed: u64) -> Vec<u8> {
    let mut layer = Dropout::new(NS_RATE, shape.to_vec()).unwrap();
    if let Some(noise_shape) = noise_shape {
        layer = layer.with_noise_shape(noise_shape).unwrap();
    }
    let mut layer = layer.with_random_state(seed);
    layer.set_training_if_mode_dependent(true);
    keep_mask(&layer.forward(&ones(shape)).unwrap())
}

/// An all-None noise_shape reproduces the default per-element draw exactly
///
/// This is the base case of the resolution rule: with no entry of 1, the mask shape equals the
/// input shape, and the sampler consumes the same values in the same order
#[test]
fn dropout_noise_shape_of_all_none_matches_the_default() {
    for seed in [1_u64, 2, 3] {
        assert_eq!(
            masked_ones(&[2, 8], None, seed),
            masked_ones(&[2, 8], Some(vec![None, None]), seed),
            "an all-None noise_shape changed the mask at seed {seed}"
        );
        // The explicit extents are the same statement
        assert_eq!(
            masked_ones(&[2, 8], None, seed),
            masked_ones(&[2, 8], Some(vec![Some(2), Some(8)]), seed),
            "an explicit full noise_shape changed the mask at seed {seed}"
        );
    }
}

/// TRAP (a): the layer samples at the small shape, and never at the full shape
///
/// A `noise_shape` of `[2, 1, 4]` over a `[2, 3, 4]` input must draw exactly 8 values. An
/// implementation that samples 24 values at the full shape and then zeroes axis 1 reproduces
/// 1 sample by coincidence and is statistically wrong.
///
/// The check pins the draw count and the draw order together. A second layer whose input shape
/// IS `[2, 1, 4]`, at the same seed and with no noise_shape, must produce the identical mask.
/// A full-shape sampler consumes 24 values from the same stream, so its first 8 land in
/// different places and the 2 masks disagree
#[test]
fn dropout_noise_shape_draws_only_at_the_resolved_small_shape() {
    for seed in [11_u64, 12, 13, 14] {
        let broadcast = masked_ones(&[2, 3, 4], Some(vec![Some(2), Some(1), Some(4)]), seed);
        let small = masked_ones(&[2, 1, 4], None, seed);

        // The 8 distinct draws of the broadcast mask, read at position 0 of the shared axis
        let distinct: Vec<u8> = (0..2)
            .flat_map(|b| (0..4).map(move |c| (b, c)))
            .map(|(b, c)| broadcast[b * 12 + c])
            .collect();

        assert_eq!(
            distinct, small,
            "the draws at seed {seed} do not match a sampler running at [2, 1, 4]"
        );
    }
}

/// TRAP (a), statistically: the draw is correlated, not an average of independent draws
///
/// With a `[4, 1, 3]` noise_shape over a `[4, 200, 3]` input, the mask holds 12 draws. The mean
/// keep rate over many seeds therefore varies with the spread of 12 coin flips, about
/// `0.25 / 12 = 0.0208`. A sampler that drew 2400 independent values and then zeroed axis 1
/// would still average 0.5, but its spread would be about `0.25 / 2400 = 0.000104`, which is
/// 200 times smaller. Keras measures 0.0209 here
#[test]
fn dropout_noise_shape_gives_a_correlated_draw_not_an_average() {
    let shape = [4_usize, 200, 3];
    let seeds = 200_u64;

    let rates: Vec<f64> = (0..seeds)
        .map(|seed| {
            let mask = masked_ones(&shape, Some(vec![Some(4), Some(1), Some(3)]), seed);
            // Every timestep of a batch item and channel must carry the same draw
            for b in 0..4 {
                for t in 0..200 {
                    for c in 0..3 {
                        assert_eq!(
                            mask[(b * 200 + t) * 3 + c],
                            mask[b * 600 + c],
                            "axis 1 was not shared at seed {seed}"
                        );
                    }
                }
            }
            f64::from(mask.iter().map(|&m| u32::from(m)).sum::<u32>()) / 2400.0
        })
        .collect();

    let mean = rates.iter().sum::<f64>() / f64::from(seeds as u32);
    let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / f64::from(seeds as u32);

    assert!(
        (0.35..0.65).contains(&mean),
        "the mean keep rate {mean} is far from 0.5"
    );
    // 0.25 / 12 = 0.0208. A full-shape sampler would land near 0.25 / 2400 = 0.000104
    assert!(
        (0.008..0.045).contains(&variance),
        "the keep-rate variance {variance} does not match 12 correlated draws"
    );
}

/// An entry of 1 makes the whole axis share 1 draw
///
/// Keras: input (2,3,4) with noise_shape (2,1,4) keeps axis 1 constant in every seed
#[test]
fn dropout_noise_shape_shares_one_draw_along_an_axis() {
    for seed in [21_u64, 22, 23] {
        let mask = masked_ones(&[2, 3, 4], Some(vec![Some(2), Some(1), Some(4)]), seed);
        for b in 0..2 {
            for t in 0..3 {
                for c in 0..4 {
                    assert_eq!(
                        mask[(b * 3 + t) * 4 + c],
                        mask[b * 12 + c],
                        "seed {seed}: axis 1 is not shared at ({b}, {t}, {c})"
                    );
                }
            }
        }
    }

    // The trailing axis shares instead, with noise_shape (2,3,1)
    for seed in [24_u64, 25] {
        let mask = masked_ones(&[2, 3, 4], Some(vec![Some(2), Some(3), Some(1)]), seed);
        for b in 0..2 {
            for t in 0..3 {
                for c in 0..4 {
                    assert_eq!(
                        mask[(b * 3 + t) * 4 + c],
                        mask[(b * 3 + t) * 4],
                        "seed {seed}: axis 2 is not shared at ({b}, {t}, {c})"
                    );
                }
            }
        }
    }
}

/// TRAP (b): a shorter noise_shape lines up against the LAST axes of the input
///
/// The omitted leading axes therefore become SHARED, not independent. A caller who writes the
/// per-sample shape and omits the batch axis gets 1 mask for the whole batch.
///
/// Keras: (1,4) resolves to (1,1,4), and (3,4) resolves to (1,3,4), over a (2,3,4) input
#[test]
fn dropout_noise_shape_shorter_than_the_input_is_right_aligned() {
    for seed in [31_u64, 32, 33] {
        // (1, 4) is the same mask as (1, 1, 4)
        assert_eq!(
            masked_ones(&[2, 3, 4], Some(vec![Some(1), Some(4)]), seed),
            masked_ones(&[2, 3, 4], Some(vec![Some(1), Some(1), Some(4)]), seed),
            "seed {seed}: (1, 4) is not (1, 1, 4)"
        );
        // (3, 4) is the same mask as (1, 3, 4)
        assert_eq!(
            masked_ones(&[2, 3, 4], Some(vec![Some(3), Some(4)]), seed),
            masked_ones(&[2, 3, 4], Some(vec![Some(1), Some(3), Some(4)]), seed),
            "seed {seed}: (3, 4) is not (1, 3, 4)"
        );
        // (4,) is the same mask as (1, 1, 4)
        assert_eq!(
            masked_ones(&[2, 3, 4], Some(vec![Some(4)]), seed),
            masked_ones(&[2, 3, 4], Some(vec![Some(1), Some(1), Some(4)]), seed),
            "seed {seed}: (4,) is not (1, 1, 4)"
        );
        // A left-aligned reading of (3, 4) would demand 3 on the batch axis of extent 2 and
        // reject the call. The short form must instead succeed and share the batch axis
        let batch_shared = masked_ones(&[2, 3, 4], Some(vec![Some(3), Some(4)]), seed);
        assert_eq!(
            batch_shared[..12],
            batch_shared[12..],
            "seed {seed}: the omitted batch axis was not shared"
        );
    }
}

/// A rank-1 noise_shape of 1 shares 1 draw over the whole tensor
///
/// Keras: input (2,3,4) with noise_shape (1,) resolves to a mask of (1,1,1)
#[test]
fn dropout_noise_shape_of_one_shares_the_whole_tensor() {
    let mut all_kept = 0;
    let mut all_dropped = 0;
    for seed in 0..30_u64 {
        let mask = masked_ones(&[2, 3, 4], Some(vec![Some(1)]), seed);
        assert!(
            mask.iter().all(|&m| m == mask[0]),
            "seed {seed}: 1 draw did not cover the whole tensor"
        );
        if mask[0] == 1 {
            all_kept += 1;
        } else {
            all_dropped += 1;
        }
    }
    // Both outcomes appear, so the single draw is a real draw and not a constant
    assert!(
        all_kept > 0 && all_dropped > 0,
        "the single draw never moved"
    );
}

/// A `None` entry takes the extent of the input on its axis, so that axis stays independent
///
/// Keras: (None,1,4) and (None,1,None) both resolve to (2,1,4) over a (2,3,4) input
#[test]
fn dropout_noise_shape_none_entry_takes_the_input_extent() {
    let mut batch_rows_differ = false;
    for seed in 0..20_u64 {
        let explicit = masked_ones(&[2, 3, 4], Some(vec![Some(2), Some(1), Some(4)]), seed);
        assert_eq!(
            masked_ones(&[2, 3, 4], Some(vec![None, Some(1), Some(4)]), seed),
            explicit,
            "seed {seed}: (None, 1, 4) is not (2, 1, 4)"
        );
        assert_eq!(
            masked_ones(&[2, 3, 4], Some(vec![None, Some(1), None]), seed),
            explicit,
            "seed {seed}: (None, 1, None) is not (2, 1, 4)"
        );
        if explicit[..12] != explicit[12..] {
            batch_rows_differ = true;
        }
    }
    assert!(
        batch_rows_differ,
        "the batch axis never drew independently, so None did not take the input extent"
    );
}

/// TRAP (c): a rate of 1 with a noise_shape is legal and gives all zeros
#[test]
fn dropout_noise_shape_at_rate_one_gives_zeros() {
    let mut layer = Dropout::new(1.0, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(2), Some(1), Some(4)])
        .unwrap()
        .with_random_state(41);
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[2, 3, 4]);
    let output = layer.forward(&input).unwrap();
    assert!(output.iter().all(|&v| v == 0.0), "rate 1 kept a unit");

    let grad = layer.backward(&ones(&[2, 3, 4])).unwrap();
    assert!(grad.iter().all(|&v| v == 0.0), "rate 1 kept a gradient");

    // A rate of 0 stays the identity with a noise_shape as well
    let mut layer = Dropout::new(0.0, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(2), Some(1), Some(4)])
        .unwrap();
    layer.set_training_if_mode_dependent(true);
    assert_allclose(&layer.forward(&input).unwrap(), &input, 1e-6_f32);
}

/// The backward pass reuses the small mask and broadcasts it the same way
///
/// The gradient of an all-ones upstream must equal the forward output of an all-ones input:
/// both are `mask * scale`. The gradient must also stay constant along the shared axis
#[test]
fn dropout_noise_shape_backward_broadcasts_the_small_mask() {
    let mut layer = Dropout::new(NS_RATE, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(2), Some(1), Some(4)])
        .unwrap()
        .with_random_state(51);
    layer.set_training_if_mode_dependent(true);

    let output = layer.forward(&ones(&[2, 3, 4])).unwrap();
    let grad_in = layer.backward(&ones(&[2, 3, 4])).unwrap();

    for (o, g) in output.iter().zip(grad_in.iter()) {
        assert_eq!(o.to_bits(), g.to_bits(), "the gradient lost the mask");
    }
    for b in 0..2 {
        for t in 0..3 {
            for c in 0..4 {
                assert_eq!(
                    grad_in[[b, t, c]],
                    grad_in[[b, 0, c]],
                    "the gradient is not shared along axis 1"
                );
            }
        }
    }

    // A scaled upstream carries through with the same mask
    let upstream = filled(&[2, 3, 4], 3.0);
    let scaled = layer.backward(&upstream).unwrap();
    for (g, s) in grad_in.iter().zip(scaled.iter()) {
        assert_eq!(
            s.to_bits(),
            (g * 3.0).to_bits(),
            "the upstream did not scale"
        );
    }
}

/// Inference passes the input through unchanged, whatever the noise_shape says
#[test]
fn dropout_noise_shape_passes_through_in_inference_mode() {
    let mut layer = Dropout::new(NS_RATE, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(2), Some(1), Some(4)])
        .unwrap()
        .with_random_state(61);
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[2, 3, 4], 1.5);
    assert_allclose(&layer.forward(&input).unwrap(), &input, 1e-6_f32);
    assert_allclose(&layer.predict(&input).unwrap(), &input, 1e-6_f32);
}

/// The same seed gives the same mask, in this process and in any other
///
/// The recorded pattern is the seeded answer for this configuration. Running the test binary a
/// second time reproduces it, which is what pins the RNG through the builder
#[test]
fn dropout_noise_shape_seed_pins_the_mask() {
    let mask = masked_ones(&[2, 3, 4], Some(vec![Some(2), Some(1), Some(4)]), 27);
    // 8 draws, each repeated at all 3 positions of axis 1
    let expected: Vec<u8> = vec![
        1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, // batch item 0 draws [1, 0, 0, 1]
        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, // batch item 1 draws [0, 1, 1, 0]
    ];
    println!("noise_shape mask fingerprint: {mask:?}");
    assert_eq!(mask, expected, "the seeded mask moved");

    // 2 layers built the same way agree
    assert_eq!(
        mask,
        masked_ones(&[2, 3, 4], Some(vec![Some(2), Some(1), Some(4)]), 27)
    );
}

/// An entry that is neither 1 nor the input extent on its axis is rejected
#[test]
fn dropout_noise_shape_rejects_an_entry_that_is_not_the_input_extent() {
    let mut layer = Dropout::new(NS_RATE, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(2), Some(2), Some(4)])
        .unwrap();
    layer.set_training_if_mode_dependent(true);

    let err = layer.forward(&ones(&[2, 3, 4])).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );

    // The right-aligned reading applies to the rejection as well: 3 is the extent of axis 1,
    // but the short form lines it up against axis 2, whose extent is 4
    let mut layer = Dropout::new(NS_RATE, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(1), Some(3)])
        .unwrap();
    layer.set_training_if_mode_dependent(true);
    assert!(matches!(
        layer.forward(&ones(&[2, 3, 4])).unwrap_err(),
        Error::InvalidParameter { .. }
    ));
}

/// A noise_shape of a higher rank than the input is rejected
#[test]
fn dropout_noise_shape_rejects_a_higher_rank_than_the_input() {
    let mut layer = Dropout::new(NS_RATE, vec![2, 3, 4])
        .unwrap()
        .with_noise_shape(vec![Some(1), Some(2), Some(3), Some(4)])
        .unwrap();
    layer.set_training_if_mode_dependent(true);

    assert!(matches!(
        layer.forward(&ones(&[2, 3, 4])).unwrap_err(),
        Error::InvalidParameter { .. }
    ));
}

/// The builder rejects an empty noise_shape and an entry of 0
#[test]
fn dropout_noise_shape_builder_rejects_an_empty_vector_and_a_zero_entry() {
    let layer = Dropout::new(NS_RATE, vec![2, 3, 4]).unwrap();
    assert!(matches!(
        layer.with_noise_shape(vec![]).unwrap_err(),
        Error::EmptyInput(_)
    ));

    let layer = Dropout::new(NS_RATE, vec![2, 3, 4]).unwrap();
    assert!(matches!(
        layer
            .with_noise_shape(vec![Some(2), Some(0), Some(4)])
            .unwrap_err(),
        Error::InvalidParameter { .. }
    ));
}

/// A rank-4 input takes the same rule, with 2 shared axes at once
///
/// Keras: input (2,4,3,5) with noise_shape (1,3,1) resolves to a mask of (1,1,3,1)
#[test]
fn dropout_noise_shape_shares_two_axes_of_a_rank_four_input() {
    for seed in [71_u64, 72] {
        assert_eq!(
            masked_ones(&[2, 4, 3, 5], Some(vec![Some(1), Some(3), Some(1)]), seed),
            masked_ones(
                &[2, 4, 3, 5],
                Some(vec![Some(1), Some(1), Some(3), Some(1)]),
                seed
            ),
            "seed {seed}: (1, 3, 1) is not (1, 1, 3, 1)"
        );

        let mask = masked_ones(&[2, 4, 3, 5], Some(vec![Some(1), Some(3), Some(1)]), seed);
        // Only axis 2 varies: 3 draws cover the whole tensor
        for b in 0..2 {
            for t in 0..4 {
                for h in 0..3 {
                    for c in 0..5 {
                        let flat = ((b * 4 + t) * 3 + h) * 5 + c;
                        assert_eq!(mask[flat], mask[h * 5], "seed {seed}: axis {h} moved");
                    }
                }
            }
        }
    }
}

/// The layer keeps its mask at the small shape, so a whole batch of shared draws stays cheap
///
/// The check reads the mask through the observable behavior: a `[1, 1, 4]` mask over a large
/// input has exactly 4 distinct draws, whatever the input size
#[test]
fn dropout_noise_shape_keeps_the_mask_at_its_own_shape() {
    let mut layer = Dropout::new(NS_RATE, vec![8, 64, 4])
        .unwrap()
        .with_noise_shape(vec![Some(1), Some(1), Some(4)])
        .unwrap()
        .with_random_state(81);
    layer.set_training_if_mode_dependent(true);

    let mask = keep_mask(&layer.forward(&ones(&[8, 64, 4])).unwrap());
    for (flat, &m) in mask.iter().enumerate() {
        assert_eq!(m, mask[flat % 4], "the 4 draws did not tile the tensor");
    }
}
