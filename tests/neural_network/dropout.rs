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

use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::dropout::dropout::Dropout;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_1d::SpatialDropout1D;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_2d::SpatialDropout2D;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_3d::SpatialDropout3D;
use rustyml::neural_network::traits::Layer;

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
    // rate=0 -> no units dropped -> output == input (dedicated early-return path)
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
    // rate=1 -> every unit dropped -> output = zeros
    let mut layer = Dropout::new(1.0, vec![2, 5]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 5], 3.0);
    let output = layer.forward(&input).unwrap();
    let expected = filled(&[2, 5], 0.0);
    assert_allclose(&output, &expected, 1e-6_f32);
}

#[test]
fn dropout_eval_mode_is_exact_identity() {
    // Inverted dropout: inference passes input through unchanged
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
    // predict() must equal forward() in eval mode (both are identity)
    let mut layer = Dropout::new(0.3, vec![2, 4]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = Array::from_shape_vec((2, 4), (0..8).map(|v| v as f32 * 0.5).collect())
        .unwrap()
        .into_dyn();

    let out_forward = layer.forward(&input).unwrap();
    let out_predict = layer.predict(&input).unwrap();
    assert_allclose(&out_forward, &out_predict, 1e-6_f32);
    // Both must equal input exactly
    assert_allclose(&out_forward, &input, 1e-6_f32);
}

#[test]
fn dropout_predict_is_identity_in_training_mode() {
    // predict() is always identity, regardless of training flag
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
    // Inverted dropout: kept units scaled by 1/(1-rate); with rate=0.5, scale = 2.0
    let mut layer = Dropout::new(0.5, vec![200]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[200]);
    let output = layer.forward(&input).unwrap();

    let mut zero_count = 0usize;
    for &v in output.iter() {
        if v == 0.0 {
            zero_count += 1;
        } else {
            // Kept element == input * 1/(1-0.5) = 2.0
            approx::assert_abs_diff_eq!(v, 2.0_f32, epsilon = 1e-5);
        }
    }
    // Loose window [20%, 80%] keeps the ~50% zeroed-fraction check statistically robust
    let zero_frac = zero_count as f32 / 200.0;
    assert!(
        (0.20..=0.80).contains(&zero_frac),
        "Expected ~50% zeros, got {:.1}%",
        zero_frac * 100.0
    );
}

#[test]
fn dropout_rate_one_backward_returns_zeros() {
    // rate=1 -> forward zeros, backward returns zeros (distinct early-return path)
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
    // Kept units in backward receive grad_output * 1/(1-rate); with rate=0.5 and
    // all-ones input, kept output == 2.0 and kept gradient == 2.0, dropped == 0.0
    let mut layer = Dropout::new(0.5, vec![50]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[50]);
    let output = layer.forward(&input).unwrap();
    let grad_upstream = ones(&[50]);
    let grad_in = layer.backward(&grad_upstream).unwrap();

    for (i, (&out_v, &grad_v)) in output.iter().zip(grad_in.iter()).enumerate() {
        if out_v == 0.0 {
            // Dropped unit: gradient must be 0
            approx::assert_abs_diff_eq!(grad_v, 0.0_f32, epsilon = 1e-5);
        } else {
            // Kept unit: output = 2.0 and gradient = 1.0 * 2.0 = 2.0 (same scale factor)
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
    // Folds negative-rate and above-one-rate fan-out: same constructor, same error,
    // differing only by the invalid scalar
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
    let mut d1 = SpatialDropout1D::new(0.5, vec![2, 4, 8]).unwrap();
    d1.set_training_if_mode_dependent(true);
    let err1 = d1.backward(&filled(&[2, 4, 8], 1.0)).unwrap_err();
    assert!(
        matches!(
            err1,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SpatialDropout1D"))
        ),
        "expected ForwardPassNotRun(\"SpatialDropout1D\"), got {:?}",
        err1
    );

    let mut d2 = SpatialDropout2D::new(0.5, vec![2, 3, 4, 4]).unwrap();
    d2.set_training_if_mode_dependent(true);
    let err2 = d2.backward(&filled(&[2, 3, 4, 4], 1.0)).unwrap_err();
    assert!(
        matches!(
            err2,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SpatialDropout2D"))
        ),
        "expected ForwardPassNotRun(\"SpatialDropout2D\"), got {:?}",
        err2
    );

    let mut d3 = SpatialDropout3D::new(0.5, vec![1, 2, 3, 3, 3]).unwrap();
    d3.set_training_if_mode_dependent(true);
    let err3 = d3.backward(&filled(&[1, 2, 3, 3, 3], 1.0)).unwrap_err();
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
    // input has wrong first dim
    let input = Array::ones((3, 4)).into_dyn();
    let err = layer.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {:?}",
        err
    );
}

#[test]
fn dropout_predict_rejects_shape_mismatch() {
    let layer = Dropout::new(0.5, vec![2, 4]).unwrap();
    let input = Array::ones((3, 4)).into_dyn();
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
    let mut layer = SpatialDropout1D::new(0.0, vec![2, 4, 8]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 4, 8], 1.5_f32);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_rate_one_yields_zeros() {
    // rate=1 -> all zeros
    let mut layer = SpatialDropout1D::new(1.0, vec![1, 3, 5]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 3, 5], 2.0);
    let output = layer.forward(&input).unwrap();
    let expected = filled(&[1, 3, 5], 0.0);
    assert_allclose(&output, &expected, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_eval_is_identity() {
    let mut layer = SpatialDropout1D::new(0.5, vec![2, 4, 6]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[2, 4, 6], 3.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_predict_is_identity() {
    let layer = SpatialDropout1D::new(0.8, vec![1, 4, 6]).unwrap();
    let input = filled(&[1, 4, 6], 2.0);
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_1d_channel_consistency() {
    // For each (batch, channel), all length positions share one value: either 0 (dropped)
    // or input * scale (kept); all-ones input gives kept value 1/(1-rate), dropped value 0
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate); // = 2.0

    let mut layer = SpatialDropout1D::new(rate, vec![1, 8, 10]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 8, 10]);
    let output = layer.forward(&input).unwrap();

    let batch_size = 1;
    let channels = 8;
    let length = 10;

    for b in 0..batch_size {
        for c in 0..channels {
            // State is taken from the first position: must be 0 (dropped) or scale (kept)
            let first = output[[b, c, 0]];
            assert!(
                first == 0.0 || (first - scale).abs() < 1e-5,
                "channel ({b},{c}) first element {first} is neither 0 nor {scale}"
            );
            // All other positions in this channel must match
            for l in 1..length {
                let v = output[[b, c, l]];
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
    // Kept channel: output = input * 1/(1-rate); distinct per-position values catch
    // partial masking
    let rate = 0.4_f32;
    let scale = 1.0 / (1.0 - rate); // ~=1.6667

    let mut layer = SpatialDropout1D::new(rate, vec![1, 10, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    // Give each spatial position a distinct value
    let data: Vec<f32> = (0..40).map(|i| i as f32 + 1.0).collect();
    let input = Tensor::from_shape_vec(ndarray::IxDyn(&[1, 10, 4]), data).unwrap();

    let output = layer.forward(&input).unwrap();

    let mut found_kept = false;
    for c in 0..10 {
        let first = output[[0, c, 0]];
        if first != 0.0 {
            // Kept channel: every position == input * scale
            found_kept = true;
            for l in 0..4 {
                let expected = input[[0, c, l]] * scale;
                assert!(
                    (output[[0, c, l]] - expected).abs() < 1e-4,
                    "channel {c} position {l}: {} != {expected}",
                    output[[0, c, l]]
                );
            }
        }
    }
    // With rate=0.4 and 10 channels, the chance of all channels being dropped is
    // 0.4^10 ~= 0.0001, so at least one kept channel is expected; failure signals a bug
    assert!(
        found_kept,
        "all channels were dropped — likely a bug (rate=0.4, 10 channels)"
    );
}

#[test]
fn spatial_dropout_1d_backward_channel_consistency() {
    // Backward: gradient zeroed for dropped channels, scaled for kept channels
    let rate = 0.5_f32;
    let mut layer = SpatialDropout1D::new(rate, vec![1, 6, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 6, 4]);
    let output = layer.forward(&input).unwrap();
    let grad_up = ones(&[1, 6, 4]);
    let grad_in = layer.backward(&grad_up).unwrap();

    let scale = 1.0 / (1.0 - rate);
    for c in 0..6 {
        let out_first = output[[0, c, 0]];
        let grad_first = grad_in[[0, c, 0]];
        // Dropped channel: gradient must be 0
        if out_first == 0.0 {
            for l in 0..4 {
                approx::assert_abs_diff_eq!(grad_in[[0, c, l]], 0.0_f32, epsilon = 1e-5);
            }
        } else {
            // Kept channel: gradient = upstream * scale
            approx::assert_abs_diff_eq!(grad_first, scale, epsilon = 1e-4);
            for l in 1..4 {
                approx::assert_abs_diff_eq!(grad_in[[0, c, l]], grad_first, epsilon = 1e-5);
            }
        }
    }
}

// SpatialDropout1D - error paths

#[test]
fn spatial_dropout_1d_rejects_invalid_rate() {
    assert!(matches!(
        SpatialDropout1D::new(-0.1, vec![1, 4, 8]).unwrap_err(),
        Error::InvalidParameter { .. }
    ));
    assert!(matches!(
        SpatialDropout1D::new(1.5, vec![1, 4, 8]).unwrap_err(),
        Error::InvalidParameter { .. }
    ));
}

#[test]
fn spatial_dropout_1d_accepts_boundary_rates() {
    assert!(SpatialDropout1D::new(0.0, vec![1, 4, 8]).is_ok());
    assert!(SpatialDropout1D::new(1.0, vec![1, 4, 8]).is_ok());
}

#[test]
fn spatial_dropout_1d_rejects_wrong_ndim_forward() {
    // SpatialDropout1D requires 3D input
    let mut layer = SpatialDropout1D::new(0.5, vec![2, 4, 8]).unwrap();
    let input_2d = Array::ones((2, 4)).into_dyn();
    assert!(layer.forward(&input_2d).is_err());

    let input_4d = Array::ones((2, 4, 8, 3)).into_dyn();
    assert!(layer.forward(&input_4d).is_err());
}

#[test]
fn spatial_dropout_1d_rejects_wrong_ndim_predict() {
    // predict() also enforces ndim
    let layer = SpatialDropout1D::new(0.5, vec![2, 4, 8]).unwrap();
    let input_2d = Array::ones((2, 4)).into_dyn();
    assert!(layer.predict(&input_2d).is_err());
}

#[test]
fn spatial_dropout_1d_backward_before_forward_returns_error() {
    let mut layer = SpatialDropout1D::new(0.5, vec![1, 4, 8]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[1, 4, 8], 1.0);
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
    let mut layer = SpatialDropout2D::new(0.0, vec![2, 3, 4, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[2, 3, 4, 4], 1.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_rate_one_yields_zeros() {
    let mut layer = SpatialDropout2D::new(1.0, vec![1, 2, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 2, 3, 3], 5.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &filled(&[1, 2, 3, 3], 0.0), 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_eval_is_identity() {
    let mut layer = SpatialDropout2D::new(0.5, vec![1, 2, 4, 4]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[1, 2, 4, 4], 2.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_predict_is_identity() {
    let layer = SpatialDropout2D::new(0.7, vec![1, 2, 3, 3]).unwrap();
    let input = filled(&[1, 2, 3, 3], 2.5);
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_2d_channel_consistency() {
    // For each (batch, channel): all (height, width) positions share the same value
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout2D::new(rate, vec![1, 8, 4, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 8, 4, 4]);
    let output = layer.forward(&input).unwrap();

    for c in 0..8 {
        let first = output[[0, c, 0, 0]];
        assert!(
            first == 0.0 || (first - scale).abs() < 1e-5,
            "channel {c} first = {first}, expected 0 or {scale}"
        );
        for h in 0..4 {
            for w in 0..4 {
                assert!(
                    (output[[0, c, h, w]] - first).abs() < 1e-5,
                    "channel {c} ({h},{w}): {} != first={first}",
                    output[[0, c, h, w]]
                );
            }
        }
    }
}

#[test]
fn spatial_dropout_2d_rejects_wrong_ndim() {
    let mut layer = SpatialDropout2D::new(0.5, vec![1, 2, 4, 4]).unwrap();
    let input_3d = Array::ones((1, 2, 4)).into_dyn();
    assert!(layer.forward(&input_3d).is_err());
}

#[test]
fn spatial_dropout_2d_predict_rejects_wrong_ndim() {
    let layer = SpatialDropout2D::new(0.5, vec![1, 2, 4, 4]).unwrap();
    let input_3d = Array::ones((1, 2, 4)).into_dyn();
    assert!(layer.predict(&input_3d).is_err());
}

#[test]
fn spatial_dropout_2d_rejects_invalid_rate() {
    assert!(SpatialDropout2D::new(-0.5, vec![1, 2, 4, 4]).is_err());
    assert!(SpatialDropout2D::new(1.1, vec![1, 2, 4, 4]).is_err());
}

#[test]
fn spatial_dropout_2d_backward_before_forward_returns_error() {
    let mut layer = SpatialDropout2D::new(0.5, vec![1, 2, 4, 4]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[1, 2, 4, 4], 1.0);
    let err = layer.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

#[test]
fn spatial_dropout_2d_backward_channel_consistency() {
    // Backward: dropped-channel gradient == 0 at all (h,w); kept-channel gradient ==
    // upstream * scale at all (h,w)
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout2D::new(rate, vec![1, 6, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 6, 3, 3]);
    let output = layer.forward(&input).unwrap();
    let grad_up = ones(&[1, 6, 3, 3]);
    let grad_in = layer.backward(&grad_up).unwrap();

    for c in 0..6 {
        let out_first = output[[0, c, 0, 0]];
        let expected_grad = if out_first == 0.0 { 0.0 } else { scale };
        for h in 0..3 {
            for w in 0..3 {
                assert!(
                    (grad_in[[0, c, h, w]] - expected_grad).abs() < 1e-4,
                    "channel {c} ({h},{w}): grad {} != expected {expected_grad}",
                    grad_in[[0, c, h, w]]
                );
            }
        }
    }
}

// SpatialDropout3D - channel-consistency and values

#[test]
fn spatial_dropout_3d_rate_zero_is_identity() {
    let mut layer = SpatialDropout3D::new(0.0, vec![1, 3, 2, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 3, 2, 3, 3], 1.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_rate_one_yields_zeros() {
    let mut layer = SpatialDropout3D::new(1.0, vec![1, 2, 2, 2, 2]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = filled(&[1, 2, 2, 2, 2], 3.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &filled(&[1, 2, 2, 2, 2], 0.0), 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_eval_is_identity() {
    let mut layer = SpatialDropout3D::new(0.5, vec![1, 2, 2, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(false);

    let input = filled(&[1, 2, 2, 3, 3], 4.0);
    let output = layer.forward(&input).unwrap();
    assert_allclose(&output, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_predict_is_identity() {
    let layer = SpatialDropout3D::new(0.6, vec![1, 2, 2, 3, 3]).unwrap();
    let input = filled(&[1, 2, 2, 3, 3], 1.5);
    let out = layer.predict(&input).unwrap();
    assert_allclose(&out, &input, 1e-6_f32);
}

#[test]
fn spatial_dropout_3d_channel_consistency() {
    // For each (batch, channel): all (depth, height, width) voxels share the same value
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout3D::new(rate, vec![1, 8, 2, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 8, 2, 3, 3]);
    let output = layer.forward(&input).unwrap();

    for c in 0..8 {
        let first = output[[0, c, 0, 0, 0]];
        assert!(
            first == 0.0 || (first - scale).abs() < 1e-5,
            "channel {c} first = {first}"
        );
        for d in 0..2 {
            for h in 0..3 {
                for w in 0..3 {
                    assert!(
                        (output[[0, c, d, h, w]] - first).abs() < 1e-5,
                        "channel {c} ({d},{h},{w}): {} != first={first}",
                        output[[0, c, d, h, w]]
                    );
                }
            }
        }
    }
}

#[test]
fn spatial_dropout_3d_rejects_wrong_ndim() {
    let mut layer = SpatialDropout3D::new(0.5, vec![1, 2, 2, 3, 3]).unwrap();
    let input_4d = Array::ones((1, 2, 2, 3)).into_dyn();
    assert!(layer.forward(&input_4d).is_err());
}

#[test]
fn spatial_dropout_3d_predict_rejects_wrong_ndim() {
    let layer = SpatialDropout3D::new(0.5, vec![1, 2, 2, 3, 3]).unwrap();
    let input_4d = Array::ones((1, 2, 2, 3)).into_dyn();
    assert!(layer.predict(&input_4d).is_err());
}

#[test]
fn spatial_dropout_3d_rejects_invalid_rate() {
    assert!(SpatialDropout3D::new(-0.1, vec![1, 2, 2, 3, 3]).is_err());
    assert!(SpatialDropout3D::new(2.0, vec![1, 2, 2, 3, 3]).is_err());
}

#[test]
fn spatial_dropout_3d_backward_before_forward_returns_error() {
    let mut layer = SpatialDropout3D::new(0.5, vec![1, 2, 2, 3, 3]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let grad = filled(&[1, 2, 2, 3, 3], 1.0);
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
    let rate = 0.5_f32;
    let scale = 1.0 / (1.0 - rate);

    let mut layer = SpatialDropout3D::new(rate, vec![1, 4, 2, 2, 2]).unwrap();
    layer.set_training_if_mode_dependent(true);

    let input = ones(&[1, 4, 2, 2, 2]);
    let output = layer.forward(&input).unwrap();
    let grad_up = ones(&[1, 4, 2, 2, 2]);
    let grad_in = layer.backward(&grad_up).unwrap();

    for c in 0..4 {
        let out_first = output[[0, c, 0, 0, 0]];
        let expected_grad = if out_first == 0.0 { 0.0 } else { scale };
        for d in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    assert!(
                        (grad_in[[0, c, d, h, w]] - expected_grad).abs() < 1e-4,
                        "channel {c} ({d},{h},{w}): grad {} != expected {expected_grad}",
                        grad_in[[0, c, d, h, w]]
                    );
                }
            }
        }
    }
}

// Cross-type: predict() never caches a mask; backward after predict uses the mask
// from the previous forward(), not the predict call

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
