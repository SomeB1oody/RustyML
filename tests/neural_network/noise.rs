//! Integration tests for GaussianNoise (additive, output = input + N(0, stddev))
//! and GaussianDropout (multiplicative, output = input * N(1, stddev) with
//! stddev = sqrt(rate / (1 - rate))) layers
//!
//! Statistical assertions use large (10 000-element) all-ones tensors so the
//! sample mean and sample std converge tightly to their population parameters,
//! with a +/-0.05 tolerance around the expected values

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise;
use rustyml::neural_network::traits::Layer;

// helpers

/// Compute the sample mean of every element in a Tensor
fn tensor_mean(t: &Tensor) -> f64 {
    let sum: f64 = t.iter().map(|&v| v as f64).sum();
    sum / t.len() as f64
}

/// Compute the sample standard deviation of every element in a Tensor
fn tensor_std(t: &Tensor) -> f64 {
    let n = t.len() as f64;
    let mean = tensor_mean(t);
    let var: f64 = t.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

// GaussianNoise: constructor validation

/// Negative stddev returns Err(InvalidParameter) rather than panicking
#[test]
fn gaussian_noise_negative_stddev_returns_err() {
    let result = GaussianNoise::new(-0.1, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for negative stddev, got {:?}",
        result
    );
}

/// stddev = 0 is accepted at construction
#[test]
fn gaussian_noise_zero_stddev_construction_ok() {
    assert!(GaussianNoise::new(0.0, vec![4, 4], None).is_ok());
}

/// Positive stddev is accepted at construction
#[test]
fn gaussian_noise_positive_stddev_construction_ok() {
    assert!(GaussianNoise::new(0.5, vec![4, 4], None).is_ok());
}

// GaussianNoise: identity paths

/// In eval mode the layer returns a tensor element-wise equal to the input
#[test]
fn gaussian_noise_eval_mode_is_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(1.0, vec![2, 3], None).unwrap();
    layer.set_training_if_mode_dependent(false);

    let output = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// predict() is always identity regardless of the training flag
#[test]
fn gaussian_noise_predict_is_always_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    // Training flag = true: predict must still be identity
    let mut layer = GaussianNoise::new(1.0, vec![2, 3], None).unwrap();
    layer.set_training_if_mode_dependent(true);
    let output = layer.predict(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// predict() in eval mode equals forward() in eval mode (both identity)
#[test]
fn gaussian_noise_predict_equals_forward_in_eval_mode() {
    let input_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 1.0).collect();
    let input = Array::from_shape_vec((3, 4), input_data)
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(0.5, vec![3, 4], None).unwrap();
    layer.set_training_if_mode_dependent(false);

    let forward_out = layer.forward(&input).unwrap();
    let predict_out = layer.predict(&input).unwrap();
    crate::common::assert_allclose(&predict_out, &forward_out, 1e-6f32);
}

/// stddev = 0 is the identity even in training mode (N(0, 0) = 0)
#[test]
fn gaussian_noise_zero_stddev_training_is_identity() {
    let input: Tensor = Array::from_shape_vec((3,), vec![1.0f32, -2.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(0.0, vec![3], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let output = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

// GaussianNoise: statistical properties in training mode

/// E[output] ~= input when training, since output = input + N(0, stddev)
#[test]
fn gaussian_noise_training_preserves_mean() {
    const N: usize = 10_000;
    let input_val = 2.0f32;
    let stddev = 0.5f32;

    // All-ones input scaled by input_val
    let input = Array::from_shape_vec((N,), vec![input_val; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(stddev, vec![N], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let output = layer.forward(&input).unwrap();
    let mean = tensor_mean(&output);

    assert!(
        (mean - input_val as f64).abs() < 0.05,
        "GaussianNoise training mean: expected ≈{:.4}, got {:.6}",
        input_val,
        mean
    );
}

/// std(output - input) ~= stddev when training, since output - input = N(0, stddev)
#[test]
fn gaussian_noise_training_noise_std_matches_stddev() {
    const N: usize = 10_000;

    for &stddev in &[0.5f32, 0.2f32] {
        let input = Array::from_shape_vec((N,), vec![3.0f32; N])
            .unwrap()
            .into_dyn();

        let mut layer = GaussianNoise::new(stddev, vec![N], None).unwrap();
        layer.set_training_if_mode_dependent(true);

        let output = layer.forward(&input).unwrap();
        // Noise = output - input element-wise
        let noise: Tensor = &output - &input;
        let std_val = tensor_std(&noise);

        assert!(
            (std_val - stddev as f64).abs() < 0.05,
            "GaussianNoise stddev={}: expected std≈{:.4}, got {:.6}",
            stddev,
            stddev,
            std_val
        );
    }
}

/// Additive model: a large positive input plus large stddev can yield negative outputs
#[test]
fn gaussian_noise_training_can_produce_negative_values() {
    const N: usize = 10_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(5.0, vec![N], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let output = layer.forward(&input).unwrap();
    let neg_count = output.iter().filter(|&&v| v < 0.0).count();

    assert!(
        neg_count > 100,
        "GaussianNoise(stddev=5) on input=1: expected >100 negative outputs, got {}",
        neg_count
    );
}

/// Two consecutive training-mode forward() calls do not produce identical outputs
#[test]
fn gaussian_noise_training_consecutive_calls_differ() {
    const N: usize = 10_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(0.5, vec![N], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let out1 = layer.forward(&input).unwrap();
    let out2 = layer.forward(&input).unwrap();

    let all_equal = out1.iter().zip(out2.iter()).all(|(a, b)| a == b);
    assert!(
        !all_equal,
        "two consecutive training forward() calls must differ"
    );
}

// GaussianNoise: error paths

/// forward() with a shape-mismatched input returns Err(ShapeMismatch)
#[test]
fn gaussian_noise_forward_shape_mismatch_returns_err() {
    let mut layer = GaussianNoise::new(0.1, vec![4, 4], None).unwrap();
    // Supply a [3, 3] tensor to a layer expecting [4, 4]
    let wrong_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();

    let result = layer.forward(&wrong_input);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected Err(ShapeMismatch), got {:?}",
        result
    );
}

/// predict() with a shape-mismatched input returns Err(ShapeMismatch)
#[test]
fn gaussian_noise_predict_shape_mismatch_returns_err() {
    let layer = GaussianNoise::new(0.1, vec![4, 4], None).unwrap();
    let wrong_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();

    let result = layer.predict(&wrong_input);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected Err(ShapeMismatch), got {:?}",
        result
    );
}

/// Empty input_shape is a wildcard: any tensor shape is accepted
#[test]
fn gaussian_noise_empty_input_shape_accepts_any_tensor() {
    let mut layer = GaussianNoise::new(0.1, vec![], None).unwrap();
    // Supply tensors of completely different shapes - both should succeed
    let a: Tensor = Array::from_shape_vec((5,), vec![1.0f32; 5])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec((2, 3, 4), vec![1.0f32; 24])
        .unwrap()
        .into_dyn();
    assert!(layer.forward(&a).is_ok());
    assert!(layer.forward(&b).is_ok());
    assert!(layer.predict(&a).is_ok());
    assert!(layer.predict(&b).is_ok());
}

// GaussianNoise: metadata

#[test]
fn gaussian_noise_layer_type_and_output_shape() {
    let layer = GaussianNoise::new(0.3, vec![32, 128], None).unwrap();
    assert_eq!(layer.layer_type(), "GaussianNoise");
    assert_eq!(layer.output_shape(), "(32, 128)");
}

#[test]
fn gaussian_noise_empty_input_shape_output_shape_unknown() {
    let layer = GaussianNoise::new(0.3, vec![], None).unwrap();
    assert_eq!(layer.output_shape(), "Unknown");
}

// GaussianDropout

// GaussianDropout: constructor validation

/// rate = 1.0 is rejected: the valid interval [0, 1) excludes 1.0
#[test]
fn gaussian_dropout_rate_one_returns_err() {
    let result = GaussianDropout::new(1.0, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for rate=1.0, got {:?}",
        result
    );
}

/// rate > 1.0 is rejected
#[test]
fn gaussian_dropout_rate_above_one_returns_err() {
    let result = GaussianDropout::new(1.5, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for rate=1.5, got {:?}",
        result
    );
}

/// Negative rate is rejected
#[test]
fn gaussian_dropout_negative_rate_returns_err() {
    let result = GaussianDropout::new(-0.1, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for rate=-0.1, got {:?}",
        result
    );
}

/// rate = 0.0 is accepted at construction
#[test]
fn gaussian_dropout_rate_zero_construction_ok() {
    assert!(GaussianDropout::new(0.0, vec![4, 4], None).is_ok());
}

/// rate = 0.999 is accepted (just below 1.0)
#[test]
fn gaussian_dropout_rate_near_one_construction_ok() {
    assert!(GaussianDropout::new(0.999, vec![4, 4], None).is_ok());
}

// GaussianDropout: identity paths

/// In eval mode the layer returns a tensor element-wise equal to the input
#[test]
fn gaussian_dropout_eval_mode_is_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5, vec![2, 3], None).unwrap();
    layer.set_training_if_mode_dependent(false);

    let output = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// predict() is always identity regardless of the training flag
#[test]
fn gaussian_dropout_predict_is_always_identity() {
    let input_data = vec![2.0f32, -1.5, 0.3, 4.0, -3.0, 1.1];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.3, vec![2, 3], None).unwrap();
    layer.set_training_if_mode_dependent(true);
    let output = layer.predict(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// predict() in eval mode equals forward() in eval mode (both identity)
#[test]
fn gaussian_dropout_predict_equals_forward_in_eval_mode() {
    let input_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 1.0).collect();
    let input = Array::from_shape_vec((3, 4), input_data)
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.4, vec![3, 4], None).unwrap();
    layer.set_training_if_mode_dependent(false);

    let forward_out = layer.forward(&input).unwrap();
    let predict_out = layer.predict(&input).unwrap();
    crate::common::assert_allclose(&predict_out, &forward_out, 1e-6f32);
}

/// rate = 0 is the identity even in training mode (stddev = 0, noise = N(1, 0) = 1)
#[test]
fn gaussian_dropout_rate_zero_training_is_identity() {
    let input: Tensor = Array::from_shape_vec((3,), vec![1.0f32, -2.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.0, vec![3], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let output = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

// GaussianDropout: statistical properties in training mode

/// E[output] ~= input when training, since output = input * N(1, stddev)
#[test]
fn gaussian_dropout_training_preserves_mean() {
    const N: usize = 10_000;
    let input_val = 1.0f32;
    // rate = 0.5 => stddev = sqrt(0.5 / 0.5) = 1.0
    let rate = 0.5f32;

    let input = Array::from_shape_vec((N,), vec![input_val; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(rate, vec![N], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let output = layer.forward(&input).unwrap();
    let mean = tensor_mean(&output);

    assert!(
        (mean - input_val as f64).abs() < 0.05,
        "GaussianDropout training mean: expected ≈{:.4}, got {:.6}",
        input_val,
        mean
    );
}

/// std(output) ~= sqrt(rate / (1 - rate)) when training with all-ones input,
/// validating the stddev formula defined in the source
#[test]
fn gaussian_dropout_training_noise_std_matches_formula() {
    const N: usize = 10_000;
    // Two distinct rates pin the formula: rate=0.5 => stddev = 1.0,
    // rate=0.25 => stddev = sqrt(1/3) ~= 0.5774
    let cases: &[(f32, f64)] = &[
        (0.5, 1.0),
        (0.25, (1.0_f64 / 3.0).sqrt()), // ~= 0.5774
    ];

    for &(rate, expected_std) in cases {
        let input = Array::from_shape_vec((N,), vec![1.0f32; N])
            .unwrap()
            .into_dyn();

        let mut layer = GaussianDropout::new(rate, vec![N], None).unwrap();
        layer.set_training_if_mode_dependent(true);

        let output = layer.forward(&input).unwrap();
        // With input=1, output = noise, so std(output) = std(noise)
        let std_val = tensor_std(&output);

        assert!(
            (std_val - expected_std).abs() < 0.05,
            "GaussianDropout rate={}: expected std≈{:.4}, got {:.6}",
            rate,
            expected_std,
            std_val
        );
    }
}

/// Two consecutive training-mode forward() calls do not produce identical outputs
#[test]
fn gaussian_dropout_training_consecutive_calls_differ() {
    const N: usize = 10_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5, vec![N], None).unwrap();
    layer.set_training_if_mode_dependent(true);

    let out1 = layer.forward(&input).unwrap();
    let out2 = layer.forward(&input).unwrap();

    let all_equal = out1.iter().zip(out2.iter()).all(|(a, b)| a == b);
    assert!(
        !all_equal,
        "two consecutive training forward() calls must differ"
    );
}

// GaussianDropout: error paths

/// forward() with a shape-mismatched input returns Err(ShapeMismatch)
#[test]
fn gaussian_dropout_forward_shape_mismatch_returns_err() {
    let mut layer = GaussianDropout::new(0.3, vec![4, 4], None).unwrap();
    let wrong_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();

    let result = layer.forward(&wrong_input);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected Err(ShapeMismatch), got {:?}",
        result
    );
}

/// predict() with a shape-mismatched input returns Err(ShapeMismatch)
#[test]
fn gaussian_dropout_predict_shape_mismatch_returns_err() {
    let layer = GaussianDropout::new(0.3, vec![4, 4], None).unwrap();
    let wrong_input: Tensor = Array::from_shape_vec((3, 3), vec![1.0f32; 9])
        .unwrap()
        .into_dyn();

    let result = layer.predict(&wrong_input);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected Err(ShapeMismatch), got {:?}",
        result
    );
}

/// Empty input_shape is a wildcard: any tensor shape is accepted
#[test]
fn gaussian_dropout_empty_input_shape_accepts_any_tensor() {
    let mut layer = GaussianDropout::new(0.3, vec![], None).unwrap();
    let a: Tensor = Array::from_shape_vec((5,), vec![1.0f32; 5])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec((2, 3, 4), vec![1.0f32; 24])
        .unwrap()
        .into_dyn();
    assert!(layer.forward(&a).is_ok());
    assert!(layer.forward(&b).is_ok());
    assert!(layer.predict(&a).is_ok());
    assert!(layer.predict(&b).is_ok());
}

/// In training mode, backward multiplies by the cached forward noise, so without a prior forward()
/// there is no noise to reuse and it errors, matching Dense/Dropout's forward-pass-not-run contract
#[test]
fn gaussian_dropout_backward_without_forward_errors() {
    let grad: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32; 6])
        .unwrap()
        .into_dyn();
    let mut layer = GaussianDropout::new(0.3, vec![2, 3], None).unwrap();

    let result = layer.backward(&grad);
    assert!(
        matches!(result, Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))),
        "expected Err(ForwardPassNotRun) without a prior forward, got {:?}",
        result
    );
}

/// backward reuses the exact multiplicative noise from forward: with input = ones, output == noise,
/// so backward(g) must equal g * output elementwise (y = x * noise => dx = g * noise)
#[test]
fn gaussian_dropout_backward_multiplies_by_forward_noise() {
    let input: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32; 6])
        .unwrap()
        .into_dyn();
    let mut layer = GaussianDropout::new(0.3, vec![2, 3], Some(42)).unwrap();

    // With input = ones, output = ones * noise = noise
    let output = layer.forward(&input).unwrap();

    let grad: Tensor = Array::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&grad).unwrap();

    // grad_input should be grad * noise == grad * output
    let expected = &grad * &output;
    for (got, exp) in grad_input.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
    }
}

/// During inference (training = false) forward is the identity, so backward passes the gradient
/// through unchanged
#[test]
fn gaussian_dropout_backward_inference_is_passthrough() {
    let grad: Tensor = Array::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        .unwrap()
        .into_dyn();
    let mut layer = GaussianDropout::new(0.3, vec![2, 3], None).unwrap();
    layer.set_training(false);

    let grad_input = layer.backward(&grad).unwrap();
    for (got, exp) in grad_input.iter().zip(grad.iter()) {
        assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
    }
}

// GaussianDropout: metadata

#[test]
fn gaussian_dropout_layer_type_and_output_shape() {
    let layer = GaussianDropout::new(0.3, vec![32, 128], None).unwrap();
    assert_eq!(layer.layer_type(), "GaussianDropout");
    assert_eq!(layer.output_shape(), "(32, 128)");
}

#[test]
fn gaussian_dropout_empty_input_shape_output_shape_unknown() {
    let layer = GaussianDropout::new(0.3, vec![], None).unwrap();
    assert_eq!(layer.output_shape(), "Unknown");
}

// Mode-switching symmetry

/// set_training_if_mode_dependent(true) re-enables noise and (false) disables it
#[test]
fn gaussian_noise_mode_switching_routes_correctly() {
    const N: usize = 1_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(2.0, vec![N], None).unwrap();

    // Eval mode is the identity
    layer.set_training_if_mode_dependent(false);
    let eval_out = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out, &input, 1e-6f32);

    // Training mode now differs from the input
    layer.set_training_if_mode_dependent(true);
    let train_out = layer.forward(&input).unwrap();
    let any_different = train_out.iter().zip(input.iter()).any(|(a, b)| a != b);
    assert!(any_different, "training mode must add noise");

    // Back to eval mode: the identity again
    layer.set_training_if_mode_dependent(false);
    let eval_out2 = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out2, &input, 1e-6f32);
}

/// Same mode-switching symmetry for GaussianDropout
#[test]
fn gaussian_dropout_mode_switching_routes_correctly() {
    const N: usize = 1_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5, vec![N], None).unwrap();

    // Eval mode is the identity
    layer.set_training_if_mode_dependent(false);
    let eval_out = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out, &input, 1e-6f32);

    // Training mode now differs from the input
    layer.set_training_if_mode_dependent(true);
    let train_out = layer.forward(&input).unwrap();
    let any_different = train_out.iter().zip(input.iter()).any(|(a, b)| a != b);
    assert!(any_different, "training mode must multiply by noise");

    // Back to eval mode: the identity again
    layer.set_training_if_mode_dependent(false);
    let eval_out2 = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out2, &input, 1e-6f32);
}

/// GaussianNoise::backward is a pure gradient pass-through: since y = x + noise and the
/// noise does not depend on x, d(y)/dx = 1, so the upstream gradient returns unchanged
#[test]
fn gaussian_noise_backward_passes_gradient_through_unchanged() {
    let mut layer = GaussianNoise::new(1.0, vec![2, 3], None).unwrap();
    // Realistic call order: a training-mode forward first, then backward
    layer.set_training_if_mode_dependent(true);
    let x: Tensor = Array::from_elem((2, 3), 1.0f32).into_dyn();
    let _ = layer.forward(&x).unwrap();

    let grad: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad).unwrap();

    // Pure pass-through => byte-identical gradient (zero epsilon)
    crate::common::assert_allclose(&grad_in, &grad, 0.0f32);
}
