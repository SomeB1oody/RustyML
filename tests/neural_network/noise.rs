//! Integration tests for GaussianNoise and GaussianDropout layers.
//!
//! All expected values are derived from the mathematical definitions:
//!   GaussianNoise  : output = input + N(0, stddev)     [additive]
//!   GaussianDropout: output = input * N(1, stddev)     [multiplicative]
//!     where stddev = sqrt(rate / (1 - rate))
//!
//! Statistical assertions use large (10 000-element) all-ones tensors so that the
//! sample mean and sample std converge tightly to their population parameters; we
//! accept ±0.05 around the expected values. Determinism tests additionally use
//! `crate::common::seeded_rng` so they are reproducible, but note that neither
//! layer accepts an external RNG — they call Tensor::random internally — so we
//! rely on statistical tolerances rather than bit-for-bit reproducibility for those
//! specific tests.

use ndarray::Array;
use rustyml::error::Error;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise;
use rustyml::neural_network::traits::Layer;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Compute the sample mean of every element in a Tensor.
fn tensor_mean(t: &Tensor) -> f64 {
    let sum: f64 = t.iter().map(|&v| v as f64).sum();
    sum / t.len() as f64
}

/// Compute the sample standard deviation of every element in a Tensor.
fn tensor_std(t: &Tensor) -> f64 {
    let n = t.len() as f64;
    let mean = tensor_mean(t);
    let var: f64 = t.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

// ─── GaussianNoise: constructor validation ───────────────────────────────────

/// Negative stddev must return Err(InvalidParameter), not panic.
/// Mathematical basis: the normal distribution N(0, stddev) is undefined for stddev < 0.
#[test]
fn gaussian_noise_negative_stddev_returns_err() {
    let result = GaussianNoise::new(-0.1, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for negative stddev, got {:?}",
        result
    );
}

/// stddev = 0 is explicitly valid (identity fast-path in forward).
#[test]
fn gaussian_noise_zero_stddev_construction_ok() {
    assert!(GaussianNoise::new(0.0, vec![4, 4], None).is_ok());
}

/// Positive stddev must be accepted.
#[test]
fn gaussian_noise_positive_stddev_construction_ok() {
    assert!(GaussianNoise::new(0.5, vec![4, 4], None).is_ok());
}

// ─── GaussianNoise: identity paths ───────────────────────────────────────────

/// In EVAL mode the layer must return a tensor element-wise equal to the input.
/// Mathematical basis: the layer documentation states inference is the identity map.
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

/// predict() must always be identity regardless of the training flag.
/// Mathematical basis: predict() never samples noise (immutable self).
#[test]
fn gaussian_noise_predict_is_always_identity() {
    let input_data = vec![1.0f32, -2.0, 3.5, 0.0, -0.5, 7.0];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    // Training flag = true: predict must still be identity.
    let mut layer = GaussianNoise::new(1.0, vec![2, 3], None).unwrap();
    layer.set_training_if_mode_dependent(true); // explicitly training
    let output = layer.predict(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// predict() in eval mode must equal forward() in eval mode (both are identity).
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

/// stddev = 0 forces the identity fast-path even in training mode.
/// Mathematical basis: N(0, 0) = 0 always, so output = input + 0 = input.
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

// ─── GaussianNoise: statistical properties in training mode ──────────────────

/// E[output] ≈ input when training.
/// Mathematical basis: output = input + N(0, stddev) ⟹ E[output] = input + 0 = input.
/// With 10 000 independent samples the CLT gives std of sample mean ≈ stddev / sqrt(n) ≈ 0.005,
/// so tolerance 0.05 is roughly 10 σ — very unlikely to fail.
#[test]
fn gaussian_noise_training_preserves_mean() {
    const N: usize = 10_000;
    let input_val = 2.0f32;
    let stddev = 0.5f32;

    // All-ones input scaled by input_val.
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

/// std(output - input) ≈ stddev when training.
/// Mathematical basis: output - input = N(0, stddev), so std of the difference = stddev.
/// Tolerance 0.05 is ~10% of the expected stddev=0.5 for the first case.
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
        // Noise = output - input element-wise.
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

/// Additive model: large positive input + large stddev can produce negative outputs.
/// Mathematical basis: output = input + noise; when noise < -input the result is negative.
/// With input=1.0 and stddev=5.0, P(output < 0) = P(noise < -1.0) = Φ(-0.2) ≈ 0.42.
/// So with N=10 000 at least 1 000 negative outputs are virtually certain.
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

/// Two consecutive forward() calls in training mode must not produce identical outputs.
/// (The probability of a perfect collision in 10 000 f32 samples is negligible.)
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

    // With 10 000 elements and stddev=0.5, probability all match is astronomically small.
    let all_equal = out1.iter().zip(out2.iter()).all(|(a, b)| a == b);
    assert!(
        !all_equal,
        "two consecutive training forward() calls must differ"
    );
}

// ─── GaussianNoise: error paths ───────────────────────────────────────────────

/// forward() with a shape-mismatched input must return Err(ShapeMismatch).
#[test]
fn gaussian_noise_forward_shape_mismatch_returns_err() {
    let mut layer = GaussianNoise::new(0.1, vec![4, 4], None).unwrap();
    // Supply a [3, 3] tensor to a layer expecting [4, 4].
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

/// predict() with a shape-mismatched input must return Err(ShapeMismatch).
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

/// Empty input_shape is a wildcard: any tensor shape is accepted.
/// Mathematical basis: validate_input_shape fast-paths when expected_shape is empty.
#[test]
fn gaussian_noise_empty_input_shape_accepts_any_tensor() {
    let mut layer = GaussianNoise::new(0.1, vec![], None).unwrap();
    // Supply tensors of completely different shapes — both should succeed.
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

// ─── GaussianNoise: metadata ──────────────────────────────────────────────────

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

// ═══════════════════════════════════════════════════════════════════════════════
// GaussianDropout
// ═══════════════════════════════════════════════════════════════════════════════

// ─── GaussianDropout: constructor validation ─────────────────────────────────

/// rate = 1.0 must be rejected: [0, 1) is exclusive of 1.0.
#[test]
fn gaussian_dropout_rate_one_returns_err() {
    let result = GaussianDropout::new(1.0, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for rate=1.0, got {:?}",
        result
    );
}

/// rate > 1.0 must be rejected.
#[test]
fn gaussian_dropout_rate_above_one_returns_err() {
    let result = GaussianDropout::new(1.5, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for rate=1.5, got {:?}",
        result
    );
}

/// Negative rate must be rejected.
#[test]
fn gaussian_dropout_negative_rate_returns_err() {
    let result = GaussianDropout::new(-0.1, vec![4, 4], None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected Err(InvalidParameter) for rate=-0.1, got {:?}",
        result
    );
}

/// rate = 0.0 is valid: it triggers the identity fast-path.
#[test]
fn gaussian_dropout_rate_zero_construction_ok() {
    assert!(GaussianDropout::new(0.0, vec![4, 4], None).is_ok());
}

/// rate = 0.999 is valid (just below 1.0).
#[test]
fn gaussian_dropout_rate_near_one_construction_ok() {
    assert!(GaussianDropout::new(0.999, vec![4, 4], None).is_ok());
}

// ─── GaussianDropout: identity paths ─────────────────────────────────────────

/// In EVAL mode the layer must return a tensor element-wise equal to the input.
/// Mathematical basis: the layer documentation states inference is the identity map.
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

/// predict() must always be identity regardless of the training flag.
/// Mathematical basis: predict() never samples noise (immutable self).
#[test]
fn gaussian_dropout_predict_is_always_identity() {
    let input_data = vec![2.0f32, -1.5, 0.3, 4.0, -3.0, 1.1];
    let input = Array::from_shape_vec((2, 3), input_data.clone())
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.3, vec![2, 3], None).unwrap();
    layer.set_training_if_mode_dependent(true); // explicitly training
    let output = layer.predict(&input).unwrap();
    crate::common::assert_allclose(&output, &input, 1e-6f32);
}

/// predict() in eval mode must equal forward() in eval mode (both are identity).
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

/// rate = 0 forces the identity fast-path even in training mode.
/// Mathematical basis: stddev = sqrt(0 / 1) = 0, so noise = N(1, 0) = 1, and output = input * 1.
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

// ─── GaussianDropout: statistical properties in training mode ─────────────────

/// E[output] ≈ input when training.
/// Mathematical basis: output = input * N(1, stddev) ⟹ E[output] = input * 1 = input.
/// With all-ones input: E[output_i] = 1. The sample mean over 10 000 elements
/// converges tightly; tolerance ±0.05 is roughly 10σ (σ ≈ stddev/sqrt(N)).
#[test]
fn gaussian_dropout_training_preserves_mean() {
    const N: usize = 10_000;
    let input_val = 1.0f32;
    // rate = 0.5  →  stddev = sqrt(0.5 / 0.5) = 1.0
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

/// std(output/input) ≈ sqrt(rate / (1 - rate)) when training and input = 1.
/// This directly validates the stddev formula defined in the source.
/// Mathematical basis: noise_i ~ N(1, σ), so std(noise) = σ = sqrt(rate/(1-rate)).
/// With all-ones input, output_i = noise_i, so std(output) = std(noise) = σ.
/// Tolerance ±0.05 is well within typical large-sample convergence.
#[test]
fn gaussian_dropout_training_noise_std_matches_formula() {
    const N: usize = 10_000;
    // Test two distinct rates to pin the formula.
    // rate=0.5  → expected stddev = sqrt(0.5/0.5)  = 1.0
    // rate=0.25 → expected stddev = sqrt(0.25/0.75) = sqrt(1/3) ≈ 0.5774
    let cases: &[(f32, f64)] = &[
        (0.5, 1.0),
        (0.25, (1.0_f64 / 3.0).sqrt()), // ≈ 0.5774
    ];

    for &(rate, expected_std) in cases {
        let input = Array::from_shape_vec((N,), vec![1.0f32; N])
            .unwrap()
            .into_dyn();

        let mut layer = GaussianDropout::new(rate, vec![N], None).unwrap();
        layer.set_training_if_mode_dependent(true);

        let output = layer.forward(&input).unwrap();
        // With input=1, output = noise, so std(output) = std(noise).
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

/// Two consecutive forward() calls in training mode must not produce identical outputs.
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

// ─── GaussianDropout: error paths ────────────────────────────────────────────

/// forward() with a shape-mismatched input must return Err(ShapeMismatch).
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

/// predict() with a shape-mismatched input must return Err(ShapeMismatch).
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

/// Empty input_shape is a wildcard: any tensor shape is accepted.
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

/// backward() does not reference any forward-pass state; it must succeed even without a prior
/// forward() call. Mathematical basis: d(x * noise)/dx = noise, but the implementation simply
/// passes grad_output through unchanged (stateless gradient).
#[test]
fn gaussian_dropout_backward_without_forward_does_not_panic() {
    let grad: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32; 6])
        .unwrap()
        .into_dyn();
    let mut layer = GaussianDropout::new(0.3, vec![2, 3], None).unwrap();

    let result = layer.backward(&grad);
    assert!(
        result.is_ok(),
        "backward() without prior forward() should succeed, got {:?}",
        result
    );
}

// ─── GaussianDropout: metadata ────────────────────────────────────────────────

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

// ─── Mode-switching symmetry ──────────────────────────────────────────────────

/// set_training_if_mode_dependent(true) must re-enable noise; (false) must disable it.
/// This tests that the macro-generated set_training_if_mode_dependent routes correctly.
#[test]
fn gaussian_noise_mode_switching_routes_correctly() {
    const N: usize = 1_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianNoise::new(2.0, vec![N], None).unwrap();

    // Start in eval mode: must be identity.
    layer.set_training_if_mode_dependent(false);
    let eval_out = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out, &input, 1e-6f32);

    // Switch to training: must now differ from input.
    layer.set_training_if_mode_dependent(true);
    let train_out = layer.forward(&input).unwrap();
    let any_different = train_out.iter().zip(input.iter()).any(|(a, b)| a != b);
    assert!(any_different, "training mode must add noise");

    // Switch back to eval: must be identity again.
    layer.set_training_if_mode_dependent(false);
    let eval_out2 = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out2, &input, 1e-6f32);
}

/// Same symmetry test for GaussianDropout.
#[test]
fn gaussian_dropout_mode_switching_routes_correctly() {
    const N: usize = 1_000;
    let input = Array::from_shape_vec((N,), vec![1.0f32; N])
        .unwrap()
        .into_dyn();

    let mut layer = GaussianDropout::new(0.5, vec![N], None).unwrap();

    // Start in eval mode.
    layer.set_training_if_mode_dependent(false);
    let eval_out = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out, &input, 1e-6f32);

    // Switch to training.
    layer.set_training_if_mode_dependent(true);
    let train_out = layer.forward(&input).unwrap();
    let any_different = train_out.iter().zip(input.iter()).any(|(a, b)| a != b);
    assert!(any_different, "training mode must multiply by noise");

    // Switch back to eval.
    layer.set_training_if_mode_dependent(false);
    let eval_out2 = layer.forward(&input).unwrap();
    crate::common::assert_allclose(&eval_out2, &input, 1e-6f32);
}

/// GaussianNoise::backward is a pure gradient pass-through: since y = x + noise and the noise does
/// not depend on x, d(y)/dx = 1, so the upstream gradient must come back unchanged — element for
/// element. (The only existing noise backward test exercises GaussianDropout, not GaussianNoise.)
#[test]
fn gaussian_noise_backward_passes_gradient_through_unchanged() {
    let mut layer = GaussianNoise::new(1.0, vec![2, 3], None).unwrap();
    // A realistic call order: a training-mode forward first, then backward.
    layer.set_training_if_mode_dependent(true);
    let x: Tensor = Array::from_elem((2, 3), 1.0f32).into_dyn();
    let _ = layer.forward(&x).unwrap();

    let grad: Tensor = Array::from_shape_vec((2, 3), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0])
        .unwrap()
        .into_dyn();
    let grad_in = layer.backward(&grad).unwrap();

    // Pure pass-through ⇒ byte-identical gradient (zero epsilon).
    crate::common::assert_allclose(&grad_in, &grad, 0.0f32);
}
