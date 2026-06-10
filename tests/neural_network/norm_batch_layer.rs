//! Integration tests for BatchNormalization and LayerNormalization.
//!
//! Every expected value is derived from the MATHEMATICAL DEFINITION or a HAND CALCULATION;
//! no value was obtained by running the layer and recording its output.
//!
//! Gradient correctness is already covered by the finite-difference harness in
//! tests/neural_network/gradient_check.rs; this file does NOT duplicate those checks.

use ndarray::ArrayD;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::regularization::normalization::layer_normalization::{
    LayerNormalization, LayerNormalizationAxis,
};
use rustyml::neural_network::traits::Layer;

use crate::common::assert_allclose;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 1-D Tensor (shape [n]) from a flat Vec.
fn tensor1(data: Vec<f32>) -> rustyml::neural_network::Tensor {
    let n = data.len();
    ArrayD::from_shape_vec(vec![n], data).expect("tensor1: shape/data mismatch")
}

/// Build a 2-D Tensor (shape [rows, cols]) from a flat Vec (row-major).
fn tensor2(data: Vec<f32>, rows: usize, cols: usize) -> rustyml::neural_network::Tensor {
    ArrayD::from_shape_vec(vec![rows, cols], data).expect("tensor2: shape/data mismatch")
}

// ══════════════════════════════════════════════════════════════════════════════
//
//  BATCH NORMALIZATION
//
// ══════════════════════════════════════════════════════════════════════════════

// ─── Constructor validation ──────────────────────────────────────────────────

/// BatchNormalization::new returns Err(EmptyInput) when input_shape is empty.
///
/// Derivation: validate_input_shape_not_empty([]) returns Error::EmptyInput.
#[test]
fn bn_constructor_rejects_empty_input_shape() {
    let result = BatchNormalization::new(vec![], 0.9, 1e-5);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput, got {:?}",
        result
    );
}

/// BatchNormalization::new returns Err(InvalidParameter) for momentum > 1.0.
///
/// Derivation: validate_momentum(1.5) fails because 1.5 > 1.0 is outside [0.0, 1.0].
#[test]
fn bn_constructor_rejects_momentum_above_one() {
    let result = BatchNormalization::new(vec![4, 3], 1.5, 1e-5);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// BatchNormalization::new returns Err(InvalidParameter) for momentum < 0.0.
///
/// Derivation: validate_momentum(-0.1) fails because -0.1 < 0.0.
#[test]
fn bn_constructor_rejects_negative_momentum() {
    let result = BatchNormalization::new(vec![4, 3], -0.1, 1e-5);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// BatchNormalization::new returns Err(InvalidParameter) for epsilon <= 0.
///
/// Derivation: validate_epsilon(0.0) fails because epsilon must be strictly positive.
#[test]
fn bn_constructor_rejects_zero_epsilon() {
    let result = BatchNormalization::new(vec![4, 3], 0.9, 0.0);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// BatchNormalization::new returns Err(InvalidParameter) for negative epsilon.
///
/// Derivation: validate_epsilon(-1e-5) fails because epsilon must be positive.
#[test]
fn bn_constructor_rejects_negative_epsilon() {
    let result = BatchNormalization::new(vec![4, 3], 0.9, -1e-5);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// BatchNormalization::new succeeds for valid boundary inputs (momentum=0.0 and momentum=1.0).
#[test]
fn bn_constructor_accepts_boundary_momentum_values() {
    assert!(BatchNormalization::new(vec![4, 3], 0.0, 1e-5).is_ok());
    assert!(BatchNormalization::new(vec![4, 3], 1.0, 1e-5).is_ok());
}

// ─── Shape mismatch on forward ───────────────────────────────────────────────

/// BatchNormalization forward returns Err(ShapeMismatch) when input shape differs.
///
/// Layer is built for [4, 3]; we feed [2, 3]. validate_input_shape fires.
#[test]
fn bn_forward_rejects_wrong_input_shape() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    let wrong = tensor2(vec![1.0f32; 6], 2, 3);
    let result = bn.forward(&wrong);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {:?}",
        result
    );
}

// ─── Forward in TRAINING mode: output has batch-mean ≈ 0 and batch-var ≈ 1 ──

/// BN training-mode forward produces outputs with batch-mean ≈ 0 and batch-var ≈ 1
/// per feature (with default gamma=1, beta=0).
///
/// Derivation:
///   BN in training mode computes, for each feature j:
///     μ_j  = mean over batch of x[:,j]
///     σ²_j = mean over batch of (x[:,j] - μ_j)^2
///     y[:,j] = (x[:,j] - μ_j) / sqrt(σ²_j + ε) * γ_j + β_j
///   With γ=1, β=0: output = x_normalized.  By construction:
///     mean_batch(x_normalized[:,j]) = 0          (exact up to floating-point)
///     var_batch(x_normalized[:,j])  ≈ 1          (exact ignoring the ε term)
///   We assert |mean| < 1e-5 and |var - 1| < 5e-4.
#[test]
fn bn_train_output_has_batch_mean_zero_and_var_one() {
    // Input: 4 samples, 3 features — deliberately non-uniform so each column
    // has a non-trivial mean and variance.
    let data = vec![
        2.0f32, 4.0, 6.0, // row 0
        4.0, 2.0, 8.0, // row 1
        6.0, 4.0, 4.0, // row 2
        8.0, 2.0, 6.0, // row 3
    ];
    let input = tensor2(data, 4, 3);

    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    // Default: training=true, gamma=1, beta=0.
    let output = bn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // Check batch-mean ≈ 0 and batch-variance ≈ 1 for each feature column.
    for feat in 0..3 {
        let col: Vec<f32> = (0..4).map(|r| output[[r, feat]]).collect();
        let mean: f32 = col.iter().sum::<f32>() / 4.0;
        let var: f32 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "feature {feat}: batch mean too far from 0, got {mean}"
        );
        // var = sigma^2 / (sigma^2 + eps), which is < 1 but very close to 1.
        // Tolerance of 5e-4 is generous enough for eps=1e-5 with typical values.
        assert!(
            (var - 1.0).abs() < 5e-4,
            "feature {feat}: batch var too far from 1, got {var}"
        );
    }
}

// ─── Forward in TRAINING mode: concrete numerical output values ───────────────

/// BN training-mode forward, concrete hand-computed values for a 4×1 batch.
///
/// Derivation:
///   Input: [[2], [4], [6], [8]]   (4 samples, 1 feature)
///   batch_mean  = (2+4+6+8)/4 = 20/4 = 5
///   x_centered  = [-3, -1, 1, 3]
///   batch_var   = (9+1+1+9)/4 = 5
///   std_dev     = sqrt(5 + 1e-5) ≈ 2.2360687…
///   x_norm[i]   = x_centered[i] / std_dev
///   With gamma=1, beta=0 → output = x_norm:
///     out[0] = -3 / sqrt(5+1e-5) ≈ -1.3416405
///     out[1] = -1 / sqrt(5+1e-5) ≈ -0.4472135
///     out[2] =  1 / sqrt(5+1e-5) ≈  0.4472135
///     out[3] =  3 / sqrt(5+1e-5) ≈  1.3416405
#[test]
fn bn_train_forward_concrete_values_4x1() {
    let input = tensor2(vec![2.0f32, 4.0, 6.0, 8.0], 4, 1);
    let mut bn = BatchNormalization::new(vec![4, 1], 0.9, 1e-5).unwrap();

    let output = bn.forward(&input).unwrap();

    // Compute the expected values analytically.
    let std = (5.0f32 + 1e-5f32).sqrt();
    let expected = tensor2(vec![-3.0 / std, -1.0 / std, 1.0 / std, 3.0 / std], 4, 1);
    assert_allclose(&output, &expected, 1e-5);
}

// ─── Running statistics update after training forward ─────────────────────────

/// After one training-mode forward, running_mean and running_var are updated.
///
/// Derivation:
///   Input (4 samples, 1 feature): [[2], [4], [6], [8]]
///   batch_mean = 5, batch_var = 5
///   Initial running_mean = 0, running_var = 1, momentum = 0.9
///   Updated running_mean = 0.9 * 0 + 0.1 * 5 = 0.5
///   Updated running_var  = 0.9 * 1 + 0.1 * 5 = 0.9 + 0.5 = 1.4
///
///   We verify in EVAL mode by feeding input [[5.0]]:
///   output = (5.0 - 0.5) / sqrt(1.4 + 1e-5) = 4.5 / sqrt(1.4 + 1e-5)
///          ≈ 4.5 / 1.18322…  ≈ 3.80256…
#[test]
fn bn_running_stats_update_after_one_forward() {
    let input_train = tensor2(vec![2.0f32, 4.0, 6.0, 8.0], 4, 1);
    let mut bn = BatchNormalization::new(vec![4, 1], 0.9, 1e-5).unwrap();

    // Training forward; updates running stats.
    bn.forward(&input_train).unwrap();

    // Switch to eval and feed the same shape [4,1] (as declared in new()).
    bn.set_training_if_mode_dependent(false);
    // All rows = 5.0; each will produce the same eval output.
    let input_eval = tensor2(vec![5.0f32, 5.0, 5.0, 5.0], 4, 1);
    let output = bn.forward(&input_eval).unwrap();

    // running_mean = 0.5, running_var = 1.4 (see derivation above).
    let expected_val = 4.5_f32 / (1.4f32 + 1e-5f32).sqrt();
    let expected = tensor2(vec![expected_val; 4], 4, 1);
    assert_allclose(&output, &expected, 1e-4);
}

// ─── EVAL mode: uses running stats from set_weights ───────────────────────────

/// In eval mode, BN uses running_mean and running_var injected via set_weights.
///
/// Derivation:
///   running_mean = [1, 2, 3], running_var = [4, 9, 1], gamma = [1, 1, 1], beta = [0, 0, 0]
///   Input: [[1, 2, 3], [5, 8, 4]]
///   eval output[i,j] = (x[i,j] - running_mean[j]) / sqrt(running_var[j] + eps)
///
///   Row 0: x = [1, 2, 3]
///     col0: (1-1) / sqrt(4+1e-5) = 0
///     col1: (2-2) / sqrt(9+1e-5) = 0
///     col2: (3-3) / sqrt(1+1e-5) = 0
///   Row 1: x = [5, 8, 4]
///     col0: (5-1) / sqrt(4+1e-5) = 4 / sqrt(4+1e-5) ≈ 1.9999900
///     col1: (8-2) / sqrt(9+1e-5) = 6 / sqrt(9+1e-5) ≈ 1.9999944
///     col2: (4-3) / sqrt(1+1e-5) = 1 / sqrt(1+1e-5) ≈ 0.9999950
#[test]
fn bn_eval_uses_running_stats_from_set_weights() {
    // Layer built for shape [2, 3].
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    // Inject known running statistics and trivial gamma/beta.
    let gamma = tensor1(vec![1.0f32, 1.0, 1.0]);
    let beta = tensor1(vec![0.0f32, 0.0, 0.0]);
    let running_mean = tensor1(vec![1.0f32, 2.0, 3.0]);
    let running_var = tensor1(vec![4.0f32, 9.0, 1.0]);
    bn.set_weights(gamma, beta, running_mean, running_var)
        .unwrap();

    // Switch to eval mode.
    bn.set_training_if_mode_dependent(false);

    let input = tensor2(vec![1.0f32, 2.0, 3.0, 5.0, 8.0, 4.0], 2, 3);
    let output = bn.forward(&input).unwrap();

    let eps = 1e-5f32;
    let e00 = 0.0f32;
    let e01 = 0.0f32;
    let e02 = 0.0f32;
    let e10 = 4.0 / (4.0f32 + eps).sqrt();
    let e11 = 6.0 / (9.0f32 + eps).sqrt();
    let e12 = 1.0 / (1.0f32 + eps).sqrt();
    let expected = tensor2(vec![e00, e01, e02, e10, e11, e12], 2, 3);
    assert_allclose(&output, &expected, 1e-4);
}

// ─── predict() == forward() in eval mode ──────────────────────────────────────

/// predict() produces the same output as forward() when the layer is in eval mode,
/// without modifying any internal state.
///
/// Derivation:
///   Both predict and forward (in eval mode) compute:
///     output = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
///   They are logically identical; we verify bit-exact equality.
#[test]
fn bn_predict_equals_forward_in_eval_mode() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let gamma = tensor1(vec![1.0f32, 1.0, 1.0]);
    let beta = tensor1(vec![0.0f32, 0.0, 0.0]);
    let running_mean = tensor1(vec![1.0f32, 2.0, 3.0]);
    let running_var = tensor1(vec![4.0f32, 9.0, 1.0]);
    bn.set_weights(gamma, beta, running_mean, running_var)
        .unwrap();
    bn.set_training_if_mode_dependent(false);

    let input = tensor2(vec![1.0f32, 2.0, 3.0, 5.0, 8.0, 4.0], 2, 3);

    let out_forward = bn.forward(&input).unwrap();
    let out_predict = bn.predict(&input).unwrap();

    // Identical outputs: element-wise exact equality.
    assert_allclose(&out_forward, &out_predict, 0.0f32);
}

// ─── set_weights rejects wrong shapes ─────────────────────────────────────────

/// set_weights returns Err(NeuralNetwork(WeightShape)) when gamma has the wrong shape.
///
/// Derivation: BN with input_shape=[4,3] has gamma shape [3]; we supply shape [2].
///   validate_weight_shape("gamma", [3], [2]) returns NnError::WeightShape.
#[test]
fn bn_set_weights_rejects_wrong_gamma_shape() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();

    let gamma_bad = tensor1(vec![1.0f32, 1.0]); // wrong: [2] instead of [3]
    let beta = tensor1(vec![0.0f32, 0.0, 0.0]);
    let rm = tensor1(vec![0.0f32, 0.0, 0.0]);
    let rv = tensor1(vec![1.0f32, 1.0, 1.0]);

    let result = bn.set_weights(gamma_bad, beta, rm, rv);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected NnError::WeightShape, got {:?}",
        result
    );
}

// ─── Custom gamma and beta scale/shift output ─────────────────────────────────

/// BN eval mode: custom gamma and beta are applied correctly.
///
/// Derivation:
///   running_mean = [0], running_var = [1], gamma = [2], beta = [1], eps = 1e-5
///   Input: [[3.0]]
///   x_norm = (3 - 0) / sqrt(1 + 1e-5) ≈ 3 / 1.000005 ≈ 2.9999850
///   output = x_norm * 2 + 1 ≈ 6.9999700
#[test]
fn bn_eval_applies_custom_gamma_and_beta() {
    let mut bn = BatchNormalization::new(vec![1, 1], 0.9, 1e-5).unwrap();

    let gamma = tensor1(vec![2.0f32]);
    let beta = tensor1(vec![1.0f32]);
    let running_mean = tensor1(vec![0.0f32]);
    let running_var = tensor1(vec![1.0f32]);
    bn.set_weights(gamma, beta, running_mean, running_var)
        .unwrap();
    bn.set_training_if_mode_dependent(false);

    let input = tensor2(vec![3.0f32], 1, 1);
    let output = bn.forward(&input).unwrap();

    let eps = 1e-5f32;
    let x_norm = 3.0_f32 / (1.0f32 + eps).sqrt();
    let expected_val = x_norm * 2.0 + 1.0;
    let expected = tensor2(vec![expected_val], 1, 1);
    assert_allclose(&output, &expected, 1e-4);
}

// ─── Multiple training forwards accumulate running stats ──────────────────────

/// Running stats accumulate correctly over two training forwards.
///
/// Derivation:
///   Input is always [[0], [2]] (batch=2, features=1).
///   batch_mean = 1, batch_var = 1, initial running_mean=0, running_var=1, momentum=0.5
///
///   After 1st forward:
///     running_mean = 0.5*0 + 0.5*1 = 0.5
///     running_var  = 0.5*1 + 0.5*1 = 1.0
///   After 2nd forward:
///     running_mean = 0.5*0.5 + 0.5*1 = 0.75
///     running_var  = 0.5*1.0 + 0.5*1 = 1.0
///
///   In eval mode feeding [[1]] (= running_mean → output ≈ 0):
///     output = (1 - 0.75) / sqrt(1.0 + 1e-5) = 0.25 / sqrt(1.0 + 1e-5)
#[test]
fn bn_running_stats_accumulate_over_multiple_forwards() {
    let mut bn = BatchNormalization::new(vec![2, 1], 0.5, 1e-5).unwrap();

    let x = tensor2(vec![0.0f32, 2.0], 2, 1);
    bn.forward(&x).unwrap();
    bn.forward(&x).unwrap();

    bn.set_training_if_mode_dependent(false);

    // running_mean ≈ 0.75, running_var ≈ 1.0
    // Must feed shape [2,1] to match declared input_shape.
    let x_eval = tensor2(vec![1.0f32, 5.0], 2, 1);
    let output = bn.forward(&x_eval).unwrap();

    let denom = (1.0f32 + 1e-5f32).sqrt();
    let e0 = (1.0 - 0.75) / denom;
    let e1 = (5.0 - 0.75) / denom;
    let expected = tensor2(vec![e0, e1], 2, 1);
    assert_allclose(&output, &expected, 1e-4);
}

// ─── Uniform batch: BN output is all zeros ────────────────────────────────────

/// When all rows of the batch are identical, BN output is all zeros.
///
/// Derivation:
///   If x[0,j] = x[1,j] = … = x[N-1,j] = c for all i:
///     batch_mean[j] = c
///     x_centered[:,j] = 0
///     batch_var[j]  = 0
///     std_dev[j]    = sqrt(0 + eps) = sqrt(eps)
///     x_norm[:,j]   = 0 / sqrt(eps) = 0
///   With gamma=1, beta=0: output = 0 everywhere.
#[test]
fn bn_uniform_batch_output_is_zero() {
    let data = vec![
        3.0f32, 7.0, -2.0, // row 0 — same as all others
        3.0, 7.0, -2.0, // row 1
        3.0, 7.0, -2.0, // row 2
        3.0, 7.0, -2.0, // row 3
    ];
    let input = tensor2(data, 4, 3);
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();

    let output = bn.forward(&input).unwrap();
    let zeros = tensor2(vec![0.0f32; 12], 4, 3);
    assert_allclose(&output, &zeros, 1e-6);
}

// ─── Training mode vs eval mode produce different outputs ─────────────────────

/// With a non-trivial input, BN training mode produces different output from eval mode.
///
/// Derivation:
///   Training mode normalizes using the BATCH statistics (mean/var of the current mini-batch).
///   Eval mode normalizes using the stored RUNNING statistics (0 and 1 by default).
///   Unless the batch happens to have the same statistics as the running stats, the outputs differ.
///   With running_mean=0 and running_var=1 but batch_mean ≠ 0:
///     eval output[i,j]  = (x[i,j] - 0) / sqrt(1 + eps)     ≈ x[i,j]
///     train output[i,j] = (x[i,j] - batch_mean[j]) / sqrt(batch_var[j] + eps)
///   For x = [[1,1],[3,3]], batch_mean=[2,2], so train output[0]≈-1/1=-1, eval output[0]≈1.
#[test]
fn bn_training_and_eval_modes_produce_different_outputs() {
    let input = tensor2(vec![1.0f32, 1.0, 3.0, 3.0], 2, 2);

    let mut bn_train = BatchNormalization::new(vec![2, 2], 0.9, 1e-5).unwrap();
    // Training mode (default), running stats untouched (mean=0, var=1).
    let out_train = bn_train.forward(&input).unwrap();

    let mut bn_eval = BatchNormalization::new(vec![2, 2], 0.9, 1e-5).unwrap();
    bn_eval.set_training_if_mode_dependent(false);
    let out_eval = bn_eval.forward(&input).unwrap();

    // The two outputs must differ; check at least one element.
    let differs = out_train
        .iter()
        .zip(out_eval.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(
        differs,
        "training and eval outputs are unexpectedly identical"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
//
//  LAYER NORMALIZATION
//
// ══════════════════════════════════════════════════════════════════════════════

// ─── Constructor validation ──────────────────────────────────────────────────

/// LayerNormalization::new returns Err(InvalidParameter) for epsilon <= 0.
///
/// Derivation: validate_epsilon(0.0) → epsilon must be strictly positive.
#[test]
fn ln_constructor_rejects_zero_epsilon() {
    let result = LayerNormalization::new(vec![4, 3], LayerNormalizationAxis::Default, 0.0);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LayerNormalization::new returns Err(InvalidParameter) for negative epsilon.
#[test]
fn ln_constructor_rejects_negative_epsilon() {
    let result = LayerNormalization::new(vec![4, 3], LayerNormalizationAxis::Default, -1e-5);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LayerNormalization::new with Multiple([]) returns Err(InvalidParameter).
///
/// Derivation: the Multiple variant checks axes.is_empty() and returns
///   Error::InvalidParameter("normalized_axis", "…non-empty").
#[test]
fn ln_constructor_rejects_multiple_empty_axes() {
    let result =
        LayerNormalization::new(vec![4, 3], LayerNormalizationAxis::Multiple(vec![]), 1e-5);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LayerNormalization::new with Multiple(axes) where an axis is out of bounds.
///
/// Derivation: input_shape=[4,3] has ndim=2; axis 2 is out of bounds.
///   The Multiple branch checks a >= input_shape.len() and returns InvalidParameter.
#[test]
fn ln_constructor_rejects_multiple_out_of_bounds_axis() {
    let result = LayerNormalization::new(
        vec![4, 3],
        LayerNormalizationAxis::Multiple(vec![0, 2]),
        1e-5,
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LayerNormalization::new with Multiple(axes) containing a duplicate axis.
///
/// Derivation: [0, 0] has a duplicate; the Multiple branch detects axes[..i].contains(&a).
#[test]
fn ln_constructor_rejects_multiple_duplicate_axes() {
    let result = LayerNormalization::new(
        vec![4, 3],
        LayerNormalizationAxis::Multiple(vec![0, 0]),
        1e-5,
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

// ─── Default axis: each sample has mean ≈ 0 and variance ≈ 1 ─────────────────

/// LN Default: each sample row has batch-mean ≈ 0 and variance ≈ 1.
///
/// Derivation:
///   Default normalizes along the LAST dimension (axis 1 for 2D input).
///   For each row i: y[i,:] = (x[i,:] - mean(x[i,:])) / sqrt(var(x[i,:]) + eps).
///   By construction: mean(y[i,:]) = 0 and var(y[i,:]) ≈ 1.
///   We assert |mean| < 1e-5 and |var - 1| < 5e-4 for each row.
#[test]
fn ln_default_each_sample_has_mean_zero_and_var_one() {
    let data = vec![
        1.0f32, 3.0, 5.0, 7.0, // row 0: mean=4, var=5
        2.0, -2.0, 0.0, 4.0, // row 1: mean=1, var=5
    ];
    let input = tensor2(data, 2, 4);
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    for row in 0..2 {
        let vals: Vec<f32> = (0..4).map(|c| output[[row, c]]).collect();
        let mean: f32 = vals.iter().sum::<f32>() / 4.0;
        let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row {row}: mean {mean} too far from 0");
        assert!(
            (var - 1.0).abs() < 5e-4,
            "row {row}: var {var} too far from 1"
        );
    }
}

// ─── Default axis: concrete hand-computed values ──────────────────────────────

/// LN Default: concrete output values for one row [1, 3, 5, 7].
///
/// Derivation for row [1, 3, 5, 7] (4 features):
///   mean = (1+3+5+7)/4 = 16/4 = 4.0
///   x_centered = [-3, -1, 1, 3]
///   var = (9+1+1+9)/4 = 20/4 = 5.0
///   std = sqrt(5.0 + 1e-5) ≈ 2.23607...
///   x_norm = [-3, -1, 1, 3] / std  ≈ [-1.34164, -0.44721, 0.44721, 1.34164]
///   With gamma=1, beta=0: output = x_norm.
///
/// For a second row [2, -2, 0, 4]:
///   mean = (2 + (-2) + 0 + 4)/4 = 4/4 = 1.0
///   x_centered = [1, -3, -1, 3]
///   var = (1+9+1+9)/4 = 20/4 = 5.0
///   std = sqrt(5 + 1e-5) ≈ 2.23607
///   x_norm ≈ [0.44721, -1.34164, -0.44721, 1.34164]
#[test]
fn ln_default_forward_concrete_values() {
    let data = vec![
        1.0f32, 3.0, 5.0, 7.0, // row 0
        2.0, -2.0, 0.0, 4.0, // row 1
    ];
    let input = tensor2(data, 2, 4);
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();

    let std_val = (5.0f32 + 1e-5f32).sqrt();
    let expected = tensor2(
        vec![
            -3.0 / std_val,
            -1.0 / std_val,
            1.0 / std_val,
            3.0 / std_val,
            1.0 / std_val,
            -3.0 / std_val,
            -1.0 / std_val,
            3.0 / std_val,
        ],
        2,
        4,
    );
    assert_allclose(&output, &expected, 1e-5);
}

// ─── Custom axis ──────────────────────────────────────────────────────────────

/// LN Custom(axis=0): normalizes along axis 0 (across rows for each column).
///
/// Derivation:
///   Input (3x2): [[1, 4], [3, 2], [5, 6]]   rows=3, cols=2
///   Axis 0 normalization: for each column j, compute mean and var over the 3 rows.
///
///   Col 0: values [1, 3, 5], mean=3, var=(4+0+4)/3=8/3≈2.6667, std=sqrt(8/3+1e-5)≈1.63299
///     x_norm[:,0] = [-2, 0, 2] / std ≈ [-1.22474, 0, 1.22474]
///   Col 1: values [4, 2, 6], mean=4, var=(0+4+4)/3=8/3≈2.6667, std=sqrt(8/3+1e-5)≈1.63299
///     x_norm[:,1] = [0, -2, 2] / std ≈ [0, -1.22474, 1.22474]
///   With gamma=1, beta=0: output = x_norm.
#[test]
fn ln_custom_axis0_concrete_values() {
    // rows=3, cols=2; normalize across rows for each column (axis=0).
    let data = vec![1.0f32, 4.0, 3.0, 2.0, 5.0, 6.0];
    let input = tensor2(data, 3, 2);
    let mut ln =
        LayerNormalization::new(vec![3, 2], LayerNormalizationAxis::Custom(0), 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();

    let std_val = (8.0f32 / 3.0 + 1e-5f32).sqrt();
    // Col 0: centered = [-2, 0, 2], col 1: centered = [0, -2, 2]
    let expected = tensor2(
        vec![
            -2.0 / std_val,
            0.0 / std_val,
            0.0 / std_val,
            -2.0 / std_val,
            2.0 / std_val,
            2.0 / std_val,
        ],
        3,
        2,
    );
    assert_allclose(&output, &expected, 1e-5);
}

// ─── Custom axis: each column has zero mean and unit variance ─────────────────

/// LN Custom(axis=0): each COLUMN of the output has mean ≈ 0 and var ≈ 1.
///
/// Derivation:
///   With Custom(0) on a [5, 3] input, normalization happens across the 5 rows for each
///   of the 3 columns. By the definition of normalization, each column of the output
///   has mean exactly 0 and variance exactly σ² / (σ² + eps) ≈ 1.
#[test]
fn ln_custom_axis0_each_column_has_mean_zero_and_var_one() {
    let data: Vec<f32> = (0..15).map(|v| v as f32 * 1.3 - 4.0).collect();
    let input = tensor2(data, 5, 3);
    let mut ln =
        LayerNormalization::new(vec![5, 3], LayerNormalizationAxis::Custom(0), 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[5, 3]);

    for col in 0..3 {
        let vals: Vec<f32> = (0..5).map(|r| output[[r, col]]).collect();
        let mean: f32 = vals.iter().sum::<f32>() / 5.0;
        let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-5, "col {col}: mean {mean} too far from 0");
        assert!(
            (var - 1.0).abs() < 5e-4,
            "col {col}: var {var} too far from 1"
        );
    }
}

// ─── Multiple axes ─────────────────────────────────────────────────────────────

/// LN Multiple([0, 1]) on a 2D input: normalizes over all elements of each sample.
///
/// Derivation:
///   input_shape=[3, 4], axes=[0, 1] → normalize over all 12 elements jointly.
///   The entire tensor is treated as one group; output has mean=0, var≈1 over all elements.
///
///   input (3×4): elements 0..11 mapped to (0.5*i - 2.75)
///   total mean = mean of arithmetic sequence 0..11 mapped = mean of (0.5*i-2.75) for i in 0..12
///              = 0.5*5.5 - 2.75 = 2.75 - 2.75 = 0   ← so batch-mean=0 is not trivial here
///   We just assert the statistical properties.
#[test]
fn ln_multiple_axes_output_has_mean_zero_and_var_one() {
    let data: Vec<f32> = (0..12).map(|i| 0.5 * i as f32 - 2.75).collect();
    let input = tensor2(data, 3, 4);
    let mut ln = LayerNormalization::new(
        vec![3, 4],
        LayerNormalizationAxis::Multiple(vec![0, 1]),
        1e-5,
    )
    .unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[3, 4]);

    // With Multiple([0,1]) on a 2D tensor, the ENTIRE output is normalized jointly.
    let flat: Vec<f32> = output.iter().cloned().collect();
    let n = flat.len() as f32;
    let mean: f32 = flat.iter().sum::<f32>() / n;
    let var: f32 = flat.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    assert!(mean.abs() < 1e-5, "global mean {mean} too far from 0");
    assert!((var - 1.0).abs() < 5e-4, "global var {var} too far from 1");
}

/// LN Multiple([1]) on a 3D input normalizes along the axis=1 dimension for each [batch, *, spatial].
///
/// Derivation:
///   input_shape=[2, 3, 4], axes=[1]. The merged layout puts axis=1 last: [2, 4, 3] → merged [2, 4, 3].
///   Actually: non-normalized = [0, 2] = axes 0 and 2; wait: axes=[1], non-normalized = [0, 2],
///   perm = [0, 2, 1], permuted shape = [2, 4, 3], merged shape = [2, 4, 3] (inner=3, outer shape=[2,4]).
///   Each position (i, k) across the 3 elements of axis=1 is normalized. We check mean≈0, var≈1
///   along axis=1 of the output.
#[test]
fn ln_multiple_single_axis_on_3d_input() {
    // shape [2, 3, 4]: 2 batches, 3 channels, 4 spatial
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 5.0).collect();
    let shape = vec![2, 3, 4];
    let input = ArrayD::from_shape_vec(shape.clone(), data).unwrap();

    let mut ln = LayerNormalization::new(
        vec![2, 3, 4],
        LayerNormalizationAxis::Multiple(vec![1]),
        1e-5,
    )
    .unwrap();

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Axis=1 has 3 elements; for each (batch, spatial) pair, those 3 values should be normalized.
    for b in 0..2 {
        for s in 0..4 {
            let vals: Vec<f32> = (0..3).map(|c| output[[b, c, s]]).collect();
            let mean: f32 = vals.iter().sum::<f32>() / 3.0;
            let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5, "b={b} s={s}: mean {mean} too far from 0");
            assert!(
                (var - 1.0).abs() < 5e-4,
                "b={b} s={s}: var {var} too far from 1"
            );
        }
    }
}

// ─── Zero-variance row: epsilon prevents NaN ──────────────────────────────────

/// LN Default: a row with all identical values produces finite output (no NaN/Inf).
///
/// Derivation:
///   If x[i,:] = c (constant), then mean = c, x_centered = 0, var = 0.
///   std = sqrt(0 + eps) = sqrt(eps) > 0 (not zero thanks to epsilon).
///   x_norm = 0 / sqrt(eps) = 0.
///   output = 0 * gamma + beta = 0 (with default gamma=1, beta=0).
#[test]
fn ln_default_constant_row_is_finite_and_zero() {
    let data = vec![
        5.0f32, 5.0, 5.0, 5.0, // constant row
        1.0, 2.0, 3.0, 4.0, // non-constant row
    ];
    let input = tensor2(data, 2, 4);
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let output = ln.forward(&input).unwrap();

    // Row 0 (constant): all outputs should be exactly 0.
    for c in 0..4 {
        assert!(
            output[[0, c]].is_finite(),
            "output[0,{c}] = {} is not finite",
            output[[0, c]]
        );
        assert_eq!(
            output[[0, c]],
            0.0,
            "output[0,{c}] should be 0 for constant row"
        );
    }

    // Row 1 should be finite (non-constant row produces well-defined normalized values).
    for c in 0..4 {
        assert!(
            output[[1, c]].is_finite(),
            "output[1,{c}] = {} is not finite",
            output[[1, c]]
        );
    }
}

// ─── predict() == forward() for LN ────────────────────────────────────────────

/// LN predict() produces the same output as forward() (LN has no running stats).
///
/// Derivation:
///   LN has no running stats; its forward always computes stats from the current input.
///   In eval mode, the backward cache is not written, but the forward arithmetic is
///   identical.  In the current implementation, LN's training field does NOT affect
///   the forward computation (only the backward), so predict() == forward() for any mode.
#[test]
fn ln_predict_equals_forward() {
    let data = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0];
    let input = tensor2(data, 2, 4);
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    // Run forward first (writes caches).
    let out_forward = ln.forward(&input).unwrap();
    // predict() should compute the same values without touching the cache.
    let out_predict = ln.predict(&input).unwrap();

    assert_allclose(&out_forward, &out_predict, 1e-6);
}

// ─── set_weights: custom gamma/beta applied ────────────────────────────────────

/// LN set_weights: custom gamma and beta scale and shift the normalized output.
///
/// Derivation:
///   Input: [[0, 4]] (1 sample, 2 features)
///   mean = (0+4)/2 = 2
///   x_centered = [-2, 2]
///   var = (4+4)/2 = 4
///   std = sqrt(4 + 1e-5) ≈ 2.0
///   x_norm = [-1, 1]
///   With gamma = [3, 3] and beta = [10, 10]:
///     output = x_norm * gamma + beta = [-3+10, 3+10] = [7, 13]
///   More precisely: [-1 * 3 + 10, 1 * 3 + 10] = [7, 13].
///   We verify against the explicit formula.
#[test]
fn ln_set_weights_custom_gamma_beta() {
    let input = tensor2(vec![0.0f32, 4.0], 1, 2);
    let mut ln =
        LayerNormalization::new(vec![1, 2], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let gamma = tensor1(vec![3.0f32, 3.0]);
    let beta = tensor1(vec![10.0f32, 10.0]);
    ln.set_weights(gamma, beta).unwrap();

    let output = ln.forward(&input).unwrap();

    let eps = 1e-5f32;
    let std = (4.0f32 + eps).sqrt();
    let x_norm_0 = -2.0_f32 / std;
    let x_norm_1 = 2.0_f32 / std;
    let e0 = x_norm_0 * 3.0 + 10.0;
    let e1 = x_norm_1 * 3.0 + 10.0;
    let expected = tensor2(vec![e0, e1], 1, 2);
    assert_allclose(&output, &expected, 1e-4);
}

/// LN set_weights rejects wrong gamma shape.
///
/// Derivation: LN with Default axis on shape [2, 4] has gamma shape [4];
///   supplying shape [3] triggers validate_weight_shape("gamma", [4], [3]).
#[test]
fn ln_set_weights_rejects_wrong_gamma_shape() {
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let bad_gamma = tensor1(vec![1.0f32, 1.0, 1.0]); // [3] instead of [4]
    let beta = tensor1(vec![0.0f32; 4]);
    let result = ln.set_weights(bad_gamma, beta);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected NnError::WeightShape, got {:?}",
        result
    );
}

// ─── Shape-mismatch on LN forward ─────────────────────────────────────────────

/// LN forward rejects input whose shape doesn't match the declared input_shape.
///
/// Derivation: layer built for [2, 4]; feeding [3, 4] triggers validate_input_shape.
#[test]
fn ln_forward_rejects_wrong_input_shape() {
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let wrong = tensor2(vec![1.0f32; 12], 3, 4);
    let result = ln.forward(&wrong);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {:?}",
        result
    );
}

// ─── LN eval mode: set_training_if_mode_dependent ─────────────────────────────

/// set_training_if_mode_dependent toggles the mode without changing the forward computation.
///
/// Derivation:
///   LayerNormalization's forward arithmetic is the same in both modes (it always uses
///   the current-input statistics — there are no running stats). Only backward behaviour
///   changes. So forward and predict must agree both before and after the mode switch.
#[test]
fn ln_mode_switch_does_not_change_forward_output() {
    let data = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0];
    let input = tensor2(data, 2, 4);
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    // Training mode output.
    let out_train = ln.forward(&input).unwrap();

    // Switch to eval mode.
    ln.set_training_if_mode_dependent(false);
    let out_eval = ln.forward(&input).unwrap();

    // LN forward result must be identical in both modes.
    assert_allclose(&out_train, &out_eval, 1e-6);
}

// ─── LN predict() in eval mode equals forward() in eval mode ──────────────────

/// In eval mode, predict() and forward() agree exactly.
#[test]
fn ln_predict_equals_forward_in_eval_mode() {
    let data = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0];
    let input = tensor2(data, 2, 4);
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    ln.set_training_if_mode_dependent(false);

    let out_forward = ln.forward(&input).unwrap();
    let out_predict = ln.predict(&input).unwrap();

    assert_allclose(&out_forward, &out_predict, 0.0f32);
}

// ─── LN Multiple axes: forward-time axis validation ───────────────────────────

/// LN forward with Multiple(axes) where an axis is out-of-bounds returns Err.
///
/// Derivation:
///   The constructor with axes within bounds succeeds. But if the INPUT tensor has fewer
///   dimensions than expected, the axis can become out-of-bounds at runtime via the
///   merge_normalized_axes helper.
///   Here we build LN for [3, 4] with Multiple([0, 1]) — both valid — then forward succeeds.
///   We test the constructor-level error instead (covered already), and confirm forward succeeds
///   for in-bounds axes.
#[test]
fn ln_multiple_valid_axes_forward_succeeds() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let input = tensor2(data, 3, 4);
    let mut ln = LayerNormalization::new(
        vec![3, 4],
        LayerNormalizationAxis::Multiple(vec![0, 1]),
        1e-5,
    )
    .unwrap();
    let result = ln.forward(&input);
    assert!(
        result.is_ok(),
        "forward with valid Multiple axes failed: {:?}",
        result
    );
}
// ─── backward() before forward() must error ───────────────────────────────────

/// BatchNormalization::backward called BEFORE any forward returns ForwardPassNotRun.
///
/// Derivation:
///   The layer defaults to training=true, so backward does NOT take the inference
///   short-circuit. It then tries to read the cached x_normalized/x_centered/batch_var,
///   which are None until a forward runs, so `.ok_or_else(forward_pass_not_run(..))?`
///   yields Error::NeuralNetwork(NnError::ForwardPassNotRun("BatchNormalization")).
///   (Mirrors the existing GroupNormalization/InstanceNormalization tests.)
#[test]
fn bn_backward_before_forward_errors() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();
    let grad = tensor2(vec![1.0f32; 6], 2, 3);
    let err = bn.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("BatchNormalization"))
        ),
        "expected ForwardPassNotRun(\"BatchNormalization\"), got {:?}",
        err
    );
}

/// LayerNormalization::backward called BEFORE any forward returns ForwardPassNotRun.
///
/// Derivation:
///   The layer defaults to training=true, so backward skips the inference short-circuit
///   and reads the cached x_normalized/x_centered/std_dev, which are None until a forward
///   runs, producing Error::NeuralNetwork(NnError::ForwardPassNotRun("LayerNormalization")).
#[test]
fn ln_backward_before_forward_errors() {
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let grad = tensor2(vec![1.0f32; 8], 2, 4);
    let err = ln.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("LayerNormalization"))
        ),
        "expected ForwardPassNotRun(\"LayerNormalization\"), got {:?}",
        err
    );
}

// ─── LN runtime axis validation on forward / predict ──────────────────────────

/// LN Custom(axis) with axis >= input.ndim() is rejected at forward time with InvalidParameter.
///
/// Derivation:
///   The Custom constructor branch does NOT validate the axis against input_shape (it falls
///   back to a param shape of [1] when input_shape.len() <= axis), so new() succeeds.
///   At forward time, with input_shape=[4] declared (so validate_input_shape passes for a
///   shape-[4] input) and a 1-D input (ndim=1), axis=5 satisfies 5 >= 1, so the Custom branch
///   returns Error::invalid_parameter("normalized_axis", ...) → Error::InvalidParameter.
#[test]
fn ln_custom_axis_out_of_bounds_forward_errors() {
    let mut ln = LayerNormalization::new(vec![4], LayerNormalizationAxis::Custom(5), 1e-5).unwrap();
    let input = tensor1(vec![1.0f32, 2.0, 3.0, 4.0]); // ndim = 1, axis 5 is out of bounds
    let result = ln.forward(&input);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LN Custom(axis) out-of-bounds is also rejected by predict() with InvalidParameter.
///
/// Derivation: predict() duplicates the same axis-resolution logic as forward(), so the
///   identical out-of-bounds Custom axis triggers Error::invalid_parameter.
#[test]
fn ln_custom_axis_out_of_bounds_predict_errors() {
    let ln = LayerNormalization::new(vec![4], LayerNormalizationAxis::Custom(5), 1e-5).unwrap();
    let input = tensor1(vec![1.0f32, 2.0, 3.0, 4.0]); // ndim = 1, axis 5 is out of bounds
    let result = ln.predict(&input);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// LN Default on a 0-dim (scalar) tensor is rejected at forward time with InvalidInput.
///
/// Derivation:
///   new(vec![], Default, ..) succeeds (Default param shape for an empty input_shape is [1]).
///   forward calls validate_input_shape, which short-circuits to Ok when the declared shape
///   is empty. The Default branch then checks input.ndim() == 0 and returns
///   Error::invalid_input("Cannot normalize a scalar tensor") → Error::InvalidInput.
#[test]
fn ln_default_scalar_input_forward_errors() {
    let mut ln = LayerNormalization::new(vec![], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let scalar = ArrayD::from_shape_vec(vec![], vec![3.0f32]).unwrap(); // 0-dim tensor
    let result = ln.forward(&scalar);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        result
    );
}

/// LN Default on a 0-dim (scalar) tensor is also rejected by predict() with InvalidInput.
///
/// Derivation: predict() duplicates the Default scalar guard from forward().
#[test]
fn ln_default_scalar_input_predict_errors() {
    let ln = LayerNormalization::new(vec![], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let scalar = ArrayD::from_shape_vec(vec![], vec![3.0f32]).unwrap(); // 0-dim tensor
    let result = ln.predict(&scalar);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        result
    );
}
// ─── EVAL-mode backward: gradient passes through unchanged ─────────────────────

/// BatchNormalization::backward in EVAL mode returns grad_output unchanged.
///
/// Derivation:
///   In eval mode (training == false) backward takes the early return
///   `if !self.training { return Ok(grad_output.clone()) }` (BN source 293-296),
///   so the returned input-gradient is a bit-exact copy of grad_output: the layer
///   contributes no Jacobian during inference. We feed a grad with all-distinct
///   values so any accidental transformation would be detected, and compare with eps=0.
#[test]
fn bn_backward_eval_mode_passes_gradient_through() {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();
    bn.set_training_if_mode_dependent(false);

    // A forward in eval mode (uses running stats); does not affect the passthrough.
    let input = tensor2(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    bn.forward(&input).unwrap();

    // Distinct gradient values so an accidental transform would change them.
    let grad = tensor2(vec![0.5f32, -1.5, 2.0, -3.0, 4.5, -6.0], 2, 3);
    let grad_input = bn.backward(&grad).unwrap();

    // Eval-mode backward returns grad_output.clone(): bit-exact equality.
    assert_allclose(&grad_input, &grad, 0.0f32);
}

/// LayerNormalization::backward in EVAL mode returns grad_output unchanged.
///
/// Derivation:
///   In eval mode (training == false) backward takes the early return
///   `if !self.training { return Ok(grad_output.clone()) }` (LN source 624-627),
///   so the input-gradient is a bit-exact copy of grad_output. We assert exact equality.
#[test]
fn ln_backward_eval_mode_passes_gradient_through() {
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    ln.set_training_if_mode_dependent(false);

    let input = tensor2(vec![1.0f32, 3.0, 5.0, 7.0, 2.0, -2.0, 0.0, 4.0], 2, 4);
    ln.forward(&input).unwrap();

    let grad = tensor2(vec![0.5f32, -1.5, 2.0, -3.0, 4.5, -6.0, 7.0, -8.5], 2, 4);
    let grad_input = ln.backward(&grad).unwrap();

    assert_allclose(&grad_input, &grad, 0.0f32);
}

// ─── BN constructor with 1-D input_shape: scalar-parameter branch ─────────────

/// BatchNormalization::new with a 1-D input_shape takes the scalar-param branch.
///
/// Derivation:
///   For a 1-D `input_shape` (here vec![4]) the constructor sets param_shape = vec![1]
///   (the `else` branch of `if input_shape.len() > 1`, BN source 108-112), so
///   gamma/beta/running_mean/running_var are all length-1 scalars broadcast over the
///   whole input.
///
///   Training-mode forward over input [1, 2, 3, 4] (shape [4]) then computes, with the
///   batch axis = axis 0 (every element is its own "sample", one scalar feature):
///     batch_mean = (1+2+3+4)/4 = 2.5
///     x_centered = [-1.5, -0.5, 0.5, 1.5]
///     batch_var  = (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4 = 5.0/4 = 1.25
///     std        = sqrt(1.25 + 1e-5)
///     output     = x_centered / std  (gamma=1, beta=0)
///   Output shape must equal the input shape [4] (ndarray broadcasts [4] with the
///   length-1 gamma/beta to [4]).
#[test]
fn bn_new_scalar_param_branch_forward_1d() {
    let mut bn = BatchNormalization::new(vec![4], 0.9, 1e-5).unwrap();

    let input = tensor1(vec![1.0f32, 2.0, 3.0, 4.0]);
    let output = bn.forward(&input).unwrap();

    // Shape is preserved (sane shapes through the scalar-param broadcast).
    assert_eq!(output.shape(), &[4]);

    let std = (1.25f32 + 1e-5f32).sqrt();
    let expected = tensor1(vec![-1.5 / std, -0.5 / std, 0.5 / std, 1.5 / std]);
    assert_allclose(&output, &expected, 1e-5);
}
