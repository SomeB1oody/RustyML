//! Integration tests for the neural-network loss functions.
//!
//! Every expected value is derived from the closed-form mathematical definition or a
//! hand calculation, **not** by running the implementation and recording its output.
//! Gradient-correctness is NOT duplicated here — that is covered by the finite-difference
//! harness in gradient_check.rs.  These tests focus on:
//!
//!  - Forward loss VALUES (hand-computed).
//!  - Gradient VALUES (closed-form; a minimal smoke-check that the formula is wired correctly).
//!  - Perfect-prediction → zero loss AND zero gradient.
//!  - Numerical stability: BCE/CCE/SCCE with predictions exactly at 0.0 / 1.0 clip to finite.
//!  - Shape mismatch → Err for both compute_loss and compute_grad.
//!  - SCCE-specific label validation (out-of-range, negative, wrong shape).

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::error::Error;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::losses::{
    BinaryCrossEntropy, CategoricalCrossEntropy, MeanAbsoluteError, MeanSquaredError,
    SparseCategoricalCrossEntropy,
};
use rustyml::neural_network::traits::Loss;

use crate::common::assert_allclose;

// ═══════════════════════════════════════════════════════════════════
// Mean Squared Error
// ═══════════════════════════════════════════════════════════════════

/// Hand derivation:
///   y_true = [[1,2],[3,4]]   y_pred = [[1,3],[5,4]]
///   (pred - true)^2 = [0, 1, 4, 0]   mean = 5/4 = 1.25
#[test]
fn mse_forward_value() {
    let mse = MeanSquaredError::new();

    let y_true: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 3.0, 5.0, 4.0])
        .unwrap()
        .into_dyn();

    let loss = mse.compute_loss(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(loss, 1.25_f32, epsilon = 1e-6);
}

/// Gradient = 2*(pred - true) / n
///   n = 4, differences = [0, 1, 2, 0]
///   grad = [0.0, 0.5, 1.0, 0.0]
#[test]
fn mse_gradient_value() {
    let mse = MeanSquaredError::new();

    let y_true: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 3.0, 5.0, 4.0])
        .unwrap()
        .into_dyn();

    let grad = mse.compute_grad(&y_true, &y_pred).unwrap();
    let expected: Tensor = Array::from_shape_vec(vec![2, 2], vec![0.0_f32, 0.5, 1.0, 0.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&grad, &expected, 1e-6_f32);
}

/// Perfect prediction: loss == 0.0 and gradient == 0.0 everywhere.
#[test]
fn mse_perfect_prediction_zero_loss_and_grad() {
    let mse = MeanSquaredError::new();

    let y: Tensor = Array::from_shape_vec(vec![3], vec![1.0_f32, 2.0, 3.0])
        .unwrap()
        .into_dyn();

    let loss = mse.compute_loss(&y, &y).unwrap();
    assert_abs_diff_eq!(loss, 0.0_f32, epsilon = 1e-7);

    let grad = mse.compute_grad(&y, &y).unwrap();
    let zeros: Tensor = Tensor::zeros(y.raw_dim());
    assert_allclose(&grad, &zeros, 1e-7_f32);
}

/// Shape mismatch is rejected with an Err for both compute_loss and compute_grad.
#[test]
fn mse_shape_mismatch_returns_err() {
    let mse = MeanSquaredError::new();

    let a: Tensor = Array::from_shape_vec(vec![2], vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec(vec![3], vec![1.0_f32, 2.0, 3.0])
        .unwrap()
        .into_dyn();

    assert!(matches!(
        mse.compute_loss(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        mse.compute_grad(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════
// Mean Absolute Error
// ═══════════════════════════════════════════════════════════════════

/// Hand derivation:
///   y_true = [1,3,5,4]   y_pred = [1,4,3,4]
///   |diff| = [0, 1, 2, 0]   mean = 3/4 = 0.75
#[test]
fn mae_forward_value() {
    let mae = MeanAbsoluteError::new();

    let y_true: Tensor = Array::from_shape_vec(vec![4], vec![1.0_f32, 3.0, 5.0, 4.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![4], vec![1.0_f32, 4.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();

    let loss = mae.compute_loss(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(loss, 0.75_f32, epsilon = 1e-6);
}

/// Gradient = sign(pred - true) / n
///   n = 4, sign([0, 1, -2, 0]) = [0, 1, -1, 0]
///   grad = [0.0, 0.25, -0.25, 0.0]
///
/// Critically: when pred == true the gradient is 0, not ±1/n.
#[test]
fn mae_gradient_value_including_zero_tie() {
    let mae = MeanAbsoluteError::new();

    let y_true: Tensor = Array::from_shape_vec(vec![4], vec![1.0_f32, 3.0, 5.0, 4.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![4], vec![1.0_f32, 4.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();

    let grad = mae.compute_grad(&y_true, &y_pred).unwrap();
    let expected: Tensor = Array::from_shape_vec(vec![4], vec![0.0_f32, 0.25, -0.25, 0.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&grad, &expected, 1e-6_f32);
}

/// Perfect prediction: every element matches → loss = 0, grad = 0 everywhere.
#[test]
fn mae_perfect_prediction_zero_loss_and_grad() {
    let mae = MeanAbsoluteError::new();

    let y: Tensor = Array::from_shape_vec(vec![3], vec![2.0_f32, 5.0, -1.0])
        .unwrap()
        .into_dyn();

    let loss = mae.compute_loss(&y, &y).unwrap();
    assert_abs_diff_eq!(loss, 0.0_f32, epsilon = 1e-7);

    let grad = mae.compute_grad(&y, &y).unwrap();
    let zeros: Tensor = Tensor::zeros(y.raw_dim());
    assert_allclose(&grad, &zeros, 1e-7_f32);
}

/// Shape mismatch is rejected with an Err.
#[test]
fn mae_shape_mismatch_returns_err() {
    let mae = MeanAbsoluteError::new();

    let a: Tensor = Array::from_shape_vec(vec![2], vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec(vec![3], vec![1.0_f32, 2.0, 3.0])
        .unwrap()
        .into_dyn();

    assert!(matches!(
        mae.compute_loss(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        mae.compute_grad(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════
// Binary Cross-Entropy
// ═══════════════════════════════════════════════════════════════════

/// Hand derivation (element-wise, then mean):
///   y_true = [0, 1, 1, 0]   y_pred = [0.1, 0.9, 0.8, 0.2]
///   per-element BCE = -[y*ln(p) + (1-y)*ln(1-p)]
///     i=0: -[0*ln(0.1) + 1*ln(0.9)] = -ln(0.9) ≈ 0.10536
///     i=1: -[1*ln(0.9) + 0*ln(0.1)] = -ln(0.9) ≈ 0.10536
///     i=2: -[1*ln(0.8) + 0*ln(0.2)] = -ln(0.8) ≈ 0.22314
///     i=3: -[0*ln(0.2) + 1*ln(0.8)] = -ln(0.8) ≈ 0.22314
///   mean ≈ 0.16425
#[test]
fn bce_forward_value() {
    let bce = BinaryCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![4], vec![0.0_f32, 1.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![4], vec![0.1_f32, 0.9, 0.8, 0.2])
        .unwrap()
        .into_dyn();

    let loss = bce.compute_loss(&y_true, &y_pred).unwrap();
    // expected ≈ 0.16425203
    assert_abs_diff_eq!(loss, 0.164252_f32, epsilon = 1e-5);
}

/// Gradient = (-y/p + (1-y)/(1-p)) / n   (with p clipped)
///   n = 4
///   i=0: (0/0.1 + 1/0.9) / 4 = (10/9) / 4 ≈  0.27778
///   i=1: (-1/0.9 + 0/0.1) / 4 = (-10/9)/4 ≈ -0.27778
///   i=2: (-1/0.8 + 0/0.2) / 4 = (-5/4)/4  = -0.31250
///   i=3: (0/0.2 + 1/0.8) / 4 = (5/4)/4    =  0.31250
#[test]
fn bce_gradient_value() {
    let bce = BinaryCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![4], vec![0.0_f32, 1.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![4], vec![0.1_f32, 0.9, 0.8, 0.2])
        .unwrap()
        .into_dyn();

    let grad = bce.compute_grad(&y_true, &y_pred).unwrap();
    let expected: Tensor =
        Array::from_shape_vec(vec![4], vec![0.277778_f32, -0.277778, -0.3125, 0.3125])
            .unwrap()
            .into_dyn();
    assert_allclose(&grad, &expected, 1e-5_f32);
}

/// Perfect binary prediction: loss and gradient are both near-zero.
/// Clipping keeps both finite even when pred == true (values at 0 or 1 are clipped).
#[test]
fn bce_perfect_prediction_near_zero_loss_and_finite_grad() {
    let bce = BinaryCrossEntropy::new();

    // Unambiguous correct labels at both 0 and 1.
    let y_true: Tensor = Array::from_shape_vec(vec![2], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    // Exact 0.0 and 1.0 predictions get clipped to (eps, 1-eps).
    let y_pred: Tensor = Array::from_shape_vec(vec![2], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();

    let loss = bce.compute_loss(&y_true, &y_pred).unwrap();
    // After clipping at eps=1e-7 each element is -ln(1-eps) ≈ 1e-7; mean ≈ 1e-7.
    assert!(
        loss.is_finite(),
        "BCE loss must not be NaN or Inf at extreme predictions"
    );
    assert!(loss >= 0.0, "BCE loss must be non-negative");
    assert!(
        loss < 1e-5,
        "BCE loss at perfect (clipped) prediction should be near zero"
    );

    let grad = bce.compute_grad(&y_true, &y_pred).unwrap();
    for g in grad.iter() {
        assert!(
            g.is_finite(),
            "BCE gradient must be finite at extreme predictions, got {g}"
        );
    }
}

/// Predictions exactly 0.0 and 1.0 are clipped → loss and gradients remain finite (no NaN/Inf).
#[test]
fn bce_extreme_predictions_clipped_to_finite() {
    let bce = BinaryCrossEntropy::new();

    // Worst case: every prediction hits the boundary.
    let y_true: Tensor = Array::from_shape_vec(vec![4], vec![0.0_f32, 1.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![4], vec![0.0_f32, 1.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    let loss = bce.compute_loss(&y_true, &y_pred).unwrap();
    assert!(
        loss.is_finite(),
        "BCE loss must be finite at extreme predictions"
    );

    let grad = bce.compute_grad(&y_true, &y_pred).unwrap();
    for g in grad.iter() {
        assert!(
            g.is_finite(),
            "BCE gradient element must be finite at extreme predictions"
        );
    }
}

/// Shape mismatch is rejected with an Err.
#[test]
fn bce_shape_mismatch_returns_err() {
    let bce = BinaryCrossEntropy::new();

    let a: Tensor = Array::from_shape_vec(vec![2], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec(vec![3], vec![0.2_f32, 0.5, 0.8])
        .unwrap()
        .into_dyn();

    assert!(matches!(
        bce.compute_loss(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        bce.compute_grad(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════
// Categorical Cross-Entropy
// ═══════════════════════════════════════════════════════════════════

/// Hand derivation (batch=3):
///   y_true (one-hot) and y_pred:
///     row 0: [1,0,0] vs [0.8,0.1,0.1] → -ln(0.8) ≈ 0.22314
///     row 1: [0,1,0] vs [0.2,0.7,0.1] → -ln(0.7) ≈ 0.35667
///     row 2: [0,0,1] vs [0.1,0.2,0.7] → -ln(0.7) ≈ 0.35667
///   mean over batch = (0.22314 + 0.35667 + 0.35667) / 3 ≈ 0.31216
///
/// IMPORTANT: CCE divides by shape[0] (batch size = 3), NOT by the total element count (9).
#[test]
fn cce_forward_value_divided_by_batch() {
    let cce = CategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap()
    .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![0.8_f32, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7],
    )
    .unwrap()
    .into_dyn();

    let loss = cce.compute_loss(&y_true, &y_pred).unwrap();
    // Expected: (-ln(0.8) - ln(0.7) - ln(0.7)) / 3 ≈ 0.31216
    assert_abs_diff_eq!(loss, 0.31216_f32, epsilon = 1e-4);
}

/// Gradient = -y_true / y_pred / batch_size
///   Row 0 non-zero: -1/0.8 / 3 ≈ -0.41667
///   Row 1 non-zero: -1/0.7 / 3 ≈ -0.47619
///   Row 2 non-zero: -1/0.7 / 3 ≈ -0.47619
///   All other elements (where y_true == 0): 0
#[test]
fn cce_gradient_value() {
    let cce = CategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap()
    .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![0.8_f32, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7],
    )
    .unwrap()
    .into_dyn();

    let grad = cce.compute_grad(&y_true, &y_pred).unwrap();

    // Flatten to check individual elements.
    let flat: Vec<f32> = grad.iter().cloned().collect();
    assert_eq!(flat.len(), 9);

    // Row 0: only class 0 is non-zero.
    assert_abs_diff_eq!(flat[0], -1.0_f32 / 0.8 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[1], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[2], 0.0_f32, epsilon = 1e-7);

    // Row 1: only class 1 is non-zero.
    assert_abs_diff_eq!(flat[3], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[4], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[5], 0.0_f32, epsilon = 1e-7);

    // Row 2: only class 2 is non-zero.
    assert_abs_diff_eq!(flat[6], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[7], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[8], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
}

/// Perfect one-hot prediction (after clipping at eps=1e-7):
///   loss is extremely small (near eps * ln(eps)^{-1} ≈ 1.6e-6) and definitely < 1e-4.
#[test]
fn cce_perfect_prediction_near_zero_loss_and_finite_grad() {
    let cce = CategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 0.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();
    // Exact 0.0 / 1.0 predictions are clipped internally.
    let y_pred: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 0.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();

    let loss = cce.compute_loss(&y_true, &y_pred).unwrap();
    assert!(
        loss.is_finite(),
        "CCE loss must be finite at perfect (clipped) predictions"
    );
    assert!(loss >= 0.0);
    assert!(
        loss < 1e-4,
        "CCE loss at perfect prediction should be near zero, got {loss}"
    );

    let grad = cce.compute_grad(&y_true, &y_pred).unwrap();
    for g in grad.iter() {
        assert!(
            g.is_finite(),
            "CCE gradient must be finite at extreme predictions"
        );
    }
}

/// Predictions at exactly 0.0 and 1.0 must not produce NaN or Inf.
#[test]
fn cce_extreme_predictions_clipped_to_finite() {
    let cce = CategoricalCrossEntropy::new();

    // One sample, 3 classes; predicted probability at 0.0 on a hot class.
    let y_true: Tensor = Array::from_shape_vec(vec![1, 3], vec![1.0_f32, 0.0, 0.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![1, 3], vec![0.0_f32, 0.5, 0.5])
        .unwrap()
        .into_dyn();

    let loss = cce.compute_loss(&y_true, &y_pred).unwrap();
    assert!(
        loss.is_finite(),
        "CCE loss must be finite when y_pred contains 0.0"
    );

    let grad = cce.compute_grad(&y_true, &y_pred).unwrap();
    for g in grad.iter() {
        assert!(
            g.is_finite(),
            "CCE gradient must be finite when y_pred contains 0.0"
        );
    }
}

/// Shape mismatch is rejected with an Err.
#[test]
fn cce_shape_mismatch_returns_err() {
    let cce = CategoricalCrossEntropy::new();

    let a: Tensor = Array::from_shape_vec(vec![2, 3], vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();
    let b: Tensor = Array::from_shape_vec(vec![2, 2], vec![0.7_f32, 0.3, 0.4, 0.6])
        .unwrap()
        .into_dyn();

    assert!(matches!(
        cce.compute_loss(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        cce.compute_grad(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
}

/// Empty y_true is rejected with an Err.
#[test]
fn cce_empty_input_returns_err() {
    let cce = CategoricalCrossEntropy::new();

    // Shape [0, 3] is technically valid shape but has zero elements.
    let y_true: Tensor = Array::from_shape_vec(vec![0, 3], vec![])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![0, 3], vec![])
        .unwrap()
        .into_dyn();

    assert!(matches!(
        cce.compute_loss(&y_true, &y_pred),
        Err(Error::EmptyInput(_))
    ));
    assert!(matches!(
        cce.compute_grad(&y_true, &y_pred),
        Err(Error::EmptyInput(_))
    ));
}

// ═══════════════════════════════════════════════════════════════════
// Sparse Categorical Cross-Entropy
// ═══════════════════════════════════════════════════════════════════

/// SCCE with integer labels must equal CCE on the equivalent one-hot encoding.
///
/// y_true_labels = [[0],[1],[2]]   y_pred = [[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.2,0.7]]
///   sample 0: -ln(0.8) ≈ 0.22314
///   sample 1: -ln(0.7) ≈ 0.35667
///   sample 2: -ln(0.7) ≈ 0.35667
///   mean ≈ 0.31216  (same as CCE test above — verified by hand)
#[test]
fn scce_forward_value_equals_cce_equivalent() {
    let scce = SparseCategoricalCrossEntropy::new();

    // Labels as [batch, 1] float tensor of class indices.
    let y_true: Tensor = Array::from_shape_vec(vec![3, 1], vec![0.0_f32, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![0.8_f32, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7],
    )
    .unwrap()
    .into_dyn();

    let loss = scce.compute_loss(&y_true, &y_pred).unwrap();
    // Must agree with CCE on the same data to within floating-point round-off.
    assert_abs_diff_eq!(loss, 0.31216_f32, epsilon = 1e-4);
}

/// SCCE gradient: for each sample i, only the entry at the true class is non-zero.
///   grad[i, c_i] = -1/y_pred[i, c_i] / batch
///   Row 0: -1/0.8/3 ≈ -0.41667 at col 0, else 0
///   Row 1: -1/0.7/3 ≈ -0.47619 at col 1, else 0
///   Row 2: -1/0.7/3 ≈ -0.47619 at col 2, else 0
#[test]
fn scce_gradient_value() {
    let scce = SparseCategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![3, 1], vec![0.0_f32, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![0.8_f32, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7],
    )
    .unwrap()
    .into_dyn();

    let grad = scce.compute_grad(&y_true, &y_pred).unwrap();
    let flat: Vec<f32> = grad.iter().cloned().collect();
    assert_eq!(flat.len(), 9);

    // Row 0.
    assert_abs_diff_eq!(flat[0], -1.0_f32 / 0.8 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[1], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[2], 0.0_f32, epsilon = 1e-7);

    // Row 1.
    assert_abs_diff_eq!(flat[3], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[4], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[5], 0.0_f32, epsilon = 1e-7);

    // Row 2.
    assert_abs_diff_eq!(flat[6], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[7], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[8], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
}

/// SCCE gradient matches CCE gradient on the equivalent one-hot encoding.
/// Both must produce identical non-zero entries.
#[test]
fn scce_gradient_matches_cce_equivalent() {
    let scce = SparseCategoricalCrossEntropy::new();
    let cce = CategoricalCrossEntropy::new();

    let y_pred: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![0.8_f32, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7],
    )
    .unwrap()
    .into_dyn();

    let y_true_sparse: Tensor = Array::from_shape_vec(vec![3, 1], vec![0.0_f32, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let y_true_onehot: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap()
    .into_dyn();

    let scce_grad = scce.compute_grad(&y_true_sparse, &y_pred).unwrap();
    let cce_grad = cce.compute_grad(&y_true_onehot, &y_pred).unwrap();

    assert_allclose(&scce_grad, &cce_grad, 1e-5_f32);
}

/// SCCE with a prediction at exactly 0.0 on the true class is clipped → finite loss/grad.
#[test]
fn scce_extreme_prediction_clipped_to_finite() {
    let scce = SparseCategoricalCrossEntropy::new();

    // Label 0, but predicted probability for class 0 is 0.0 (worst case).
    let y_true: Tensor = Array::from_shape_vec(vec![1, 1], vec![0.0_f32])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![1, 3], vec![0.0_f32, 0.6, 0.4])
        .unwrap()
        .into_dyn();

    let loss = scce.compute_loss(&y_true, &y_pred).unwrap();
    assert!(
        loss.is_finite(),
        "SCCE loss must be finite when y_pred contains 0.0"
    );

    let grad = scce.compute_grad(&y_true, &y_pred).unwrap();
    for g in grad.iter() {
        assert!(
            g.is_finite(),
            "SCCE gradient must be finite when y_pred contains 0.0"
        );
    }
}

/// SCCE rejects a negative label.
#[test]
fn scce_negative_label_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![2, 1], vec![-1.0_f32, 0.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![2, 3], vec![0.3_f32, 0.4, 0.3, 0.1, 0.8, 0.1])
        .unwrap()
        .into_dyn();

    assert!(
        matches!(
            scce.compute_loss(&y_true, &y_pred),
            Err(Error::InvalidInput(_))
        ),
        "negative label must return InvalidInput"
    );
    assert!(
        matches!(
            scce.compute_grad(&y_true, &y_pred),
            Err(Error::InvalidInput(_))
        ),
        "negative label must return InvalidInput"
    );
}

/// SCCE rejects an out-of-range label (class index >= num_classes).
#[test]
fn scce_out_of_range_label_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new();

    // 3 classes but label = 3 (valid range is 0..=2).
    let y_true: Tensor = Array::from_shape_vec(vec![1, 1], vec![3.0_f32])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![1, 3], vec![0.3_f32, 0.4, 0.3])
        .unwrap()
        .into_dyn();

    assert!(
        matches!(
            scce.compute_loss(&y_true, &y_pred),
            Err(Error::InvalidInput(_))
        ),
        "out-of-range label must return InvalidInput"
    );
}

/// SCCE rejects y_true with wrong shape (not [batch, 1]).
#[test]
fn scce_wrong_label_shape_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new();

    // Labels are 1D — must be [batch, 1].
    let y_true_1d: Tensor = Array::from_shape_vec(vec![2], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(vec![2, 3], vec![0.3_f32, 0.4, 0.3, 0.1, 0.8, 0.1])
        .unwrap()
        .into_dyn();

    assert!(
        scce.compute_loss(&y_true_1d, &y_pred).is_err(),
        "1D labels must return an Err"
    );

    // Labels are [batch, 2] — wrong second dimension.
    let y_true_wrong: Tensor = Array::from_shape_vec(vec![2, 2], vec![0.0_f32, 0.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    assert!(
        scce.compute_loss(&y_true_wrong, &y_pred).is_err(),
        "[batch, 2] labels must return an Err"
    );
}

/// SCCE rejects a batch-size mismatch between labels and predictions.
#[test]
fn scce_batch_mismatch_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![2, 1], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec(
        vec![3, 3],
        vec![0.3_f32, 0.4, 0.3, 0.1, 0.8, 0.1, 0.2, 0.3, 0.5],
    )
    .unwrap()
    .into_dyn();

    assert!(
        scce.compute_loss(&y_true, &y_pred).is_err(),
        "batch size mismatch must return an Err"
    );
    assert!(
        scce.compute_grad(&y_true, &y_pred).is_err(),
        "batch size mismatch must return an Err (grad)"
    );
}

/// SCCE rejects 3-D predictions (not [batch, num_classes]).
#[test]
fn scce_non_2d_predictions_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new();

    let y_true: Tensor = Array::from_shape_vec(vec![2, 1], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    // 3-D predictions — must be rejected.
    let y_pred: Tensor = Array::from_shape_vec(vec![2, 2, 3], vec![0.1_f32; 12])
        .unwrap()
        .into_dyn();

    assert!(
        matches!(
            scce.compute_loss(&y_true, &y_pred),
            Err(Error::InvalidInput(_))
        ),
        "3D predictions must return InvalidInput"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Cross-loss consistency: SCCE loss == CCE loss (batch normalisation)
// ═══════════════════════════════════════════════════════════════════

/// For any predictions, SCCE(integer labels) must produce the same loss value as
/// CCE(equivalent one-hot encoding) to machine precision.
///
/// This validates the "divides by batch_size" convention is consistent across the two
/// implementations.
#[test]
fn scce_loss_equals_cce_loss_on_equivalent_inputs() {
    let cce = CategoricalCrossEntropy::new();
    let scce = SparseCategoricalCrossEntropy::new();

    let y_pred: Tensor = Array::from_shape_vec(
        vec![4, 3],
        vec![
            0.7_f32, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.3, 0.5, 0.33, 0.34, 0.33,
        ],
    )
    .unwrap()
    .into_dyn();

    // One-hot for CCE.
    let y_true_onehot: Tensor = Array::from_shape_vec(
        vec![4, 3],
        vec![
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ],
    )
    .unwrap()
    .into_dyn();

    // Integer labels for SCCE (same classes as one-hot above).
    let y_true_labels: Tensor = Array::from_shape_vec(vec![4, 1], vec![0.0_f32, 1.0, 2.0, 1.0])
        .unwrap()
        .into_dyn();

    let cce_loss = cce.compute_loss(&y_true_onehot, &y_pred).unwrap();
    let scce_loss = scce.compute_loss(&y_true_labels, &y_pred).unwrap();

    assert_abs_diff_eq!(scce_loss, cce_loss, epsilon = 1e-5);
}
