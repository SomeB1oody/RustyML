//! Integration tests for the neural-network loss functions: hand-computed forward
//! and gradient values, perfect-prediction zero loss/grad, numerical stability under
//! clipping, shape-mismatch errors, and SCCE label validation

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

// Mean Squared Error

/// MSE forward value equals the mean of squared element-wise differences
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

/// MSE gradient equals 2*(pred - true) / n element-wise
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

/// Perfect prediction gives zero MSE loss and zero gradient everywhere
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

/// Shape mismatch is rejected with an Err for both compute_loss and compute_grad
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

// Mean Absolute Error

/// MAE forward value equals the mean of absolute element-wise differences
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

/// MAE gradient equals sign(pred - true) / n, with a zero (not +/-1/n) at exact ties
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

/// Perfect prediction gives zero MAE loss and zero gradient everywhere
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

/// Shape mismatch is rejected with an Err
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

// Binary Cross-Entropy

/// BCE forward value equals the mean of -[y*ln(p) + (1-y)*ln(1-p)]
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
    // expected ~= 0.16425203
    assert_abs_diff_eq!(loss, 0.164252_f32, epsilon = 1e-5);
}

/// BCE gradient equals (-y/p + (1-y)/(1-p)) / n with p clipped
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

/// Perfect binary prediction gives near-zero BCE loss and finite gradient under clipping
#[test]
fn bce_perfect_prediction_near_zero_loss_and_finite_grad() {
    let bce = BinaryCrossEntropy::new();

    // Unambiguous correct labels at both 0 and 1
    let y_true: Tensor = Array::from_shape_vec(vec![2], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    // Exact 0.0 and 1.0 predictions get clipped to (eps, 1-eps)
    let y_pred: Tensor = Array::from_shape_vec(vec![2], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();

    let loss = bce.compute_loss(&y_true, &y_pred).unwrap();
    // After clipping at eps=1e-7 each element is -ln(1-eps) ~= 1e-7; mean ~= 1e-7
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

/// Predictions exactly 0.0 and 1.0 are clipped so BCE loss and gradients stay finite
#[test]
fn bce_extreme_predictions_clipped_to_finite() {
    let bce = BinaryCrossEntropy::new();

    // Worst case: every prediction hits the boundary
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

/// Shape mismatch is rejected with an Err
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

// Categorical Cross-Entropy

/// CCE forward value is the mean per-sample cross-entropy, divided by batch size (not element count)
#[test]
fn cce_forward_value_divided_by_batch() {
    let cce = CategoricalCrossEntropy::new(false);

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
    // Expected: (-ln(0.8) - ln(0.7) - ln(0.7)) / 3 ~= 0.31216
    assert_abs_diff_eq!(loss, 0.31216_f32, epsilon = 1e-4);
}

/// CCE gradient equals -y_true / y_pred / batch_size, zero where y_true is zero
#[test]
fn cce_gradient_value() {
    let cce = CategoricalCrossEntropy::new(false);

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

    // Flatten to check individual elements
    let flat: Vec<f32> = grad.iter().cloned().collect();
    assert_eq!(flat.len(), 9);

    // Row 0: only class 0 is non-zero
    assert_abs_diff_eq!(flat[0], -1.0_f32 / 0.8 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[1], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[2], 0.0_f32, epsilon = 1e-7);

    // Row 1: only class 1 is non-zero
    assert_abs_diff_eq!(flat[3], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[4], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[5], 0.0_f32, epsilon = 1e-7);

    // Row 2: only class 2 is non-zero
    assert_abs_diff_eq!(flat[6], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[7], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[8], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
}

/// Perfect one-hot prediction gives near-zero CCE loss (< 1e-4) and finite gradient under clipping
#[test]
fn cce_perfect_prediction_near_zero_loss_and_finite_grad() {
    let cce = CategoricalCrossEntropy::new(false);

    let y_true: Tensor = Array::from_shape_vec(vec![2, 2], vec![1.0_f32, 0.0, 0.0, 1.0])
        .unwrap()
        .into_dyn();
    // Exact 0.0 / 1.0 predictions are clipped internally
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

/// Predictions at exactly 0.0 and 1.0 do not produce NaN or Inf in CCE loss or gradient
#[test]
fn cce_extreme_predictions_clipped_to_finite() {
    let cce = CategoricalCrossEntropy::new(false);

    // One sample, 3 classes; predicted probability at 0.0 on a hot class
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

/// Shape mismatch is rejected with an Err
#[test]
fn cce_shape_mismatch_returns_err() {
    let cce = CategoricalCrossEntropy::new(false);

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

/// Empty y_true is rejected with an Err
#[test]
fn cce_empty_input_returns_err() {
    let cce = CategoricalCrossEntropy::new(false);

    // Shape [0, 3] is a valid shape but has zero elements
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

/// 1D inputs are rejected: `shape()[0]` would be the element count rather than the batch size,
/// silently rescaling the loss/gradient
#[test]
fn cce_rejects_1d_input() {
    let cce = CategoricalCrossEntropy::new(false);
    let y_true: Tensor = Array::from_shape_vec((3,), vec![1.0, 0.0, 0.0])
        .unwrap()
        .into_dyn();
    let y_pred: Tensor = Array::from_shape_vec((3,), vec![0.5, 0.3, 0.2])
        .unwrap()
        .into_dyn();
    assert!(matches!(
        cce.compute_loss(&y_true, &y_pred),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        cce.compute_grad(&y_true, &y_pred),
        Err(Error::InvalidInput(_))
    ));
}

/// from_logits CCE applies softmax internally: its loss matches probability-mode CCE on the same
/// softmax, and its gradient is the fused `(softmax(z) - y) / batch`
#[test]
fn cce_from_logits_loss_and_grad() {
    let logits: Tensor = Array::from_shape_vec(vec![1, 3], vec![1.0_f32, 2.0, 0.5])
        .unwrap()
        .into_dyn();
    let y_true: Tensor = Array::from_shape_vec(vec![1, 3], vec![0.0_f32, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    // Hand-computed softmax of [1, 2, 0.5]
    let exp = [1.0_f32.exp(), 2.0_f32.exp(), 0.5_f32.exp()];
    let sum: f32 = exp.iter().sum();
    let p = [exp[0] / sum, exp[1] / sum, exp[2] / sum];

    let cce_logits = CategoricalCrossEntropy::new(true);
    let loss = cce_logits.compute_loss(&y_true, &logits).unwrap();
    assert_abs_diff_eq!(loss, -p[1].ln(), epsilon = 1e-5);

    // Gradient w.r.t. logits is softmax - one_hot
    let grad = cce_logits.compute_grad(&y_true, &logits).unwrap();
    let expected: Tensor = Array::from_shape_vec(vec![1, 3], vec![p[0], p[1] - 1.0, p[2]])
        .unwrap()
        .into_dyn();
    assert_allclose(&grad, &expected, 1e-5_f32);

    // Consistency: from_logits loss == probability-mode loss on the same softmax
    let probs: Tensor = Array::from_shape_vec(vec![1, 3], p.to_vec())
        .unwrap()
        .into_dyn();
    let cce_probs = CategoricalCrossEntropy::new(false);
    assert_abs_diff_eq!(
        cce_probs.compute_loss(&y_true, &probs).unwrap(),
        loss,
        epsilon = 1e-5
    );
}

// Sparse Categorical Cross-Entropy

/// SCCE forward value with integer labels equals CCE on the equivalent one-hot encoding
#[test]
fn scce_forward_value_equals_cce_equivalent() {
    let scce = SparseCategoricalCrossEntropy::new(false);

    // Labels as [batch, 1] float tensor of class indices
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
    // Must agree with CCE on the same data to within floating-point round-off
    assert_abs_diff_eq!(loss, 0.31216_f32, epsilon = 1e-4);
}

/// SCCE gradient is -1/y_pred / batch at each sample's true class and zero elsewhere
#[test]
fn scce_gradient_value() {
    let scce = SparseCategoricalCrossEntropy::new(false);

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

    // Row 0
    assert_abs_diff_eq!(flat[0], -1.0_f32 / 0.8 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[1], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[2], 0.0_f32, epsilon = 1e-7);

    // Row 1
    assert_abs_diff_eq!(flat[3], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[4], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(flat[5], 0.0_f32, epsilon = 1e-7);

    // Row 2
    assert_abs_diff_eq!(flat[6], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[7], 0.0_f32, epsilon = 1e-7);
    assert_abs_diff_eq!(flat[8], -1.0_f32 / 0.7 / 3.0, epsilon = 1e-5);
}

/// SCCE gradient matches CCE gradient on the equivalent one-hot encoding
#[test]
fn scce_gradient_matches_cce_equivalent() {
    let scce = SparseCategoricalCrossEntropy::new(false);
    let cce = CategoricalCrossEntropy::new(false);

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

/// A prediction at exactly 0.0 on the true class is clipped so SCCE loss and grad stay finite
#[test]
fn scce_extreme_prediction_clipped_to_finite() {
    let scce = SparseCategoricalCrossEntropy::new(false);

    // Label 0, but predicted probability for class 0 is 0.0 (worst case)
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

/// SCCE rejects a negative label with InvalidInput
#[test]
fn scce_negative_label_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new(false);

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

/// SCCE rejects an out-of-range label (class index >= num_classes) with InvalidInput
#[test]
fn scce_out_of_range_label_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new(false);

    // 3 classes but label = 3 (valid range is 0..=2)
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

/// SCCE rejects y_true with wrong shape (1D or wrong second dimension, not [batch, 1])
#[test]
fn scce_wrong_label_shape_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new(false);

    // Labels are 1D - must be [batch, 1]
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

    // Labels are [batch, 2] - wrong second dimension
    let y_true_wrong: Tensor = Array::from_shape_vec(vec![2, 2], vec![0.0_f32, 0.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    assert!(
        scce.compute_loss(&y_true_wrong, &y_pred).is_err(),
        "[batch, 2] labels must return an Err"
    );
}

/// SCCE rejects a batch-size mismatch between labels and predictions
#[test]
fn scce_batch_mismatch_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new(false);

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

/// SCCE rejects 3-D predictions (not [batch, num_classes]) with InvalidInput
#[test]
fn scce_non_2d_predictions_returns_err() {
    let scce = SparseCategoricalCrossEntropy::new(false);

    let y_true: Tensor = Array::from_shape_vec(vec![2, 1], vec![0.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    // 3-D predictions - must be rejected
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

// Cross-loss consistency: SCCE loss == CCE loss (batch normalisation)

/// SCCE(integer labels) loss equals CCE(equivalent one-hot) loss, confirming a shared
/// divide-by-batch_size convention
#[test]
fn scce_loss_equals_cce_loss_on_equivalent_inputs() {
    let cce = CategoricalCrossEntropy::new(false);
    let scce = SparseCategoricalCrossEntropy::new(false);

    let y_pred: Tensor = Array::from_shape_vec(
        vec![4, 3],
        vec![
            0.7_f32, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.3, 0.5, 0.33, 0.34, 0.33,
        ],
    )
    .unwrap()
    .into_dyn();

    // One-hot for CCE
    let y_true_onehot: Tensor = Array::from_shape_vec(
        vec![4, 3],
        vec![
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ],
    )
    .unwrap()
    .into_dyn();

    // Integer labels for SCCE (same classes as one-hot above)
    let y_true_labels: Tensor = Array::from_shape_vec(vec![4, 1], vec![0.0_f32, 1.0, 2.0, 1.0])
        .unwrap()
        .into_dyn();

    let cce_loss = cce.compute_loss(&y_true_onehot, &y_pred).unwrap();
    let scce_loss = scce.compute_loss(&y_true_labels, &y_pred).unwrap();

    assert_abs_diff_eq!(scce_loss, cce_loss, epsilon = 1e-5);
}

/// from_logits SCCE: loss equals -log_softmax at the label, and the gradient w.r.t. the logits is
/// softmax with 1 subtracted at the true class
#[test]
fn scce_from_logits_loss_and_grad() {
    let logits: Tensor = Array::from_shape_vec(vec![1, 3], vec![1.0_f32, 2.0, 0.5])
        .unwrap()
        .into_dyn();
    let labels: Tensor = Array::from_shape_vec(vec![1, 1], vec![1.0_f32])
        .unwrap()
        .into_dyn();

    let exp = [1.0_f32.exp(), 2.0_f32.exp(), 0.5_f32.exp()];
    let sum: f32 = exp.iter().sum();
    let p = [exp[0] / sum, exp[1] / sum, exp[2] / sum];

    let scce = SparseCategoricalCrossEntropy::new(true);
    let loss = scce.compute_loss(&labels, &logits).unwrap();
    assert_abs_diff_eq!(loss, -p[1].ln(), epsilon = 1e-5);

    let grad = scce.compute_grad(&labels, &logits).unwrap();
    let expected: Tensor = Array::from_shape_vec(vec![1, 3], vec![p[0], p[1] - 1.0, p[2]])
        .unwrap()
        .into_dyn();
    assert_allclose(&grad, &expected, 1e-5_f32);
}
