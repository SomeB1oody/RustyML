#![cfg(feature = "neural_network")]

//! Numerical tests for the loss functions: every `compute_loss` / `compute_grad` is checked
//! against hand-computed expected values, and the `Result` error paths (shape mismatch, bad
//! sparse labels) are exercised. This replaces the previous tests, which only trained a model
//! and printed the prediction without asserting anything.

use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::loss_function::binary_cross_entropy::BinaryCrossEntropy;
use rustyml::neural_network::loss_function::categorical_cross_entropy::CategoricalCrossEntropy;
use rustyml::neural_network::loss_function::mean_absolute_error::MeanAbsoluteError;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::loss_function::sparse_categorical_cross_entropy::SparseCategoricalCrossEntropy;
use rustyml::neural_network::neural_network_trait::LossFunction;

const EPS: f32 = 1e-5;

fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
    ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap()
}

fn assert_grad_eq(grad: &Tensor, expected: &[f32]) {
    assert_eq!(grad.len(), expected.len(), "gradient length mismatch");
    for (g, e) in grad.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = EPS);
    }
}

#[test]
fn mse_matches_hand_computed_values() {
    let y_true = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let y_pred = tensor(&[2, 2], &[1.5, 2.0, 2.0, 4.0]);
    // diff = [0.5, 0, -1, 0]; mean(diff^2) = (0.25 + 0 + 1 + 0) / 4 = 0.3125
    let mse = MeanSquaredError::new();
    assert_abs_diff_eq!(
        mse.compute_loss(&y_true, &y_pred).unwrap(),
        0.3125,
        epsilon = EPS
    );
    // grad = 2 * diff / 4
    assert_grad_eq(
        &mse.compute_grad(&y_true, &y_pred).unwrap(),
        &[0.25, 0.0, -0.5, 0.0],
    );
}

#[test]
fn mae_matches_hand_computed_values() {
    let y_true = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let y_pred = tensor(&[2, 2], &[1.5, 2.0, 2.0, 4.0]);
    // mean(|diff|) = (0.5 + 0 + 1 + 0) / 4 = 0.375
    let mae = MeanAbsoluteError::new();
    assert_abs_diff_eq!(
        mae.compute_loss(&y_true, &y_pred).unwrap(),
        0.375,
        epsilon = EPS
    );
    // grad = sign(diff) / 4 (0 where diff == 0)
    assert_grad_eq(
        &mae.compute_grad(&y_true, &y_pred).unwrap(),
        &[0.25, 0.0, -0.25, 0.0],
    );
}

#[test]
fn bce_matches_hand_computed_values() {
    let y_true = tensor(&[1, 2], &[1.0, 0.0]);
    let y_pred = tensor(&[1, 2], &[0.8, 0.3]);
    // loss = -(ln 0.8 + ln 0.7) / 2
    let expected_loss = -((0.8f32).ln() + (0.7f32).ln()) / 2.0;
    let bce = BinaryCrossEntropy::new();
    assert_abs_diff_eq!(
        bce.compute_loss(&y_true, &y_pred).unwrap(),
        expected_loss,
        epsilon = EPS
    );
    // grad = [-1/0.8, 1/0.7] / 2
    assert_grad_eq(
        &bce.compute_grad(&y_true, &y_pred).unwrap(),
        &[-1.0 / 0.8 / 2.0, 1.0 / 0.7 / 2.0],
    );
}

#[test]
fn cce_matches_hand_computed_values() {
    let y_true = tensor(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let y_pred = tensor(&[2, 3], &[0.7, 0.2, 0.1, 0.1, 0.6, 0.3]);
    // loss = -(ln 0.7 + ln 0.6) / 2  (averaged over the batch, not all elements)
    let expected_loss = -((0.7f32).ln() + (0.6f32).ln()) / 2.0;
    let cce = CategoricalCrossEntropy::new();
    assert_abs_diff_eq!(
        cce.compute_loss(&y_true, &y_pred).unwrap(),
        expected_loss,
        epsilon = EPS
    );
    // grad = -y_true / y_pred / batch (nonzero only at the true classes)
    assert_grad_eq(
        &cce.compute_grad(&y_true, &y_pred).unwrap(),
        &[-1.0 / 0.7 / 2.0, 0.0, 0.0, 0.0, -1.0 / 0.6 / 2.0, 0.0],
    );
}

#[test]
fn scce_matches_hand_computed_values() {
    let y_true = tensor(&[2, 1], &[0.0, 2.0]);
    let y_pred = tensor(&[2, 3], &[0.7, 0.2, 0.1, 0.1, 0.3, 0.6]);
    // loss = -(ln 0.7 + ln 0.6) / 2  (true classes are 0 and 2)
    let expected_loss = -((0.7f32).ln() + (0.6f32).ln()) / 2.0;
    let scce = SparseCategoricalCrossEntropy::new();
    assert_abs_diff_eq!(
        scce.compute_loss(&y_true, &y_pred).unwrap(),
        expected_loss,
        epsilon = EPS
    );
    // grad has a single nonzero entry per sample, at the true class.
    assert_grad_eq(
        &scce.compute_grad(&y_true, &y_pred).unwrap(),
        &[-1.0 / 0.7 / 2.0, 0.0, 0.0, 0.0, 0.0, -1.0 / 0.6 / 2.0],
    );
}

#[test]
fn elementwise_losses_reject_shape_mismatch() {
    let y_true = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let y_pred = tensor(&[2, 3], &[0.0; 6]);
    assert!(MeanSquaredError::new().compute_loss(&y_true, &y_pred).is_err());
    assert!(MeanAbsoluteError::new().compute_grad(&y_true, &y_pred).is_err());
    assert!(BinaryCrossEntropy::new().compute_loss(&y_true, &y_pred).is_err());
    assert!(
        CategoricalCrossEntropy::new()
            .compute_grad(&y_true, &y_pred)
            .is_err()
    );
}

#[test]
fn scce_rejects_out_of_range_and_malformed_labels() {
    let y_pred = tensor(&[2, 3], &[0.7, 0.2, 0.1, 0.1, 0.3, 0.6]);
    let scce = SparseCategoricalCrossEntropy::new();

    // Label 3 is out of range for 3 classes (valid indices are 0..=2).
    let bad_label = tensor(&[2, 1], &[0.0, 3.0]);
    assert!(scce.compute_loss(&bad_label, &y_pred).is_err());

    // Labels must be shape [batch, 1]; a [batch, 2] target is malformed.
    let wrong_shape = tensor(&[2, 2], &[0.0, 0.0, 1.0, 0.0]);
    assert!(scce.compute_grad(&wrong_shape, &y_pred).is_err());

    // Negative label is invalid.
    let negative = tensor(&[2, 1], &[0.0, -1.0]);
    assert!(scce.compute_loss(&negative, &y_pred).is_err());
}
