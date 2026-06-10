//! Integration tests for `src/metrics/regression.rs`: error metrics (MSE, RMSE,
//! MAE, MedAE, MAPE) and variance-explained scores (R2, EVS)

use approx::assert_abs_diff_eq;
use ndarray::array;
use rustyml::metrics::*;

// mean_squared_error

/// MSE of a four-sample case equals 0.375
#[test]
fn test_mse_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    assert_abs_diff_eq!(mean_squared_error(&y_true, &y_pred), 0.375, epsilon = 1e-9);
}

/// Perfect predictions give MSE = 0
#[test]
fn test_mse_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(mean_squared_error(&y_true, &y_pred), 0.0, epsilon = 1e-15);
}

/// MSE of a single-element pair equals the squared difference
#[test]
fn test_mse_single_element() {
    let y_true = array![5.0];
    let y_pred = array![3.0];
    assert_abs_diff_eq!(mean_squared_error(&y_true, &y_pred), 4.0, epsilon = 1e-15);
}

/// MSE is symmetric under swapping y_true and y_pred
#[test]
fn test_mse_symmetry() {
    let a = array![1.0, 3.0, 5.0];
    let b = array![2.0, 2.0, 6.0];
    assert_abs_diff_eq!(
        mean_squared_error(&a, &b),
        mean_squared_error(&b, &a),
        epsilon = 1e-15
    );
}

/// MSE is always non-negative
#[test]
fn test_mse_nonnegative() {
    let y_true = array![-5.0, -2.0, 0.0, 3.0, 7.0];
    let y_pred = array![1.0, -3.0, 0.5, 2.0, 9.0];
    assert!(mean_squared_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_mse_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = mean_squared_error(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_mse_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = mean_squared_error(&y_true, &y_pred);
}

// root_mean_squared_error

/// RMSE of the four-sample case equals sqrt(MSE) = sqrt(0.375)
#[test]
fn test_rmse_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    let expected = 0.375_f64.sqrt();
    assert_abs_diff_eq!(
        root_mean_squared_error(&y_true, &y_pred),
        expected,
        epsilon = 1e-12
    );
}

/// Perfect predictions give RMSE = 0
#[test]
fn test_rmse_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 3.0];
    assert_abs_diff_eq!(
        root_mean_squared_error(&y_true, &y_pred),
        0.0,
        epsilon = 1e-15
    );
}

/// Uniform unit errors give RMSE = 1
#[test]
fn test_rmse_unit_errors() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 3.0, 4.0];
    assert_abs_diff_eq!(
        root_mean_squared_error(&y_true, &y_pred),
        1.0,
        epsilon = 1e-12
    );
}

/// RMSE >= MAE always holds
#[test]
fn test_rmse_ge_mae() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    assert!(root_mean_squared_error(&y_true, &y_pred) >= mean_absolute_error(&y_true, &y_pred));
}

/// RMSE is always non-negative
#[test]
fn test_rmse_nonnegative() {
    let y_true = array![0.0, 1.0, 2.0];
    let y_pred = array![1.0, 0.0, 2.0];
    assert!(root_mean_squared_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_rmse_length_mismatch_panics() {
    let y_true = array![1.0, 2.0];
    let y_pred = array![1.0, 2.0, 3.0];
    let _ = root_mean_squared_error(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_rmse_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = root_mean_squared_error(&y_true, &y_pred);
}

// mean_absolute_error

/// MAE of the four-sample case equals 0.5
#[test]
fn test_mae_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true, &y_pred), 0.5, epsilon = 1e-9);
}

/// Perfect predictions give MAE = 0
#[test]
fn test_mae_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true, &y_pred), 0.0, epsilon = 1e-15);
}

/// MAE of a single-element pair equals the absolute difference
#[test]
fn test_mae_single_element() {
    let y_true = array![5.0];
    let y_pred = array![3.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true, &y_pred), 2.0, epsilon = 1e-15);
}

/// MAE is symmetric under swapping arguments
#[test]
fn test_mae_symmetry() {
    let a = array![1.0, 4.0, 6.0];
    let b = array![2.0, 3.0, 8.0];
    assert_abs_diff_eq!(
        mean_absolute_error(&a, &b),
        mean_absolute_error(&b, &a),
        epsilon = 1e-15
    );
}

/// MAE is always non-negative
#[test]
fn test_mae_nonnegative() {
    let y_true = array![-3.0, 0.0, 2.0];
    let y_pred = array![0.0, -1.0, 5.0];
    assert!(mean_absolute_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_mae_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = mean_absolute_error(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_mae_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = mean_absolute_error(&y_true, &y_pred);
}

// r2_score

/// Perfect predictions give R2 = 1
#[test]
fn test_r2_perfect_fit() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

/// Predicting the mean for every sample gives R2 = 0
#[test]
fn test_r2_mean_predictor_zero() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 2.0, 2.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

/// A worse-than-mean predictor gives negative R2 (here -3.0)
#[test]
fn test_r2_worse_than_mean_negative() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![3.0, 2.0, 1.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), -3.0, epsilon = 1e-12);
}

/// R2 matches a known fractional value (0.75)
#[test]
fn test_r2_known_fractional() {
    let y_true = array![1.0, 3.0, 5.0];
    let y_pred = array![2.0, 3.0, 4.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 0.75, epsilon = 1e-9);
}

/// R2 of the four-sample case equals 1 - 1.5/29.1875
#[test]
fn test_r2_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    let expected = 1.0 - 1.5_f64 / 29.1875_f64;
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), expected, epsilon = 1e-9);
}

/// R2 is not symmetric since SST depends only on y_true: R2(a, b) = 4/7 but R2(b, a) = 0
#[test]
fn test_r2_not_symmetric() {
    let a = array![1.0, 2.0, 4.0];
    let b = array![2.0, 3.0, 4.0];
    let r2_ab = r2_score(&a, &b);
    let r2_ba = r2_score(&b, &a);
    assert!(
        (r2_ab - r2_ba).abs() > 1e-9,
        "R2 should differ when arguments are swapped; got r2_ab={r2_ab}, r2_ba={r2_ba}"
    );
    assert_abs_diff_eq!(r2_ab, 4.0_f64 / 7.0_f64, epsilon = 1e-9);
    assert_abs_diff_eq!(r2_ba, 0.0, epsilon = 1e-9);
}

/// Constant y_true with zero SSE returns 1.0 by convention
#[test]
fn test_r2_constant_y_true_perfect_sse() {
    let y_true = array![3.0, 3.0, 3.0, 3.0];
    let y_pred = array![3.0, 3.0, 3.0, 3.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

/// Constant y_true with non-zero SSE returns 0.0 by convention
#[test]
fn test_r2_constant_y_true_nonzero_sse() {
    let y_true = array![3.0, 3.0, 3.0];
    let y_pred = array![2.0, 4.0, 5.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

/// R2 is bounded above by 1.0 for inputs with variance
#[test]
fn test_r2_bounded_above_by_one() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 1.9, 3.2, 3.8, 5.1];
    assert!(r2_score(&y_true, &y_pred) <= 1.0);
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_r2_length_mismatch_panics() {
    let y_true = array![1.0, 2.0];
    let y_pred = array![1.0];
    let _ = r2_score(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_r2_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = r2_score(&y_true, &y_pred);
}

// explained_variance_score

/// Perfect predictions give EVS = 1
#[test]
fn test_evs_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 3.0];
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        1.0,
        epsilon = 1e-12
    );
}

/// Constant-bias predictions score 1.0 since EVS ignores bias
#[test]
fn test_evs_constant_bias_scores_one() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 3.0, 4.0];
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        1.0,
        epsilon = 1e-12
    );
}

/// EVS exceeds R2 when a constant bias is present
#[test]
fn test_evs_greater_than_r2_with_bias() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 3.0, 4.0];
    let evs = explained_variance_score(&y_true, &y_pred);
    let r2 = r2_score(&y_true, &y_pred);
    assert!(
        evs > r2,
        "EVS ({evs}) should exceed R2 ({r2}) with a constant prediction bias"
    );
}

/// EVS matches a known fractional value (0.6)
#[test]
fn test_evs_known_fractional() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 3.0, 2.0, 4.0];
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        0.6,
        epsilon = 1e-9
    );
}

/// Constant y_true with zero residual variance returns 1.0
#[test]
fn test_evs_constant_y_true_zero_residual_variance() {
    let y_true = array![5.0, 5.0, 5.0];
    let y_pred = array![5.0, 5.0, 5.0];
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        1.0,
        epsilon = 1e-12
    );
}

/// Constant y_true with non-zero residual variance returns 0.0
#[test]
fn test_evs_constant_y_true_nonzero_residual_variance() {
    let y_true = array![5.0, 5.0, 5.0];
    let y_pred = array![3.0, 5.0, 7.0];
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        0.0,
        epsilon = 1e-12
    );
}

/// EVS is bounded above by 1.0
#[test]
fn test_evs_bounded_above_by_one() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 2.2, 2.8, 4.0, 4.9];
    assert!(explained_variance_score(&y_true, &y_pred) <= 1.0);
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_evs_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = explained_variance_score(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_evs_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = explained_variance_score(&y_true, &y_pred);
}

// median_absolute_error

/// Odd sample count: median is the middle sorted absolute error (2.0)
#[test]
fn test_medae_odd_samples() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.0, 1.0, 1.0, 1.0, 1.0];
    assert_abs_diff_eq!(
        median_absolute_error(&y_true, &y_pred),
        2.0,
        epsilon = 1e-12
    );
}

/// Even sample count: median averages the two middle sorted errors (1.5)
#[test]
fn test_medae_even_samples() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![3.0, 1.0, 4.0, 1.0];
    assert_abs_diff_eq!(
        median_absolute_error(&y_true, &y_pred),
        1.5,
        epsilon = 1e-12
    );
}

/// Perfect predictions give MedAE = 0
#[test]
fn test_medae_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 3.0];
    assert_abs_diff_eq!(
        median_absolute_error(&y_true, &y_pred),
        0.0,
        epsilon = 1e-12
    );
}

/// Single element: median equals the only absolute error
#[test]
fn test_medae_single_element() {
    let y_true = array![4.0];
    let y_pred = array![1.0];
    assert_abs_diff_eq!(
        median_absolute_error(&y_true, &y_pred),
        3.0,
        epsilon = 1e-12
    );
}

/// A single large error does not dominate the median (stays 0.0)
#[test]
fn test_medae_robust_to_outlier() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 10.0];
    assert_abs_diff_eq!(
        median_absolute_error(&y_true, &y_pred),
        0.0,
        epsilon = 1e-12
    );
}

/// MedAE is symmetric under swapping arguments
#[test]
fn test_medae_symmetry() {
    let a = array![1.0, 3.0, 5.0, 7.0];
    let b = array![2.0, 2.0, 6.0, 9.0];
    assert_abs_diff_eq!(
        median_absolute_error(&a, &b),
        median_absolute_error(&b, &a),
        epsilon = 1e-15
    );
}

/// MedAE is always non-negative
#[test]
fn test_medae_nonnegative() {
    let y_true = array![-2.0, 0.0, 3.0];
    let y_pred = array![1.0, -1.0, 2.0];
    assert!(median_absolute_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_medae_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = median_absolute_error(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_medae_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = median_absolute_error(&y_true, &y_pred);
}

// mean_absolute_percentage_error

/// MAPE of the standard three-sample case equals 1/6
#[test]
fn test_mape_three_samples() {
    let y_true = array![2.0, 4.0, 5.0];
    let y_pred = array![1.0, 4.0, 5.0];
    assert_abs_diff_eq!(
        mean_absolute_percentage_error(&y_true, &y_pred),
        1.0_f64 / 6.0_f64,
        epsilon = 1e-9
    );
}

/// Perfect predictions give MAPE = 0
#[test]
fn test_mape_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(
        mean_absolute_percentage_error(&y_true, &y_pred),
        0.0,
        epsilon = 1e-15
    );
}

/// MAPE matches a known two-sample value (0.1)
#[test]
fn test_mape_two_samples_known() {
    let y_true = array![10.0, 20.0];
    let y_pred = array![11.0, 18.0];
    assert_abs_diff_eq!(
        mean_absolute_percentage_error(&y_true, &y_pred),
        0.1,
        epsilon = 1e-12
    );
}

/// MAPE of a single-element pair equals the relative error (0.25)
#[test]
fn test_mape_single_element() {
    let y_true = array![4.0];
    let y_pred = array![3.0];
    assert_abs_diff_eq!(
        mean_absolute_percentage_error(&y_true, &y_pred),
        0.25,
        epsilon = 1e-12
    );
}

/// Negative y_true: the denominator uses abs(y_true)
#[test]
fn test_mape_negative_y_true() {
    let y_true = array![-4.0];
    let y_pred = array![-2.0];
    assert_abs_diff_eq!(
        mean_absolute_percentage_error(&y_true, &y_pred),
        0.5,
        epsilon = 1e-12
    );
}

/// MAPE is always non-negative
#[test]
fn test_mape_nonnegative() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 1.0, 4.0];
    assert!(mean_absolute_percentage_error(&y_true, &y_pred) >= 0.0);
}

/// When y_true = 0 the denominator is floored at 1e-15, so MAPE blows up (> 1e13)
#[test]
fn test_mape_zero_y_true_eps_floor() {
    let y_true = array![0.0, 2.0];
    let y_pred = array![1.0, 2.0];
    let mape = mean_absolute_percentage_error(&y_true, &y_pred);
    assert!(
        mape > 1e13,
        "MAPE with y_true=0 should be very large due to 1e-15 floor; got {mape}"
    );
}

/// Length mismatch panics
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_mape_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = mean_absolute_percentage_error(&y_true, &y_pred);
}

/// Empty arrays panic
#[test]
#[should_panic(expected = "input is empty")]
fn test_mape_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = mean_absolute_percentage_error(&y_true, &y_pred);
}
