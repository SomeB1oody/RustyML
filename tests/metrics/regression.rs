//! Integration tests for `src/metrics/regression.rs`.
//!
//! Every expected value is derived from the mathematical definition of the metric or a
//! closed-form hand calculation. No value is obtained by reading the implementation.

use approx::assert_abs_diff_eq;
use ndarray::array;
use rustyml::metrics::*;

// ---------------------------------------------------------------------------
// mean_squared_error
// ---------------------------------------------------------------------------

/// Canonical four-sample example from the task brief.
/// errors: (3-2.5)^2=0.25, (-0.5-0)^2=0.25, (2-2)^2=0, (7-8)^2=1
/// MSE = (0.25+0.25+0+1)/4 = 1.5/4 = 0.375
#[test]
fn test_mse_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    assert_abs_diff_eq!(mean_squared_error(&y_true, &y_pred), 0.375, epsilon = 1e-9);
}

/// Perfect predictions: every squared error is 0, so MSE = 0.
#[test]
fn test_mse_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(mean_squared_error(&y_true, &y_pred), 0.0, epsilon = 1e-15);
}

/// Single-element arrays.
/// MSE([5.0], [3.0]) = (5-3)^2/1 = 4.0
#[test]
fn test_mse_single_element() {
    let y_true = array![5.0];
    let y_pred = array![3.0];
    assert_abs_diff_eq!(mean_squared_error(&y_true, &y_pred), 4.0, epsilon = 1e-15);
}

/// MSE is symmetric: swapping y_true and y_pred does not change the result
/// because each squared difference (a-b)^2 = (b-a)^2.
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

/// MSE is always non-negative (mean of squares).
#[test]
fn test_mse_nonnegative() {
    let y_true = array![-5.0, -2.0, 0.0, 3.0, 7.0];
    let y_pred = array![1.0, -3.0, 0.5, 2.0, 9.0];
    assert!(mean_squared_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_mse_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = mean_squared_error(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_mse_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = mean_squared_error(&y_true, &y_pred);
}

// ---------------------------------------------------------------------------
// root_mean_squared_error
// ---------------------------------------------------------------------------

/// RMSE = sqrt(MSE). Using the same four-sample input:
/// MSE = 0.375, so RMSE = sqrt(0.375) ≈ 0.6123724356957945
#[test]
fn test_rmse_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    let expected = 0.375_f64.sqrt(); // ≈ 0.6123724356957945
    assert_abs_diff_eq!(
        root_mean_squared_error(&y_true, &y_pred),
        expected,
        epsilon = 1e-12
    );
}

/// Perfect predictions: RMSE = sqrt(0) = 0.
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

/// Uniform unit errors: errors=[1,1,1], MSE=1, RMSE=1.
#[test]
fn test_rmse_unit_errors() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 3.0, 4.0];
    // MSE = ((2-1)^2+(3-2)^2+(4-3)^2)/3 = 3/3 = 1, RMSE = sqrt(1) = 1.0
    assert_abs_diff_eq!(
        root_mean_squared_error(&y_true, &y_pred),
        1.0,
        epsilon = 1e-12
    );
}

/// RMSE >= MAE always (by Cauchy-Schwarz / Jensen's inequality for n > 0).
#[test]
fn test_rmse_ge_mae() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    assert!(root_mean_squared_error(&y_true, &y_pred) >= mean_absolute_error(&y_true, &y_pred));
}

/// RMSE is always non-negative.
#[test]
fn test_rmse_nonnegative() {
    let y_true = array![0.0, 1.0, 2.0];
    let y_pred = array![1.0, 0.0, 2.0];
    assert!(root_mean_squared_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_rmse_length_mismatch_panics() {
    let y_true = array![1.0, 2.0];
    let y_pred = array![1.0, 2.0, 3.0];
    let _ = root_mean_squared_error(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_rmse_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = root_mean_squared_error(&y_true, &y_pred);
}

// ---------------------------------------------------------------------------
// mean_absolute_error
// ---------------------------------------------------------------------------

/// Four-sample canonical example:
/// |3-2.5|=0.5, |-0.5-0|=0.5, |2-2|=0, |7-8|=1
/// MAE = (0.5+0.5+0+1)/4 = 2.0/4 = 0.5
#[test]
fn test_mae_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true, &y_pred), 0.5, epsilon = 1e-9);
}

/// Perfect predictions: MAE = 0.
#[test]
fn test_mae_perfect_predictions() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true, &y_pred), 0.0, epsilon = 1e-15);
}

/// Single element with known error.
/// MAE([5.0], [3.0]) = |5-3|/1 = 2.0
#[test]
fn test_mae_single_element() {
    let y_true = array![5.0];
    let y_pred = array![3.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true, &y_pred), 2.0, epsilon = 1e-15);
}

/// MAE is symmetric (absolute difference is symmetric).
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

/// MAE is always non-negative.
#[test]
fn test_mae_nonnegative() {
    let y_true = array![-3.0, 0.0, 2.0];
    let y_pred = array![0.0, -1.0, 5.0];
    assert!(mean_absolute_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_mae_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = mean_absolute_error(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_mae_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = mean_absolute_error(&y_true, &y_pred);
}

// ---------------------------------------------------------------------------
// r2_score
// ---------------------------------------------------------------------------

/// Perfect predictions give R2 = 1.0.
/// SSE = 0, R2 = 1 - 0/SST = 1.0
#[test]
fn test_r2_perfect_fit() {
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

/// Mean-predictor (predict mean for all) gives R2 = 0.
/// y_true=[1,2,3], mean=2; y_pred=[2,2,2]
/// SST = (1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1 = 2
/// SSE = (2-1)^2+(2-2)^2+(2-3)^2 = 1+0+1 = 2
/// R2 = 1 - 2/2 = 0.0
#[test]
fn test_r2_mean_predictor_zero() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 2.0, 2.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

/// Worse-than-mean predictor gives negative R2.
/// y_true=[1,2,3], y_pred=[3,2,1] (reversed)
/// SST = 2 (as above)
/// SSE = (3-1)^2+(2-2)^2+(1-3)^2 = 4+0+4 = 8
/// R2 = 1 - 8/2 = -3.0
#[test]
fn test_r2_worse_than_mean_negative() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![3.0, 2.0, 1.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), -3.0, epsilon = 1e-12);
}

/// Known fractional R2 value.
/// y_true=[1,3,5], y_pred=[2,3,4]
/// mean(y_true)=3, SST=(1-3)^2+(3-3)^2+(5-3)^2=4+0+4=8
/// SSE=(2-1)^2+(3-3)^2+(4-5)^2=1+0+1=2
/// R2 = 1 - 2/8 = 0.75
#[test]
fn test_r2_known_fractional() {
    let y_true = array![1.0, 3.0, 5.0];
    let y_pred = array![2.0, 3.0, 4.0];
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 0.75, epsilon = 1e-9);
}

/// Four-sample canonical example.
/// y_true=[3,-0.5,2,7], y_pred=[2.5,0,2,8]
/// mean(y_true)=2.875
/// SST=(3-2.875)^2+(-0.5-2.875)^2+(2-2.875)^2+(7-2.875)^2
///    =0.015625+11.390625+0.765625+17.015625=29.1875
/// SSE=(2.5-3)^2+(0-(-0.5))^2+(2-2)^2+(8-7)^2=0.25+0.25+0+1=1.5
/// R2 = 1 - 1.5/29.1875
#[test]
fn test_r2_four_samples() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];
    let expected = 1.0 - 1.5_f64 / 29.1875_f64;
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), expected, epsilon = 1e-9);
}

/// R2 is NOT symmetric: argument order matters because SST is computed from y_true only.
/// With y_true=[1,2,3], y_pred=[1,3,2]:
///   R2 != R2 with args swapped.
/// Concretely:
///   R2(y_true=[1,2,3], y_pred=[1,3,2]):
///     mean=2, SST=2, SSE=(1-1)^2+(3-2)^2+(2-3)^2=0+1+1=2, R2=0.0
///   R2(y_true=[1,3,2], y_pred=[1,2,3]):
///     mean=2, SST=(1-2)^2+(3-2)^2+(2-2)^2=1+1+0=2, SSE=(1-1)^2+(2-3)^2+(3-2)^2=0+1+1=2, R2=0.0
/// Use a clearly asymmetric case instead:
///   y_true=[1,2,3], y_pred=[3,2,1]:  R2 = -3.0 (computed above)
///   y_true=[3,2,1], y_pred=[1,2,3]:
///     mean(y_true)=2, SST=(3-2)^2+(2-2)^2+(1-2)^2=1+0+1=2
///     SSE=(1-3)^2+(2-2)^2+(3-1)^2=4+0+4=8
///     R2 = 1 - 8/2 = -3.0  (same here because data is symmetric around 2)
/// Use a genuinely asymmetric pair:
///   y_true=[1,2,4], y_pred=[2,3,4]:
///     mean=7/3, SST=(1-7/3)^2+(2-7/3)^2+(4-7/3)^2=(4/3)^2+(1/3)^2+(5/3)^2=16/9+1/9+25/9=42/9=14/3
///     SSE=(2-1)^2+(3-2)^2+(4-4)^2=1+1+0=2
///     R2 = 1 - 2/(14/3) = 1 - 6/14 = 1 - 3/7 = 4/7
///   y_true=[2,3,4], y_pred=[1,2,4]:
///     mean=3, SST=(2-3)^2+(3-3)^2+(4-3)^2=1+0+1=2
///     SSE=(1-2)^2+(2-3)^2+(4-4)^2=1+1+0=2
///     R2 = 1 - 2/2 = 0.0
/// So 4/7 != 0.0 — argument order matters.
#[test]
fn test_r2_not_symmetric() {
    let a = array![1.0, 2.0, 4.0];
    let b = array![2.0, 3.0, 4.0];
    // R2(a, b) = 4/7 ≈ 0.5714..., R2(b, a) = 0.0
    let r2_ab = r2_score(&a, &b);
    let r2_ba = r2_score(&b, &a);
    assert!(
        (r2_ab - r2_ba).abs() > 1e-9,
        "R2 should differ when arguments are swapped; got r2_ab={r2_ab}, r2_ba={r2_ba}"
    );
    assert_abs_diff_eq!(r2_ab, 4.0_f64 / 7.0_f64, epsilon = 1e-9);
    assert_abs_diff_eq!(r2_ba, 0.0, epsilon = 1e-9);
}

/// Constant y_true, perfect SSE (SSE ≈ 0): returns 1.0 per the convention.
#[test]
fn test_r2_constant_y_true_perfect_sse() {
    let y_true = array![3.0, 3.0, 3.0, 3.0];
    let y_pred = array![3.0, 3.0, 3.0, 3.0];
    // SST < SST_EPSILON, SSE = 0 < SST_EPSILON → return 1.0
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

/// Constant y_true, imperfect prediction: returns 0.0 per the convention.
#[test]
fn test_r2_constant_y_true_nonzero_sse() {
    let y_true = array![3.0, 3.0, 3.0];
    let y_pred = array![2.0, 4.0, 5.0];
    // SST ≈ 0, SSE > 0 → return 0.0
    assert_abs_diff_eq!(r2_score(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

/// R2 is <= 1.0 for any inputs with variance.
#[test]
fn test_r2_bounded_above_by_one() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 1.9, 3.2, 3.8, 5.1];
    assert!(r2_score(&y_true, &y_pred) <= 1.0);
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_r2_length_mismatch_panics() {
    let y_true = array![1.0, 2.0];
    let y_pred = array![1.0];
    let _ = r2_score(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_r2_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = r2_score(&y_true, &y_pred);
}

// ---------------------------------------------------------------------------
// explained_variance_score
// ---------------------------------------------------------------------------

/// Perfect predictions: residuals = 0, EVS = 1.0.
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

/// Constant-bias predictions score 1.0 (EVS ignores bias, R2 does not).
/// y_true=[1,2,3], y_pred=[2,3,4] (constant +1 bias)
/// residuals=[-1,-1,-1], Var(residuals)=0
/// EVS = 1 - 0/Var([1,2,3]) = 1.0
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

/// EVS > R2 when a constant bias is present.
/// For y_true=[1,2,3], y_pred=[2,3,4]:
///   EVS = 1.0 (shown above)
///   R2: SST=2, SSE=(2-1)^2+(3-2)^2+(4-3)^2=3, R2=1-3/2=-0.5
///   So EVS=1.0 > R2=-0.5
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

/// Known fractional EVS.
/// y_true=[1,2,3,4], y_pred=[1,3,2,4]:
/// residuals = [0,-1,1,0], mean=0
/// Var(residuals) = (0+1+1+0)/4 = 0.5
/// mean(y_true)=2.5, Var(y_true)=((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4
///                              =(2.25+0.25+0.25+2.25)/4 = 5/4 = 1.25
/// EVS = 1 - 0.5/1.25 = 1 - 0.4 = 0.6
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

/// Constant y_true with zero residual variance: returns 1.0.
#[test]
fn test_evs_constant_y_true_zero_residual_variance() {
    let y_true = array![5.0, 5.0, 5.0];
    let y_pred = array![5.0, 5.0, 5.0];
    // Var(y_true) < SST_EPSILON, Var(residuals)=0 → returns 1.0
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        1.0,
        epsilon = 1e-12
    );
}

/// Constant y_true with non-zero residual variance: returns 0.0.
#[test]
fn test_evs_constant_y_true_nonzero_residual_variance() {
    let y_true = array![5.0, 5.0, 5.0];
    let y_pred = array![3.0, 5.0, 7.0];
    // Var(y_true) ≈ 0, residuals=[2,0,-2], Var(residuals)>0 → returns 0.0
    assert_abs_diff_eq!(
        explained_variance_score(&y_true, &y_pred),
        0.0,
        epsilon = 1e-12
    );
}

/// EVS is <= 1.0 for any inputs.
#[test]
fn test_evs_bounded_above_by_one() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 2.2, 2.8, 4.0, 4.9];
    assert!(explained_variance_score(&y_true, &y_pred) <= 1.0);
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_evs_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = explained_variance_score(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_evs_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = explained_variance_score(&y_true, &y_pred);
}

// ---------------------------------------------------------------------------
// median_absolute_error
// ---------------------------------------------------------------------------

/// Odd number of samples: median is the middle value of sorted absolute errors.
/// y_true=[1,2,3,4,5], y_pred=[1,1,1,1,1]
/// abs errors = [0,1,2,3,4], sorted same, n=5, median = errors[2] = 2.0
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

/// Even number of samples: median is average of two middle values.
/// y_true=[1,2,3,4], y_pred=[3,1,4,1]
/// abs errors = [|1-3|,|2-1|,|3-4|,|4-1|] = [2,1,1,3]
/// sorted = [1,1,2,3], n=4, median = (errors[1]+errors[2])/2 = (1+2)/2 = 1.5
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

/// Perfect predictions: all absolute errors are 0, median is 0.
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

/// Single element: median equals the only absolute error.
/// y_true=[4.0], y_pred=[1.0]: abs error = 3.0, n=1 (odd), median = errors[0] = 3.0
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

/// Robustness to outlier: single large error does not dominate the median.
/// y_true=[1,2,3,4], y_pred=[1,2,3,10]
/// abs errors = [0,0,0,6], sorted same, n=4 (even)
/// median = (errors[1]+errors[2])/2 = (0+0)/2 = 0.0
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

/// MedAE is symmetric (absolute error is symmetric).
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

/// MedAE is always non-negative.
#[test]
fn test_medae_nonnegative() {
    let y_true = array![-2.0, 0.0, 3.0];
    let y_pred = array![1.0, -1.0, 2.0];
    assert!(median_absolute_error(&y_true, &y_pred) >= 0.0);
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_medae_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = median_absolute_error(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_medae_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = median_absolute_error(&y_true, &y_pred);
}

// ---------------------------------------------------------------------------
// mean_absolute_percentage_error
// ---------------------------------------------------------------------------

/// Standard three-sample case from docstring.
/// y_true=[2,4,5], y_pred=[1,4,5]
/// per-sample: |2-1|/2=0.5, |4-4|/4=0, |5-5|/5=0
/// MAPE = (0.5+0+0)/3 = 1/6 ≈ 0.16666...
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

/// Perfect predictions: every relative error is 0, MAPE = 0.
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

/// Known two-sample MAPE.
/// y_true=[10, 20], y_pred=[11, 18]
/// per-sample: |10-11|/10=0.1, |20-18|/20=0.1
/// MAPE = (0.1+0.1)/2 = 0.1
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

/// Single-element MAPE.
/// y_true=[4.0], y_pred=[3.0]: |4-3|/4 = 0.25
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

/// Negative y_true values: denominator uses abs(y_true).
/// y_true=[-4], y_pred=[-2]: |(-4)-(-2)|/|-4| = 2/4 = 0.5
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

/// MAPE is always non-negative.
#[test]
fn test_mape_nonnegative() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 1.0, 4.0];
    assert!(mean_absolute_percentage_error(&y_true, &y_pred) >= 0.0);
}

/// When y_true = 0 the denominator is floored at EPS = 1e-15.
/// y_true=[0,2], y_pred=[1,2]
/// sample 0: |0-1|/max(|0|,1e-15) = 1/1e-15 = 1e15
/// sample 1: |2-2|/2 = 0
/// MAPE = (1e15 + 0) / 2 = 5e14
/// The result should be very large (> 1e13).
#[test]
fn test_mape_zero_y_true_eps_floor() {
    let y_true = array![0.0, 2.0];
    let y_pred = array![1.0, 2.0];
    let mape = mean_absolute_percentage_error(&y_true, &y_pred);
    // The floor is 1e-15, so the first term is ~1e15; total MAPE ~5e14
    assert!(
        mape > 1e13,
        "MAPE with y_true=0 should be very large due to 1e-15 floor; got {mape}"
    );
}

/// Length mismatch must panic.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_mape_length_mismatch_panics() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];
    let _ = mean_absolute_percentage_error(&y_true, &y_pred);
}

/// Empty arrays must panic.
#[test]
#[should_panic(expected = "input is empty")]
fn test_mape_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = mean_absolute_percentage_error(&y_true, &y_pred);
}
