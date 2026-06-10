//! Integration tests for [`rustyml::machine_learning::linear_regression::LinearRegression`].
//!
//! Every expected value is derived from the problem design or a closed-form analytic result —
//! never recorded from model output.  See the per-test derivation comments.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::linear_regression::{LinearRegression, RegularizationType};

use crate::common::assert_allclose;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor validation
// ─────────────────────────────────────────────────────────────────────────────

/// learning_rate = 0.0 is not positive → InvalidParameter
#[test]
fn constructor_zero_learning_rate_is_invalid() {
    let result = LinearRegression::new(true, 0.0, 100, 1e-6, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// learning_rate < 0 → InvalidParameter
#[test]
fn constructor_negative_learning_rate_is_invalid() {
    let result = LinearRegression::new(true, -0.01, 100, 1e-6, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// learning_rate = NaN → InvalidParameter
#[test]
fn constructor_nan_learning_rate_is_invalid() {
    let result = LinearRegression::new(true, f64::NAN, 100, 1e-6, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// learning_rate = +inf → InvalidParameter
#[test]
fn constructor_inf_learning_rate_is_invalid() {
    let result = LinearRegression::new(true, f64::INFINITY, 100, 1e-6, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// max_iterations = 0 → InvalidParameter
#[test]
fn constructor_zero_max_iter_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 0, 1e-6, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tolerance = 0.0 is not positive → InvalidParameter
#[test]
fn constructor_zero_tolerance_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 100, 0.0, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tolerance < 0 → InvalidParameter
#[test]
fn constructor_negative_tolerance_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 100, -1e-6, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tolerance = NaN → InvalidParameter
#[test]
fn constructor_nan_tolerance_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 100, f64::NAN, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tolerance = +inf → InvalidParameter
#[test]
fn constructor_inf_tolerance_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 100, f64::INFINITY, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Negative L2 alpha → InvalidParameter
#[test]
fn constructor_negative_l2_alpha_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 100, 1e-6, Some(RegularizationType::L2(-0.1)));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Negative L1 alpha → InvalidParameter
#[test]
fn constructor_negative_l1_alpha_is_invalid() {
    let result = LinearRegression::new(true, 0.01, 100, 1e-6, Some(RegularizationType::L1(-0.5)));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// NaN L2 alpha → InvalidParameter
#[test]
fn constructor_nan_l2_alpha_is_invalid() {
    let result = LinearRegression::new(
        true,
        0.01,
        100,
        1e-6,
        Some(RegularizationType::L2(f64::NAN)),
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Valid constructor with all legal parameters → Ok
#[test]
fn constructor_valid_parameters_succeeds() {
    let result = LinearRegression::new(true, 0.01, 1000, 1e-6, None);
    assert!(result.is_ok(), "expected Ok, got {:?}", result);
}

/// Getters on a freshly constructed model return the supplied values.
#[test]
fn constructor_getters_round_trip() {
    let model = LinearRegression::new(false, 0.05, 500, 1e-4, None).unwrap();
    assert!(!model.get_fit_intercept());
    assert_abs_diff_eq!(model.get_learning_rate(), 0.05, epsilon = 1e-15);
    assert_eq!(model.get_max_iterations(), 500);
    assert_abs_diff_eq!(model.get_tolerance(), 1e-4, epsilon = 1e-20);
    assert!(model.get_coefficients().is_none());
    assert!(model.get_intercept().is_none());
    assert!(model.get_actual_iterations().is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// NotFitted errors before fit
// ─────────────────────────────────────────────────────────────────────────────

/// predict() on an unfitted model → NotFitted
#[test]
fn predict_before_fit_returns_not_fitted() {
    let model = LinearRegression::new(true, 0.01, 100, 1e-6, None).unwrap();
    let x = array![[1.0, 2.0]];
    let result = model.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// fit() input-validation errors
// ─────────────────────────────────────────────────────────────────────────────

/// fit() with empty X → EmptyInput
#[test]
fn fit_empty_x_returns_empty_input() {
    let mut model = LinearRegression::new(true, 0.01, 100, 1e-6, None).unwrap();
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<f64> = Array1::zeros(0);
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput, got {:?}",
        result
    );
}

/// fit() with NaN in X → NonFinite
#[test]
fn fit_nan_in_x_returns_non_finite() {
    let mut model = LinearRegression::new(true, 0.01, 100, 1e-6, None).unwrap();
    let x = array![[1.0, f64::NAN], [2.0, 3.0]];
    let y = array![1.0, 2.0];
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// fit() with Inf in X → NonFinite
#[test]
fn fit_inf_in_x_returns_non_finite() {
    let mut model = LinearRegression::new(true, 0.01, 100, 1e-6, None).unwrap();
    let x = array![[1.0, f64::INFINITY], [2.0, 3.0]];
    let y = array![1.0, 2.0];
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// fit() with mismatched y length → DimensionMismatch
#[test]
fn fit_y_length_mismatch_returns_dimension_mismatch() {
    let mut model = LinearRegression::new(true, 0.01, 100, 1e-6, None).unwrap();
    // x has 3 rows but y has 2 elements
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![1.0, 2.0];
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// predict() input-validation errors (after fit)
// ─────────────────────────────────────────────────────────────────────────────

/// predict() with empty matrix → EmptyInput
#[test]
fn predict_empty_matrix_returns_empty_input() {
    let mut model = LinearRegression::new(true, 0.01, 5000, 1e-8, None).unwrap();
    // Train on y=2x+1
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x_train, &y_train).unwrap();

    let x_empty: Array2<f64> = Array2::zeros((0, 1));
    let result = model.predict(&x_empty);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput, got {:?}",
        result
    );
}

/// predict() with wrong number of columns → DimensionMismatch
#[test]
fn predict_wrong_feature_count_returns_dimension_mismatch() {
    let mut model = LinearRegression::new(true, 0.01, 5000, 1e-8, None).unwrap();
    // Trained on 1 feature
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x_train, &y_train).unwrap();

    // Predict with 2 features (wrong)
    let x_wrong = array![[1.0, 2.0]];
    let result = model.predict(&x_wrong);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {:?}",
        result
    );
}

/// predict() with NaN in X → NonFinite
#[test]
fn predict_nan_in_x_returns_non_finite() {
    let mut model = LinearRegression::new(true, 0.01, 5000, 1e-8, None).unwrap();
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x_train, &y_train).unwrap();

    let x_nan = array![[f64::NAN]];
    let result = model.predict(&x_nan);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// predict() with +Inf in X → NonFinite
#[test]
fn predict_inf_in_x_returns_non_finite() {
    let mut model = LinearRegression::new(true, 0.01, 5000, 1e-8, None).unwrap();
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x_train, &y_train).unwrap();

    let x_inf = array![[f64::INFINITY]];
    let result = model.predict(&x_inf);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness: univariate y = 2x + 1
// ─────────────────────────────────────────────────────────────────────────────
//
// Closed-form OLS (or gradient-descent converged to OLS):
//   x = [1,2,3,4,5], y = [3,5,7,9,11]
//   mean(x)=3, mean(y)=7
//   sum((xi-3)(yi-7)) = (-2)(-4)+(-1)(-2)+0+1*2+2*4 = 8+2+0+2+8 = 20
//   sum((xi-3)^2) = 4+1+0+1+4 = 10
//   slope = 20/10 = 2.0
//   intercept = 7 - 2*3 = 1.0
//   predict(x=6) = 2*6+1 = 13.0
//   predict(x=0) = 2*0+1 = 1.0

/// After fit, coefficient ≈ 2.0 and intercept ≈ 1.0 (tight tolerance).
#[test]
fn univariate_y_equals_2x_plus_1_coefficient_and_intercept() {
    // Use a small learning rate and many iterations to ensure gradient descent converges.
    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x, &y).unwrap();

    // Gradient descent converges to the closed-form OLS solution (coef=2, intercept=1) to ~1e-3;
    // assert within a tolerance appropriate for an iterative solver, not exact-OLS precision.
    let coeff = model.get_coefficients().unwrap();
    assert_abs_diff_eq!(coeff[0], 2.0, epsilon = 3e-3);

    let intercept = model.get_intercept().unwrap();
    assert_abs_diff_eq!(intercept, 1.0, epsilon = 3e-3);
}

/// predict on x=6 → 13.0; predict on x=0 → 1.0
#[test]
fn univariate_y_equals_2x_plus_1_predictions() {
    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x, &y).unwrap();

    // predict x=6 → 2*6+1 = 13.0
    // predict x=0 → 2*0+1 = 1.0
    let preds = model.predict(&array![[6.0], [0.0]]).unwrap();
    let expected = array![13.0, 1.0];
    assert_allclose(&preds, &expected, 1e-3);
}

/// After fit is done, n_iter is set (model ran at least 1 iteration).
#[test]
fn fit_sets_n_iter() {
    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
    model.fit(&x, &y).unwrap();

    let n_iter = model.get_actual_iterations();
    assert!(n_iter.is_some(), "n_iter should be set after fit");
    assert!(n_iter.unwrap() >= 1, "n_iter must be at least 1");
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness: multivariate y = 2*x1 + 3*x2 + 1
// ─────────────────────────────────────────────────────────────────────────────
//
// By construction: coefficients should converge to [2.0, 3.0], intercept to 1.0.
// Prediction at (1,1): 2*1+3*1+1 = 6.0
// Prediction at (2,3): 2*2+3*3+1 = 14.0

/// Multivariate coefficients and intercept converge to known values.
#[test]
fn multivariate_y_equals_2x1_plus_3x2_plus_1_coefficients() {
    let mut model = LinearRegression::new(true, 0.01, 20_000, 1e-10, None).unwrap();
    // 6 training points spanning the feature space well
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, // y = 2+3+1 = 6
            2.0, 1.0, // y = 4+3+1 = 8
            1.0, 2.0, // y = 2+6+1 = 9
            3.0, 2.0, // y = 6+6+1 = 13
            2.0, 3.0, // y = 4+9+1 = 14
            4.0, 1.0, // y = 8+3+1 = 12
        ],
    )
    .unwrap();
    let y = array![6.0, 8.0, 9.0, 13.0, 14.0, 12.0];
    model.fit(&x, &y).unwrap();

    let coeff = model.get_coefficients().unwrap();
    assert_abs_diff_eq!(coeff[0], 2.0, epsilon = 3e-3);
    assert_abs_diff_eq!(coeff[1], 3.0, epsilon = 3e-3);

    let intercept = model.get_intercept().unwrap();
    assert_abs_diff_eq!(intercept, 1.0, epsilon = 3e-3);
}

/// Multivariate predictions match closed-form y = 2*x1 + 3*x2 + 1.
#[test]
fn multivariate_predictions_match_closed_form() {
    let mut model = LinearRegression::new(true, 0.01, 20_000, 1e-10, None).unwrap();
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 1.0],
    )
    .unwrap();
    let y = array![6.0, 8.0, 9.0, 13.0, 14.0, 12.0];
    model.fit(&x, &y).unwrap();

    // predict at (1,1) → 6.0, (2,3) → 14.0
    let x_new = array![[1.0, 1.0], [2.0, 3.0]];
    let preds = model.predict(&x_new).unwrap();
    let expected = array![6.0, 14.0];
    assert_allclose(&preds, &expected, 5e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// fit_intercept = false
// ─────────────────────────────────────────────────────────────────────────────
//
// Data y = 2x, passing through origin (no intercept needed).
// OLS solution: slope = 2.0, intercept stored = 0.0 (by contract).

/// With fit_intercept=false the stored intercept is exactly 0.0.
#[test]
fn no_intercept_stored_intercept_is_zero() {
    let mut model = LinearRegression::new(false, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    // y = 2x, perfectly passes through origin
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    model.fit(&x, &y).unwrap();

    // The source code sets intercept = 0.0 when fit_intercept=false
    let intercept = model.get_intercept().unwrap();
    assert_abs_diff_eq!(intercept, 0.0, epsilon = 1e-15);
}

/// With fit_intercept=false the coefficient converges to slope ≈ 2.0.
#[test]
fn no_intercept_coefficient_converges_to_slope() {
    let mut model = LinearRegression::new(false, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    model.fit(&x, &y).unwrap();

    let coeff = model.get_coefficients().unwrap();
    assert_abs_diff_eq!(coeff[0], 2.0, epsilon = 1e-4);
}

/// With fit_intercept=false get_fit_intercept() returns false.
#[test]
fn no_intercept_getter_returns_false() {
    let model = LinearRegression::new(false, 0.01, 1000, 1e-6, None).unwrap();
    assert!(!model.get_fit_intercept());
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytic OLS sanity: y = 3x + 2
// ─────────────────────────────────────────────────────────────────────────────
//
// x = [1,2,3,4,5], y = [5,8,11,14,17]
// mean(x)=3, mean(y)=11
// sum((xi-3)(yi-11)) = (-2)(-6)+(-1)(-3)+0+(1)(3)+(2)(6) = 12+3+0+3+12 = 30
// sum((xi-3)^2) = 4+1+0+1+4 = 10
// slope_OLS = 30/10 = 3.0
// intercept_OLS = 11 - 3*3 = 2.0
// predict(x=6) = 3*6+2 = 20.0
// predict(x=10) = 3*10+2 = 32.0

/// OLS converges to slope=3.0, intercept=2.0 on y=3x+2.
#[test]
fn ols_sanity_y_equals_3x_plus_2_parameters() {
    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![5.0, 8.0, 11.0, 14.0, 17.0];
    model.fit(&x, &y).unwrap();

    let coeff = model.get_coefficients().unwrap();
    // analytic slope = 3.0 (gradient descent converges to ~1e-3)
    assert_abs_diff_eq!(coeff[0], 3.0, epsilon = 3e-3);

    // analytic intercept = 2.0
    let intercept = model.get_intercept().unwrap();
    assert_abs_diff_eq!(intercept, 2.0, epsilon = 3e-3);
}

/// OLS prediction at x=6 → 20.0, x=10 → 32.0 on y=3x+2.
#[test]
fn ols_sanity_y_equals_3x_plus_2_predictions() {
    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![5.0, 8.0, 11.0, 14.0, 17.0];
    model.fit(&x, &y).unwrap();

    let preds = model.predict(&array![[6.0], [10.0]]).unwrap();
    let expected = array![20.0, 32.0];
    assert_allclose(&preds, &expected, 3e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// fit_predict matches separate fit + predict
// ─────────────────────────────────────────────────────────────────────────────

/// fit_predict() returns the same predictions as fit() + predict() on training data.
#[test]
fn fit_predict_matches_fit_then_predict() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    // Path 1: fit_predict
    let mut model_a = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let preds_a = model_a.fit_predict(&x, &y).unwrap();

    // Path 2: fit then predict
    let mut model_b = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model_b.fit(&x, &y).unwrap();
    let preds_b = model_b.predict(&x).unwrap();

    assert_allclose(&preds_a, &preds_b, 1e-12);
}

/// fit_predict() predictions match known true values from y=2x+1.
#[test]
fn fit_predict_values_match_known_true_values() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let preds = model.fit_predict(&x, &y).unwrap();

    // The model is trained on exactly this data; predictions on training data
    // should closely match y = 2x+1.
    let expected = array![3.0, 5.0, 7.0, 9.0, 11.0];
    assert_allclose(&preds, &expected, 5e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Regularization: L2 shrinks the coefficient
// ─────────────────────────────────────────────────────────────────────────────
//
// Ridge penalty adds alpha * w^2 to the cost; the optimal ridge coefficient on
// univariate data x=[1..5], y=2x+1 with alpha > 0 is
//   w_ridge = (X'X + alpha*I)^{-1} X'y
// which is strictly smaller in magnitude than the OLS estimate w_ols = 2.0.
// We do not compute the exact ridge solution; we only assert |w_ridge| < |w_ols|.

/// L2 regularization shrinks the L2-norm of coefficients below the unregularized value.
#[test]
fn l2_regularization_shrinks_coefficient_norm() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut unregularized = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    unregularized.fit(&x, &y).unwrap();
    let coeff_unreg = unregularized.get_coefficients().unwrap()[0].abs();

    let mut ridge =
        LinearRegression::new(true, 0.01, 10_000, 1e-10, Some(RegularizationType::L2(5.0)))
            .unwrap();
    ridge.fit(&x, &y).unwrap();
    let coeff_ridge = ridge.get_coefficients().unwrap()[0].abs();

    assert!(
        coeff_ridge < coeff_unreg,
        "Ridge coefficient {coeff_ridge} should be smaller than unregularized {coeff_unreg}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Regularization: L1 shrinks the coefficient
// ─────────────────────────────────────────────────────────────────────────────
//
// Lasso penalty adds alpha * |w| to the cost; for sufficiently large alpha the
// coefficient is driven toward zero, so |w_lasso| < |w_ols|.

/// L1 regularization shrinks the coefficient below the unregularized value.
#[test]
fn l1_regularization_shrinks_coefficient() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut unregularized = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    unregularized.fit(&x, &y).unwrap();
    let coeff_unreg = unregularized.get_coefficients().unwrap()[0].abs();

    let mut lasso =
        LinearRegression::new(true, 0.01, 10_000, 1e-10, Some(RegularizationType::L1(5.0)))
            .unwrap();
    lasso.fit(&x, &y).unwrap();
    let coeff_lasso = lasso.get_coefficients().unwrap()[0].abs();

    assert!(
        coeff_lasso < coeff_unreg,
        "Lasso coefficient {coeff_lasso} should be smaller than unregularized {coeff_unreg}"
    );
}

/// L2 regularized model still recovers intercept in the right ball-park (fit_intercept=true).
/// For moderate alpha the intercept should not deviate too far from 1.0.
#[test]
fn l2_regularization_intercept_within_reasonable_range() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut ridge =
        LinearRegression::new(true, 0.01, 10_000, 1e-10, Some(RegularizationType::L2(0.1)))
            .unwrap();
    ridge.fit(&x, &y).unwrap();

    // With small alpha=0.1 the intercept should still be close to 1.0 (within ±0.5)
    let intercept = ridge.get_intercept().unwrap();
    assert!(
        (intercept - 1.0).abs() < 0.5,
        "Intercept {intercept} deviates too far from 1.0 under weak L2 regularization"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism: same seed / same data → identical results
// ─────────────────────────────────────────────────────────────────────────────
//
// LinearRegression uses gradient descent which is deterministic given the data;
// there is no internal randomness.  Two identical constructions on the same data
// must produce bit-identical predictions.

/// Two identical LinearRegression models trained on the same data produce identical predictions.
#[test]
fn determinism_same_data_identical_predictions() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
    let x_test = array![[6.0], [7.0], [8.0]];

    let mut model_a = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model_a.fit(&x, &y).unwrap();
    let preds_a = model_a.predict(&x_test).unwrap();

    let mut model_b = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model_b.fit(&x, &y).unwrap();
    let preds_b = model_b.predict(&x_test).unwrap();

    // Gradient descent is deterministic — results must be bit-identical
    assert_allclose(&preds_a, &preds_b, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Save / load round-trip
// ─────────────────────────────────────────────────────────────────────────────

/// save_to_path + load_from_path round-trip yields identical predictions.
#[test]
fn save_load_round_trip_identical_predictions() {
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];
    let x_test = array![[6.0], [7.0], [0.5]];

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model.fit(&x_train, &y_train).unwrap();
    let preds_before = model.predict(&x_test).unwrap();

    // Use a tmp path unique to this test to avoid collisions
    let path = "/tmp/rustyml_linear_regression_test_round_trip.json";
    model.save_to_path(path).unwrap();

    let loaded = LinearRegression::load_from_path(path).unwrap();
    let preds_after = loaded.predict(&x_test).unwrap();

    assert_allclose(&preds_before, &preds_after, 0.0);

    // Clean up
    let _ = std::fs::remove_file(path);
}

/// After load, getter values match those of the original model.
#[test]
fn save_load_preserves_model_state() {
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut model = LinearRegression::new(false, 0.005, 8_000, 1e-9, None).unwrap();
    model.fit(&x_train, &y_train).unwrap();

    let path = "/tmp/rustyml_linear_regression_test_state.json";
    model.save_to_path(path).unwrap();
    let loaded = LinearRegression::load_from_path(path).unwrap();

    assert_eq!(loaded.get_fit_intercept(), model.get_fit_intercept());
    assert_abs_diff_eq!(
        loaded.get_learning_rate(),
        model.get_learning_rate(),
        epsilon = 1e-15
    );
    assert_eq!(loaded.get_max_iterations(), model.get_max_iterations());

    // Coefficients must survive round-trip exactly
    let orig_coeff = model.get_coefficients().unwrap();
    let load_coeff = loaded.get_coefficients().unwrap();
    assert_allclose(orig_coeff, load_coeff, 0.0);

    let _ = std::fs::remove_file(path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Default constructor
// ─────────────────────────────────────────────────────────────────────────────

/// Default model has fit_intercept=true, learning_rate=0.01, max_iter=1000, tol=1e-5.
#[test]
fn default_constructor_has_expected_hyperparameters() {
    let model = LinearRegression::default();
    assert!(model.get_fit_intercept());
    assert_abs_diff_eq!(model.get_learning_rate(), 0.01, epsilon = 1e-15);
    assert_eq!(model.get_max_iterations(), 1000);
    assert_abs_diff_eq!(model.get_tolerance(), 1e-5, epsilon = 1e-20);
    assert!(model.get_coefficients().is_none());
    assert!(model.get_intercept().is_none());
    assert!(model.get_actual_iterations().is_none());
}

/// Default model can be fit and predict without error.
#[test]
fn default_constructor_can_fit_and_predict() {
    let mut model = LinearRegression::default();
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![3.0, 5.0, 7.0];
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&array![[4.0]]).unwrap();
    // y=2x+1, predict(4) ≈ 9.0
    assert_abs_diff_eq!(preds[0], 9.0, epsilon = 5e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Clone
// ─────────────────────────────────────────────────────────────────────────────

/// Clone of a fitted model makes identical predictions.
#[test]
fn clone_of_fitted_model_makes_identical_predictions() {
    let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![3.0, 5.0, 7.0, 9.0, 11.0];
    let x_test = array![[6.0], [0.0]];

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model.fit(&x_train, &y_train).unwrap();
    let preds_orig = model.predict(&x_test).unwrap();

    let cloned = model.clone();
    let preds_clone = cloned.predict(&x_test).unwrap();

    assert_allclose(&preds_orig, &preds_clone, 0.0);
}
// ─────────────────────────────────────────────────────────────────────────────
// In-loop NonFinite divergence guards
// ─────────────────────────────────────────────────────────────────────────────
//
// The data here is finite and dimensionally valid, so the up-front `preliminary_check`
// passes cleanly.  The failure is generated INSIDE the gradient-descent loop: an absurdly
// large learning_rate makes each weight update over-shoot, so the parameters grow
// geometrically and overflow f64 to ±inf within a handful of iterations.  That is caught by
// the in-loop finiteness guards on the cost / gradients / updated parameters, all of which
// return `Error::NonFinite(_)`.  This is a DIFFERENT code path from the existing
// `fit_nan_in_x` / `fit_inf_in_x` tests, which inject non-finite values into X and are
// rejected up-front by `preliminary_check`.
//
// Growth argument (pure arithmetic, no model output):
//   Weights start at 0.  With x = [1,2,3,4,5], y = 2x+1 = [3,5,7,9,11], the first update is
//     w[0] <- w[0] - lr * (Xᵀ(Xw - y)/n)[0]
//           = 0      - 1e8 * (Σ xᵢ·(0 - yᵢ) / 5)
//           = 0      - 1e8 * (-(1·3 + 2·5 + 3·7 + 4·9 + 5·11)/5)
//           = 0      - 1e8 * (-125/5) = +1e8 * 25 = 2.5e9   after a SINGLE step.
//   Thereafter the homogeneous part scales each step by lr·(XᵀX/n) = 1e8·(55/5) = 1.1e9, so
//   the magnitude multiplies by ~1e9 per iteration; starting near 2.5e9 it passes f64::MAX
//   (~1.8e308) within well under 35 iterations — far inside the 1000-iteration budget — so an
//   inf is guaranteed and one of the in-loop guards returns Error::NonFinite(_).

/// fit() on finite, valid data but with a huge learning_rate diverges to ±inf inside the
/// gradient-descent loop → Error::NonFinite (in-loop guard, not the up-front X check).
#[test]
fn fit_huge_learning_rate_diverges_returns_non_finite() {
    // learning_rate = 1e8 is positive and finite, so the constructor accepts it.
    let mut model = LinearRegression::new(true, 1e8, 1000, 1e-10, None).unwrap();

    // Perfectly clean, finite data: y = 2x + 1 on x = [1..5].
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite from in-loop divergence guard, got {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// L1 regularization parallel branch (n_features >= 200)
// ─────────────────────────────────────────────────────────────────────────────
//
// When n_features >= LINEAR_REGRESSION_PARALLEL_THRESHOLD (200) the L1 cost term and the L1
// gradient term switch to their parallel (rayon) code paths, which additionally call
// `weights.as_slice().unwrap()` / `weight_gradients.as_slice_mut().unwrap()`.  This is the
// ONLY path that exercises that >=200-feature parallel L1 branch, so we build a dataset with
// exactly 200 features (just crossing the `>=` threshold: 200 >= 200).
//
// Design / ground truth (no model output used):
//   * fit_intercept = false, so the target needs no bias term.
//   * Feature column 0 carries ALL of the signal: y[i] = 3.0 * x[i][0] exactly.
//   * The other 199 columns are tiny, bounded, deterministic noise in [-0.03, 0.03] that does
//     not track y (a fixed integer zig-zag pattern), so they are uninformative.
//
// Scale separation (drives every assertion below).  With x0[i] = i - 5.5 (i = 0..11):
//     Σ x0²  = 2*(0.5²+1.5²+2.5²+3.5²+4.5²+5.5²) = 2*71.5 = 143,
//     Σ|x0|  = 2*(0.5+1.5+2.5+3.5+4.5+5.5)        = 2*18   = 36.
//   At w = 0 the gradient on the informative column is
//     grad[0] = -Σ(x0[i]·y[i])/n = -3·Σx0²/12 = -3·143/12 ≈ -35.75,
//   so the very first step moves w[0] toward +3.0 (the true slope).  For any noise column j,
//   |x_j| ≤ 0.03, hence
//     |grad[j]| ≤ 0.03·Σ|y[i]|/12 = 0.03·(3·Σ|x0|)/12 = 0.03·108/12 = 0.27,
//   a ≈130:1 separation from grad[0].  Gradient descent from all-zeros therefore drives
//   coefficient[0] up toward ~3.0 while every noise coefficient stays small.  We assert the
//   structural facts implied by this construction (not an exact value, since the 200-feature /
//   12-sample system is under-determined and only the dominant coefficient is unambiguous):
//     (a) coefficient[0] is substantial and positive (well above 1.0; true slope 3.0),
//     (b) every other coefficient is small in magnitude (|·| < 0.5, far below the ~3.0 signal),
//     (c) coefficient[0] strictly dominates the largest other |coefficient|.

/// L1 regularization with >=200 features hits the parallel L1 cost/gradient branch; the one
/// informative feature (column 0, y = 3*x0) ends up with the dominant coefficient while the
/// uninformative noise columns stay small.
#[test]
fn l1_regularization_parallel_branch_recovers_informative_feature() {
    let n_samples = 12usize;
    let n_features = 200usize; // exactly the parallel threshold (>= 200 triggers the branch)

    // Build X deterministically: column 0 = centered signal, columns 1.. = tiny noise.
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        if j == 0 {
            // centered, varying signal in [-5.5, 5.5]
            (i as f64) - 5.5
        } else {
            // deterministic, bounded zig-zag noise in {-0.03, ..., 0.03}, uncorrelated with y
            0.01 * (((i * 31 + j * 17) % 7) as f64 - 3.0)
        }
    });

    // y depends only on column 0:  y = 3 * x0  (no intercept needed).
    let y = Array1::from_shape_fn(n_samples, |i| 3.0 * ((i as f64) - 5.5));

    // Very weak L1 so the dominant coefficient is shrunk only slightly. fit_intercept = false.
    let mut model = LinearRegression::new(
        false,
        0.01,
        20_000,
        1e-12,
        Some(RegularizationType::L1(1e-3)),
    )
    .unwrap();
    model
        .fit(&x, &y)
        .expect("fit with >=200 features and L1 should succeed");

    let coeffs = model.get_coefficients().unwrap();
    assert_eq!(
        coeffs.len(),
        n_features,
        "coefficient vector length must equal feature count"
    );

    let c0 = coeffs[0];
    // (a) The informative coefficient is substantial and positive (true slope is 3.0).
    assert!(
        c0 > 1.0,
        "informative coefficient[0] = {c0} should be a large positive value (true slope 3.0)"
    );

    // (b) Every uninformative noise coefficient stays small, well below the ~3.0 signal. The
    // bound 0.5 is comfortably above any value the bounded (|x_j| <= 0.03) noise columns can
    // earn, yet far below coefficient[0], so it is a design-derived correctness check.
    let max_other = coeffs
        .iter()
        .skip(1)
        .fold(0.0_f64, |acc, &w| acc.max(w.abs()));
    assert!(
        max_other < 0.5,
        "uninformative coefficients should stay small; largest |other| = {max_other}"
    );

    // (c) coefficient[0] strictly dominates every other |coefficient|.
    assert!(
        c0.abs() > max_other,
        "|coefficient[0]| = {} should dominate the largest other |coefficient| = {}",
        c0.abs(),
        max_other
    );
}
