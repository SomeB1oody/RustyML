//! Internal input-validation helpers shared across machine-learning estimators
//!
//! Provides checks for feature/target matrices and common hyperparameters
//! (learning rate, max iterations, tolerance, regularization) plus a fitted-state
//! guard, each returning a consistent [`Error`] on failure

use super::RegularizationType;
use crate::error::Error;
use ndarray::{ArrayBase, Data, Dimension, Ix1, Ix2};

/// Performs validation checks on the input data matrices
///
/// Validates that:
/// - The input data matrix is not empty
/// - The input data does not contain NaN or infinite values
/// - When a target vector is provided:
///   - The target vector is not empty
///   - The target vector length matches the number of rows in the input data
///
/// # Parameters
///
/// - `x` - A 2D array of feature values where rows represent samples and columns represent features
/// - `y` - An optional 1D array of target variables or labels corresponding to each sample
///
/// # Returns
///
/// - `Ok(())` if all validation checks pass, otherwise an `Error`
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If the input data or target vector is empty
/// - [`Error::NonFinite`] - If the input data contains NaN or infinite values
/// - [`Error::DimensionMismatch`] - If the dimensions of `x` and `y` mismatch
pub(super) fn preliminary_check<S>(
    x: &ArrayBase<S, Ix2>,
    y: Option<&ArrayBase<S, Ix1>>,
) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    if x.nrows() == 0 {
        return Err(Error::empty_input("input data"));
    }

    for (i, row) in x.outer_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(Error::non_finite(format!(
                    "input data at position [{}][{}]",
                    i, j
                )));
            }
        }
    }

    if let Some(y) = y {
        if y.is_empty() {
            return Err(Error::empty_input("target vector"));
        }

        if y.len() != x.nrows() {
            return Err(Error::dimension_mismatch(x.nrows(), y.len()));
        }
    }
    Ok(())
}

/// Validates that the learning rate parameter is positive and finite
///
/// The learning rate controls the step size in gradient descent optimization,
/// so it must be a positive, finite value to ensure proper convergence behavior
///
/// # Parameters
///
/// - `learning_rate` - The learning rate value to validate
///
/// # Returns
///
/// - `Ok(())` if the learning rate is valid, otherwise an `Error`
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If the learning rate is non-positive, NaN, or infinite
pub(super) fn validate_learning_rate(learning_rate: f64) -> Result<(), Error> {
    if learning_rate <= 0.0 || !learning_rate.is_finite() {
        return Err(Error::invalid_parameter(
            "learning_rate",
            format!("must be positive and finite, got {}", learning_rate),
        ));
    }

    Ok(())
}

/// Validates that the maximum iterations parameter is greater than zero
///
/// The maximum iterations parameter determines the upper bound on the number
/// of training iterations. It must be at least 1 to allow the algorithm to run
///
/// # Parameters
///
/// - `max_iterations` - The maximum number of iterations to validate
///
/// # Returns
///
/// - `Ok(())` if the maximum iterations value is valid, otherwise an `Error`
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If the maximum iterations value is 0
pub(super) fn validate_max_iterations(max_iterations: usize) -> Result<(), Error> {
    if max_iterations == 0 {
        return Err(Error::invalid_parameter(
            "max_iterations",
            "must be greater than 0",
        ));
    }

    Ok(())
}

/// Validates that the tolerance parameter is positive and finite
///
/// The tolerance parameter defines the convergence criterion for iterative algorithms:
/// training stops when the change in loss between iterations falls below this threshold,
/// so it must be a positive, finite value to ensure meaningful convergence detection
///
/// # Parameters
///
/// - `tolerance` - The convergence tolerance value to validate
///
/// # Returns
///
/// - `Ok(())` if the tolerance is valid, otherwise an `Error`
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If the tolerance is non-positive, NaN, or infinite
pub(super) fn validate_tolerance(tolerance: f64) -> Result<(), Error> {
    if tolerance <= 0.0 || !tolerance.is_finite() {
        return Err(Error::invalid_parameter(
            "tolerance",
            format!("must be positive and finite, got {}", tolerance),
        ));
    }

    Ok(())
}

/// Validates the regularization type and its associated parameters
///
/// Checks that regularization parameters are properly configured:
/// - For L1 and L2 regularization, the alpha (regularization strength) parameter
///   must be non-negative and finite
/// - None (no regularization) is always valid
///
/// # Parameters
///
/// - `reg_type` - An optional regularization type with its strength parameter
///
/// # Returns
///
/// - `Ok(())` if the regularization configuration is valid, otherwise an `Error`
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If the regularization alpha is negative, NaN, or infinite
pub(super) fn validate_regularization_type(
    reg_type: Option<RegularizationType>,
) -> Result<(), Error> {
    if let Some(reg) = &reg_type {
        match reg {
            RegularizationType::L1(alpha) | RegularizationType::L2(alpha) => {
                if *alpha < 0.0 || !alpha.is_finite() {
                    return Err(Error::invalid_parameter(
                        "alpha",
                        format!("must be non-negative and finite, got {}", alpha),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Returns [`Error::NotFitted`] when a model has not been fitted yet
///
/// Centralizes the "is this model fitted?" guard so every estimator emits the
/// same error instead of hand-rolling the check at each call site
///
/// # Parameters
///
/// - `is_fitted` - Whether the model has already been trained (e.g. `self.weights.is_some()`)
/// - `model` - The model's name, used to build the [`Error::NotFitted`] payload
///
/// # Returns
///
/// - `Ok(())` if the model is fitted, otherwise [`Error::NotFitted`]
///
/// # Errors
///
/// - [`Error::NotFitted`] - If `is_fitted` is `false`
#[inline]
pub(super) fn check_is_fitted(
    is_fitted: bool,
    model: &'static str,
) -> crate::error::RustymlResult<()> {
    if is_fitted {
        Ok(())
    } else {
        Err(Error::not_fitted(model))
    }
}

/// Validates a feature matrix passed to a `predict`-style method
///
/// Performs the three checks every estimator needs before predicting, with a
/// single consistent error message for each failure mode:
/// - the matrix is not empty
/// - the feature count matches the training data
/// - no value is NaN or infinite
///
/// # Parameters
///
/// - `x` - Feature matrix where rows are samples and columns are features
/// - `expected_features` - Number of features the fitted model was trained on
///
/// # Returns
///
/// - `Ok(())` if all checks pass, otherwise an `Error`
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty
/// - [`Error::DimensionMismatch`] - If `x` has a mismatched feature count
/// - [`Error::NonFinite`] - If `x` contains non-finite values
pub(super) fn validate_predict_input<S>(
    x: &ArrayBase<S, Ix2>,
    expected_features: usize,
) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    if x.is_empty() {
        return Err(Error::empty_input("dataset to predict on"));
    }

    if x.ncols() != expected_features {
        return Err(Error::dimension_mismatch(expected_features, x.ncols()));
    }

    if x.iter().any(|&val| !val.is_finite()) {
        return Err(Error::non_finite("input data"));
    }

    Ok(())
}

/// Returns an error if `x` contains no elements
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty
pub(super) fn check_non_empty<S, D>(x: &ArrayBase<S, D>) -> Result<(), Error>
where
    S: Data,
    D: Dimension,
{
    if x.is_empty() {
        return Err(Error::empty_input("input data"));
    }
    Ok(())
}

/// Returns an error if `x` contains any NaN or infinite value
///
/// # Errors
///
/// - [`Error::NonFinite`] - If any element is not finite
pub(super) fn check_finite<S, D>(x: &ArrayBase<S, D>) -> Result<(), Error>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::non_finite("input data"));
    }
    Ok(())
}

/// Returns an error if `x` has fewer than `min_samples` rows
///
/// `model` names the estimator so the message reads naturally (e.g. `"PCA
/// requires at least 2 samples"`)
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If `x` has fewer than `min_samples` rows
pub(super) fn check_min_samples<S>(
    x: &ArrayBase<S, Ix2>,
    min_samples: usize,
    model: &str,
) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    if x.nrows() < min_samples {
        return Err(Error::invalid_input(format!(
            "{} requires at least {} samples",
            model, min_samples
        )));
    }
    Ok(())
}

/// Validates a feature matrix passed to a `fit`-style method
///
/// Checks, in order, that the matrix is non-empty, has at least one feature
/// column, and contains only finite values
///
/// # Parameters
///
/// - `x` - Feature matrix with samples as rows and features as columns
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty or has zero columns
/// - [`Error::NonFinite`] - If `x` contains non-finite values
pub(super) fn validate_fit_matrix<S>(x: &ArrayBase<S, Ix2>) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    check_non_empty(x)?;
    if x.ncols() == 0 {
        return Err(Error::empty_input("features"));
    }
    check_finite(x)
}

/// Validates a feature matrix passed to a `transform`-style method
///
/// Checks that the matrix is non-empty, that its feature count matches the
/// fitted model, and that it contains only finite values. Companion to
/// [`validate_predict_input`] for the dimensionality-reduction estimators, whose
/// projection method is named `transform` rather than `predict`
///
/// # Parameters
///
/// - `x` - Feature matrix with samples as rows and features as columns
/// - `expected_features` - Number of features the fitted model was trained on
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty
/// - [`Error::DimensionMismatch`] - If `x` has a mismatched feature count
/// - [`Error::NonFinite`] - If `x` contains non-finite values
pub(super) fn validate_transform_matrix<S>(
    x: &ArrayBase<S, Ix2>,
    expected_features: usize,
) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    check_non_empty(x)?;
    if x.ncols() != expected_features {
        return Err(Error::dimension_mismatch(expected_features, x.ncols()));
    }
    check_finite(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use ndarray::{Array1, Array2};

    // preliminary_check

    /// preliminary_check with an empty x (0 rows) returns Error::EmptyInput
    #[test]
    fn preliminary_check_empty_x_gives_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 3));
        let err = preliminary_check(&x, None).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// preliminary_check with NaN in x returns Error::NonFinite
    #[test]
    fn preliminary_check_x_with_nan_gives_non_finite() {
        let x = ndarray::array![[1.0, f64::NAN], [2.0, 3.0]];
        let err = preliminary_check(&x, None).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// preliminary_check with infinity in x returns Error::NonFinite
    #[test]
    fn preliminary_check_x_with_inf_gives_non_finite() {
        let x = ndarray::array![[1.0, f64::INFINITY]];
        let err = preliminary_check(&x, None).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// preliminary_check with y.len() != x.nrows() returns Error::DimensionMismatch
    #[test]
    fn preliminary_check_y_len_mismatch_gives_dimension_mismatch() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = Array1::from_vec(vec![1.0, 2.0]); // len 2, but x has 3 rows
        let err = preliminary_check(&x, Some(&y)).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 3, "expected count should be x.nrows()=3");
                assert_eq!(found, 2, "found count should be y.len()=2");
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    /// preliminary_check with a valid finite x and no y returns Ok
    #[test]
    fn preliminary_check_valid_x_no_y_gives_ok() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        assert!(preliminary_check(&x, None).is_ok());
    }

    /// preliminary_check with a valid x and matching-length y returns Ok
    #[test]
    fn preliminary_check_valid_x_and_y_gives_ok() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);
        assert!(preliminary_check(&x, Some(&y)).is_ok());
    }

    // validate_predict_input

    /// validate_predict_input with an empty x returns Error::EmptyInput
    #[test]
    fn validate_predict_input_empty_gives_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 3));
        let err = validate_predict_input(&x, 3).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// validate_predict_input with a feature-count mismatch returns Error::DimensionMismatch
    #[test]
    fn validate_predict_input_ncols_mismatch_gives_dimension_mismatch() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]]; // ncols = 2
        let err = validate_predict_input(&x, 3).unwrap_err(); // expected 3 cols
        match err {
            Error::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 3);
                assert_eq!(found, 2);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    /// validate_predict_input with NaN in x returns Error::NonFinite
    #[test]
    fn validate_predict_input_nan_gives_non_finite() {
        let x = ndarray::array![[1.0, f64::NAN], [2.0, 3.0]];
        let err = validate_predict_input(&x, 2).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// validate_predict_input with a valid x returns Ok
    #[test]
    fn validate_predict_input_valid_gives_ok() {
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(validate_predict_input(&x, 3).is_ok());
    }

    // validate_learning_rate

    /// validate_learning_rate with a positive finite value returns Ok
    #[test]
    fn validate_learning_rate_positive_gives_ok() {
        assert!(validate_learning_rate(0.1).is_ok());
    }

    /// validate_learning_rate with 0.0 returns Error::InvalidParameter
    #[test]
    fn validate_learning_rate_zero_gives_invalid_parameter() {
        let err = validate_learning_rate(0.0).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => {
                assert_eq!(name, "learning_rate");
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_learning_rate with a negative value returns Error::InvalidParameter
    #[test]
    fn validate_learning_rate_negative_gives_invalid_parameter() {
        let err = validate_learning_rate(-1.0).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "learning_rate"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_learning_rate with NaN returns Error::InvalidParameter
    #[test]
    fn validate_learning_rate_nan_gives_invalid_parameter() {
        let err = validate_learning_rate(f64::NAN).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "learning_rate"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_learning_rate with infinity returns Error::InvalidParameter
    #[test]
    fn validate_learning_rate_inf_gives_invalid_parameter() {
        let err = validate_learning_rate(f64::INFINITY).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "learning_rate"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // validate_max_iterations

    /// validate_max_iterations with a value >= 1 returns Ok
    #[test]
    fn validate_max_iterations_one_gives_ok() {
        assert!(validate_max_iterations(1).is_ok());
    }

    /// validate_max_iterations with 0 returns Error::InvalidParameter
    #[test]
    fn validate_max_iterations_zero_gives_invalid_parameter() {
        let err = validate_max_iterations(0).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => {
                assert_eq!(name, "max_iterations");
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // validate_tolerance

    /// validate_tolerance with a positive finite value returns Ok
    #[test]
    fn validate_tolerance_positive_gives_ok() {
        assert!(validate_tolerance(1e-4).is_ok());
    }

    /// validate_tolerance with 0.0 returns Error::InvalidParameter
    #[test]
    fn validate_tolerance_zero_gives_invalid_parameter() {
        let err = validate_tolerance(0.0).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_tolerance with a negative value returns Error::InvalidParameter
    #[test]
    fn validate_tolerance_negative_gives_invalid_parameter() {
        let err = validate_tolerance(-0.5).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_tolerance with NaN returns Error::InvalidParameter
    #[test]
    fn validate_tolerance_nan_gives_invalid_parameter() {
        let err = validate_tolerance(f64::NAN).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_tolerance with infinity returns Error::InvalidParameter
    #[test]
    fn validate_tolerance_inf_gives_invalid_parameter() {
        let err = validate_tolerance(f64::INFINITY).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // validate_regularization_type

    /// validate_regularization_type with None returns Ok
    #[test]
    fn validate_regularization_type_none_gives_ok() {
        assert!(validate_regularization_type(None).is_ok());
    }

    /// validate_regularization_type with a zero L1 alpha returns Ok
    #[test]
    fn validate_regularization_type_l1_zero_gives_ok() {
        assert!(validate_regularization_type(Some(RegularizationType::L1(0.0))).is_ok());
    }

    /// validate_regularization_type with a positive L2 alpha returns Ok
    #[test]
    fn validate_regularization_type_l2_positive_gives_ok() {
        assert!(validate_regularization_type(Some(RegularizationType::L2(0.5))).is_ok());
    }

    /// validate_regularization_type with a negative L1 alpha returns Error::InvalidParameter
    #[test]
    fn validate_regularization_type_l1_negative_gives_invalid_parameter() {
        let err = validate_regularization_type(Some(RegularizationType::L1(-1.0))).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_regularization_type with a NaN L1 alpha returns Error::InvalidParameter
    #[test]
    fn validate_regularization_type_l1_nan_gives_invalid_parameter() {
        let err = validate_regularization_type(Some(RegularizationType::L1(f64::NAN))).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// validate_regularization_type with an infinite L1 alpha returns Error::InvalidParameter
    #[test]
    fn validate_regularization_type_l1_inf_gives_invalid_parameter() {
        let err =
            validate_regularization_type(Some(RegularizationType::L1(f64::INFINITY))).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // check_is_fitted

    /// check_is_fitted with is_fitted == true returns Ok
    #[test]
    fn check_is_fitted_true_gives_ok() {
        assert!(check_is_fitted(true, "SomeModel").is_ok());
    }

    /// check_is_fitted with is_fitted == false returns Error::NotFitted carrying the model name
    #[test]
    fn check_is_fitted_false_gives_not_fitted() {
        let err = check_is_fitted(false, "SomeModel").unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "SomeModel"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }

    // check_non_empty

    /// An empty array produces EmptyInput
    #[test]
    fn check_non_empty_empty_array_gives_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 3));
        let err = check_non_empty(&empty).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// A non-empty array returns Ok
    #[test]
    fn check_non_empty_non_empty_array_gives_ok() {
        let a: Array2<f64> = Array2::zeros((2, 3));
        assert!(check_non_empty(&a).is_ok());
    }

    // check_finite

    /// An array containing NaN produces NonFinite
    #[test]
    fn check_finite_nan_gives_non_finite() {
        let a = ndarray::array![[1.0, f64::NAN], [2.0, 3.0]];
        let err = check_finite(&a).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// An array containing infinity produces NonFinite
    #[test]
    fn check_finite_inf_gives_non_finite() {
        let a = ndarray::array![[f64::INFINITY, 2.0]];
        let err = check_finite(&a).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// An array of finite values returns Ok
    #[test]
    fn check_finite_all_finite_gives_ok() {
        let a = ndarray::array![[0.0, -1.5, 1e10], [f64::MIN, f64::MAX, 0.0]];
        assert!(check_finite(&a).is_ok());
    }

    // check_min_samples

    /// Too few rows produces InvalidInput
    #[test]
    fn check_min_samples_too_few_gives_invalid_input() {
        let a: Array2<f64> = Array2::zeros((1, 3));
        let err = check_min_samples(&a, 2, "TestModel").unwrap_err();
        match err {
            Error::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    /// Rows equal to min_samples returns Ok
    #[test]
    fn check_min_samples_exactly_min_gives_ok() {
        let a: Array2<f64> = Array2::zeros((2, 3));
        assert!(check_min_samples(&a, 2, "TestModel").is_ok());
    }

    // validate_fit_matrix

    /// A matrix with zero feature columns produces EmptyInput
    #[test]
    fn validate_fit_matrix_zero_cols_gives_empty_input() {
        let a: Array2<f64> = Array2::zeros((3, 0));
        let err = validate_fit_matrix(&a).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// A valid finite matrix returns Ok
    #[test]
    fn validate_fit_matrix_valid_gives_ok() {
        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        assert!(validate_fit_matrix(&a).is_ok());
    }

    // validate_transform_matrix

    /// A feature-count mismatch produces DimensionMismatch
    #[test]
    fn validate_transform_matrix_ncols_mismatch_gives_dimension_mismatch() {
        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]]; // ncols = 2
        let err = validate_transform_matrix(&a, 3).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 3);
                assert_eq!(found, 2);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    /// A matching feature count returns Ok
    #[test]
    fn validate_transform_matrix_match_gives_ok() {
        let a = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(validate_transform_matrix(&a, 3).is_ok());
    }
}
