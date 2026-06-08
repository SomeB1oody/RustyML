use super::RegularizationType;
use crate::error::Error;
use ndarray::{ArrayBase, Data, Ix1, Ix2};

/// Performs validation checks on the input data matrices.
///
/// This function validates that:
/// - The input data matrix is not empty
/// - The input data does not contain NaN or infinite values
/// - When a target vector is provided:
///   - The target vector is not empty
///   - The target vector length matches the number of rows in the input data
///
/// # Parameters
///
/// - `x` - A 2D array of feature values where rows represent samples and columns represent features
/// - `y` - An optional 1D array representing the target variables or labels corresponding to each sample
///
/// # Returns
///
/// - `result` - `Ok(())` if all validation checks pass, otherwise an `Error`
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

/// Validates that the learning rate parameter is positive and finite.
///
/// The learning rate controls the step size in gradient descent optimization.
/// It must be a positive, finite value to ensure proper convergence behavior.
///
/// # Parameters
///
/// - `learning_rate` - The learning rate value to validate
///
/// # Returns
///
/// - `result` - `Ok(())` if the learning rate is valid, otherwise an `Error`
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

/// Validates that the maximum iterations parameter is greater than zero.
///
/// The maximum iterations parameter determines the upper bound on the number
/// of training iterations. It must be at least 1 to allow the algorithm to run.
///
/// # Parameters
///
/// - `max_iterations` - The maximum number of iterations to validate
///
/// # Returns
///
/// - `result` - `Ok(())` if the maximum iterations value is valid, otherwise an `Error`
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

/// Validates that the tolerance parameter is positive and finite.
///
/// The tolerance parameter defines the convergence criterion for iterative algorithms.
/// Training stops when the change in loss between iterations falls below this threshold.
/// It must be a positive, finite value to ensure meaningful convergence detection.
///
/// # Parameters
///
/// - `tolerance` - The convergence tolerance value to validate
///
/// # Returns
///
/// - `result` - `Ok(())` if the tolerance is valid, otherwise an `Error`
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

/// Validates the regularization type and its associated parameters.
///
/// This function checks that regularization parameters are properly configured:
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
/// - `result` - `Ok(())` if the regularization configuration is valid, otherwise an `Error`
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

/// Returns [`Error::NotFitted`] when a model has not been fitted yet.
///
/// Centralizes the "is this model fitted?" guard so every estimator emits the
/// same error instead of hand-rolling the check at each call site.
///
/// # Parameters
///
/// - `is_fitted` - Whether the model has already been trained (e.g. `self.weights.is_some()`)
/// - `model` - The model's name, used to build the [`Error::NotFitted`] payload
///
/// # Returns
///
/// - `result` - `Ok(())` if the model is fitted, otherwise [`Error::NotFitted`]
///
/// # Errors
///
/// - [`Error::NotFitted`] - If `is_fitted` is `false`
#[inline]
pub(super) fn check_is_fitted(is_fitted: bool, model: &'static str) -> crate::error::RustymlResult<()> {
    if is_fitted {
        Ok(())
    } else {
        Err(Error::not_fitted(model))
    }
}

/// Validates a feature matrix passed to a `predict`-style method.
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
/// - `result` - `Ok(())` if all checks pass, otherwise an `Error`
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
