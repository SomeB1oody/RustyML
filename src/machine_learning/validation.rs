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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use ndarray::{Array1, Array2};

    // --- preliminary_check ---

    /// An empty x (0 rows) must produce EmptyInput.
    /// Derivation: x.nrows() == 0 triggers the first guard; code returns
    /// Err(Error::empty_input("input data")) → Error::EmptyInput variant.
    #[test]
    fn preliminary_check_empty_x_gives_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 3));
        let err = preliminary_check(&x, None).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// An x with NaN values must produce NonFinite.
    /// Derivation: x.nrows() > 0 passes the first guard; the nested loop finds
    /// NaN via is_nan() and returns Err(Error::non_finite(...)) → NonFinite variant.
    #[test]
    fn preliminary_check_x_with_nan_gives_non_finite() {
        let x = ndarray::array![[1.0, f64::NAN], [2.0, 3.0]];
        let err = preliminary_check(&x, None).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// An x with infinity must produce NonFinite.
    /// Derivation: is_infinite() is true for f64::INFINITY; the inner loop
    /// catches it and returns NonFinite.
    #[test]
    fn preliminary_check_x_with_inf_gives_non_finite() {
        let x = ndarray::array![[1.0, f64::INFINITY]];
        let err = preliminary_check(&x, None).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// When y.len() != x.nrows() a DimensionMismatch error must be returned.
    /// Derivation: x has 3 rows; y has len 2; 2 != 3 triggers
    /// Err(Error::dimension_mismatch(3, 2)) → DimensionMismatch{ expected:3, found:2 }.
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

    /// y=None with a valid finite x must return Ok(()).
    /// Derivation: x.nrows() > 0, all finite, y is None so the y-check is
    /// skipped; function reaches Ok(()) unconditionally.
    #[test]
    fn preliminary_check_valid_x_no_y_gives_ok() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        assert!(preliminary_check(&x, None).is_ok());
    }

    /// y with matching length and valid x must return Ok(()).
    /// Derivation: 2 rows == y.len()==2; all values finite; Ok(()) is returned.
    #[test]
    fn preliminary_check_valid_x_and_y_gives_ok() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![0.0, 1.0]);
        assert!(preliminary_check(&x, Some(&y)).is_ok());
    }

    // --- validate_predict_input ---

    /// An empty x (0 elements) must produce EmptyInput.
    /// Derivation: x.is_empty() is true for shape (0,3); first guard fires.
    #[test]
    fn validate_predict_input_empty_gives_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 3));
        let err = validate_predict_input(&x, 3).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// A feature-count mismatch must produce DimensionMismatch.
    /// Derivation: x has ncols=2; expected_features=3; 2 != 3 triggers
    /// Err(Error::dimension_mismatch(3, 2)) → DimensionMismatch{expected:3, found:2}.
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

    /// An x with NaN (correct col count) must produce NonFinite.
    /// Derivation: empty guard passes (non-empty), col guard passes (ncols=2==2),
    /// then the iter check finds NaN and returns NonFinite.
    #[test]
    fn validate_predict_input_nan_gives_non_finite() {
        let x = ndarray::array![[1.0, f64::NAN], [2.0, 3.0]];
        let err = validate_predict_input(&x, 2).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// A valid x (non-empty, correct col count, finite) must return Ok(()).
    /// Derivation: all three guards pass; Ok(()) is returned.
    #[test]
    fn validate_predict_input_valid_gives_ok() {
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(validate_predict_input(&x, 3).is_ok());
    }
    // --- validate_learning_rate ---

    /// A positive finite learning rate must return Ok(()).
    /// Derivation: 0.1 > 0.0 and is_finite() is true, so neither half of the
    /// guard (`<= 0.0 || !is_finite()`) holds; function reaches Ok(()).
    #[test]
    fn validate_learning_rate_positive_gives_ok() {
        assert!(validate_learning_rate(0.1).is_ok());
    }

    /// learning_rate == 0.0 must be rejected as InvalidParameter.
    /// Derivation: 0.0 <= 0.0 is true, so the guard fires and the code returns
    /// Err(Error::invalid_parameter("learning_rate", ...)) -> InvalidParameter.
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

    /// A negative learning rate must be rejected as InvalidParameter.
    /// Derivation: -1.0 <= 0.0 is true -> guard fires -> InvalidParameter.
    #[test]
    fn validate_learning_rate_negative_gives_invalid_parameter() {
        let err = validate_learning_rate(-1.0).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "learning_rate"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// A NaN learning rate must be rejected as InvalidParameter.
    /// Derivation: NaN <= 0.0 is false, but !NaN.is_finite() is true, so the
    /// second half of the guard fires -> InvalidParameter.
    #[test]
    fn validate_learning_rate_nan_gives_invalid_parameter() {
        let err = validate_learning_rate(f64::NAN).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "learning_rate"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// An infinite learning rate must be rejected as InvalidParameter.
    /// Derivation: INFINITY <= 0.0 is false, but !INFINITY.is_finite() is true,
    /// so the second half of the guard fires -> InvalidParameter.
    #[test]
    fn validate_learning_rate_inf_gives_invalid_parameter() {
        let err = validate_learning_rate(f64::INFINITY).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "learning_rate"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // --- validate_max_iterations ---

    /// max_iterations >= 1 must return Ok(()).
    /// Derivation: 1 == 0 is false, so the guard does not fire; Ok(()) is returned.
    #[test]
    fn validate_max_iterations_one_gives_ok() {
        assert!(validate_max_iterations(1).is_ok());
    }

    /// max_iterations == 0 must be rejected as InvalidParameter.
    /// Derivation: 0 == 0 is true -> guard fires -> Err(Error::invalid_parameter(
    /// "max_iterations", ...)) -> InvalidParameter.
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

    // --- validate_tolerance ---

    /// A positive finite tolerance must return Ok(()).
    /// Derivation: 1e-4 > 0.0 and finite, so the guard (`<= 0.0 || !is_finite()`)
    /// does not fire; Ok(()) is returned.
    #[test]
    fn validate_tolerance_positive_gives_ok() {
        assert!(validate_tolerance(1e-4).is_ok());
    }

    /// tolerance == 0.0 must be rejected as InvalidParameter.
    /// Derivation: 0.0 <= 0.0 is true -> guard fires -> InvalidParameter.
    #[test]
    fn validate_tolerance_zero_gives_invalid_parameter() {
        let err = validate_tolerance(0.0).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// A negative tolerance must be rejected as InvalidParameter.
    /// Derivation: -0.5 <= 0.0 is true -> guard fires -> InvalidParameter.
    #[test]
    fn validate_tolerance_negative_gives_invalid_parameter() {
        let err = validate_tolerance(-0.5).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// A NaN tolerance must be rejected as InvalidParameter.
    /// Derivation: NaN <= 0.0 is false, but !NaN.is_finite() is true -> guard fires.
    #[test]
    fn validate_tolerance_nan_gives_invalid_parameter() {
        let err = validate_tolerance(f64::NAN).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// An infinite tolerance must be rejected as InvalidParameter.
    /// Derivation: INFINITY <= 0.0 is false, but !INFINITY.is_finite() is true ->
    /// guard fires -> InvalidParameter.
    #[test]
    fn validate_tolerance_inf_gives_invalid_parameter() {
        let err = validate_tolerance(f64::INFINITY).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "tolerance"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // --- validate_regularization_type ---

    /// None (no regularization) must always return Ok(()).
    /// Derivation: the `if let Some(reg)` block is skipped when reg_type is None;
    /// function falls through to Ok(()).
    #[test]
    fn validate_regularization_type_none_gives_ok() {
        assert!(validate_regularization_type(None).is_ok());
    }

    /// A non-negative finite L1 alpha must return Ok(()).
    /// Derivation: 0.0 < 0.0 is false and 0.0 is finite, so the inner guard does
    /// not fire for the L1 arm; Ok(()) is returned.
    #[test]
    fn validate_regularization_type_l1_zero_gives_ok() {
        assert!(validate_regularization_type(Some(RegularizationType::L1(0.0))).is_ok());
    }

    /// A positive finite L2 alpha must return Ok(()).
    /// Derivation: 0.5 < 0.0 is false and 0.5 is finite, so the inner guard does
    /// not fire for the L2 arm; Ok(()) is returned.
    #[test]
    fn validate_regularization_type_l2_positive_gives_ok() {
        assert!(validate_regularization_type(Some(RegularizationType::L2(0.5))).is_ok());
    }

    /// A negative L1 alpha must be rejected as InvalidParameter.
    /// Derivation: for the L1 arm, -1.0 < 0.0 is true -> inner guard fires ->
    /// Err(Error::invalid_parameter("alpha", ...)) -> InvalidParameter.
    #[test]
    fn validate_regularization_type_l1_negative_gives_invalid_parameter() {
        let err = validate_regularization_type(Some(RegularizationType::L1(-1.0))).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// A NaN L1 alpha must be rejected as InvalidParameter.
    /// Derivation: for the L1 arm specifically, NaN < 0.0 is false, but
    /// !NaN.is_finite() is true -> inner guard fires -> InvalidParameter.
    #[test]
    fn validate_regularization_type_l1_nan_gives_invalid_parameter() {
        let err = validate_regularization_type(Some(RegularizationType::L1(f64::NAN))).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// An infinite L1 alpha must be rejected as InvalidParameter.
    /// Derivation: for the L1 arm, INFINITY < 0.0 is false, but
    /// !INFINITY.is_finite() is true -> inner guard fires -> InvalidParameter.
    #[test]
    fn validate_regularization_type_l1_inf_gives_invalid_parameter() {
        let err =
            validate_regularization_type(Some(RegularizationType::L1(f64::INFINITY))).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    // --- check_is_fitted ---

    /// is_fitted == true must return Ok(()).
    /// Derivation: the `if is_fitted` branch is taken, returning Ok(()).
    #[test]
    fn check_is_fitted_true_gives_ok() {
        assert!(check_is_fitted(true, "SomeModel").is_ok());
    }

    /// is_fitted == false must return NotFitted carrying the model name.
    /// Derivation: the else branch returns Err(Error::not_fitted("SomeModel"))
    /// -> Error::NotFitted("SomeModel").
    #[test]
    fn check_is_fitted_false_gives_not_fitted() {
        let err = check_is_fitted(false, "SomeModel").unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "SomeModel"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }
}
