use super::*;

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
/// - `Ok(())` - If all validation checks pass
/// - `Err(ModelError::InputValidationError)` - If any validation check fails, with an informative error message
pub fn preliminary_check<S>(
    x: &ArrayBase<S, Ix2>,
    y: Option<&ArrayBase<S, Ix1>>,
) -> Result<(), ModelError>
where
    S: Data<Elem = f64>,
{
    if x.nrows() == 0 {
        return Err(ModelError::InputValidationError(
            "Input data is empty".to_string(),
        ));
    }

    for (i, row) in x.outer_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(ModelError::InputValidationError(format!(
                    "Input data contains NaN or infinite value at position [{}][{}]",
                    i, j
                )));
            }
        }
    }

    if let Some(y) = y {
        if y.len() == 0 {
            return Err(ModelError::InputValidationError(
                "Target vector is empty".to_string(),
            ));
        }

        if y.len() != x.nrows() {
            return Err(ModelError::InputValidationError(format!(
                "Input data and target vector have different lengths, x columns: {}, y length: {}",
                x.nrows(),
                y.len()
            )));
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
/// * `learning_rate` - The learning rate value to validate
///
/// # Returns
///
/// - `Ok(())` - If the learning rate is valid (positive and finite)
/// - `Err(ModelError::InputValidationError)` - If the learning rate is invalid (non-positive, NaN, or infinite)
pub fn validate_learning_rate(learning_rate: f64) -> Result<(), ModelError> {
    if learning_rate <= 0.0 || !learning_rate.is_finite() {
        return Err(ModelError::InputValidationError(format!(
            "learning_rate must be positive and finite, got {}",
            learning_rate
        )));
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
/// * `max_iterations` - The maximum number of iterations to validate
///
/// # Returns
///
/// - `Ok(())` - If the maximum iterations value is valid (greater than 0)
/// - `Err(ModelError::InputValidationError)` - If the maximum iterations value is 0
pub fn validate_max_iterations(max_iterations: usize) -> Result<(), ModelError> {
    if max_iterations == 0 {
        return Err(ModelError::InputValidationError(
            "max_iterations must be greater than 0".to_string(),
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
/// * `tolerance` - The convergence tolerance value to validate
///
/// # Returns
///
/// - `Ok(())` - If the tolerance is valid (positive and finite)
/// - `Err(ModelError::InputValidationError)` - If the tolerance is invalid (non-positive, NaN, or infinite)
pub fn validate_tolerance(tolerance: f64) -> Result<(), ModelError> {
    if tolerance <= 0.0 || !tolerance.is_finite() {
        return Err(ModelError::InputValidationError(format!(
            "tolerance must be positive and finite, got {}",
            tolerance
        )));
    }

    Ok(())
}

/// Validates the regularization type and its associated parameters.
///
/// This function checks that regularization parameters are properly configured:
/// - For L1 and L2 regularization, the alpha (regularization strength) parameter
///   must be non-negative and finite
/// - If alpha is 0, a warning is printed suggesting to use None instead
/// - None (no regularization) is always valid
///
/// # Parameters
///
/// * `reg_type` - An optional regularization type with its strength parameter
///
/// # Returns
///
/// - `Ok(())` - If the regularization configuration is valid
/// - `Err(ModelError::InputValidationError)` - If the regularization alpha is negative, NaN, or infinite
///
/// # Side Effects
///
/// - Prints a warning to stderr if alpha is 0.0, recommending to use None instead
pub fn validate_regulation_type(reg_type: Option<RegularizationType>) -> Result<(), ModelError> {
    if let Some(reg) = &reg_type {
        match reg {
            RegularizationType::L1(alpha) | RegularizationType::L2(alpha) => {
                if *alpha < 0.0 || !alpha.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Regularization alpha must be non-negative and finite, got {}",
                        alpha
                    )));
                }
                if *alpha == 0.0 {
                    eprintln!("Warning: regularization alpha is 0, consider using None instead");
                }
            }
        }
    }

    Ok(())
}
