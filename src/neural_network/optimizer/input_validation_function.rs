use super::*;

/// Validates that the learning rate is positive.
///
/// # Parameters
///
/// * `learning_rate` - The learning rate value to validate
///
/// # Returns
///
/// - `Ok(())` if the learning rate is positive
/// - `Err(ModelError::InputValidationError)` if the learning rate is not positive
pub(super) fn validate_learning_rate(learning_rate: f32) -> Result<(), ModelError> {
    if learning_rate <= 0.0 {
        return Err(ModelError::InputValidationError(format!(
            "learning_rate must be positive, got {}",
            learning_rate
        )));
    }
    Ok(())
}

/// Validates that a decay rate (beta or rho) is in the range [0, 1).
///
/// # Parameters
///
/// - `value` - The decay rate value to validate
/// - `param_name` - The name of the parameter (for error messages)
///
/// # Returns
///
/// - `Ok(())` if the value is in the valid range
/// - `Err(ModelError::InputValidationError)` if the value is out of range
pub(super) fn validate_decay_rate(value: f32, param_name: &str) -> Result<(), ModelError> {
    if !(0.0..1.0).contains(&value) {
        return Err(ModelError::InputValidationError(format!(
            "{} must be in range [0, 1), got {}",
            param_name, value
        )));
    }
    Ok(())
}

/// Validates that epsilon is positive.
///
/// # Parameters
///
/// - `epsilon` - The epsilon value to validate
///
/// # Returns
///
/// - `Ok(())` if epsilon is positive
/// - `Err(ModelError::InputValidationError)` if epsilon is not positive
pub(super) fn validate_epsilon(epsilon: f32) -> Result<(), ModelError> {
    if epsilon <= 0.0 {
        return Err(ModelError::InputValidationError(format!(
            "epsilon must be positive, got {}",
            epsilon
        )));
    }
    Ok(())
}

/// Validates that a value is positive and finite.
///
/// # Parameters
///
/// - `value` - The value to validate
/// - `param_name` - The name of the parameter (for error messages)
///
/// # Returns
///
/// - `Ok(())` if the value is positive and finite
/// - `Err(ModelError::InputValidationError)` if the value is not positive or not finite
pub(super) fn validate_positive_finite(value: f32, param_name: &str) -> Result<(), ModelError> {
    if !(value > 0.0 && value.is_finite()) {
        return Err(ModelError::InputValidationError(format!(
            "{} must be positive and finite, got {}",
            param_name, value
        )));
    }
    Ok(())
}
