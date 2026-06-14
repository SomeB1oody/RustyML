//! Shared parameter validators for optimizer constructors

use crate::error::Error;

/// Validates that the learning rate is positive and finite
///
/// # Parameters
///
/// - `learning_rate` - The learning rate value to validate
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if the learning rate is not positive or not finite
pub(super) fn validate_learning_rate(learning_rate: f32) -> Result<(), Error> {
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        return Err(Error::invalid_parameter(
            "learning_rate",
            format!("must be positive and finite, got {}", learning_rate),
        ));
    }
    Ok(())
}

/// Validates that a decay rate (beta or rho) is in the range [0, 1) and finite
///
/// # Parameters
///
/// - `value` - The decay rate value to validate
/// - `param_name` - The name of the parameter (for error messages)
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if the value is out of range or not finite
pub(super) fn validate_decay_rate(value: f32, param_name: &str) -> Result<(), Error> {
    if !((0.0..1.0).contains(&value) && value.is_finite()) {
        return Err(Error::invalid_parameter(
            param_name.to_string(),
            format!("must be in range [0, 1) and finite, got {}", value),
        ));
    }
    Ok(())
}

/// Validates that epsilon is positive and finite
///
/// # Parameters
///
/// - `epsilon` - The epsilon value to validate
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if epsilon is not positive or not finite
pub(super) fn validate_epsilon(epsilon: f32) -> Result<(), Error> {
    if !(epsilon > 0.0 && epsilon.is_finite()) {
        return Err(Error::invalid_parameter(
            "epsilon",
            format!("must be positive and finite, got {}", epsilon),
        ));
    }
    Ok(())
}

/// Validates that a value is positive and finite
///
/// # Parameters
///
/// - `value` - The value to validate
/// - `param_name` - The name of the parameter (for error messages)
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if the value is not positive or not finite
pub(super) fn validate_positive_finite(value: f32, param_name: &str) -> Result<(), Error> {
    if !(value > 0.0 && value.is_finite()) {
        return Err(Error::invalid_parameter(
            param_name.to_string(),
            format!("must be positive and finite, got {}", value),
        ));
    }
    Ok(())
}

/// Validates that a value is non-negative and finite (e.g. `momentum`, `weight_decay`, where 0
/// disables the feature)
///
/// # Parameters
///
/// - `value` - The value to validate
/// - `param_name` - The name of the parameter (for error messages)
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if the value is negative or not finite
pub(super) fn validate_non_negative_finite(value: f32, param_name: &str) -> Result<(), Error> {
    if !(value >= 0.0 && value.is_finite()) {
        return Err(Error::invalid_parameter(
            param_name.to_string(),
            format!("must be non-negative and finite, got {}", value),
        ));
    }
    Ok(())
}

/// Validates an optional clip-by-global-norm threshold
///
/// `None` (clipping disabled) is always valid; a `Some(max_norm)` must be positive and finite
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if `clip_norm` is `Some` value that is not positive and finite
pub(super) fn validate_clip_norm(clip_norm: Option<f32>) -> Result<(), Error> {
    if let Some(max_norm) = clip_norm {
        validate_positive_finite(max_norm, "clip_norm")?;
    }
    Ok(())
}
