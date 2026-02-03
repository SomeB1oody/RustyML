use crate::error::ModelError;

/// Validates that a rate parameter is between 0.0 and 1.0 (inclusive)
pub(super) fn validate_rate(rate: f32, param_name: &str) -> Result<(), ModelError> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(ModelError::InputValidationError(format!(
            "{} must be between 0 and 1, got {}",
            param_name, rate
        )));
    }
    Ok(())
}

/// Validates that a rate parameter is between 0.0 and 1.0 (exclusive of 1.0)
pub(super) fn validate_rate_exclusive(rate: f32, param_name: &str) -> Result<(), ModelError> {
    if rate < 0.0 || rate >= 1.0 {
        return Err(ModelError::InputValidationError(format!(
            "{} must be in range [0, 1), got {}",
            param_name, rate
        )));
    }
    Ok(())
}

/// Validates that input shape matches expected shape
pub(super) fn validate_input_shape(
    input_shape: &[usize],
    expected_shape: &[usize],
) -> Result<(), ModelError> {
    if !expected_shape.is_empty() && input_shape != expected_shape {
        return Err(ModelError::InputValidationError(format!(
            "Input shape mismatch: expected {:?}, got {:?}",
            expected_shape, input_shape
        )));
    }
    Ok(())
}

/// Validates that input has the expected number of dimensions
pub(super) fn validate_input_ndim(
    input_ndim: usize,
    expected_ndim: usize,
    layer_name: &str,
) -> Result<(), ModelError> {
    if input_ndim != expected_ndim {
        return Err(ModelError::InputValidationError(format!(
            "{} expects {}D input, got {}D",
            layer_name, expected_ndim, input_ndim
        )));
    }
    Ok(())
}

/// Validates that input has at least the minimum number of dimensions
pub(super) fn validate_min_input_ndim(
    input_ndim: usize,
    min_ndim: usize,
    layer_name: &str,
) -> Result<(), ModelError> {
    if input_ndim < min_ndim {
        return Err(ModelError::InputValidationError(format!(
            "{} expects at least {}D input, got {}D",
            layer_name, min_ndim, input_ndim
        )));
    }
    Ok(())
}

/// Validates that a standard deviation parameter is non-negative
pub(super) fn validate_stddev(stddev: f32) -> Result<(), ModelError> {
    if stddev < 0.0 {
        return Err(ModelError::InputValidationError(
            "Standard deviation cannot be negative".to_string(),
        ));
    }
    Ok(())
}

/// Validates that epsilon is positive and finite
pub(super) fn validate_epsilon(epsilon: f32) -> Result<(), ModelError> {
    if epsilon <= 0.0 {
        return Err(ModelError::InputValidationError(format!(
            "Epsilon must be positive, got {}",
            epsilon
        )));
    }
    if !epsilon.is_finite() {
        return Err(ModelError::InputValidationError(format!(
            "Epsilon must be finite, got {}",
            epsilon
        )));
    }
    Ok(())
}

/// Validates that momentum is between 0.0 and 1.0 (inclusive)
pub(super) fn validate_momentum(momentum: f32) -> Result<(), ModelError> {
    if !(0.0..=1.0).contains(&momentum) {
        return Err(ModelError::InputValidationError(format!(
            "Momentum must be between 0.0 and 1.0, got {}",
            momentum
        )));
    }
    Ok(())
}

/// Validates that channel axis is valid (not 0, within bounds)
pub(super) fn validate_channel_axis(
    channel_axis: usize,
    input_ndim: usize,
) -> Result<(), ModelError> {
    if channel_axis == 0 {
        return Err(ModelError::InputValidationError(
            "Channel axis cannot be 0 (batch axis)".to_string(),
        ));
    }
    if channel_axis >= input_ndim {
        return Err(ModelError::InputValidationError(format!(
            "Channel axis {} is out of bounds for input with {} dimensions",
            channel_axis, input_ndim
        )));
    }
    Ok(())
}

/// Validates that num_groups divides num_channels evenly
pub(super) fn validate_num_groups(
    num_channels: usize,
    num_groups: usize,
) -> Result<(), ModelError> {
    if num_channels % num_groups != 0 {
        return Err(ModelError::InputValidationError(format!(
            "Number of channels ({}) must be divisible by num_groups ({})",
            num_channels, num_groups
        )));
    }
    Ok(())
}

/// Validates that input_shape is not empty
pub(super) fn validate_input_shape_not_empty(input_shape: &[usize]) -> Result<(), ModelError> {
    if input_shape.is_empty() {
        return Err(ModelError::InputValidationError(
            "Input shape cannot be empty".to_string(),
        ));
    }
    Ok(())
}

/// Validates that num_groups is greater than 0
pub(super) fn validate_num_groups_positive(num_groups: usize) -> Result<(), ModelError> {
    if num_groups == 0 {
        return Err(ModelError::InputValidationError(
            "Number of groups must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates channel axis for normalization layers (checks bounds and that it's not 0)
/// This is used during layer construction when we have input_shape
pub(super) fn validate_channel_axis_with_shape(
    channel_axis: usize,
    input_shape: &[usize],
) -> Result<(), ModelError> {
    if channel_axis >= input_shape.len() {
        return Err(ModelError::InputValidationError(format!(
            "Channel axis {} is out of bounds for input shape with {} dimensions",
            channel_axis,
            input_shape.len()
        )));
    }
    if channel_axis == 0 {
        return Err(ModelError::InputValidationError(
            "Channel axis cannot be 0 (batch axis)".to_string(),
        ));
    }
    Ok(())
}
