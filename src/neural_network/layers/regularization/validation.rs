//! Shared parameter and shape validation helpers for regularization layers

use crate::error::Error;

/// Validates that a rate parameter is between 0.0 and 1.0 (inclusive)
pub(super) fn validate_rate(rate: f32, param_name: &str) -> Result<(), Error> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(Error::invalid_parameter(
            param_name.to_string(),
            format!("must be between 0 and 1, got {}", rate),
        ));
    }
    Ok(())
}

/// Validates that a rate parameter is between 0.0 and 1.0 (exclusive of 1.0)
pub(super) fn validate_rate_exclusive(rate: f32, param_name: &str) -> Result<(), Error> {
    if !(0.0..1.0).contains(&rate) {
        return Err(Error::invalid_parameter(
            param_name.to_string(),
            format!("must be in range [0, 1), got {}", rate),
        ));
    }
    Ok(())
}

/// Validates that input shape matches expected shape
pub(super) fn validate_input_shape(
    input_shape: &[usize],
    expected_shape: &[usize],
) -> Result<(), Error> {
    if !expected_shape.is_empty() && input_shape != expected_shape {
        return Err(Error::shape_mismatch(expected_shape, input_shape));
    }
    Ok(())
}

/// Validates that input has the expected number of dimensions
pub(super) fn validate_input_ndim(
    input_ndim: usize,
    expected_ndim: usize,
    layer_name: &str,
) -> Result<(), Error> {
    if input_ndim != expected_ndim {
        return Err(Error::invalid_input(format!(
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
) -> Result<(), Error> {
    if input_ndim < min_ndim {
        return Err(Error::invalid_input(format!(
            "{} expects at least {}D input, got {}D",
            layer_name, min_ndim, input_ndim
        )));
    }
    Ok(())
}

/// Validates that a standard deviation parameter is non-negative and finite
pub(super) fn validate_stddev(stddev: f32) -> Result<(), Error> {
    if stddev < 0.0 {
        return Err(Error::invalid_parameter(
            "stddev",
            "Standard deviation cannot be negative",
        ));
    }
    // A non-finite stddev (NaN / +inf) would otherwise reach `Normal::new(..).unwrap()` in the forward pass and panic, so reject it up front
    if !stddev.is_finite() {
        return Err(Error::invalid_parameter(
            "stddev",
            format!("Standard deviation must be finite, got {}", stddev),
        ));
    }
    Ok(())
}

/// Validates that epsilon is positive and finite
pub(super) fn validate_epsilon(epsilon: f32) -> Result<(), Error> {
    if epsilon <= 0.0 {
        return Err(Error::invalid_parameter(
            "epsilon",
            format!("Epsilon must be positive, got {}", epsilon),
        ));
    }
    if !epsilon.is_finite() {
        return Err(Error::invalid_parameter(
            "epsilon",
            format!("Epsilon must be finite, got {}", epsilon),
        ));
    }
    Ok(())
}

/// Validates that momentum is between 0.0 and 1.0 (inclusive)
pub(super) fn validate_momentum(momentum: f32) -> Result<(), Error> {
    if !(0.0..=1.0).contains(&momentum) {
        return Err(Error::invalid_parameter(
            "momentum",
            format!("Momentum must be between 0.0 and 1.0, got {}", momentum),
        ));
    }
    Ok(())
}

/// Validates that channel axis is valid (not 0, within bounds)
pub(super) fn validate_channel_axis(channel_axis: usize, input_ndim: usize) -> Result<(), Error> {
    if channel_axis == 0 {
        return Err(Error::invalid_parameter(
            "channel_axis",
            "Channel axis cannot be 0 (batch axis)",
        ));
    }
    if channel_axis >= input_ndim {
        return Err(Error::invalid_parameter(
            "channel_axis",
            format!(
                "Channel axis {} is out of bounds for input with {} dimensions",
                channel_axis, input_ndim
            ),
        ));
    }
    Ok(())
}

/// Validates that num_groups divides num_channels evenly
pub(super) fn validate_num_groups(num_channels: usize, num_groups: usize) -> Result<(), Error> {
    if !num_channels.is_multiple_of(num_groups) {
        return Err(Error::invalid_parameter(
            "num_groups",
            format!(
                "Number of channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            ),
        ));
    }
    Ok(())
}

/// Validates that input_shape is not empty
pub(super) fn validate_input_shape_not_empty(input_shape: &[usize]) -> Result<(), Error> {
    if input_shape.is_empty() {
        return Err(Error::empty_input("input shape"));
    }
    Ok(())
}

/// Validates that num_groups is greater than 0
pub(super) fn validate_num_groups_positive(num_groups: usize) -> Result<(), Error> {
    if num_groups == 0 {
        return Err(Error::invalid_parameter(
            "num_groups",
            "Number of groups must be greater than 0",
        ));
    }
    Ok(())
}
