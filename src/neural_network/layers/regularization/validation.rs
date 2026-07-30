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

/// Validates that an input's shape matches the declared one, ignoring the batch axis
///
/// The declared `expected_shape` includes a leading batch size (Keras calls it `batch_shape`,
/// not `input_shape`). That axis is not fixed for a layer.
/// [`fit_with_batches`](crate::neural_network::sequential::Sequential::fit_with_batches) feeds
/// `batch_size` rows per step, plus a shorter final chunk when the dataset does not divide evenly.
/// `predict` may also get any number of samples. This function checks only the rank and the
/// per-sample axes, and never axis 0.
///
/// An empty `expected_shape` disables the check entirely.
pub(super) fn validate_input_shape(
    input_shape: &[usize],
    expected_shape: &[usize],
) -> Result<(), Error> {
    if expected_shape.is_empty() {
        return Ok(());
    }
    if input_shape.len() != expected_shape.len() {
        return Err(Error::shape_mismatch(expected_shape, input_shape));
    }
    if input_shape[1..] != expected_shape[1..] {
        // Substitute the actual batch size into the reported shape. The printed difference then
        // shows only the axes that truly disagree, not the unenforced batch size.
        let mut reported = expected_shape.to_vec();
        reported[0] = input_shape[0];
        return Err(Error::shape_mismatch(reported, input_shape));
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
    // A non-finite stddev (NaN or infinity) would reach `Normal::new(..).unwrap()` in the forward
    // pass. That call panics, so this function rejects it up front.
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
