use super::*;

/// Validates that the input shape has the expected number of dimensions.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if the shape length doesn't match expected_dims.
pub fn validate_input_shape_dims(
    input_shape: &[usize],
    expected_dims: usize,
    layer_name: &str,
) -> Result<(), ModelError> {
    if input_shape.len() != expected_dims {
        let dim_names = match expected_dims {
            3 => "[batch_size, channels, length]",
            4 => "[batch_size, channels, height, width]",
            5 => "[batch_size, channels, depth, height, width]",
            _ => "proper dimensions",
        };
        return Err(ModelError::InputValidationError(format!(
            "Input shape must be {}-dimensional: {} for {}",
            expected_dims, dim_names, layer_name
        )));
    }
    Ok(())
}

/// Validates that all dimensions in input_shape are greater than zero.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any dimension is 0.
pub fn validate_all_dims_positive(input_shape: &[usize]) -> Result<(), ModelError> {
    if !input_shape.iter().all(|&dim| dim > 0) {
        return Err(ModelError::InputValidationError(format!(
            "All dimensions in input_shape must be greater than zero. Got: {:?}",
            input_shape
        )));
    }
    Ok(())
}

/// Validates pool size for 1D pooling.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if:
/// - pool_size is 0
/// - pool_size is greater than input length
pub fn validate_pool_size_1d(pool_size: usize, input_length: usize) -> Result<(), ModelError> {
    if pool_size == 0 {
        return Err(ModelError::InputValidationError(
            "pool_size must be greater than zero".to_string(),
        ));
    }
    if pool_size > input_length {
        return Err(ModelError::InputValidationError(format!(
            "pool_size ({}) cannot be greater than input length ({})",
            pool_size, input_length
        )));
    }
    Ok(())
}

/// Validates pool size for 2D pooling.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any dimension is 0.
pub fn validate_pool_size_2d(pool_size: (usize, usize)) -> Result<(), ModelError> {
    if pool_size.0 == 0 || pool_size.1 == 0 {
        return Err(ModelError::InputValidationError(
            "Pool size must be greater than zero in all dimensions".to_string(),
        ));
    }
    Ok(())
}

/// Validates pool size for 3D pooling.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any dimension is 0.
pub fn validate_pool_size_3d(pool_size: (usize, usize, usize)) -> Result<(), ModelError> {
    if pool_size.0 == 0 || pool_size.1 == 0 || pool_size.2 == 0 {
        return Err(ModelError::InputValidationError(
            "Pool size dimensions must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

/// Validates stride for 1D pooling.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if stride is 0.
pub fn validate_stride_1d(stride: usize) -> Result<(), ModelError> {
    if stride == 0 {
        return Err(ModelError::InputValidationError(
            "stride must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

/// Validates strides for 2D pooling.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any stride is 0.
pub fn validate_strides_2d(strides: (usize, usize)) -> Result<(), ModelError> {
    if strides.0 == 0 || strides.1 == 0 {
        return Err(ModelError::InputValidationError(
            "Strides must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

/// Validates strides for 3D pooling.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any stride is 0.
pub fn validate_strides_3d(strides: (usize, usize, usize)) -> Result<(), ModelError> {
    if strides.0 == 0 || strides.1 == 0 || strides.2 == 0 {
        return Err(ModelError::InputValidationError(
            "Stride dimensions must be greater than zero".to_string(),
        ));
    }
    Ok(())
}
