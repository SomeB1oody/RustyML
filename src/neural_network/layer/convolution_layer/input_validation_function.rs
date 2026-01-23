use super::*;

/// Validates the filters parameter.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if filters is 0.
pub(super) fn validate_filters(filters: usize) -> Result<(), ModelError> {
    if filters == 0 {
        return Err(ModelError::InputValidationError(
            "Number of filters must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates kernel size for 1D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if kernel_size is 0.
pub(super) fn validate_kernel_size_1d(kernel_size: usize) -> Result<(), ModelError> {
    if kernel_size == 0 {
        return Err(ModelError::InputValidationError(
            "Kernel size must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates kernel size for 2D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any dimension is 0.
pub(super) fn validate_kernel_size_2d(kernel_size: (usize, usize)) -> Result<(), ModelError> {
    if kernel_size.0 == 0 || kernel_size.1 == 0 {
        return Err(ModelError::InputValidationError(
            "Kernel dimensions must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates kernel size for 3D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any dimension is 0.
pub(super) fn validate_kernel_size_3d(
    kernel_size: (usize, usize, usize),
) -> Result<(), ModelError> {
    if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
        return Err(ModelError::InputValidationError(
            "Kernel dimensions must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates strides for 1D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if stride is 0.
pub(super) fn validate_strides_1d(stride: usize) -> Result<(), ModelError> {
    if stride == 0 {
        return Err(ModelError::InputValidationError(
            "Stride must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates strides for 2D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any stride is 0.
pub(super) fn validate_strides_2d(strides: (usize, usize)) -> Result<(), ModelError> {
    if strides.0 == 0 || strides.1 == 0 {
        return Err(ModelError::InputValidationError(
            "Strides must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates strides for 3D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if any stride is 0.
pub(super) fn validate_strides_3d(strides: (usize, usize, usize)) -> Result<(), ModelError> {
    if strides.0 == 0 || strides.1 == 0 || strides.2 == 0 {
        return Err(ModelError::InputValidationError(
            "Strides must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates input shape for 1D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if:
/// - Shape is not 3D
/// - Input channels is 0
/// - Input length is less than kernel size
pub(super) fn validate_input_shape_1d(
    input_shape: &[usize],
    kernel_size: usize,
) -> Result<(), ModelError> {
    if input_shape.len() != 3 {
        return Err(ModelError::InputValidationError(
            "Input shape must be 3D: [batch_size, channels, length]".to_string(),
        ));
    }
    if input_shape[1] == 0 {
        return Err(ModelError::InputValidationError(
            "Number of input channels must be greater than 0".to_string(),
        ));
    }
    if input_shape[2] < kernel_size {
        return Err(ModelError::InputValidationError(
            "Input length must be at least as large as the kernel size".to_string(),
        ));
    }
    Ok(())
}

/// Validates input shape for 2D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if:
/// - Shape is not 4D
/// - Input channels is 0
/// - Input dimensions are less than kernel size
pub(super) fn validate_input_shape_2d(
    input_shape: &[usize],
    kernel_size: (usize, usize),
) -> Result<(), ModelError> {
    if input_shape.len() != 4 {
        return Err(ModelError::InputValidationError(
            "Input shape must be 4D: [batch_size, channels, height, width]".to_string(),
        ));
    }
    if input_shape[1] == 0 {
        return Err(ModelError::InputValidationError(
            "Number of input channels must be greater than 0".to_string(),
        ));
    }
    if input_shape[2] < kernel_size.0 || input_shape[3] < kernel_size.1 {
        return Err(ModelError::InputValidationError(
            "Input dimensions must be at least as large as the kernel size".to_string(),
        ));
    }
    Ok(())
}

/// Validates input shape for 3D convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if:
/// - Shape is not 5D
/// - Any dimension is 0
pub(super) fn validate_input_shape_3d(input_shape: &[usize]) -> Result<(), ModelError> {
    if input_shape.len() != 5 {
        return Err(ModelError::InputValidationError(
            "Input shape must be 5-dimensional: [batch, channels, depth, height, width]"
                .to_string(),
        ));
    }
    if input_shape.iter().any(|&dim| dim == 0) {
        return Err(ModelError::InputValidationError(
            "All input dimensions must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

/// Validates depth multiplier for depthwise separable convolution.
///
/// # Errors
///
/// Returns `ModelError::InputValidationError` if depth_multiplier is 0.
pub(super) fn validate_depth_multiplier(depth_multiplier: usize) -> Result<(), ModelError> {
    if depth_multiplier == 0 {
        return Err(ModelError::InputValidationError(
            "Depth multiplier must be greater than 0".to_string(),
        ));
    }
    Ok(())
}
