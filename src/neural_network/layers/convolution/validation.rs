//! Shared parameter and input-shape validators for the convolution layers

use crate::error::Error;

/// Validates the filters parameter
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if filters is 0
pub(super) fn validate_filters(filters: usize) -> Result<(), Error> {
    if filters == 0 {
        return Err(Error::invalid_parameter(
            "filters",
            "Number of filters must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates kernel size for 1D convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if kernel_size is 0
pub(super) fn validate_kernel_size_1d(kernel_size: usize) -> Result<(), Error> {
    if kernel_size == 0 {
        return Err(Error::invalid_parameter(
            "kernel_size",
            "Kernel size must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates kernel size for 2D convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if any dimension is 0
pub(super) fn validate_kernel_size_2d(kernel_size: (usize, usize)) -> Result<(), Error> {
    if kernel_size.0 == 0 || kernel_size.1 == 0 {
        return Err(Error::invalid_parameter(
            "kernel_size",
            "Kernel dimensions must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates kernel size for 3D convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if any dimension is 0
pub(super) fn validate_kernel_size_3d(kernel_size: (usize, usize, usize)) -> Result<(), Error> {
    if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
        return Err(Error::invalid_parameter(
            "kernel_size",
            "Kernel dimensions must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates strides for 1D convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if stride is 0
pub(super) fn validate_strides_1d(stride: usize) -> Result<(), Error> {
    if stride == 0 {
        return Err(Error::invalid_parameter(
            "stride",
            "Stride must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates strides for 2D convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if any stride is 0
pub(super) fn validate_strides_2d(strides: (usize, usize)) -> Result<(), Error> {
    if strides.0 == 0 || strides.1 == 0 {
        return Err(Error::invalid_parameter(
            "strides",
            "Strides must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates strides for 3D convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if any stride is 0
pub(super) fn validate_strides_3d(strides: (usize, usize, usize)) -> Result<(), Error> {
    if strides.0 == 0 || strides.1 == 0 || strides.2 == 0 {
        return Err(Error::invalid_parameter(
            "strides",
            "Strides must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates input shape for 1D convolution
///
/// # Errors
///
/// Returns `Error::InvalidInput` if:
/// - Shape is not 3D
/// - Input channels is 0
/// - Input length is less than kernel size
pub(super) fn validate_input_shape_1d(
    input_shape: &[usize],
    kernel_size: usize,
) -> Result<(), Error> {
    if input_shape.len() != 3 {
        return Err(Error::invalid_input(
            "Input shape must be 3D: [batch_size, channels, length]",
        ));
    }
    if input_shape[1] == 0 {
        return Err(Error::invalid_input(
            "Number of input channels must be greater than 0",
        ));
    }
    if input_shape[2] < kernel_size {
        return Err(Error::invalid_input(
            "Input length must be at least as large as the kernel size",
        ));
    }
    Ok(())
}

/// Validates input shape for 2D convolution
///
/// # Errors
///
/// Returns `Error::InvalidInput` if:
/// - Shape is not 4D
/// - Input channels is 0
/// - Input dimensions are less than kernel size
pub(super) fn validate_input_shape_2d(
    input_shape: &[usize],
    kernel_size: (usize, usize),
) -> Result<(), Error> {
    if input_shape.len() != 4 {
        return Err(Error::invalid_input(
            "Input shape must be 4D: [batch_size, channels, height, width]",
        ));
    }
    if input_shape[1] == 0 {
        return Err(Error::invalid_input(
            "Number of input channels must be greater than 0",
        ));
    }
    if input_shape[2] < kernel_size.0 || input_shape[3] < kernel_size.1 {
        return Err(Error::invalid_input(
            "Input dimensions must be at least as large as the kernel size",
        ));
    }
    Ok(())
}

/// Validates input shape for 3D convolution
///
/// # Errors
///
/// Returns `Error::InvalidInput` if:
/// - Shape is not 5D
/// - Any dimension is 0
pub(super) fn validate_input_shape_3d(input_shape: &[usize]) -> Result<(), Error> {
    if input_shape.len() != 5 {
        return Err(Error::invalid_input(
            "Input shape must be 5-dimensional: [batch, channels, depth, height, width]",
        ));
    }
    if input_shape.contains(&0) {
        return Err(Error::invalid_input(
            "All input dimensions must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates depth multiplier for depthwise separable convolution
///
/// # Errors
///
/// Returns `Error::InvalidParameter` if depth_multiplier is 0
pub(super) fn validate_depth_multiplier(depth_multiplier: usize) -> Result<(), Error> {
    if depth_multiplier == 0 {
        return Err(Error::invalid_parameter(
            "depth_multiplier",
            "Depth multiplier must be greater than 0",
        ));
    }
    Ok(())
}
