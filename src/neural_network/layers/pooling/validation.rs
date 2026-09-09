//! Shape and parameter validation helpers shared by the pooling layers

use crate::error::Error;

/// Validates pool size for 1D pooling
///
/// # Parameters
///
/// - `pool_size` - the size of the pooling window
/// - `input_length` - the input length along the pooled axis
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if pool_size is 0, or if it exceeds the input length
/// (which would underflow the output-shape calculation)
pub(super) fn validate_pool_size_1d(pool_size: usize, input_length: usize) -> Result<(), Error> {
    if pool_size == 0 {
        return Err(Error::invalid_parameter(
            "pool_size",
            "must be greater than zero",
        ));
    }
    if pool_size > input_length {
        return Err(Error::invalid_parameter(
            "pool_size",
            "cannot exceed the corresponding input dimension",
        ));
    }
    Ok(())
}

/// Validates pool size for 2D pooling
///
/// # Parameters
///
/// - `pool_size` - the pooling window size as (height, width)
/// - `input_height` - the input height
/// - `input_width` - the input width
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any dimension is 0, or if a pool dimension exceeds
/// the corresponding input dimension (which would underflow the output-shape calculation)
pub(super) fn validate_pool_size_2d(
    pool_size: (usize, usize),
    input_height: usize,
    input_width: usize,
) -> Result<(), Error> {
    if pool_size.0 == 0 || pool_size.1 == 0 {
        return Err(Error::invalid_parameter(
            "pool_size",
            "must be greater than zero in all dimensions",
        ));
    }
    if pool_size.0 > input_height || pool_size.1 > input_width {
        return Err(Error::invalid_parameter(
            "pool_size",
            "cannot exceed the corresponding input dimension",
        ));
    }
    Ok(())
}

/// Validates pool size for 3D pooling
///
/// # Parameters
///
/// - `pool_size` - the pooling window size as (depth, height, width)
/// - `input_depth` - the input depth
/// - `input_height` - the input height
/// - `input_width` - the input width
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any dimension is 0, or if a pool dimension exceeds
/// the corresponding input dimension (which would underflow the output-shape calculation)
pub(super) fn validate_pool_size_3d(
    pool_size: (usize, usize, usize),
    input_depth: usize,
    input_height: usize,
    input_width: usize,
) -> Result<(), Error> {
    if pool_size.0 == 0 || pool_size.1 == 0 || pool_size.2 == 0 {
        return Err(Error::invalid_parameter(
            "pool_size",
            "must be greater than zero in all dimensions",
        ));
    }
    if pool_size.0 > input_depth || pool_size.1 > input_height || pool_size.2 > input_width {
        return Err(Error::invalid_parameter(
            "pool_size",
            "cannot exceed the corresponding input dimension",
        ));
    }
    Ok(())
}

/// Validates stride for 1D pooling
///
/// # Parameters
///
/// - `stride` - the step size for the pooling operation
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if stride is 0
pub(super) fn validate_stride_1d(stride: usize) -> Result<(), Error> {
    if stride == 0 {
        return Err(Error::invalid_parameter(
            "stride",
            "must be greater than zero",
        ));
    }
    Ok(())
}

/// Validates strides for 2D pooling
///
/// # Parameters
///
/// - `strides` - the step sizes as (height_step, width_step)
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any stride is 0
pub(super) fn validate_strides_2d(strides: (usize, usize)) -> Result<(), Error> {
    if strides.0 == 0 || strides.1 == 0 {
        return Err(Error::invalid_parameter(
            "strides",
            "must be greater than zero in all dimensions",
        ));
    }
    Ok(())
}

/// Validates strides for 3D pooling
///
/// # Parameters
///
/// - `strides` - the step sizes as (depth_step, height_step, width_step)
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any stride is 0
pub(super) fn validate_strides_3d(strides: (usize, usize, usize)) -> Result<(), Error> {
    if strides.0 == 0 || strides.1 == 0 || strides.2 == 0 {
        return Err(Error::invalid_parameter(
            "strides",
            "must be greater than zero in all dimensions",
        ));
    }
    Ok(())
}

/// Unit tests for the validation helpers
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// A 3D pool whose height alone exceeds the input height yields InvalidParameter(pool_size)
    #[test]
    fn test_validate_pool_size_3d_oversized_height() {
        let result = validate_pool_size_3d((2, 5, 2), 4, 4, 4);
        assert!(
            matches!(result, Err(Error::InvalidParameter { ref name, .. }) if name == "pool_size"),
            "oversized height pool must yield InvalidParameter(pool_size), got {:?}",
            result
        );
    }

    /// A 3D pool whose width alone exceeds the input width yields InvalidParameter(pool_size)
    #[test]
    fn test_validate_pool_size_3d_oversized_width() {
        let result = validate_pool_size_3d((2, 2, 5), 4, 4, 4);
        assert!(
            matches!(result, Err(Error::InvalidParameter { ref name, .. }) if name == "pool_size"),
            "oversized width pool must yield InvalidParameter(pool_size), got {:?}",
            result
        );
    }

    /// A 3D pool that fits on every axis succeeds
    #[test]
    fn test_validate_pool_size_3d_all_fitting_ok() {
        assert!(validate_pool_size_3d((2, 2, 2), 4, 4, 4).is_ok());
    }
}
