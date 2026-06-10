use crate::error::Error;

/// Validates that the input shape has the expected number of dimensions.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] if the shape length doesn't match expected_dims.
pub(super) fn validate_input_shape_dims(
    input_shape: &[usize],
    expected_dims: usize,
    _layer_name: &str,
) -> Result<(), Error> {
    if input_shape.len() != expected_dims {
        return Err(Error::dimension_mismatch(expected_dims, input_shape.len()));
    }
    Ok(())
}

/// Validates that all dimensions in input_shape are greater than zero.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if any dimension is 0.
pub(super) fn validate_all_dims_positive(input_shape: &[usize]) -> Result<(), Error> {
    if !input_shape.iter().all(|&dim| dim > 0) {
        return Err(Error::invalid_input(format!(
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
/// Returns [`Error::InvalidParameter`] if:
/// - pool_size is 0
/// - pool_size is greater than input length
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

/// Validates pool size for 2D pooling.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any dimension is 0, or if a pool dimension is
/// greater than the corresponding input dimension (which would underflow the output-shape
/// calculation).
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

/// Validates pool size for 3D pooling.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any dimension is 0, or if a pool dimension is
/// greater than the corresponding input dimension (which would underflow the output-shape
/// calculation).
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

/// Validates stride for 1D pooling.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if stride is 0.
pub(super) fn validate_stride_1d(stride: usize) -> Result<(), Error> {
    if stride == 0 {
        return Err(Error::invalid_parameter(
            "stride",
            "must be greater than zero",
        ));
    }
    Ok(())
}

/// Validates strides for 2D pooling.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any stride is 0.
pub(super) fn validate_strides_2d(strides: (usize, usize)) -> Result<(), Error> {
    if strides.0 == 0 || strides.1 == 0 {
        return Err(Error::invalid_parameter(
            "strides",
            "must be greater than zero in all dimensions",
        ));
    }
    Ok(())
}

/// Validates strides for 3D pooling.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if any stride is 0.
pub(super) fn validate_strides_3d(strides: (usize, usize, usize)) -> Result<(), Error> {
    if strides.0 == 0 || strides.1 == 0 || strides.2 == 0 {
        return Err(Error::invalid_parameter(
            "strides",
            "must be greater than zero in all dimensions",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// validate_pool_size_3d HEIGHT over-size branch.
    ///
    /// Input spatial dims depth=4, height=4, width=4; pool_size = (depth=2, height=5, width=2).
    /// Depth 2 ≤ 4 and width 2 ≤ 4 both pass; only `pool_size.1 > input_height` (5 > 4) is true,
    /// so the `cannot exceed the corresponding input dimension` guard must fire on the HEIGHT axis.
    /// (Depth-axis over-size is already covered elsewhere.) The zero-check is skipped because no
    /// dimension is 0. Expected: Err(Error::InvalidParameter { name: "pool_size", .. }).
    #[test]
    fn test_validate_pool_size_3d_oversized_height() {
        let result = validate_pool_size_3d((2, 5, 2), 4, 4, 4);
        assert!(
            matches!(result, Err(Error::InvalidParameter { ref name, .. }) if name == "pool_size"),
            "oversized height pool must yield InvalidParameter(pool_size), got {:?}",
            result
        );
    }

    /// validate_pool_size_3d WIDTH over-size branch.
    ///
    /// Input spatial dims depth=4, height=4, width=4; pool_size = (depth=2, height=2, width=5).
    /// Depth 2 ≤ 4 and height 2 ≤ 4 both pass; only `pool_size.2 > input_width` (5 > 4) is true,
    /// so the `cannot exceed the corresponding input dimension` guard must fire on the WIDTH axis.
    /// Expected: Err(Error::InvalidParameter { name: "pool_size", .. }).
    #[test]
    fn test_validate_pool_size_3d_oversized_width() {
        let result = validate_pool_size_3d((2, 2, 5), 4, 4, 4);
        assert!(
            matches!(result, Err(Error::InvalidParameter { ref name, .. }) if name == "pool_size"),
            "oversized width pool must yield InvalidParameter(pool_size), got {:?}",
            result
        );
    }

    /// Control: an all-fitting 3D pool (every axis ≤ its input dim, none zero) must succeed,
    /// confirming the two failing tests above trip specifically on the over-size comparison and
    /// not on some unrelated guard.
    #[test]
    fn test_validate_pool_size_3d_all_fitting_ok() {
        assert!(validate_pool_size_3d((2, 2, 2), 4, 4, 4).is_ok());
    }
}
