use crate::error::Error;
use crate::neural_network::Tensor;

/// Validates that a dimension value is greater than 0
///
/// # Parameters
///
/// - `value` - The dimension value to validate
/// - `name` - The name of the dimension for error messages
///
/// # Returns
///
/// - `Ok(())` if validation passes
/// - `Err(Error::InvalidParameter)` if validation fails
pub(super) fn validate_dimension_greater_than_zero(value: usize, name: &str) -> Result<(), Error> {
    if value == 0 {
        return Err(Error::invalid_parameter(name, "must be greater than 0"));
    }
    Ok(())
}

/// Validates input dimensions for recurrent layers
///
/// # Parameters
///
/// - `input_dim` - The input dimension to validate
/// - `units` - The units dimension to validate
///
/// # Returns
///
/// - `Ok(())` if validation passes
/// - `Err(Error::InvalidParameter)` if validation fails
pub(super) fn validate_recurrent_dimensions(input_dim: usize, units: usize) -> Result<(), Error> {
    validate_dimension_greater_than_zero(input_dim, "input_dim")?;
    validate_dimension_greater_than_zero(units, "units")?;
    Ok(())
}

/// Validates that input tensor is 3D for recurrent layers
///
/// # Parameters
///
/// - `input` - The input tensor to validate
///
/// # Returns
///
/// - `Ok(())` if validation passes
/// - `Err(Error::InvalidInput)` if validation fails
pub(super) fn validate_input_3d(input: &Tensor) -> Result<(), Error> {
    if input.ndim() != 3 {
        return Err(Error::invalid_input("input tensor is not 3D"));
    }
    Ok(())
}
