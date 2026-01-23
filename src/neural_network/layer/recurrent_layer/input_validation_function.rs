use super::*;

/// Validates that a dimension value is greater than 0
///
/// # Parameters
///
/// - `value` - The dimension value to validate
/// - `name` - The name of the dimension for error messages
///
/// # Returns
///
/// * `Ok(())` if validation passes
/// * `Err(ModelError)` if validation fails
pub(super) fn validate_dimension_greater_than_zero(
    value: usize,
    name: &str,
) -> Result<(), ModelError> {
    if value == 0 {
        return Err(ModelError::InputValidationError(format!(
            "{} must be greater than 0",
            name
        )));
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
/// * `Ok(())` if validation passes
/// * `Err(ModelError)` if validation fails
pub(super) fn validate_recurrent_dimensions(
    input_dim: usize,
    units: usize,
) -> Result<(), ModelError> {
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
/// * `Ok(())` if validation passes
/// * `Err(ModelError)` if validation fails
pub(super) fn validate_input_3d(input: &Tensor) -> Result<(), ModelError> {
    if input.ndim() != 3 {
        return Err(ModelError::InputValidationError(
            "input tensor is not 3D".to_string(),
        ));
    }
    Ok(())
}
