//! Shared input/weight validation for the layer module.

use crate::error::ModelError;

/// Validates that a weight array being assigned to a layer has the shape the layer expects.
///
/// Layers initialize their weight arrays with the correct shape in `new()`, so loaded or
/// user-supplied weights must match that shape. This turns a silent corruption (and a later
/// opaque panic deep inside `dot`) into a clear, recoverable error.
///
/// # Parameters
///
/// - `name` - Name of the parameter being set (for the error message, e.g. "weight" or "bias")
/// - `expected` - The shape the layer currently has for this parameter
/// - `found` - The shape of the array being assigned
///
/// # Returns
///
/// - `Ok(())` - The shapes match
/// - `Err(ModelError::InputValidationError)` - The shapes differ
pub(super) fn validate_weight_shape(
    name: &str,
    expected: &[usize],
    found: &[usize],
) -> Result<(), ModelError> {
    if expected != found {
        return Err(ModelError::InputValidationError(format!(
            "weight shape mismatch for `{}`: layer expects {:?}, got {:?}",
            name, expected, found
        )));
    }
    Ok(())
}
