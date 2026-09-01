//! Shared input/weight validation for the layer module

use crate::error::Error;
use crate::neural_network::NnError;

/// Validates that a weight array assigned to a layer has the shape the layer expects
///
/// Layers initialize their weight arrays with the correct shape in `new()`, so loaded or
/// user-supplied weights must match that shape. This turns a silent corruption (and a later
/// opaque panic deep inside `dot`) into a clear, recoverable error
///
/// # Parameters
///
/// - `name` - Name of the parameter being set, used in the error message, e.g. "kernel" or
///   "bias"
/// - `expected` - Shape the layer currently has for this parameter
/// - `found` - Shape of the array being assigned
///
/// # Errors
///
/// - `Error::NeuralNetwork(NnError::WeightShape)` if the shapes differ
pub(super) fn validate_weight_shape(
    name: &str,
    expected: &[usize],
    found: &[usize],
) -> crate::error::RustymlResult<()> {
    if expected != found {
        return Err(Error::NeuralNetwork(NnError::WeightShape {
            name: name.to_string(),
            expected: expected.to_vec(),
            found: found.to_vec(),
        }));
    }
    Ok(())
}

/// Validates that a supplied optional array agrees with the flag that decides whether the
/// layer holds it
///
/// A layer built with `use_bias` set to false holds no bias, so a caller that supplies one has
/// a value that reaches nothing. A layer built with `use_bias` left true holds a bias, so a
/// caller that supplies none leaves it at whatever it already held. Both cases are mistakes,
/// and this reports each of them by name
///
/// # Parameters
///
/// - `name` - Name of the array, such as `"bias"` or `"gamma"`
/// - `flag` - Name of the configuration flag that decides whether the layer holds the array,
///   such as `"use_bias"`
/// - `holds` - Whether the layer holds the array
/// - `supplied` - The array the caller gave, if any
///
/// # Returns
///
/// - `RustymlResult<Option<T>>` - The supplied array, when the 2 sides agree
///
/// # Errors
///
/// - `Error::InvalidParameter` - The layer holds the array and the caller gave none, or the
///   layer holds no such array and the caller gave one
pub(super) fn validate_optional_weight<T>(
    name: &str,
    flag: &str,
    holds: bool,
    supplied: Option<T>,
) -> crate::error::RustymlResult<Option<T>> {
    match (holds, supplied.is_some()) {
        (true, false) => Err(Error::invalid_parameter(
            name,
            format!("`{flag}` is true, so the layer holds a {name}. Give the {name} array"),
        )),
        (false, true) => Err(Error::invalid_parameter(
            name,
            format!(
                "`{flag}` is false, so the layer holds no {name}. Give no {name}, or build the \
                 layer with `{flag}` set to true"
            ),
        )),
        _ => Ok(supplied),
    }
}
