//! Shared input/weight validation for the layer module

use crate::error::Error;
use crate::neural_network::NnError;
use crate::neural_network::Shape;

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

/// Opens the build of a layer, and refuses a second build for another shape
///
/// Every layer that allocates from its input shape calls this first. The answer says whether
/// the layer must allocate. A layer that is already built for the same input keeps every array
/// it holds, so no weight is ever drawn twice
///
/// The comparison frees the batch axis, because 1 layer serves every batch size
///
/// # Parameters
///
/// - `built` - The shape the layer already built for, or `None` before its first build
/// - `layer` - Layer name, which the message names
/// - `input` - Shape the layer is being built for, batch axis first
///
/// # Returns
///
/// - `Result<Option<Shape>, Error>` - `Some(shape)` to allocate and then record, and `None`
///   when the layer already holds every array for this input
///
/// # Errors
///
/// - `Error::InvalidInput` - If the layer is already built for another input shape
pub(super) fn start_build(
    built: &Option<Shape>,
    layer: &str,
    input: &Shape,
) -> Result<Option<Shape>, Error> {
    let Some(held) = built else {
        return Ok(Some(input.clone()));
    };
    if held.free_batch() != input.free_batch() {
        return Err(Error::invalid_input(format!(
            "{layer} is already built for the input shape {held}, and cannot build again for \
             {input}. Build a new layer for the second shape"
        )));
    }
    Ok(None)
}

/// Opens the build of a layer with several inputs, and refuses a second build for other shapes
///
/// This is [`start_build`] for a merge layer. Such a layer records 1 shape per input, so the
/// answer carries the whole list. A layer that a graph reaches from several nodes builds once,
/// on its first node, and every later node checks its shapes against that build
///
/// The comparison frees the batch axis of every shape, because 1 layer serves every batch size
///
/// # Parameters
///
/// - `built` - The shapes the layer already built for, or `None` before its first build
/// - `layer` - Layer name, which the message names
/// - `inputs` - 1 shape per input of the layer, batch axis first
///
/// # Returns
///
/// - `Result<Option<Vec<Shape>>, Error>` - `Some(shapes)` to allocate and then record, and
///   `None` when the layer already holds every array for these inputs
///
/// # Errors
///
/// - `Error::InvalidInput` - If the layer is already built for another input count, for another
///   rank, or for another shape. Each message names the 2 sides
pub(super) fn start_build_many(
    built: &Option<Vec<Shape>>,
    layer: &str,
    inputs: &[Shape],
) -> Result<Option<Vec<Shape>>, Error> {
    let Some(held) = built else {
        return Ok(Some(inputs.to_vec()));
    };
    if held.len() != inputs.len() {
        return Err(Error::invalid_input(format!(
            "{layer} is already built for {} inputs, and cannot build again for {} inputs. \
             Build a new layer for the second input count",
            held.len(),
            inputs.len()
        )));
    }
    for (position, (held, input)) in held.iter().zip(inputs).enumerate() {
        if held.rank() != input.rank() {
            return Err(Error::invalid_input(format!(
                "{layer} is already built for the shape {held} of rank {} on input {position}, \
                 and cannot build again for {input} of rank {}. Build a new layer for the \
                 second shape",
                held.rank(),
                input.rank()
            )));
        }
        if held.free_batch() != input.free_batch() {
            return Err(Error::invalid_input(format!(
                "{layer} is already built for the shape {held} on input {position}, and cannot \
                 build again for {input}. Build a new layer for the second shape"
            )));
        }
    }
    Ok(None)
}

/// Checks a live input tensor against the shape the layer was built for
///
/// The rank must match, and every axis after the batch axis that the build shape fixes must
/// match as well. The batch axis is never checked, because 1 layer serves every batch size. An
/// axis that the build shape leaves free is not checked either, and such an axis is exactly an
/// axis that no array of the layer depends on
///
/// This is the whole input check of a built layer. What the layer prints as its output shape
/// is what this enforces, so a summary can no longer name an extent that a forward pass would
/// accept a different value for
///
/// # Parameters
///
/// - `built` - The shape the layer built for, or `None` while it holds none
/// - `layer` - Layer name, which the message names
/// - `actual` - Shape of the tensor that arrived
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok` when the tensor fits the build
///
/// # Errors
///
/// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer is not built
/// - `Error::InvalidInput` - If the rank or a fixed axis disagrees with the build shape
pub(super) fn validate_built_input(
    built: &Option<Shape>,
    layer: &'static str,
    actual: &[usize],
) -> Result<(), Error> {
    let Some(built) = built else {
        return Err(Error::not_built(layer));
    };
    let axes = built.axes();
    if axes.len() != actual.len() {
        return Err(Error::invalid_input(format!(
            "{layer} was built for the input shape {built} of rank {}, and got a tensor of \
             rank {}",
            axes.len(),
            actual.len()
        )));
    }
    for (position, (axis, &extent)) in axes.iter().zip(actual.iter()).enumerate().skip(1) {
        if let Some(wanted) = axis
            && *wanted != extent
        {
            return Err(Error::invalid_input(format!(
                "{layer} was built for the input shape {built}, and got {extent} on axis \
                 {position} where it expects {wanted}"
            )));
        }
    }
    Ok(())
}
