//! Dimension and shape validators shared by the recurrent layers

use crate::error::Error;
use crate::neural_network::Tensor;
use ndarray::{Array2, ArrayView3, Ix2, Ix3};

/// Validates that a dimension value is greater than 0
///
/// # Parameters
///
/// - `value` - The dimension value to validate
/// - `name` - The name of the dimension, used to build the error message
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok(())` when `value` is greater than 0, otherwise the failing `Error`
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If `value` is 0
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
/// - `Result<(), Error>` - `Ok(())` when both dimensions are greater than 0, otherwise the
///   failing `Error`
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If `input_dim` or `units` is 0
pub(super) fn validate_recurrent_dimensions(input_dim: usize, units: usize) -> Result<(), Error> {
    validate_dimension_greater_than_zero(input_dim, "input_dim")?;
    validate_dimension_greater_than_zero(units, "units")?;
    Ok(())
}

/// Validates that the input tensor is 3D for recurrent layers
///
/// # Parameters
///
/// - `input` - The input tensor to validate
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok(())` when `input` has 3 dimensions, otherwise the failing `Error`
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If `input` is not 3D
pub(super) fn validate_input_3d(input: &Tensor) -> Result<(), Error> {
    if input.ndim() != 3 {
        return Err(Error::invalid_input("input tensor is not 3D"));
    }
    Ok(())
}

/// Splits the upstream gradient into the running hidden-state gradient and the per-step direct
/// contributions
///
/// With `return_sequences` off, the whole upstream gradient is the gradient of the single
/// returned state, so it starts the backward recurrence directly. With `return_sequences` on,
/// the upstream gradient holds 1 direct contribution per processing step, and the backward
/// recurrence starts from 0 and adds 1 slice per step.
///
/// # Parameters
///
/// - `grad_output` - Upstream gradient from the next layer
/// - `layer` - Name of the layer, used to build the error message
/// - `return_sequences` - True when the forward pass returned every timestep
/// - `batch` - Batch size recorded by the forward pass
/// - `timesteps` - Length of the time axis recorded by the forward pass
/// - `units` - Number of units of the layer
///
/// # Returns
///
/// - `Result<(Array2<f32>, Option<ArrayView3<f32>>), Error>` - The initial running gradient, and
///   the per-step contributions when `return_sequences` is true
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If the gradient rank does not match `return_sequences`
/// - [`Error::InvalidInput`] - If `return_sequences` is true and the gradient shape is not
///   `[batch, timesteps, units]`
pub(super) fn split_grad_output<'a>(
    grad_output: &'a Tensor,
    layer: &str,
    return_sequences: bool,
    batch: usize,
    timesteps: usize,
    units: usize,
) -> Result<(Array2<f32>, Option<ArrayView3<'a, f32>>), Error> {
    if !return_sequences {
        let grad_2d = grad_output
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "{layer} backward expects a 2D gradient [batch, units], got shape {:?}",
                    grad_output.shape()
                ))
            })?;
        return Ok((grad_2d.to_owned(), None));
    }

    let grad_3d = grad_output
        .view()
        .into_dimensionality::<Ix3>()
        .map_err(|_| {
            Error::invalid_input(format!(
                "{layer} backward expects a 3D gradient [batch, timesteps, units] when \
                 return_sequences is true, got shape {:?}",
                grad_output.shape()
            ))
        })?;
    if grad_3d.shape() != [batch, timesteps, units] {
        return Err(Error::invalid_input(format!(
            "{layer} backward expects a gradient of shape {:?}, got shape {:?}",
            [batch, timesteps, units],
            grad_3d.shape()
        )));
    }
    Ok((Array2::zeros((batch, units)), Some(grad_3d)))
}
