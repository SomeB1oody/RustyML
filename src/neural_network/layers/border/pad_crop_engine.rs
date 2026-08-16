//! Pad and crop kernels shared by every border layer
//!
//! A zero-padding layer and a cropping layer move no data. One grows the spatial axes and
//! writes zeros in the new positions. The other returns the interior and drops the rest. Each
//! one is the backward pass of the other, so 1 pair of kernels serves all 6 layers
//!
//! Every function here indexes `borders` by spatial axis. Entry `i` of `borders` describes
//! axis `i + 1` of the tensor. The batch axis and the channel axis have no entry, because a
//! border layer never changes them

use crate::error::Error;
use crate::neural_network::Tensor;
use ndarray::{IxDyn, Slice};

/// Shape a padded output takes, given the shape that enters the layer
fn padded_shape(input_shape: &[usize], borders: &[(usize, usize)]) -> Vec<usize> {
    let mut shape = input_shape.to_vec();
    for (spatial, &(before, after)) in borders.iter().enumerate() {
        shape[spatial + 1] += before + after;
    }
    shape
}

/// Shape a cropped output takes, given the shape that enters the layer
///
/// The caller must check the borders against the input first. See [`validate_crop_fits`]
fn cropped_shape(input_shape: &[usize], borders: &[(usize, usize)]) -> Vec<usize> {
    let mut shape = input_shape.to_vec();
    for (spatial, &(before, after)) in borders.iter().enumerate() {
        shape[spatial + 1] -= before + after;
    }
    shape
}

/// Copies `input` into a zero tensor whose spatial axes each grow by their border
fn pad_into(input: &Tensor, borders: &[(usize, usize)]) -> Tensor {
    let input_shape = input.shape();
    let mut output = Tensor::zeros(IxDyn(&padded_shape(input_shape, borders)));

    output
        .slice_each_axis_mut(|ax| {
            let axis = ax.axis.index();
            match axis.checked_sub(1).and_then(|spatial| borders.get(spatial)) {
                Some(&(before, _)) => Slice::from(before..before + input_shape[axis]),
                None => Slice::from(..),
            }
        })
        .assign(input);

    output
}

/// Copies the interior of `input`, with each spatial axis shrunk by its border
fn crop_out(input: &Tensor, borders: &[(usize, usize)]) -> Tensor {
    let input_shape = input.shape().to_vec();

    let interior = input.slice_each_axis(|ax| {
        let axis = ax.axis.index();
        match axis.checked_sub(1).and_then(|spatial| borders.get(spatial)) {
            Some(&(before, after)) => Slice::from(before..input_shape[axis] - after),
            None => Slice::from(..),
        }
    });

    // `ArrayBase::to_owned` copies a strided view 1 element at a time. `assign` instead runs
    // through `Zip`, which copies the widest run that is contiguous on both sides
    let mut output = Tensor::zeros(interior.raw_dim());
    output.assign(&interior);
    output
}

/// Checks the rank and the element count of a tensor entering a border layer
///
/// # Errors
///
/// - `Error::InvalidInput` - If the rank is not `rank`
/// - `Error::EmptyInput` - If any axis has an extent of 0
fn validate_input(input: &Tensor, rank: usize, layer: &'static str) -> Result<(), Error> {
    if input.ndim() != rank {
        return Err(Error::invalid_input(format!(
            "{} layer expects a {}D input, got a {}D tensor",
            layer,
            rank,
            input.ndim()
        )));
    }
    if input.is_empty() {
        return Err(Error::empty_input("input tensor"));
    }
    Ok(())
}

/// Checks that every cropped axis keeps at least 1 position
///
/// # Errors
///
/// - `Error::InvalidInput` - If a border removes the whole extent of its axis or more
fn validate_crop_fits(
    input_shape: &[usize],
    borders: &[(usize, usize)],
    layer: &'static str,
) -> Result<(), Error> {
    for (spatial, &(before, after)) in borders.iter().enumerate() {
        let axis = spatial + 1;
        let extent = input_shape[axis];
        if before + after >= extent {
            return Err(Error::invalid_input(format!(
                "{} layer removes {} of the {} positions on axis {}, and at least 1 must remain",
                layer,
                before + after,
                extent,
                axis
            )));
        }
    }
    Ok(())
}

/// Formats a shape the way `summary()` prints it, with the batch axis as "None"
///
/// An input shape of `None` means no forward pass has run, so the layer cannot know its output
/// shape yet
fn format_shape(shape: Option<Vec<usize>>) -> String {
    match shape {
        Some(shape) => {
            let axes: Vec<String> = shape[1..].iter().map(|e| e.to_string()).collect();
            format!("(None, {})", axes.join(", "))
        }
        None => "Unknown".to_string(),
    }
}

/// Runs the forward pass of a zero-padding layer
///
/// # Parameters
///
/// - `input` - Tensor entering the layer
/// - `borders` - Zero positions to add at each end of each spatial axis
/// - `rank` - Rank the layer accepts, batch and channel axes included
/// - `layer` - Layer name, used in error messages
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The padded tensor
///
/// # Errors
///
/// - `Error::InvalidInput` - If the input rank is not `rank`
/// - `Error::EmptyInput` - If any axis of the input has an extent of 0
pub(super) fn pad_forward(
    input: &Tensor,
    borders: &[(usize, usize)],
    rank: usize,
    layer: &'static str,
) -> Result<Tensor, Error> {
    validate_input(input, rank, layer)?;
    Ok(pad_into(input, borders))
}

/// Runs the backward pass of a zero-padding layer
///
/// The gradient of a padded position goes nowhere, so the pass returns the interior of the
/// incoming gradient
///
/// # Parameters
///
/// - `grad_output` - Gradient from the next layer
/// - `input_shape` - Shape of the most recent forward input, or `None` if none has run
/// - `borders` - Zero positions the forward pass added
/// - `layer` - Layer name, used in error messages
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The gradient for the previous layer
///
/// # Errors
///
/// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `input_shape` is `None`
/// - `Error::ShapeMismatch` - If `grad_output` does not have the padded output shape
pub(super) fn pad_backward(
    grad_output: &Tensor,
    input_shape: Option<&[usize]>,
    borders: &[(usize, usize)],
    layer: &'static str,
) -> Result<Tensor, Error> {
    let Some(input_shape) = input_shape else {
        return Err(Error::forward_pass_not_run(layer));
    };

    let expected = padded_shape(input_shape, borders);
    if grad_output.shape() != expected.as_slice() {
        return Err(Error::shape_mismatch(expected, grad_output.shape()));
    }

    Ok(crop_out(grad_output, borders))
}

/// Formats the output shape of a zero-padding layer for `summary()`
pub(super) fn pad_summary(input_shape: Option<&[usize]>, borders: &[(usize, usize)]) -> String {
    format_shape(input_shape.map(|shape| padded_shape(shape, borders)))
}

/// Runs the forward pass of a cropping layer
///
/// # Parameters
///
/// - `input` - Tensor entering the layer
/// - `borders` - Positions to remove at each end of each spatial axis
/// - `rank` - Rank the layer accepts, batch and channel axes included
/// - `layer` - Layer name, used in error messages
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The cropped tensor
///
/// # Errors
///
/// - `Error::InvalidInput` - If the input rank is not `rank`, or if a border leaves an axis
///   with no positions
/// - `Error::EmptyInput` - If any axis of the input has an extent of 0
pub(super) fn crop_forward(
    input: &Tensor,
    borders: &[(usize, usize)],
    rank: usize,
    layer: &'static str,
) -> Result<Tensor, Error> {
    validate_input(input, rank, layer)?;
    validate_crop_fits(input.shape(), borders, layer)?;
    Ok(crop_out(input, borders))
}

/// Runs the backward pass of a cropping layer
///
/// A removed position takes no part in the output, so its gradient is 0. The pass writes the
/// incoming gradient into the interior of a zero tensor of the input shape
///
/// # Parameters
///
/// - `grad_output` - Gradient from the next layer
/// - `input_shape` - Shape of the most recent forward input, or `None` if none has run
/// - `borders` - Positions the forward pass removed
/// - `layer` - Layer name, used in error messages
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The gradient for the previous layer
///
/// # Errors
///
/// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `input_shape` is `None`
/// - `Error::ShapeMismatch` - If `grad_output` does not have the cropped output shape
pub(super) fn crop_backward(
    grad_output: &Tensor,
    input_shape: Option<&[usize]>,
    borders: &[(usize, usize)],
    layer: &'static str,
) -> Result<Tensor, Error> {
    let Some(input_shape) = input_shape else {
        return Err(Error::forward_pass_not_run(layer));
    };

    let expected = cropped_shape(input_shape, borders);
    if grad_output.shape() != expected.as_slice() {
        return Err(Error::shape_mismatch(expected, grad_output.shape()));
    }

    Ok(pad_into(grad_output, borders))
}

/// Formats the output shape of a cropping layer for `summary()`
pub(super) fn crop_summary(input_shape: Option<&[usize]>, borders: &[(usize, usize)]) -> String {
    format_shape(input_shape.map(|shape| cropped_shape(shape, borders)))
}
