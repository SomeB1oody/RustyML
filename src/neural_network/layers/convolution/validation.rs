//! Shared parameter and input-shape validators for the convolution layers

use crate::error::Error;
use crate::neural_network::layers::convolution::convolution_engine::{
    ConvPadding, effective_kernel,
};

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
/// # Notes
///
/// The kernel size is not part of this rule. A kernel longer than the input axis stays legal
/// under `Same` and `Causal` padding. See [`valid_output_size`], which applies that rule at the
/// build, under `Valid` padding alone
///
/// # Errors
///
/// Returns `Error::InvalidInput` if:
/// - Shape is not 3D
/// - Input channels is 0
pub(super) fn validate_input_shape_1d(input_shape: &[usize]) -> Result<(), Error> {
    if input_shape.len() != 3 {
        return Err(Error::invalid_input(
            "Input shape must be 3D: [batch_size, length, channels]",
        ));
    }
    if input_shape[2] == 0 {
        return Err(Error::invalid_input(
            "Number of input channels must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates input shape for 2D convolution
///
/// # Notes
///
/// The kernel size is not part of this rule. See [`validate_input_shape_1d`]
///
/// # Errors
///
/// Returns `Error::InvalidInput` if:
/// - Shape is not 4D
/// - Input channels is 0
pub(super) fn validate_input_shape_2d(input_shape: &[usize]) -> Result<(), Error> {
    if input_shape.len() != 4 {
        return Err(Error::invalid_input(
            "Input shape must be 4D: [batch_size, height, width, channels]",
        ));
    }
    if input_shape[3] == 0 {
        return Err(Error::invalid_input(
            "Number of input channels must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates input shape for 3D convolution
///
/// # Notes
///
/// The kernel size is not part of this rule. See [`validate_input_shape_1d`]
///
/// # Errors
///
/// Returns `Error::InvalidInput` if:
/// - Shape is not 5D
/// - Any dimension is 0
pub(super) fn validate_input_shape_3d(input_shape: &[usize]) -> Result<(), Error> {
    if input_shape.len() != 5 {
        return Err(Error::invalid_input(
            "Input shape must be 5-dimensional: [batch, depth, height, width, channels]",
        ));
    }
    if input_shape.contains(&0) {
        return Err(Error::invalid_input(
            "All input dimensions must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates the input shape of a transposed convolution of the given spatial rank
///
/// A transposed convolution grows its input, so it puts no lower bound on the input spatial
/// size. A 1x1 input under a 3x3 kernel is a normal decoder step. A plain convolution bounds the
/// input only under `Valid` padding, and it applies that rule at the build. See
/// [`valid_output_size`]
///
/// # Parameters
///
/// - `input_shape` - Shape of the input tensor
/// - `rank` - Number of spatial axes (1, 2, or 3)
/// - `layout` - Human-readable axis list, used in the error message
///
/// # Errors
///
/// Returns `Error::InvalidInput` if the shape does not have `rank + 2` axes, or if any axis is 0
pub(super) fn validate_transpose_input_shape(
    input_shape: &[usize],
    rank: usize,
    layout: &str,
) -> Result<(), Error> {
    if input_shape.len() != rank + 2 {
        return Err(Error::invalid_input(format!(
            "Input shape must be {}D: {layout}",
            rank + 2
        )));
    }
    if input_shape.contains(&0) {
        return Err(Error::invalid_input(
            "All input dimensions must be greater than 0",
        ));
    }
    Ok(())
}

/// Validates a dilation rate
///
/// # Parameters
///
/// - `dilation` - Tap spacing of each spatial axis
///
/// # Errors
///
/// - `Error::InvalidParameter` - If any dilation is 0
pub(super) fn validate_dilation(dilation: &[usize]) -> Result<(), Error> {
    if dilation.contains(&0) {
        return Err(Error::invalid_parameter(
            "dilation_rate",
            "Dilation rate must be greater than 0",
        ));
    }
    Ok(())
}

/// Rejects an effective kernel longer than the input axis it runs on, under `Valid` padding only
///
/// The effective extent of `k` taps spaced `dilation` apart is `(k - 1) * dilation + 1`. Under
/// `Valid` padding the layer reads only complete windows, so an extent longer than the input axis
/// gives an output size of 0. That configuration is rejected. `Same` and `Causal` padding add the
/// missing cells on the borders, so every extent stays legal and this rule does not apply. A
/// transposed convolution grows its input and puts no such bound on it, so it does not call this
///
/// This is the guard of the free functions that carry no layer name, such as
/// [`conv_forward_impl`](super::convolution_engine::conv_forward_impl). Every layer applies the
/// same rule at its build through [`valid_output_size`], which names the layer and the axis. A
/// layer therefore refuses an oversized kernel before it allocates, and this guard never fires
/// through a layer
///
/// # Parameters
///
/// - `padding` - Padding mode of the layer
/// - `kernel` - Kernel size of each spatial axis
/// - `dilation` - Tap spacing of each spatial axis
/// - `input_sp` - Input size of each spatial axis
///
/// # Errors
///
/// - `Error::InvalidInput` - If the padding is `Valid` and an effective kernel is longer than the
///   input axis it runs on
pub(super) fn validate_valid_kernel_fits(
    padding: ConvPadding,
    kernel: &[usize],
    dilation: &[usize],
    input_sp: &[usize],
) -> Result<(), Error> {
    if padding != ConvPadding::Valid {
        return Ok(());
    }
    for d in 0..kernel.len() {
        let keff = effective_kernel(kernel[d], dilation[d]);
        if input_sp[d] < keff {
            return Err(Error::invalid_input(format!(
                "Valid-padding convolution requires every input spatial dimension to be at least \
                 the dilated kernel size: axis {d} has input size {} < dilated kernel size {keff}",
                input_sp[d]
            )));
        }
    }
    Ok(())
}

/// Output size of 1 spatial axis under `Valid` padding, or the refusal of an oversized kernel
///
/// Under `Valid` padding the layer reads only complete windows. An effective kernel longer than
/// the input axis leaves no complete window, so the axis would carry 0 positions. The shape
/// algebra refuses that configuration rather than report a 0 extent, because a build must not
/// hand an empty axis to the next layer of the stack
///
/// The rule belongs to `Valid` padding alone. `Same` and `Causal` padding add the missing cells on
/// the borders, so an oversized kernel stays legal there, and those branches never call this
///
/// # Parameters
///
/// - `layer` - Layer name, which the message names
/// - `axis` - Name of the spatial axis, such as `"length"`, `"height"`, `"width"`, or `"depth"`
/// - `input` - Input size of the axis
/// - `keff` - Effective kernel extent of the axis, which is `(k - 1) * dilation + 1`
/// - `stride` - Stride of the axis
///
/// # Returns
///
/// - `Result<usize, Error>` - Number of output positions on the axis
///
/// # Errors
///
/// - `Error::InvalidInput` - If the effective kernel is longer than the input axis
pub(super) fn valid_output_size(
    layer: &str,
    axis: &str,
    input: usize,
    keff: usize,
    stride: usize,
) -> Result<usize, Error> {
    let Some(rest) = input.checked_sub(keff) else {
        return Err(Error::invalid_input(format!(
            "{layer} under Valid padding needs an input {axis} of at least the effective kernel \
             extent, which is {keff}. This input {axis} is {input}. Use Same padding, a smaller \
             kernel, a smaller dilation rate, or a larger input"
        )));
    };
    Ok(rest / stride + 1)
}

/// Rejects a stride above 1 together with a dilation above 1
///
/// The rule fires on the maximum across the axes, not axis by axis. A stride of 2 on 1 axis and
/// a dilation of 2 on another is rejected as well
///
/// # Parameters
///
/// - `strides` - Stride of each spatial axis
/// - `dilation` - Tap spacing of each spatial axis
///
/// # Notes
///
/// ONLY the plain and the transposed convolutions call this. The depthwise and the separable
/// layers accept a stride and a dilation above 1 together, so this must stay out of any
/// validator they share
///
/// # Errors
///
/// - `Error::InvalidParameter` - If any stride and any dilation are both above 1
pub(super) fn validate_stride_dilation_exclusive(
    strides: &[usize],
    dilation: &[usize],
) -> Result<(), Error> {
    let max_stride = strides.iter().copied().max().unwrap_or(1);
    let max_dilation = dilation.iter().copied().max().unwrap_or(1);
    if max_stride > 1 && max_dilation > 1 {
        return Err(Error::invalid_parameter(
            "dilation_rate",
            format!(
                "A stride above 1 does not combine with a dilation above 1: strides {strides:?} \
                 and dilation {dilation:?}"
            ),
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
