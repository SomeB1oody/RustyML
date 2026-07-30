//! Output-shape calculators for the pooling and convolution layers

use crate::neural_network::layers::convolution::PaddingType;

/// Output size of one pooling axis: `(in - pool)/stride + 1` for `Valid`, `ceil(in/stride)` for
/// `Same`
fn pool_out_dim(input: usize, pool: usize, stride: usize, padding: PaddingType) -> usize {
    match padding {
        PaddingType::Valid => (input - pool) / stride + 1,
        PaddingType::Same => input.div_ceil(stride),
    }
}

/// Calculate output shape for 1d pooling layer
///
/// # Parameters
///
/// - `input_shape` - Shape of the input tensor, in format `[batch_size, length, channels]`
/// - `pool_size` - Size of the pooling window
/// - `stride` - Step size for sliding the window across the input
/// - `padding` - Padding strategy (Valid or Same)
///
/// # Returns
///
/// - `Vec<usize>` - Output shape in the format `[batch_size, output_length, channels]`
pub(super) fn calculate_output_shape_1d_pooling(
    input_shape: &[usize],
    pool_size: usize,
    stride: usize,
    padding: PaddingType,
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let length = input_shape[1];
    let channels = input_shape[2];

    let output_length = pool_out_dim(length, pool_size, stride, padding);

    vec![batch_size, output_length, channels]
}

/// Calculates the output shape of the 2d pooling layer
///
/// # Parameters
///
/// - `input_shape` - Shape of the input tensor, in format `[batch_size, height, width, channels]`
/// - `pool_size` - Size of the pooling window as a tuple (height, width)
/// - `strides` - Step size for the pooling window as a tuple (height_step, width_step)
/// - `padding` - Padding strategy (Valid or Same)
///
/// # Returns
///
/// - `Vec<usize>` - Output shape in format `[batch_size, output_height, output_width, channels]`
pub(super) fn calculate_output_shape_2d_pooling(
    input_shape: &[usize],
    pool_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let input_height = input_shape[1];
    let input_width = input_shape[2];
    let channels = input_shape[3];

    let output_height = pool_out_dim(input_height, pool_size.0, strides.0, padding);
    let output_width = pool_out_dim(input_width, pool_size.1, strides.1, padding);

    vec![batch_size, output_height, output_width, channels]
}

/// Calculates the output shape of the 3D pooling layer
///
/// # Parameters
///
/// - `input_shape` - Shape of the input tensor, in format
///   `[batch_size, depth, height, width, channels]`
/// - `pool_size` - Size of the pooling window as a tuple (depth, height, width)
/// - `strides` - Step size for the pooling window as a tuple (depth_step, height_step, width_step)
/// - `padding` - Padding strategy (Valid or Same)
///
/// # Returns
///
/// - `Vec<usize>` - Output shape in format
///   `[batch_size, output_depth, output_height, output_width, channels]`
pub(super) fn calculate_output_shape_3d_pooling(
    input_shape: &[usize],
    pool_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    padding: PaddingType,
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let input_depth = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    let channels = input_shape[4];

    let output_depth = pool_out_dim(input_depth, pool_size.0, strides.0, padding);
    let output_height = pool_out_dim(input_height, pool_size.1, strides.1, padding);
    let output_width = pool_out_dim(input_width, pool_size.2, strides.2, padding);

    vec![
        batch_size,
        output_depth,
        output_height,
        output_width,
        channels,
    ]
}

/// Calculate the output height and width for 2D convolution or pooling operations
///
/// Determines the output dimensions from the input dimensions, kernel size, stride, and
/// padding type. Supports both `Valid` and `Same` padding strategies
///
/// # Parameters
///
/// - `padding_type` - Type of padding to apply (Valid or Same)
/// - `input_height` - Height of the input tensor
/// - `input_width` - Width of the input tensor
/// - `kernel_size` - Size of the kernel as a tuple (height, width)
/// - `strides` - Stride values as a tuple (height_stride, width_stride)
///
/// # Returns
///
/// - `(usize, usize)` - Output dimensions (output_height, output_width)
///   - For `Valid` padding: dimensions are reduced based on kernel size and stride
///   - For `Same` padding: dimensions equal the input spatial size divided by the stride
pub(super) fn calculate_output_height_and_weight(
    padding_type: PaddingType,
    input_height: usize,
    input_width: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
) -> (usize, usize) {
    let (output_height, output_width) = match padding_type {
        PaddingType::Valid => {
            let out_height = (input_height - kernel_size.0) / strides.0 + 1;
            let out_width = (input_width - kernel_size.1) / strides.1 + 1;
            (out_height, out_width)
        }
        PaddingType::Same => {
            let out_height = (input_height as f32 / strides.0 as f32).ceil() as usize;
            let out_width = (input_width as f32 / strides.1 as f32).ceil() as usize;
            (out_height, out_width)
        }
    };

    (output_height, output_width)
}

/// Calculate the output shape for 2D convolution operations
///
/// # Parameters
///
/// - `input_shape` - Input tensor shape as `[batch_size, height, width, channels]`
/// - `kernel_size` - Size of the convolution kernel as (height, width)
/// - `strides` - Stride of the convolution as (height_stride, width_stride)
/// - `padding` - Padding strategy (Valid or Same)
///
/// # Returns
///
/// - `Vec<usize>` - Output shape as `[batch_size, output_height, output_width, channels]`
///
/// # Panics
///
/// Panics if `input_shape` has fewer than 4 dimensions
pub(super) fn calculate_output_shape_2d(
    input_shape: &[usize],
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: &PaddingType,
) -> Vec<usize> {
    assert!(
        input_shape.len() >= 4,
        "Input shape must have at least 4 dimensions"
    );

    let batch_size = input_shape[0];
    let input_height = input_shape[1];
    let input_width = input_shape[2];
    let channels = input_shape[3];

    let (output_height, output_width) = calculate_output_height_and_weight(
        *padding,
        input_height,
        input_width,
        kernel_size,
        strides,
    );

    vec![batch_size, output_height, output_width, channels]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Valid padding/pooling: out = floor((in - window) / stride) + 1
    // Same padding: out = ceil(in / stride)

    /// 1D pooling output length, with the batch and channel axes unchanged
    #[test]
    fn test_calculate_output_shape_1d_pooling() {
        let out = calculate_output_shape_1d_pooling(&[8, 10, 5], 3, 2, PaddingType::Valid);
        assert_eq!(out, vec![8, 4, 5]);
    }

    /// 1D pooling with a window equal to the length yields a single output position
    #[test]
    fn test_calculate_output_shape_1d_pooling_full_window() {
        let out = calculate_output_shape_1d_pooling(&[2, 6, 3], 6, 1, PaddingType::Valid);
        assert_eq!(out, vec![2, 1, 3]);
    }

    /// 2D pooling output shape for distinct window sizes and strides per axis
    ///
    /// Every axis has a different extent, so a mix-up between axes would appear in the result.
    /// Batch 4, height 6, width 7, and channels 8 pool to height `(6 - 2) / 2 + 1 = 3`. Width
    /// pools to `(7 - 3) / 1 + 1 = 5`. The batch and channel axes stay unchanged
    #[test]
    fn test_calculate_output_shape_2d_pooling() {
        let out =
            calculate_output_shape_2d_pooling(&[4, 6, 7, 8], (2, 3), (2, 1), PaddingType::Valid);
        assert_eq!(out, vec![4, 3, 5, 8]);
    }

    /// 3D pooling output shape across depth, height, and width axes
    ///
    /// Batch 2, depth 4, height 5, width 6, and channels 9 pool to depth `(4 - 2) / 1 + 1 = 3`.
    /// Height pools to `(5 - 2) / 2 + 1 = 2`. Width pools to `(6 - 3) / 3 + 1 = 2`
    #[test]
    fn test_calculate_output_shape_3d_pooling() {
        let out = calculate_output_shape_3d_pooling(
            &[2, 4, 5, 6, 9],
            (2, 2, 3),
            (1, 2, 3),
            PaddingType::Valid,
        );
        assert_eq!(out, vec![2, 3, 2, 2, 9]);
    }

    /// `Valid` padding reduces the output dimensions based on kernel size and stride
    #[test]
    fn test_calculate_output_height_and_weight_valid() {
        let (h, w) = calculate_output_height_and_weight(PaddingType::Valid, 5, 7, (3, 2), (1, 2));
        assert_eq!((h, w), (3, 3));
    }

    /// `Same` padding yields the input divided by the stride (rounded up), independent of the
    /// kernel
    #[test]
    fn test_calculate_output_height_and_weight_same() {
        let (h, w) = calculate_output_height_and_weight(PaddingType::Same, 7, 8, (3, 3), (2, 3));
        assert_eq!((h, w), (4, 3));
    }

    /// `Same` padding with stride 1 keeps the spatial size unchanged regardless of kernel
    #[test]
    fn test_calculate_output_height_and_weight_same_stride_one() {
        let (h, w) = calculate_output_height_and_weight(PaddingType::Same, 5, 5, (3, 3), (1, 1));
        assert_eq!((h, w), (5, 5));
    }

    /// `calculate_output_shape_2d` keeps the batch size and channel count unchanged and
    /// delegates the spatial extents to the formula, here with Valid padding. The channel count
    /// is the trailing axis and stays the same. An asymmetric shape confirms the channel count
    /// is not read as a spatial extent
    #[test]
    fn test_calculate_output_shape_2d_valid() {
        let out = calculate_output_shape_2d(&[2, 5, 5, 4], (3, 3), (1, 1), &PaddingType::Valid);
        assert_eq!(out, vec![2, 3, 3, 4]);
    }
}
