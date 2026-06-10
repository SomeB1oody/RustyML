//! Output-shape calculators for the pooling and convolution layers.

use crate::neural_network::layers::convolution::PaddingType;

/// Calculate output shape for 1d pooling layer.
///
/// # Parameters
///
/// - `input_shape` - Shape of the input tensor, in format `[batch_size, channels, length]`
/// - `pool_size` - Size of the pooling/convolutional window
/// - `stride` - Step size for sliding the window across the input
///
/// # Returns
///
/// * `Vec<usize>` - A vector containing the dimensions of the output tensor in the format: `[batch_size, channels, output_length]`
pub(super) fn calculate_output_shape_1d_pooling(
    input_shape: &[usize],
    pool_size: usize,
    stride: usize,
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let length = input_shape[2];

    let output_length = (length - pool_size) / stride + 1;

    vec![batch_size, channels, output_length]
}

/// Calculates the output shape of the 2d pooling layer.
///
/// # Parameters
///
/// * `input_shape` - Shape of the input tensor, in format \[batch_size, channels, height, width\].
/// * `pool_size` - Size of the pooling window as a tuple (height, width).
/// * `strides` - Step size for the pooling window as a tuple (height_step, width_step).
///
/// # Returns
///
/// * `Vec<usize>` - A vector containing the calculated output shape, in format \[batch_size, channels, output_height, output_width\].
pub(super) fn calculate_output_shape_2d_pooling(
    input_shape: &[usize],
    pool_size: (usize, usize),
    strides: (usize, usize),
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    // Calculate the height and width of the output
    let output_height = (input_height - pool_size.0) / strides.0 + 1;
    let output_width = (input_width - pool_size.1) / strides.1 + 1;

    vec![batch_size, channels, output_height, output_width]
}

/// Calculates the output shape of the 3D pooling layer.
///
/// # Parameters
///
/// * `input_shape` - The shape of the input tensor, formatted as \[batch_size, channels, depth, height, width\].
/// * `pool_size` - Size of the pooling window as a tuple (depth, height, width).
/// * `strides` - Step size for the pooling window as a tuple (depth_step, height_step, width_step).
///
/// # Returns
///
/// * `Vec<usize>` - A vector containing the calculated output shape, formatted as \[batch_size, channels, output_depth, output_height, output_width\].
pub(super) fn calculate_output_shape_3d_pooling(
    input_shape: &[usize],
    pool_size: (usize, usize, usize),
    strides: (usize, usize, usize),
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    // Calculate the output depth, height, and width
    let output_depth = (input_depth - pool_size.0) / strides.0 + 1;
    let output_height = (input_height - pool_size.1) / strides.1 + 1;
    let output_width = (input_width - pool_size.2) / strides.2 + 1;

    vec![
        batch_size,
        channels,
        output_depth,
        output_height,
        output_width,
    ]
}

/// Calculates the output height and width for 2D convolution or pooling operations.
///
/// This function determines the output dimensions based on input dimensions, kernel size,
/// stride, and padding type. It supports both 'Valid' and 'Same' padding strategies.
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
/// * `(usize, usize)` - A tuple containing the calculated output dimensions (output_height, output_width).
///   - For `Valid` padding: Dimensions are reduced based on kernel size and stride
///   - For `Same` padding: Output dimensions are calculated to maintain the input spatial dimensions
///     divided by the stride
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
/// - `input_shape` - Input tensor shape as \[batch_size, channels, height, width\]
/// - `kernel_size` - Size of the convolution kernel as (height, width)
/// - `strides` - Stride of the convolution as (height_stride, width_stride)
/// - `padding` - Padding strategy (Valid or Same)
///
/// # Returns
///
/// * `Vec<usize>` - Output shape as \[batch_size, channels, output_height, output_width\]
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
    let channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let (output_height, output_width) = calculate_output_height_and_weight(
        *padding,
        input_height,
        input_width,
        kernel_size,
        strides,
    );

    vec![batch_size, channels, output_height, output_width]
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests verify the spatial-dimension formulas straight from their definitions.
    // For `Valid` padding / pooling the output extent along an axis is
    //   out = floor((in - window) / stride) + 1   (integer division)
    // and for `Same` padding it is
    //   out = ceil(in / stride).
    //
    // `PaddingType` is brought into the test module by `use super::*;` (the parent module's
    // `use crate::neural_network::layers::convolution::PaddingType;` is visible to this child
    // module); it derives `Copy`, so passing it by value is fine.

    /// 1D pooling: in=10, pool=3, stride=2 => floor((10-3)/2)+1 = floor(7/2)+1 = 3+1 = 4.
    /// Batch and channel axes pass through unchanged.
    #[test]
    fn test_calculate_output_shape_1d_pooling() {
        let out = calculate_output_shape_1d_pooling(&[8, 5, 10], 3, 2);
        assert_eq!(out, vec![8, 5, 4]);
    }

    /// 1D pooling with stride 1 and a window equal to the length leaves a single position:
    /// in=6, pool=6, stride=1 => floor((6-6)/1)+1 = 1.
    #[test]
    fn test_calculate_output_shape_1d_pooling_full_window() {
        let out = calculate_output_shape_1d_pooling(&[2, 3, 6], 6, 1);
        assert_eq!(out, vec![2, 3, 1]);
    }

    /// 2D pooling: height 7/pool 2/stride 2 => floor((7-2)/2)+1 = floor(5/2)+1 = 2+1 = 3;
    /// width 8/pool 3/stride 1 => floor((8-3)/1)+1 = 5+1 = 6.
    #[test]
    fn test_calculate_output_shape_2d_pooling() {
        let out = calculate_output_shape_2d_pooling(&[4, 6, 7, 8], (2, 3), (2, 1));
        assert_eq!(out, vec![4, 6, 3, 6]);
    }

    /// 3D pooling: depth 5/pool 2/stride 1 => floor(3/1)+1 = 4; height 6/pool 2/stride 2 =>
    /// floor(4/2)+1 = 3; width 9/pool 3/stride 3 => floor(6/3)+1 = 3.
    #[test]
    fn test_calculate_output_shape_3d_pooling() {
        let out = calculate_output_shape_3d_pooling(&[2, 4, 5, 6, 9], (2, 2, 3), (1, 2, 3));
        assert_eq!(out, vec![2, 4, 4, 3, 3]);
    }

    /// `Valid` padding reduces dimensions: height 5/kernel 3/stride 1 => floor((5-3)/1)+1 = 3;
    /// width 7/kernel 2/stride 2 => floor((7-2)/2)+1 = floor(5/2)+1 = 2+1 = 3.
    #[test]
    fn test_calculate_output_height_and_weight_valid() {
        let (h, w) = calculate_output_height_and_weight(PaddingType::Valid, 5, 7, (3, 2), (1, 2));
        assert_eq!((h, w), (3, 3));
    }

    /// `Same` padding preserves the input divided by the stride (rounded up), independent of
    /// the kernel: height 7/stride 2 => ceil(7/2) = 4; width 8/stride 3 => ceil(8/3) = 3.
    #[test]
    fn test_calculate_output_height_and_weight_same() {
        let (h, w) = calculate_output_height_and_weight(PaddingType::Same, 7, 8, (3, 3), (2, 3));
        assert_eq!((h, w), (4, 3));
    }

    /// `Same` with stride 1 keeps the spatial size exactly: 5x5 stays 5x5 regardless of kernel.
    #[test]
    fn test_calculate_output_height_and_weight_same_stride_one() {
        let (h, w) = calculate_output_height_and_weight(PaddingType::Same, 5, 5, (3, 3), (1, 1));
        assert_eq!((h, w), (5, 5));
    }

    /// `calculate_output_shape_2d` threads batch/channels through and delegates the spatial
    /// extents to the formula above. With Valid padding, kernel (3,3), stride (1,1) on a
    /// [2, 4, 5, 5] input: out_h = floor((5-3)/1)+1 = 3, out_w = 3.
    #[test]
    fn test_calculate_output_shape_2d_valid() {
        let out = calculate_output_shape_2d(&[2, 4, 5, 5], (3, 3), (1, 1), &PaddingType::Valid);
        assert_eq!(out, vec![2, 4, 3, 3]);
    }
}
