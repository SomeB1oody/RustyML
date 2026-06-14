//! Convolution-internal helpers: 2D/4D zero-padding for the depthwise/separable stages

use crate::neural_network::Tensor;
use ndarray::{Array2, ArrayD, s};

/// Zero-pads a 2D tensor symmetrically, with any odd remainder on the trailing (bottom/right) edge
///
/// # Parameters
///
/// - `input` - 2D input tensor to pad
/// - `pad_h` - total padding added along the height dimension
/// - `pad_w` - total padding added along the width dimension
///
/// # Returns
///
/// - `Array2<f32>` - new 2D tensor with the padding applied
pub(super) fn pad_tensor_2d(input: &Array2<f32>, pad_h: usize, pad_w: usize) -> Array2<f32> {
    let (input_height, input_width) = input.dim();

    let pad_top = pad_h / 2;
    let pad_left = pad_w / 2;

    let output_height = input_height + pad_h;
    let output_width = input_width + pad_w;

    let mut output = Array2::zeros((output_height, output_width));

    // Copy input into the center of the zero-filled output
    output
        .slice_mut(s![
            pad_top..pad_top + input_height,
            pad_left..pad_left + input_width
        ])
        .assign(input);

    output
}

/// Zero-pads the two trailing spatial (`H`, `W`) axes of a 4D `[batch, channels, H, W]` tensor,
/// leaving the batch and channel axes untouched
///
/// Padding is symmetric with any odd remainder on the trailing edge (`pad_before = pad / 2`),
/// matching [`pad_tensor_2d`] and the convolution engine. Returns a plain clone when both pads are
/// zero, so callers can treat it as a no-op for `Valid` padding
///
/// # Parameters
///
/// - `input` - 4D `[batch, channels, H, W]` tensor to pad
/// - `pad_h` - total padding added along the height axis
/// - `pad_w` - total padding added along the width axis
///
/// # Returns
///
/// - `Tensor` - new 4D tensor with the spatial axes zero-padded
pub(super) fn pad_tensor_4d_spatial(input: &Tensor, pad_h: usize, pad_w: usize) -> Tensor {
    if pad_h == 0 && pad_w == 0 {
        return input.clone();
    }

    let shape = input.shape();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let pad_top = pad_h / 2;
    let pad_left = pad_w / 2;

    let mut output: Tensor = ArrayD::zeros(vec![batch, channels, height + pad_h, width + pad_w]);
    output
        .slice_mut(s![
            ..,
            ..,
            pad_top..pad_top + height,
            pad_left..pad_left + width
        ])
        .assign(input);

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `pad_tensor_2d` centers the input with even padding and zeros around it
    #[test]
    fn test_pad_tensor_2d_symmetric() {
        let input = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let out = pad_tensor_2d(&input, 2, 2);
        let expected = array![
            [0.0_f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        assert_eq!(out.dim(), (4, 4));
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(out[[i, j]], expected[[i, j]], epsilon = 0.0);
            }
        }
    }

    /// `pad_tensor_2d` puts the odd-remainder padding on the trailing (bottom/right) edge
    #[test]
    fn test_pad_tensor_2d_odd_padding_trailing_edge() {
        let input = array![[7.0_f32]];
        let out = pad_tensor_2d(&input, 1, 1);
        let expected = array![[7.0_f32, 0.0], [0.0, 0.0]];
        assert_eq!(out.dim(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(out[[i, j]], expected[[i, j]], epsilon = 0.0);
            }
        }
    }

    /// `pad_tensor_4d_spatial` is a no-op clone when both pads are zero
    #[test]
    fn test_pad_tensor_4d_spatial_zero_is_noop() {
        let input: Tensor = array![[[[1.0_f32, 2.0], [3.0, 4.0]]]].into_dyn(); // [1,1,2,2]
        let out = pad_tensor_4d_spatial(&input, 0, 0);
        assert_eq!(out.shape(), input.shape());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(out[[0, 0, i, j]], input[[0, 0, i, j]], epsilon = 0.0);
            }
        }
    }

    /// `pad_tensor_4d_spatial` grows only the spatial axes, leaving batch/channel unchanged
    #[test]
    fn test_pad_tensor_4d_spatial_pads_only_spatial_axes() {
        let input: Tensor = array![[[[1.0_f32, 2.0], [3.0, 4.0]]]].into_dyn();
        let out = pad_tensor_4d_spatial(&input, 2, 2);
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
        // Centered block
        assert_abs_diff_eq!(out[[0, 0, 1, 1]], 1.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 1, 2]], 2.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 2, 1]], 3.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 2, 2]], 4.0, epsilon = 0.0);
        // Corners are zero padding
        assert_abs_diff_eq!(out[[0, 0, 0, 0]], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 3, 3]], 0.0, epsilon = 0.0);
    }
}
