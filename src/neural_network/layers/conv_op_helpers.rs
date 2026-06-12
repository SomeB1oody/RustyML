//! Convolution-internal helpers: weight-gradient accumulation and 2D/4D zero-padding

use crate::neural_network::Tensor;
use ndarray::{Array2, ArrayD, s};

/// Accumulates the weight-gradient sum for a single kernel row during convolution backprop
///
/// Sums gradient[b,f,i,j] * input[b,c,i_pos,j_pos] across output columns j, skipping any column
/// whose mapped input position j_pos = j * stride_1 + w falls outside the input width
///
/// # Parameters
///
/// - `gradient` - gradient tensor
/// - `input` - input tensor
/// - `b` - batch index
/// - `f` - filter index
/// - `c` - channel index
/// - `i` - output row index
/// - `i_pos` - corresponding row position in the input tensor
/// - `w` - kernel width index
/// - `grad_shape` - shape of the gradient tensor
/// - `input_shape` - shape of the input tensor
/// - `stride_1` - stride in the width direction
///
/// # Returns
///
/// - `f32` - accumulated weight-gradient sum for this row
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_row_gradient_sum(
    gradient: &Tensor,
    input: &Tensor,
    b: usize,
    f: usize,
    c: usize,
    i: usize,
    i_pos: usize,
    w: usize,
    grad_shape: &[usize],
    input_shape: &[usize],
    stride_1: usize,
) -> f32 {
    let mut sum = 0.0;

    for j in 0..grad_shape[3] {
        let j_pos = j * stride_1 + w;
        if j_pos < input_shape[3] {
            sum += gradient[[b, f, i, j]] * input[[b, c, i_pos, j_pos]];
        }
    }

    sum
}

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

    /// `compute_row_gradient_sum` accumulates gradient * input over in-bounds output columns
    #[test]
    fn test_compute_row_gradient_sum_basic() {
        let gradient: Tensor = array![[[[2.0_f32, 3.0]]]].into_dyn(); // [1,1,1,2]
        let input: Tensor = array![[[[1.0_f32, 10.0, 100.0, 1000.0]]]].into_dyn(); // [1,1,1,4]
        let grad_shape = gradient.shape().to_vec();
        let input_shape = input.shape().to_vec();

        let sum = compute_row_gradient_sum(
            &gradient,
            &input,
            0,
            0,
            0,
            0,
            0,
            1, // w (kernel width index)
            &grad_shape,
            &input_shape,
            1, // stride_1
        );
        assert_abs_diff_eq!(sum, 320.0_f32, epsilon = 1e-6);
    }

    /// `compute_row_gradient_sum` maps output columns to input positions via stride_1 and w
    #[test]
    fn test_compute_row_gradient_sum_strided() {
        let gradient: Tensor = array![[[[2.0_f32, 3.0]]]].into_dyn();
        let input: Tensor = array![[[[1.0_f32, 10.0, 100.0, 1000.0]]]].into_dyn();
        let grad_shape = gradient.shape().to_vec();
        let input_shape = input.shape().to_vec();

        let sum = compute_row_gradient_sum(
            &gradient,
            &input,
            0,
            0,
            0,
            0,
            0,
            1,
            &grad_shape,
            &input_shape,
            2,
        );
        assert_abs_diff_eq!(sum, 3020.0_f32, epsilon = 1e-6);
    }

    /// `compute_row_gradient_sum` drops any output column whose mapped position exceeds the input width
    #[test]
    fn test_compute_row_gradient_sum_drops_out_of_bounds() {
        let gradient: Tensor = array![[[[2.0_f32, 3.0]]]].into_dyn();
        let input: Tensor = array![[[[1.0_f32, 10.0]]]].into_dyn(); // width 2
        let grad_shape = gradient.shape().to_vec();
        let input_shape = input.shape().to_vec();

        let sum = compute_row_gradient_sum(
            &gradient,
            &input,
            0,
            0,
            0,
            0,
            0,
            1,
            &grad_shape,
            &input_shape,
            2,
        );
        assert_abs_diff_eq!(sum, 20.0_f32, epsilon = 1e-6);
    }

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
