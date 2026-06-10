//! Convolution-internal helpers: parallel output assembly, weight-gradient accumulation, and 2D
//! zero-padding.

use crate::neural_network::Tensor;
use ndarray::{Array2, Array3, ArrayD, Axis, s};

/// Merges parallel computation results from multiple batches into a single output tensor.
///
/// This function is designed to efficiently combine results from parallel batch processing
/// operations, particularly useful in convolutional neural network computations where
/// different batches are processed independently and their results need to be merged.
///
/// # Parameters
///
/// - `output_shape` - The desired shape of the final output tensor as a vector of dimensions.
///   Expected format: `[batch_size, filters, height, width]`
/// - `results` - A vector of tuples where each tuple contains:
///   - `usize`: The batch index indicating which batch this result belongs to
///   - `Array3<f32>`: The 3D computation result for that batch with shape `[filters, height, width]`
///
/// # Returns
///
/// * `ArrayD<f32>` - A 4D dynamic array containing the merged results with the specified output shape
pub(super) fn merge_results(
    output_shape: Vec<usize>,
    results: Vec<(usize, Array3<f32>)>,
) -> ArrayD<f32> {
    let mut output: ArrayD<f32> = ArrayD::zeros(output_shape);

    // Merge each batch's [filters, height, width] result into the output. Assigning the whole
    // sub-view in one shot is far cheaper than the previous element-by-element index writes.
    for (b, batch_output) in results {
        output.index_axis_mut(Axis(0), b).assign(&batch_output);
    }

    output
}

/// Computes the accumulated sum of weight gradients for a single row
///
/// In the weight gradient calculation during convolutional layer backpropagation,
/// this function accumulates gradients across all columns for a specified row (position i).
/// It handles gradient computation for convolution operations along the width dimension.
///
/// # Parameters
///
/// - `gradient` - Reference to the gradient tensor
/// - `input` - Reference to the input tensor
/// - `b` - Batch index
/// - `f` - Filter index
/// - `c` - Channel index
/// - `i` - Current row index
/// - `i_pos` - Corresponding row position in the input tensor
/// - `w` - Convolution kernel width index
/// - `grad_shape` - Shape of the gradient tensor
/// - `input_shape` - Shape of the input tensor
/// - `stride_1` - Stride in the width direction
///
/// # Returns
///
/// * `f32` - The computed accumulated sum of weight gradients for this row
#[allow(clippy::too_many_arguments)] // convolution geometry params are all needed
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

/// Applies padding to a 2D tensor.
///
/// This function adds zero padding to a 2D input tensor to increase its spatial dimensions.
/// The padding is applied symmetrically around the tensor, with any odd padding values
/// distributed with more padding on the right/bottom edges.
///
/// # Parameters
///
/// - `input` - A reference to the 2D input tensor to be padded
/// - `pad_h` - Total padding to be applied in the height dimension
/// - `pad_w` - Total padding to be applied in the width dimension
///
/// # Returns
///
/// * `Array2<f32>` - A new 2D tensor with the specified padding applied
pub(super) fn pad_tensor_2d(input: &Array2<f32>, pad_h: usize, pad_w: usize) -> Array2<f32> {
    let (input_height, input_width) = input.dim();

    // Calculate padding for each side
    let pad_top = pad_h / 2;
    let pad_left = pad_w / 2;

    // Calculate output dimensions
    let output_height = input_height + pad_h;
    let output_width = input_width + pad_w;

    // Create output tensor filled with zeros
    let mut output = Array2::zeros((output_height, output_width));

    // Copy input data to the center of the output tensor
    output
        .slice_mut(s![
            pad_top..pad_top + input_height,
            pad_left..pad_left + input_width
        ])
        .assign(input);

    output
}

/// Applies zero-padding to the two trailing (spatial) (`H`, `W`) axes of a 4D
/// `[batch, channels, H, W]` tensor, leaving the batch and channel axes untouched.
///
/// Padding is symmetric with any odd remainder placed on the trailing edge
/// (`pad_before = pad / 2`), matching [`pad_tensor_2d`] and the convolution engine. Returns a
/// plain clone when both pads are zero, so callers can treat it as a no-op for `Valid` padding.
///
/// # Parameters
///
/// - `input` - A 4D `[batch, channels, H, W]` tensor to be padded
/// - `pad_h` - Total padding to add along the height axis
/// - `pad_w` - Total padding to add along the width axis
///
/// # Returns
///
/// * `Tensor` - A new 4D tensor with the spatial axes zero-padded
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

    /// `merge_results` writes each batch's [filters, height, width] result into the matching
    /// batch slot of a zero-filled [batch, filters, height, width] tensor. Build two distinct
    /// per-batch results (filters=1, 2x2) and check every element lands in the slot the index
    /// names; both batches are supplied here, so no slot stays zero.
    #[test]
    fn test_merge_results_places_each_batch() {
        // batch=2, filters=1, height=2, width=2
        let b0 = array![[[1.0_f32, 2.0], [3.0, 4.0]]]; // Array3 [1,2,2]
        let b1 = array![[[5.0_f32, 6.0], [7.0, 8.0]]];
        let out = merge_results(vec![2, 1, 2, 2], vec![(0, b0.clone()), (1, b1.clone())]);

        assert_eq!(out.shape(), &[2, 1, 2, 2]);
        // Batch 0 slot holds b0, batch 1 slot holds b1, element for element (filters axis = 0).
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(out[[0, 0, i, j]], b0[[0, i, j]], epsilon = 0.0);
                assert_abs_diff_eq!(out[[1, 0, i, j]], b1[[0, i, j]], epsilon = 0.0);
            }
        }
    }

    /// A batch index that is not supplied stays zero-filled. With batch=2 but only batch 1
    /// provided, the whole batch-0 slice must remain 0.0.
    #[test]
    fn test_merge_results_missing_batch_stays_zero() {
        let b1 = array![[[9.0_f32, 9.0], [9.0, 9.0]]];
        let out = merge_results(vec![2, 1, 2, 2], vec![(1, b1)]);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(out[[0, 0, i, j]], 0.0, epsilon = 0.0);
                assert_abs_diff_eq!(out[[1, 0, i, j]], 9.0, epsilon = 0.0);
            }
        }
    }

    /// `compute_row_gradient_sum` accumulates, over output columns j with j_pos = j*stride + w
    /// inside the input width, gradient[b,f,i,j] * input[b,c,i_pos,j_pos].
    ///
    /// Setup: gradient shape [1,1,1,2] with row i=0 = [g0, g1] = [2, 3]; input shape [1,1,1,4];
    /// stride_1 = 1, kernel index w = 1, i_pos = 0, b=f=c=0.
    ///   j=0 -> j_pos = 0*1+1 = 1 (<4): term = g0 * input[..,1] = 2 * 10 = 20
    ///   j=1 -> j_pos = 1*1+1 = 2 (<4): term = g1 * input[..,2] = 3 * 100 = 300
    /// Expected sum = 320.
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

    /// Strided accumulation. With stride_1 = 2 and w = 1 on the same [1,1,1,4] input:
    ///   j=0 -> j_pos = 0*2+1 = 1 (<4): term = g0 * input[..,1] = 2 * 10 = 20
    ///   j=1 -> j_pos = 1*2+1 = 3 (<4): term = g1 * input[..,3] = 3 * 1000 = 3000
    /// Expected sum = 3020 (both terms pass the `< input_shape[3]` guard).
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

    /// A column whose j_pos lands outside the input width is dropped. gradient [1,1,1,2] = [2,3],
    /// input width 2, stride_1 = 2, w = 1:
    ///   j=0 -> j_pos = 1 (<2): term = 2 * input[..,1] = 2 * 10 = 20
    ///   j=1 -> j_pos = 3 (>=2): skipped.
    /// Expected sum = 20.
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

    /// `pad_tensor_2d` grows a HxW tensor by (pad_h, pad_w) with the original placed at offset
    /// (pad_h/2, pad_w/2) and zeros elsewhere (odd remainder on the trailing edge). A 2x2 input
    /// padded by (2, 2) yields a 4x4 with the input centered at rows/cols 1..3.
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

    /// Odd padding puts the extra row/column on the trailing (bottom/right) edge: pad_top =
    /// pad_h/2 = 0 when pad_h = 1, so a 1x1 input padded by (1, 1) becomes 2x2 with the value
    /// at [0,0] and the extra zero band on the bottom and right.
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

    /// `pad_tensor_4d_spatial` is a no-op clone when both pads are zero: shape and every element
    /// are preserved, batch and channel axes untouched.
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

    /// With nonzero pads only the trailing two (spatial) axes grow; batch/channel are unchanged.
    /// A [1,1,2,2] input padded by (2, 2) becomes [1,1,4,4] with the 2x2 block centered at
    /// spatial rows/cols 1..3 (matching pad_tensor_2d's placement).
    #[test]
    fn test_pad_tensor_4d_spatial_pads_only_spatial_axes() {
        let input: Tensor = array![[[[1.0_f32, 2.0], [3.0, 4.0]]]].into_dyn();
        let out = pad_tensor_4d_spatial(&input, 2, 2);
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
        // Centered block.
        assert_abs_diff_eq!(out[[0, 0, 1, 1]], 1.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 1, 2]], 2.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 2, 1]], 3.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 2, 2]], 4.0, epsilon = 0.0);
        // Corners are zero padding.
        assert_abs_diff_eq!(out[[0, 0, 0, 0]], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(out[[0, 0, 3, 3]], 0.0, epsilon = 0.0);
    }
}
