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
