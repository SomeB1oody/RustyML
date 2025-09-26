use crate::ModelError;
use crate::neural_network::activation::Activation;
use crate::neural_network::optimizer::*;
use crate::neural_network::traits::Layer;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Defines the padding method used in convolutional layers.
///
/// The padding type determines how the input is padded before applying convolution:
/// - `Valid`: No padding is applied, which reduces the output dimensions.
/// - `Same`: Padding is added to preserve the input spatial dimensions in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingType {
    /// No padding is applied. The convolution is only computed where the filter
    /// fully overlaps with the input, resulting in an output with reduced dimensions.
    Valid,

    /// Padding is added around the input to ensure that the output has the same
    /// spatial dimensions as the input (when stride is 1). This is done by adding
    /// zeros around the borders of the input.
    Same,
}

/// A macro that generates the SGD (Stochastic Gradient Descent) parameter update method for convolutional layers.
///
/// This macro creates a standardized `update_parameters_sgd` function that applies SGD updates
/// to both weights and biases of convolutional layers. The generated function checks for the
/// presence of gradients and delegates the actual parameter updates to the SGD optimizer.
macro_rules! update_sgd_conv {
    () => {
        fn update_parameters_sgd(&mut self, lr: f32) {
            if let (Some(weight_grads), Some(bias_grads)) =
                (&self.weight_gradients, &self.bias_gradients)
            {
                SGD::update_sgd_parameters(
                    self.weights.as_slice_mut().unwrap(),
                    weight_grads.as_slice().unwrap(),
                    self.bias.as_slice_mut().unwrap(),
                    bias_grads.as_slice().unwrap(),
                    lr,
                )
            }
        }
    };
}

/// A macro that generates standard function implementations for neural network layers
/// without trainable parameters.
///
/// This macro expands to implementations of the following functions:
/// - `param_count`: Returns 0 as the layer has no trainable parameters
/// - `update_parameters_sgd`: Empty implementation for SGD parameter updates
/// - `update_parameters_adam`: Empty implementation for Adam parameter updates
/// - `update_parameters_rmsprop`: Empty implementation for RMSProp parameter updates
/// - `get_weights`: Returns `LayerWeight::Empty` as the layer has no weights
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> usize {
            // This layer has no trainable parameters
            0
        }

        fn update_parameters_sgd(&mut self, _lr: f32) {
            // This layer have no trainable parameters
        }

        fn update_parameters_adam(
            &mut self,
            _lr: f32,
            _beta1: f32,
            _beta2: f32,
            _epsilon: f32,
            _t: u64,
        ) {
            // This layer have no trainable parameters
        }

        fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
            // This layer have no trainable parameters
        }

        fn get_weights(&self) -> LayerWeight<'_> {
            // This layer has no weights
            LayerWeight::Empty
        }
    };
}

/// A macro that generates standard function implementations for 1D pooling layers.
///
/// This macro expands to implementations of:
/// - `output_shape`: Calculates and returns the output shape after 1D pooling operations
/// - Standard layer functions for layers without trainable parameters
///
/// The macro is designed for pooling layers that operate on 3D tensors with shape
/// `[batch_size, channels, length]` and produce outputs with shape
/// `[batch_size, channels, output_length]`.
///
/// # Generated Functions
///
/// - `output_shape()`: Returns a formatted string representation of the output dimensions.
///   If the input shape is available, it calculates the actual output dimensions using
///   the pooling parameters. Otherwise, returns "Unknown".
/// - All functions from `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the following fields:
/// - `input_shape: Vec<usize>` - The shape of the input tensor
/// - `pool_size: usize` - Size of the pooling window
/// - `stride: usize` - Step size for the pooling operation
macro_rules! layer_functions_1d_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                let output_shape = calculate_output_shape_1d_pooling(
                    &self.input_shape,
                    self.pool_size,
                    self.stride,
                );
                format!(
                    "({}, {}, {})",
                    output_shape[0], output_shape[1], output_shape[2]
                )
            } else {
                String::from("Unknown")
            }
        }

        no_trainable_parameters_layer_functions!();
    };
}

/// A macro that generates standard function implementations for 2D pooling layers.
///
/// This macro expands to implementations of:
/// - `output_shape`: Calculates and returns the output shape after 2D pooling operations
/// - Standard layer functions for layers without trainable parameters
///
/// The macro is designed for pooling layers that operate on 4D tensors with shape
/// `[batch_size, channels, height, width]` and produce outputs with shape
/// `[batch_size, channels, output_height, output_width]`.
///
/// # Generated Functions
///
/// - `output_shape()`: Returns a formatted string representation of the output dimensions.
///   If the input shape is available, it calculates the actual output dimensions using
///   the pooling parameters. Otherwise, returns "Unknown".
/// - All functions from `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the following fields:
/// - `input_shape: Vec<usize>` - The shape of the input tensor
/// - `pool_size: (usize, usize)` - Size of the pooling window as (height, width)
/// - `strides: (usize, usize)` - Step size for the pooling operation as (height_step, width_step)
macro_rules! layer_functions_2d_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                let output_shape = calculate_output_shape_2d_pooling(
                    &self.input_shape,
                    self.pool_size,
                    self.strides,
                );
                format!(
                    "({}, {}, {}, {})",
                    output_shape[0], output_shape[1], output_shape[2], output_shape[3]
                )
            } else {
                String::from("Unknown")
            }
        }

        no_trainable_parameters_layer_functions!();
    };
}

/// A macro that generates standard function implementations for 3D pooling layers.
///
/// This macro expands to implementations of:
/// - `output_shape`: Calculates and returns the output shape after 3D pooling operations
/// - Standard layer functions for layers without trainable parameters
///
/// The macro is designed for pooling layers that operate on 5D tensors with shape
/// `[batch_size, channels, depth, height, width]` and produce outputs with shape
/// `[batch_size, channels, output_depth, output_height, output_width]`.
///
/// # Generated Functions
///
/// - `output_shape()`: Returns a formatted string representation of the output dimensions.
///   If the input shape is available, it calculates the actual output dimensions using
///   the pooling parameters. Otherwise, returns "Unknown".
/// - All functions from `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the following fields:
/// - `input_shape: Vec<usize>` - The shape of the input tensor
/// - `pool_size: (usize, usize, usize)` - Size of the pooling window as (depth, height, width)
/// - `strides: (usize, usize, usize)` - Step size for the pooling operation as (depth_step, height_step, width_step)
macro_rules! layer_functions_3d_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                let output_shape = calculate_output_shape_3d_pooling(
                    &self.input_shape,
                    self.pool_size,
                    self.strides,
                );
                format!(
                    "({}, {}, {}, {}, {})",
                    output_shape[0],
                    output_shape[1],
                    output_shape[2],
                    output_shape[3],
                    output_shape[4]
                )
            } else {
                String::from("Unknown")
            }
        }

        no_trainable_parameters_layer_functions!();
    };
}

/// A macro that generates standard function implementations for global pooling layers.
///
/// This macro expands to implementations of:
/// - `output_shape`: Returns the output shape after global pooling operations
/// - Standard layer functions for layers without trainable parameters
///
/// Global pooling operations reduce the spatial dimensions of the input tensor to a single value
/// per channel by applying a pooling operation (such as max or average) across all spatial
/// dimensions. The output shape preserves only the batch size and channel dimensions.
///
/// # Generated Functions
///
/// - `output_shape()`: Returns a formatted string representation of the output dimensions.
///   If the input shape is available, it returns the batch size and number of channels
///   as `"(batch_size, channels)"`. Otherwise, returns "Unknown".
/// - All functions from `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the following field:
/// - `input_shape: Vec<usize>` - The shape of the input tensor
macro_rules! layer_functions_global_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                format!("({}, {})", self.input_shape[0], self.input_shape[1])
            } else {
                String::from("Unknown")
            }
        }

        no_trainable_parameters_layer_functions!();
    };
}

/// Calculate output shape for 1d pooling layer.
///
/// # Parameters
///
/// - `batch_size` - Number of samples in the batch
/// - `channels` - Number of channels (features) in each sample
/// - `length` - Length of the input along the dimension where pooling/convolution is applied
/// - `pool_size` - Size of the pooling/convolutional window
/// - `stride` - Step size for sliding the window across the input
///
/// # Returns
///
/// * `Vec<usize>` - A vector containing the dimensions of the output tensor in the format: `[batch_size, channels, output_length]`
pub fn calculate_output_shape_1d_pooling(
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
pub fn calculate_output_shape_2d_pooling(
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

/// Calculates the output shape of the 3D layer.
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
pub fn calculate_output_shape_3d_pooling(
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

/// Updates parameters using the Adam optimization algorithm.
///
/// This function performs parameter updates using the Adam (Adaptive Moment Estimation) optimizer,
/// which adapts the learning rate for each parameter using estimates of first and second moments
/// of the gradients. The implementation uses parallel iteration via Rayon for improved performance.
///
/// # Parameters
///
/// - `params` - Mutable slice of model parameters to be updated
/// - `grads` - Slice of gradients corresponding to each parameter
/// - `m` - Mutable slice for first moment estimates (momentum)
/// - `v` - Mutable slice for second moment estimates (velocity/variance)
/// - `lr` - Learning rate for the update step
/// - `beta1` - Exponential decay rate for the first moment estimates (typically 0.9)
/// - `beta2` - Exponential decay rate for the second moment estimates (typically 0.999)
/// - `epsilon` - Small constant for numerical stability
/// - `bias_correction1` - Bias correction term for first moment estimate
/// - `bias_correction2` - Bias correction term for second moment estimate
pub fn update_adam_conv(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) {
    params
        .par_iter_mut()
        .zip(grads.par_iter())
        .zip(m.par_iter_mut())
        .zip(v.par_iter_mut())
        .for_each(|(((param, &grad), m_val), v_val)| {
            // Update momentum and variance
            *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
            *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;

            // Calculate corrected momentum and variance
            let m_corrected = *m_val / bias_correction1;
            let v_corrected = *v_val / bias_correction2;

            // Update parameter
            *param -= lr * m_corrected / (v_corrected.sqrt() + epsilon);
        });
}

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
/// - `filters` - The number of filters (channels) in the output
///
/// # Returns
///
/// * `ArrayD<f32>` - A 4D dynamic array containing the merged results with the specified output shape
pub fn merge_results(
    output_shape: Vec<usize>,
    results: Vec<(usize, Array3<f32>)>,
    filters: usize,
) -> ArrayD<f32> {
    let mut output: ArrayD<f32> = ArrayD::zeros(output_shape.clone());

    // Merge results from each batch into final output
    for (b, batch_output) in results {
        for f in 0..filters {
            for i in 0..output_shape[2] {
                for j in 0..output_shape[3] {
                    output[[b, f, i, j]] = batch_output[[f, i, j]];
                }
            }
        }
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
pub fn compute_row_gradient_sum(
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

/// Updates parameters using the RMSprop optimization algorithm.
///
/// RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm
/// that addresses Adagrad's radically diminishing learning rates by using a moving average
/// of squared gradients instead of accumulating all past squared gradients.
///
/// The algorithm maintains a running average of the squared gradients and uses this to normalize
/// the gradient updates:
/// - cache = rho * cache + (1 - rho) * grad²
/// - param = param - lr * grad / (√cache + epsilon)
///
/// # Parameters
///
/// - `params` - Mutable slice of parameters to be updated
/// - `grads` - Slice of gradients corresponding to each parameter
/// - `cache` - Mutable slice of running averages of squared gradients (cache values)
/// - `rho` - Decay rate for the moving average (typically 0.9 or 0.95)
/// - `epsilon` - Small constant added to prevent division by zero (typically 1e-8)
/// - `lr` - Learning rate that controls the step size
pub fn update_rmsprop(
    params: &mut [f32],
    grads: &[f32],
    cache: &mut [f32],
    rho: f32,
    epsilon: f32,
    lr: f32,
) {
    params
        .par_iter_mut()
        .zip(grads.par_iter())
        .zip(cache.par_iter_mut())
        .for_each(|((param, &grad), cache_val)| {
            // Update cache
            *cache_val = rho * *cache_val + (1.0 - rho) * grad * grad;
            // Update parameters
            *param -= lr * grad / (cache_val.sqrt() + epsilon);
        });
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
///     - For `Valid` padding: Dimensions are reduced based on kernel size and stride
///     - For `Same` padding: Output dimensions are calculated to maintain the input spatial dimensions
///     divided by the stride
pub fn calculate_output_height_and_weight(
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
pub fn pad_tensor_2d(input: &Array2<f32>, pad_h: usize, pad_w: usize) -> Array2<f32> {
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
pub fn calculate_output_shape_2d(
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

/// 1D Average Pooling layer for neural networks.
pub mod average_pooling_1d;
/// A 2D average pooling layer for neural networks.
pub mod average_pooling_2d;
/// 3D Average Pooling Layer
pub mod average_pooling_3d;
/// A 1D convolutional layer for neural networks.
pub mod conv1d;
/// A 2D convolutional layer for neural networks.
pub mod conv2d;
/// A 3D convolutional layer for neural networks
pub mod conv3d;
/// Dense (Fully Connected) layer implementation for neural networks.
pub mod dense;
pub mod depthwise_conv2d;
/// A layer that flattens a 4D tensor into a 2D tensor.
pub mod flatten;
/// Global Average Pooling 1D Layer
pub mod global_average_pooling_1d;
/// Global Average Pooling 2D Layer
pub mod global_average_pooling_2d;
pub mod global_average_pooling_3d;
/// Global Max Pooling 1D Layer
pub mod global_max_pooling_1d;
/// Global Max Pooling Layer
pub mod global_max_pooling_2d;
/// Global Max Pooling 3D Layer
pub mod global_max_pooling_3d;
/// Container for different types of neural network layer weights
pub mod layer_weight;
/// LSTM (Long Short-Term Memory) neural network layer implementation.
pub mod lstm;
/// 1D Max Pooling layer for neural networks.
pub mod max_pooling_1d;
/// Defines a structure for max pooling operation, used to perform max pooling on 2D data.
pub mod max_pooling_2d;
/// 3D data max pooling Layer.
mod max_pooling_3d;
/// A 2D separable convolutional layer for neural networks
pub mod separable_conv2d;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation.
pub mod simple_rnn;

use crate::neural_network::Tensor;
pub use average_pooling_1d::*;
pub use average_pooling_2d::*;
pub use average_pooling_3d::*;
pub use conv1d::*;
pub use conv2d::*;
pub use conv3d::*;
pub use dense::*;
pub use depthwise_conv2d::*;
pub use flatten::*;
pub use global_average_pooling_1d::*;
pub use global_average_pooling_2d::*;
pub use global_average_pooling_3d::*;
pub use global_max_pooling_1d::*;
pub use global_max_pooling_2d::*;
pub use global_max_pooling_3d::*;
pub use layer_weight::*;
pub use lstm::*;
pub use max_pooling_1d::*;
pub use max_pooling_2d::*;
pub use max_pooling_3d::*;
use ndarray::{Array2, Array3, ArrayD, s};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};
pub use separable_conv2d::*;
pub use simple_rnn::*;
