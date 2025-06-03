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

        fn get_weights(&self) -> LayerWeight {
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
/// A layer that flattens a 4D tensor into a 2D tensor.
pub mod flatten;
/// Global Average Pooling 1D Layer
pub mod global_average_pooling_1d;
/// Global Average Pooling 2D Layer
pub mod global_average_pooling_2d;
pub mod global_average_pooling_3d;
/// Global Max Pooling 1D Layer
mod global_max_pooling_1d;
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
/// Defines the padding method used in convolutional layers
pub mod padding_type;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation.
pub mod simple_rnn;

pub use average_pooling_1d::*;
pub use average_pooling_2d::*;
pub use average_pooling_3d::*;
pub use conv1d::*;
pub use conv2d::*;
pub use conv3d::*;
pub use dense::*;
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
pub use padding_type::*;
pub use simple_rnn::*;

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
fn calculate_output_shape_1d_pooling(
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
fn calculate_output_shape_2d_pooling(
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
/// A vector containing the calculated output shape, formatted as \[batch_size, channels, output_depth, output_height, output_width\].
fn calculate_output_shape_3d_pooling(
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
/// * `params` - Mutable slice of model parameters to be updated
/// * `grads` - Slice of gradients corresponding to each parameter
/// * `m` - Mutable slice for first moment estimates (momentum)
/// * `v` - Mutable slice for second moment estimates (velocity/variance)
/// * `lr` - Learning rate for the update step
/// * `beta1` - Exponential decay rate for the first moment estimates (typically 0.9)
/// * `beta2` - Exponential decay rate for the second moment estimates (typically 0.999)
/// * `epsilon` - Small constant for numerical stability
/// * `bias_correction1` - Bias correction term for first moment estimate
/// * `bias_correction2` - Bias correction term for second moment estimate
fn update_adam_conv(
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
    use rayon::prelude::*;

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
