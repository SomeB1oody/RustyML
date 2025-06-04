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
pub mod depthwise_conv2d;
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
/// A module containing helper functions for neural network layers.
pub mod helper_functions;
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
/// A 2D separable convolutional layer for neural networks
pub mod separable_conv2d;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation.
mod simple_rnn;

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
pub use padding_type::*;
pub use separable_conv2d::*;
pub use simple_rnn::*;
