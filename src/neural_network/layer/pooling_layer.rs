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

/// 1D Average Pooling Layer
pub mod average_pooling_1d;
/// 2D Average Pooling Layer
pub mod average_pooling_2d;
/// 3D Average Pooling Layer
pub mod average_pooling_3d;
/// Global Average Pooling 1D Layer
pub mod global_average_pooling_1d;
/// Global Average Pooling 2D Layer
pub mod global_average_pooling_2d;
/// Global Average Pooling 3D Layer
pub mod global_average_pooling_3d;
/// Global Max Pooling layer 1D Layer
pub mod global_max_pooling_1d;
/// Global Max Pooling layer 2D Layer
pub mod global_max_pooling_2d;
/// Global Max Pooling layer 3D Layer
pub mod global_max_pooling_3d;
/// Input validation functions for pooling layers
mod input_validation_function;
/// Max Pooling layer 1D Layer
pub mod max_pooling_1d;
/// Max Pooling layer 2D Layer
pub mod max_pooling_2d;
/// Max Pooling layer 3D Layer
pub mod max_pooling_3d;

pub use average_pooling_1d::AveragePooling1D;
pub use average_pooling_2d::AveragePooling2D;
pub use average_pooling_3d::AveragePooling3D;
pub use global_average_pooling_1d::GlobalAveragePooling1D;
pub use global_average_pooling_2d::GlobalAveragePooling2D;
pub use global_average_pooling_3d::GlobalAveragePooling3D;
pub use global_max_pooling_1d::GlobalMaxPooling1D;
pub use global_max_pooling_2d::GlobalMaxPooling2D;
pub use global_max_pooling_3d::GlobalMaxPooling3D;
pub use max_pooling_1d::MaxPooling1D;
pub use max_pooling_2d::MaxPooling2D;
pub use max_pooling_3d::MaxPooling3D;
