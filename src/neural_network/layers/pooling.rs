//! Pooling layers and the shared helpers that build them
//!
//! Re-exports every pooling layer (average, max, and their global variants in 1D/2D/3D)
//! and defines the macros that generate the common `Layer` implementations for them

/// 1D average pooling layer
pub mod average_pooling_1d;
/// 2D average pooling layer
pub mod average_pooling_2d;
/// 3D average pooling layer
pub mod average_pooling_3d;
/// 1D global average pooling layer
pub mod global_average_pooling_1d;
/// 2D global average pooling layer
pub mod global_average_pooling_2d;
/// 3D global average pooling layer
pub mod global_average_pooling_3d;
/// 1D global max pooling layer
pub mod global_max_pooling_1d;
/// 2D global max pooling layer
pub mod global_max_pooling_2d;
/// 3D global max pooling layer
pub mod global_max_pooling_3d;
/// 1D max pooling layer
pub mod max_pooling_1d;
/// 2D max pooling layer
pub mod max_pooling_2d;
/// 3D max pooling layer
pub mod max_pooling_3d;
/// Dimension-generic pooling engine shared by every pooling layer
pub(crate) mod pooling_engine;
/// Input validation functions for pooling layers
mod validation;

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

// The `pub(in ...) use` lines below export the macros by path, so callers import them explicitly
/// Generates the standard `Layer` function implementations for global pooling layers
///
/// Global pooling reduces the spatial dimensions of the input to a single value per channel. It
/// applies a pooling operation (max or average) across all spatial dimensions. The output shape
/// keeps only the batch size and the channel count
///
/// # Generated Functions
///
/// - `output_shape()` - returns a formatted string of the output dimensions. If the input shape
///   is known, it returns the batch size and the trailing channel count as
///   `"(batch_size, channels)"`. Otherwise it returns `"Unknown"`
/// - all functions from the `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the field:
/// - `input_shape: Vec<usize>` - shape of the input tensor
macro_rules! layer_functions_global_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                format!(
                    "({}, {})",
                    self.input_shape[0],
                    self.input_shape[self.input_shape.len() - 1]
                )
            } else {
                String::from("Unknown")
            }
        }

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}

/// Generates the standard `Layer` function implementations for 1D pooling layers
///
/// Applies to pooling layers that operate on 3D tensors with shape
/// `[batch_size, length, channels]` and produce outputs with shape
/// `[batch_size, output_length, channels]`
///
/// # Generated Functions
///
/// - `output_shape()` - returns a formatted string of the output dimensions. If the input shape
///   is known, it computes the dimensions from the pooling parameters. Otherwise it returns
///   `"Unknown"`
/// - all functions from the `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `input_shape: Vec<usize>` - shape of the input tensor
/// - `pool_size: usize` - size of the pooling window
/// - `stride: usize` - step size for the pooling operation
macro_rules! layer_functions_1d_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                let output_shape = calculate_output_shape_1d_pooling(
                    &self.input_shape,
                    self.pool_size,
                    self.stride,
                    self.padding,
                );
                format!(
                    "({}, {}, {})",
                    output_shape[0], output_shape[1], output_shape[2]
                )
            } else {
                String::from("Unknown")
            }
        }

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}

/// Generates the standard `Layer` function implementations for 2D pooling layers
///
/// Applies to pooling layers that operate on 4D tensors with shape
/// `[batch_size, height, width, channels]` and produce outputs with shape
/// `[batch_size, output_height, output_width, channels]`
///
/// # Generated Functions
///
/// - `output_shape()` - returns a formatted string of the output dimensions. If the input shape
///   is known, it computes the dimensions from the pooling parameters. Otherwise it returns
///   `"Unknown"`
/// - all functions from the `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `input_shape: Vec<usize>` - shape of the input tensor
/// - `pool_size: (usize, usize)` - size of the pooling window as (height, width)
/// - `strides: (usize, usize)` - step size for the pooling operation as (height_step, width_step)
macro_rules! layer_functions_2d_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                let output_shape = calculate_output_shape_2d_pooling(
                    &self.input_shape,
                    self.pool_size,
                    self.strides,
                    self.padding,
                );
                format!(
                    "({}, {}, {}, {})",
                    output_shape[0], output_shape[1], output_shape[2], output_shape[3]
                )
            } else {
                String::from("Unknown")
            }
        }

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}

/// Generates the standard `Layer` function implementations for 3D pooling layers
///
/// Applies to pooling layers that operate on 5D tensors with shape
/// `[batch_size, depth, height, width, channels]`. These layers produce outputs with shape
/// `[batch_size, output_depth, output_height, output_width, channels]`
///
/// # Generated Functions
///
/// - `output_shape()` - returns a formatted string of the output dimensions. If the input shape
///   is known, it computes the dimensions from the pooling parameters. Otherwise it returns
///   `"Unknown"`
/// - all functions from the `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `input_shape: Vec<usize>` - shape of the input tensor
/// - `pool_size: (usize, usize, usize)` - size of the pooling window as (depth, height, width)
/// - `strides: (usize, usize, usize)` - step size for the pooling operation as
///   (depth_step, height_step, width_step)
macro_rules! layer_functions_3d_pooling {
    () => {
        fn output_shape(&self) -> String {
            if !self.input_shape.is_empty() {
                let output_shape = calculate_output_shape_3d_pooling(
                    &self.input_shape,
                    self.pool_size,
                    self.strides,
                    self.padding,
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

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}
pub(in crate::neural_network::layers::pooling) use layer_functions_1d_pooling;
pub(in crate::neural_network::layers::pooling) use layer_functions_2d_pooling;
pub(in crate::neural_network::layers::pooling) use layer_functions_3d_pooling;
pub(in crate::neural_network::layers::pooling) use layer_functions_global_pooling;
