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

// Macros are path-exported via `pub(in ...) use`, so callers import them explicitly
/// Generate the standard `Layer` function implementations for global pooling layers
///
/// Expands to implementations of:
/// - `output_shape`: returns the output shape after global pooling
/// - the standard functions for layers without trainable parameters
///
/// Global pooling reduces the spatial dimensions of the input to a single value per channel
/// by applying a pooling operation (max or average) across all spatial dimensions, so the
/// output shape keeps only the batch size and channel dimensions
///
/// # Generated Functions
///
/// - `output_shape()`: returns a formatted string of the output dimensions; if the input
///   shape is available it returns the batch size and channel count as
///   `"(batch_size, channels)"`, otherwise `"Unknown"`
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
                format!("({}, {})", self.input_shape[0], self.input_shape[1])
            } else {
                String::from("Unknown")
            }
        }

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}

/// Generate the standard `Layer` function implementations for 1D pooling layers
///
/// Expands to implementations of:
/// - `output_shape`: computes and returns the output shape after 1D pooling
/// - the standard functions for layers without trainable parameters
///
/// Designed for pooling layers that operate on 3D tensors with shape
/// `[batch_size, channels, length]` and produce outputs with shape
/// `[batch_size, channels, output_length]`
///
/// # Generated Functions
///
/// - `output_shape()`: returns a formatted string of the output dimensions; if the input
///   shape is available it computes the dimensions from the pooling parameters, otherwise
///   returns `"Unknown"`
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

/// Generate the standard `Layer` function implementations for 2D pooling layers
///
/// Expands to implementations of:
/// - `output_shape`: computes and returns the output shape after 2D pooling
/// - the standard functions for layers without trainable parameters
///
/// Designed for pooling layers that operate on 4D tensors with shape
/// `[batch_size, channels, height, width]` and produce outputs with shape
/// `[batch_size, channels, output_height, output_width]`
///
/// # Generated Functions
///
/// - `output_shape()`: returns a formatted string of the output dimensions; if the input
///   shape is available it computes the dimensions from the pooling parameters, otherwise
///   returns `"Unknown"`
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

/// Generate the standard `Layer` function implementations for 3D pooling layers
///
/// Expands to implementations of:
/// - `output_shape`: computes and returns the output shape after 3D pooling
/// - the standard functions for layers without trainable parameters
///
/// Designed for pooling layers that operate on 5D tensors with shape
/// `[batch_size, channels, depth, height, width]` and produce outputs with shape
/// `[batch_size, channels, output_depth, output_height, output_width]`
///
/// # Generated Functions
///
/// - `output_shape()`: returns a formatted string of the output dimensions; if the input
///   shape is available it computes the dimensions from the pooling parameters, otherwise
///   returns `"Unknown"`
/// - all functions from the `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `input_shape: Vec<usize>` - shape of the input tensor
/// - `pool_size: (usize, usize, usize)` - size of the pooling window as (depth, height, width)
/// - `strides: (usize, usize, usize)` - step size for the pooling operation as (depth_step, height_step, width_step)
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
