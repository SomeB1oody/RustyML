//! Pooling layers and the shared helpers that build them
//!
//! Re-exports every pooling layer (average, max, and their global variants in 1D/2D/3D)
//! and defines the macros that generate the shared `UnaryLayer` and `LayerBase` implementations
//! for them

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
/// Generates the `UnaryLayer` function implementations for global pooling layers
///
/// Global pooling reduces the spatial dimensions of the input to 1 value per channel. It
/// applies a pooling operation (max or average) across all spatial dimensions. The output shape
/// keeps only the batch size and the channel count
///
/// # Parameters
///
/// - `layer` - the layer name, used in error messages
/// - `rank` - the input rank the layer accepts, used in error messages
///
/// # Generated Functions
///
/// - `build()` - records the shape the layer serves, and refuses a rank it cannot pool
/// - `compute_output_shape()` - drops every spatial axis and keeps the batch axis and the
///   channel axis
///
/// # Requirements
///
/// The implementing struct must have the field:
/// - `built: Option<Shape>` - the shape the layer was built for
macro_rules! layer_functions_global_pooling {
    ($layer:literal, $rank:literal) => {
        /// Records the shape the layer serves. The layer holds no array, so nothing is
        /// allocated. The shape algebra checks the rank
        fn build(
            &mut self,
            input: &$crate::neural_network::Shape,
        ) -> Result<(), $crate::error::Error> {
            let Some(built) = $crate::neural_network::layers::validation::start_build(
                &self.built,
                $layer,
                input,
            )?
            else {
                return Ok(());
            };
            $crate::neural_network::traits::UnaryLayer::compute_output_shape(self, &built)?;
            self.built = Some(built);
            Ok(())
        }

        fn compute_output_shape(
            &self,
            input: &$crate::neural_network::Shape,
        ) -> Result<$crate::neural_network::Shape, $crate::error::Error> {
            input.check_rank($layer, $rank)?;
            let (batch, tail) = input.split_batch($layer)?;
            // Global pooling reduces every spatial axis to 1 value, so the batch axis and the
            // channel axis are all that is left
            Ok($crate::neural_network::Shape::from_batch(
                batch,
                &[tail[tail.len() - 1]],
            ))
        }
    };
}

/// Generates the `LayerBase` function implementations that every pooling layer shares
///
/// # Generated Functions
///
/// - `known_input_shapes()` - the shape the layer was built for
/// - `is_built()` - whether the layer holds a build
/// - `build_config()` - the same shape with the batch axis freed, for a checkpoint
/// - all functions from the `no_trainable_parameters_layer_functions!()` macro
///
/// # Requirements
///
/// The implementing struct must have the field:
/// - `built: Option<Shape>` - the shape the layer was built for
macro_rules! layer_base_functions_pooling {
    () => {
        $crate::neural_network::layers::built_layer_shape_functions!();

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}
pub(in crate::neural_network::layers::pooling) use layer_base_functions_pooling;

/// Generates the `UnaryLayer` function implementations for 1D pooling layers
///
/// Applies to pooling layers that operate on 3D tensors with shape
/// `[batch_size, length, channels]` and produce outputs with shape
/// `[batch_size, output_length, channels]`
///
/// # Parameters
///
/// - `layer` - the layer name, used in error messages
///
/// # Generated Functions
///
/// - `build()` - records the shape the layer serves, and refuses a shape the window
///   does not fit
/// - `compute_output_shape()` - applies the pooling window to every spatial axis, and keeps the
///   batch axis and the channel axis
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `built: Option<Shape>` - the shape the layer was built for
/// - `pool_size: usize` - size of the pooling window
/// - `stride: usize` - step size for the pooling operation
/// - `padding: PaddingType` - the padding mode applied before pooling
macro_rules! layer_functions_1d_pooling {
    ($layer:literal) => {
        /// Records the shape the window runs over. The layer holds no array, so nothing is
        /// allocated. The shape algebra checks the rank, the extents, and the window
        fn build(
            &mut self,
            input: &$crate::neural_network::Shape,
        ) -> Result<(), $crate::error::Error> {
            let Some(built) = $crate::neural_network::layers::validation::start_build(
                &self.built,
                $layer,
                input,
            )?
            else {
                return Ok(());
            };
            $crate::neural_network::traits::UnaryLayer::compute_output_shape(self, &built)?;
            self.built = Some(built);
            Ok(())
        }

        fn compute_output_shape(
            &self,
            input: &$crate::neural_network::Shape,
        ) -> Result<$crate::neural_network::Shape, $crate::error::Error> {
            input.check_rank($layer, 3)?;
            let (batch, tail) = input.split_batch($layer)?;
            validate_pool_size_1d(self.pool_size, tail[0])?;
            // The calculator indexes the batch axis at 0, so a placeholder value is prepended
            let mut dims = vec![0];
            dims.extend(tail);
            let output_shape =
                calculate_output_shape_1d_pooling(&dims, self.pool_size, self.stride, self.padding);
            Ok($crate::neural_network::Shape::from_batch(
                batch,
                &output_shape[1..],
            ))
        }
    };
}

/// Generates the `UnaryLayer` function implementations for 2D pooling layers
///
/// Applies to pooling layers that operate on 4D tensors with shape
/// `[batch_size, height, width, channels]` and produce outputs with shape
/// `[batch_size, output_height, output_width, channels]`
///
/// # Parameters
///
/// - `layer` - the layer name, used in error messages
///
/// # Generated Functions
///
/// - `build()` - records the shape the layer serves, and refuses a shape the window
///   does not fit
/// - `compute_output_shape()` - applies the pooling window to every spatial axis, and keeps the
///   batch axis and the channel axis
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `built: Option<Shape>` - the shape the layer was built for
/// - `pool_size: (usize, usize)` - size of the pooling window as (height, width)
/// - `strides: (usize, usize)` - step size for the pooling operation as (height_step, width_step)
/// - `padding: PaddingType` - the padding mode applied before pooling
macro_rules! layer_functions_2d_pooling {
    ($layer:literal) => {
        /// Records the shape the window runs over. The layer holds no array, so nothing is
        /// allocated. The shape algebra checks the rank, the extents, and the window
        fn build(
            &mut self,
            input: &$crate::neural_network::Shape,
        ) -> Result<(), $crate::error::Error> {
            let Some(built) = $crate::neural_network::layers::validation::start_build(
                &self.built,
                $layer,
                input,
            )?
            else {
                return Ok(());
            };
            $crate::neural_network::traits::UnaryLayer::compute_output_shape(self, &built)?;
            self.built = Some(built);
            Ok(())
        }

        fn compute_output_shape(
            &self,
            input: &$crate::neural_network::Shape,
        ) -> Result<$crate::neural_network::Shape, $crate::error::Error> {
            input.check_rank($layer, 4)?;
            let (batch, tail) = input.split_batch($layer)?;
            validate_pool_size_2d(self.pool_size, tail[0], tail[1])?;
            // The calculator indexes the batch axis at 0, so a placeholder value is prepended
            let mut dims = vec![0];
            dims.extend(tail);
            let output_shape = calculate_output_shape_2d_pooling(
                &dims,
                self.pool_size,
                self.strides,
                self.padding,
            );
            Ok($crate::neural_network::Shape::from_batch(
                batch,
                &output_shape[1..],
            ))
        }
    };
}

/// Generates the `UnaryLayer` function implementations for 3D pooling layers
///
/// Applies to pooling layers that operate on 5D tensors with shape
/// `[batch_size, depth, height, width, channels]` and produce outputs with shape
/// `[batch_size, output_depth, output_height, output_width, channels]`
///
/// # Parameters
///
/// - `layer` - the layer name, used in error messages
///
/// # Generated Functions
///
/// - `build()` - records the shape the layer serves, and refuses a shape the window
///   does not fit
/// - `compute_output_shape()` - applies the pooling window to every spatial axis, and keeps the
///   batch axis and the channel axis
///
/// # Requirements
///
/// The implementing struct must have the fields:
/// - `built: Option<Shape>` - the shape the layer was built for
/// - `pool_size: (usize, usize, usize)` - size of the pooling window as (depth, height, width)
/// - `strides: (usize, usize, usize)` - step size for the pooling operation as
///   (depth_step, height_step, width_step)
/// - `padding: PaddingType` - the padding mode applied before pooling
macro_rules! layer_functions_3d_pooling {
    ($layer:literal) => {
        /// Records the shape the window runs over. The layer holds no array, so nothing is
        /// allocated. The shape algebra checks the rank, the extents, and the window
        fn build(
            &mut self,
            input: &$crate::neural_network::Shape,
        ) -> Result<(), $crate::error::Error> {
            let Some(built) = $crate::neural_network::layers::validation::start_build(
                &self.built,
                $layer,
                input,
            )?
            else {
                return Ok(());
            };
            $crate::neural_network::traits::UnaryLayer::compute_output_shape(self, &built)?;
            self.built = Some(built);
            Ok(())
        }

        fn compute_output_shape(
            &self,
            input: &$crate::neural_network::Shape,
        ) -> Result<$crate::neural_network::Shape, $crate::error::Error> {
            input.check_rank($layer, 5)?;
            let (batch, tail) = input.split_batch($layer)?;
            validate_pool_size_3d(self.pool_size, tail[0], tail[1], tail[2])?;
            // The calculator indexes the batch axis at 0, so a placeholder value is prepended
            let mut dims = vec![0];
            dims.extend(tail);
            let output_shape = calculate_output_shape_3d_pooling(
                &dims,
                self.pool_size,
                self.strides,
                self.padding,
            );
            Ok($crate::neural_network::Shape::from_batch(
                batch,
                &output_shape[1..],
            ))
        }
    };
}
pub(in crate::neural_network::layers::pooling) use layer_functions_1d_pooling;
pub(in crate::neural_network::layers::pooling) use layer_functions_2d_pooling;
pub(in crate::neural_network::layers::pooling) use layer_functions_3d_pooling;
pub(in crate::neural_network::layers::pooling) use layer_functions_global_pooling;
