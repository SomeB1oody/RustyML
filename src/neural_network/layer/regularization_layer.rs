use super::*;

/// Common backward pass implementation for all dropout layers.
///
/// This function implements the backward pass logic that is shared across
/// all dropout variants (Dropout, SpatialDropout1D/2D/3D).
///
/// # Parameters
///
/// - `grad_output` - Gradient from the next layer
/// - `mask` - The dropout mask applied during forward pass
/// - `training` - Whether the layer is in training mode
/// - `rate` - The dropout rate
///
/// # Returns
///
/// * `Result<Tensor, ModelError>` - Gradient to pass to previous layer
pub fn dropout_backward(
    grad_output: &Tensor,
    mask: &Option<Tensor>,
    training: bool,
    rate: f32,
) -> Result<Tensor, ModelError> {
    if !training || rate == 0.0 {
        // During inference or if rate is 0, pass gradient through unchanged
        return Ok(grad_output.clone());
    }

    if rate == 1.0 {
        // If dropout rate is 1.0, return zero gradients
        return Ok(Tensor::zeros(grad_output.raw_dim()));
    }

    // Apply the same mask to the gradient
    if let Some(mask) = mask {
        let scale = 1.0 / (1.0 - rate);
        let grad_input = grad_output * mask * scale;
        Ok(grad_input)
    } else {
        Err(ModelError::ProcessingError(
            "Forward pass has not been run".to_string(),
        ))
    }
}

/// Common output shape implementation for all dropout layers.
///
/// This function formats the input shape into a string representation
/// that is shared across all dropout variants.
///
/// # Parameters
///
/// - `input_shape` - The input shape vector
///
/// # Returns
///
/// * `String` - Formatted output shape string
pub fn dropout_output_shape(input_shape: &[usize]) -> String {
    if !input_shape.is_empty() {
        format!(
            "({})",
            input_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::from("Unknown")
    }
}

/// Applies threshold to create binary mask with parallel or sequential computation.
///
/// This function is used by all spatial dropout layers to convert a random mask
/// into a binary mask based on the dropout rate. For larger masks, parallel computation
/// is used for better performance.
///
/// # Parameters
///
/// - `mask_2d` - The random mask to convert to binary (modified in place)
/// - `rate` - The dropout rate threshold
/// - `parallel_threshold` - The threshold for using parallel computation
pub fn apply_spatial_dropout_threshold(mask_2d: &mut Tensor, rate: f32, parallel_threshold: usize) {
    let total_elements = mask_2d.len();

    // Apply threshold to create binary mask with parallel or sequential computation
    if total_elements >= parallel_threshold {
        mask_2d.par_mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    } else {
        mask_2d.mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    }
}

/// A macro to define a layer-specific method for setting the training mode.
///
/// This macro generates a `set_training` method within the implementing object
/// to allow toggling the training mode between training (`true`) and inference (`false`).
///
/// The generated method is used to manage the state of `training` within the object,
/// which can be critical for operations like forward and backward passes in machine learning models.
macro_rules! mode_dependent_layer_set_training {
    () => {
        /// Sets the training mode for the object.
        ///
        /// # Arguments
        ///
        /// * `is_training` - A boolean value indicating whether the object should be
        ///   in training mode (`true`) or not (`false`).
        ///
        /// # Effects
        ///
        /// This method modifies the `training` field of the object to reflect the provided value.
        /// It is commonly used to toggle the state of an object between training and inference modes.
        pub fn set_training(&mut self, is_training: bool) {
            self.training = is_training;
        }
    };
}

/// A macro that defines a method `set_training_if_mode_dependent` for a layer that may have
/// behavior dependent on whether it is in training or inference mode.
macro_rules! mode_dependent_layer_trait {
    () => {
        fn set_training_if_mode_dependent(&mut self, is_training: bool) {
            self.set_training(is_training);
        }
    };
}

/// Dropout layer for neural networks
pub mod dropout;
/// Spatial Dropout layer for 1D data
pub mod spatial_dropout_1d;
/// Spatial Dropout layer for 2D data
pub mod spatial_dropout_2d;
/// Spatial Dropout layer for 3D data
pub mod spatial_dropout_3d;

pub use dropout::*;
pub use spatial_dropout_1d::*;
pub use spatial_dropout_2d::*;
pub use spatial_dropout_3d::*;
