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
fn dropout_backward(
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
fn dropout_output_shape(input_shape: &[usize]) -> String {
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
fn apply_spatial_dropout_threshold(mask_2d: &mut Tensor, rate: f32, parallel_threshold: usize) {
    let total_elements = mask_2d.len();

    // Apply threshold to create binary mask with parallel or sequential computation
    if total_elements >= parallel_threshold {
        mask_2d.par_mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    } else {
        mask_2d.mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    }
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
