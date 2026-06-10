//! Regularization layers (Dropout and SpatialDropout1D/2D/3D) plus the shared
//! backward, output-shape, and masking helpers they share

use crate::error::Error;
use crate::neural_network::Tensor;

/// Common backward pass shared by all dropout layers
///
/// Implements the backward logic used by every dropout variant (Dropout and
/// SpatialDropout1D/2D/3D)
///
/// # Parameters
///
/// - `grad_output` - Gradient from the next layer
/// - `mask` - The dropout mask applied during the forward pass
/// - `training` - Whether the layer is in training mode
/// - `rate` - The dropout rate
/// - `layer_name` - Concrete layer name, used in the "forward pass not run" error message so the
///   error identifies the actual layer (e.g. `SpatialDropout2D`) rather than always `Dropout`
///
/// # Returns
///
/// - `Result<Tensor, Error>` - Gradient to pass to the previous layer
///
/// # Errors
///
/// Returns an error when the forward pass has not been run and no mask is available
fn dropout_backward(
    grad_output: &Tensor,
    mask: &Option<Tensor>,
    training: bool,
    rate: f32,
    layer_name: &'static str,
) -> Result<Tensor, Error> {
    if !training || rate == 0.0 {
        // During inference or zero rate, pass the gradient through unchanged
        return Ok(grad_output.clone());
    }

    if rate == 1.0 {
        // Rate of 1.0 drops everything, so the gradient is zero
        return Ok(Tensor::zeros(grad_output.raw_dim()));
    }

    // Apply the same mask to the gradient
    if let Some(mask) = mask {
        let scale = 1.0 / (1.0 - rate);
        let grad_input = grad_output * mask * scale;
        Ok(grad_input)
    } else {
        Err(Error::forward_pass_not_run(layer_name))
    }
}

/// Common output-shape formatting shared by all dropout layers
///
/// Formats the input shape into a string representation, as the output shape
/// equals the input shape for every dropout variant
///
/// # Parameters
///
/// - `input_shape` - The input shape vector
///
/// # Returns
///
/// - `String` - Formatted output shape string
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

/// Thresholds a random mask into a binary mask, in parallel or sequentially
///
/// Used by all spatial dropout layers to convert a random mask into a binary
/// mask based on the dropout rate. Larger masks use parallel computation
///
/// # Parameters
///
/// - `mask_2d` - The random mask to convert to binary (modified in place)
/// - `rate` - The dropout rate threshold
/// - `parallel_threshold` - Element count at or above which parallel computation is used
fn apply_spatial_dropout_threshold(mask_2d: &mut Tensor, rate: f32, parallel_threshold: usize) {
    let total_elements = mask_2d.len();

    // Use parallel computation for large masks, sequential otherwise
    if total_elements >= parallel_threshold {
        mask_2d.par_mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    } else {
        mask_2d.mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });
    }
}

/// Dropout layer for neural networks
// `Dropout` lives in a `dropout` submodule beside the sibling spatial-dropout modules; the
// repeated name is the intended file layout
#[allow(clippy::module_inception)]
pub mod dropout;
/// Spatial Dropout layer for 1D data
pub mod spatial_dropout_1d;
/// Spatial Dropout layer for 2D data
pub mod spatial_dropout_2d;
/// Spatial Dropout layer for 3D data
pub mod spatial_dropout_3d;

pub use dropout::Dropout;
pub use spatial_dropout_1d::SpatialDropout1D;
pub use spatial_dropout_2d::SpatialDropout2D;
pub use spatial_dropout_3d::SpatialDropout3D;
