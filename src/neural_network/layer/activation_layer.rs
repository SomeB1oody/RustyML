use super::*;

/// Helper function to format the output shape for activation layers.
///
/// Returns a formatted string representing the shape of the cached tensor,
/// or "Unknown" if no tensor has been cached yet.
fn format_output_shape(cached_tensor: &Option<Tensor>) -> String {
    if let Some(tensor) = cached_tensor {
        let shape = tensor.shape();
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        "Unknown".to_string()
    }
}

/// Linear (Identity) activation layer.
pub mod linear;
/// ReLU (Rectified Linear Unit) activation layer
pub mod relu;
/// Sigmoid activation layer
pub mod sigmoid;
/// Softmax activation layer
pub mod softmax;
/// Tanh (Hyperbolic Tangent) activation layer
pub mod tanh;

pub use linear::*;
pub use relu::*;
pub use sigmoid::*;
pub use softmax::*;
pub use tanh::*;
