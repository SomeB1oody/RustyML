use super::*;

/// Defines the padding method used in convolutional layers.
///
/// The padding type determines how the input is padded before applying convolution:
/// - `Valid`: No padding is applied, which reduces the output dimensions.
/// - `Same`: Padding is added to preserve the input spatial dimensions in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingType {
    /// No padding is applied. The convolution is only computed where the filter
    /// fully overlaps with the input, resulting in an output with reduced dimensions.
    Valid,

    /// Padding is added around the input to ensure that the output has the same
    /// spatial dimensions as the input (when stride is 1). This is done by adding
    /// zeros around the borders of the input.
    Same,
}

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

/// 1D Convolutional Layer
pub mod conv_1d;
/// 2D Convolutional Layer
pub mod conv_2d;
/// 3D Convolutional Layer
pub mod conv_3d;
/// 2D Depthwise Convolutional Layer
pub mod depthwise_conv_2d;
/// 2D Separable Convolutional Layer
pub mod separable_conv_2d;

pub use conv_1d::*;
pub use conv_2d::*;
pub use conv_3d::*;
pub use depthwise_conv_2d::*;
pub use separable_conv_2d::*;
