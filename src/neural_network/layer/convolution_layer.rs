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

/// A macro that generates the AdaGrad parameter update logic for convolutional layers.
///
/// This macro creates a standardized block of code that applies AdaGrad updates
/// to both weights and biases of convolutional layers. It defines a closure that
/// updates the accumulator and parameters in parallel, then applies it to weights and biases.
///
/// # Usage
/// This macro should be used inside the AdaGrad update condition where:
/// - `self.optimizer_cache.ada_grad_cache` exists
/// - `weight_gradients` and `bias_gradients` are available
/// - `lr` and `epsilon` are defined
macro_rules! update_adagrad_conv {
    ($self:expr, $weight_gradients:expr, $bias_gradients:expr, $lr:expr, $epsilon:expr) => {
        if let Some(ada_grad_cache) = &mut $self.optimizer_cache.ada_grad_cache {
            // Define a generic parameter update closure for AdaGrad
            let update_parameters = |params: &mut [f32], accumulator: &mut [f32], grads: &[f32]| {
                // Update accumulator (accumulated squared gradients) in parallel
                accumulator
                    .par_iter_mut()
                    .zip(grads.par_iter())
                    .for_each(|(acc, &grad)| {
                        *acc += grad * grad;
                    });

                // Update parameters in parallel
                params
                    .par_iter_mut()
                    .zip(grads.par_iter())
                    .zip(accumulator.par_iter())
                    .for_each(|((param, &grad), &acc_val)| {
                        *param -= $lr * grad / (acc_val.sqrt() + $epsilon);
                    });
            };

            // Update weight parameters
            update_parameters(
                $self.weights.as_slice_mut().unwrap(),
                ada_grad_cache.accumulator.as_slice_mut().unwrap(),
                $weight_gradients.as_slice().unwrap(),
            );

            // Update bias parameters
            update_parameters(
                $self.bias.as_slice_mut().unwrap(),
                ada_grad_cache.accumulator_bias.as_slice_mut().unwrap(),
                $bias_gradients.as_slice().unwrap(),
            );
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
