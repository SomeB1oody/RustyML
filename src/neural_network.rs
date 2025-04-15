/// Module that contains neural network layer implementations.
pub mod layer;
/// Module that contains loss function implementations.
pub mod loss_function;
/// Module that contains optimization algorithms for neural network training.
pub mod optimizer;
/// Module that contains implementations for sequential model architecture.
pub mod sequential;

pub use layer::*;
pub use loss_function::*;
pub use optimizer::*;
pub use sequential::*;

use crate::ModelError;
use ndarray::{Array2, ArrayD, Axis};

/// Type alias for n-dimensional arrays used as tensors in the neural network.
pub type Tensor = ArrayD<f32>;

/// Defines the interface for neural network layers.
///
/// This trait provides the core functionality that all neural network layers must implement,
/// including forward and backward propagation, as well as parameter updates for different
/// optimization algorithms.
pub trait Layer {
    /// Performs forward propagation through the layer.
    ///
    /// # Parameters
    ///
    /// * `input` - The input tensor to the layer
    ///
    /// # Returns
    ///
    /// The output tensor after forward computation
    fn forward(&mut self, input: &Tensor) -> Tensor;

    /// Performs backward propagation through the layer.
    ///
    /// # Parameters
    ///
    /// * `grad_output` - The gradient tensor from the next layer
    ///
    /// # Returns
    ///
    /// The gradient tensor to be passed to the previous layer
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError>;

    /// Returns the type name of the layer (e.g., "Dense").
    ///
    /// # Returns
    ///
    /// A string slice representing the layer type
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Returns a description of the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A string describing the output dimensions
    fn output_shape(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the total number of trainable parameters in the layer.
    ///
    /// # Returns
    ///
    /// The count of parameters as a usize
    fn param_count(&self) -> usize {
        0
    }

    /// Updates the layer parameters using Stochastic Gradient Descent.
    ///
    /// # Parameters
    ///
    /// * `_lr` - Learning rate for parameter updates
    fn update_parameters_sgd(&mut self, _lr: f32) {
        // Default implementation does nothing
    }

    /// Updates the layer parameters using Adam optimizer.
    ///
    /// # Parameters
    ///
    /// - `_lr` - Learning rate for parameter updates
    /// - `_beta1` - Exponential decay rate for the first moment estimates
    /// - `_beta2` - Exponential decay rate for the second moment estimates
    /// - `_epsilon` - Small constant for numerical stability
    /// - `_t` - Current training iteration
    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // Default implementation does nothing
    }

    /// Updates the layer parameters using RMSprop optimizer.
    ///
    /// # Parameters
    ///
    /// - `_lr` - Learning rate for parameter updates
    /// - `_rho` - Decay rate for moving average of squared gradients
    /// - `_epsilon` - Small constant for numerical stability
    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // Default implementation does nothing
    }
}

/// Defines the interface for loss functions used in neural network training.
///
/// This trait provides methods to compute both the loss value and its gradient
/// with respect to the predicted values.
pub trait LossFunction {
    /// Computes the loss between true and predicted values.
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// The scalar loss value
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32;

    /// Computes the gradient of the loss with respect to the predictions.
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// Tensor containing the gradient of the loss
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor;
}

/// Defines the interface for optimization algorithms.
///
/// This trait provides methods to update layer parameters during
/// the training process.
pub trait Optimizer {
    /// Updates the parameters of a layer according to the optimization algorithm.
    ///
    /// # Parameters
    ///
    /// * `layer` - The layer whose parameters should be updated
    fn update(&mut self, layer: &mut dyn Layer);
}

/// Activation function enum, supporting ReLU, Tanh, Sigmoid, and Softmax
#[derive(Debug, PartialEq)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl Activation {
    /// Forward application of activation functions
    ///
    /// Applies the specified activation function to the input tensor.
    ///
    /// # Parameters
    ///
    /// - `z` - Input tensor to apply activation function to
    /// - `activation` - The activation function to apply
    ///
    /// # Returns
    /// A new tensor with the activation function applied
    pub fn apply_activation(z: &Array2<f32>, activation: &Activation) -> Array2<f32> {
        match activation {
            Activation::ReLU => z.mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Activation::Sigmoid => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::Tanh => z.mapv(|x| x.tanh()),
            Activation::Softmax => {
                let mut out = z.clone();
                // Apply softmax to each row
                for mut row in out.outer_iter_mut() {
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    row.map_inplace(|x| *x = (*x - max_val).exp());
                    let sum = row.sum();
                    row.map_inplace(|x| *x /= sum);
                }
                out
            }
        }
    }

    /// Computes derivatives for ReLU, Sigmoid, and Tanh activation functions
    ///
    /// Returns the derivative of the activation function given the activated output.
    /// For Softmax, the gradient is handled separately in backward propagation.
    ///
    /// # Parameters
    ///
    /// - `activation_output` - The output after activation function has been applied
    /// - `activation` - The activation function whose derivative to compute
    ///
    /// # Returns
    /// A tensor containing the derivative values
    pub fn activation_derivative(
        activation_output: &Array2<f32>,
        activation: &Activation,
    ) -> Array2<f32> {
        match activation {
            Activation::ReLU => activation_output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => activation_output.mapv(|a| a * (1.0 - a)),
            Activation::Tanh => activation_output.mapv(|a| 1.0 - a * a),
            Activation::Softmax => Array2::ones(activation_output.dim()),
        }
    }

    /// Backward propagation for Softmax activation
    ///
    /// For each row, computes:
    /// new_grad\[i\] = a\[i\] * (upstream\[i\] - sum_j(a\[j\]*upstream\[j\]))
    ///
    /// # Parameters
    ///
    /// - `a` - The output from the softmax activation
    /// - `upstream` - The gradient flowing from the next layer
    ///
    /// # Returns
    /// The gradient with respect to the input of the softmax function
    pub fn softmax_backward(a: &Array2<f32>, upstream: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::<f32>::zeros(a.raw_dim());
        for (mut out_row, (a_row, up_row)) in result
            .axis_iter_mut(Axis(0))
            .zip(a.axis_iter(Axis(0)).zip(upstream.axis_iter(Axis(0))))
        {
            let dot = a_row
                .iter()
                .zip(up_row.iter())
                .map(|(&ai, &gi)| ai * gi)
                .sum::<f32>();
            for (j, r) in out_row.iter_mut().enumerate() {
                *r = a_row[j] * (up_row[j] - dot);
            }
        }
        result
    }
}
