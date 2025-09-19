use crate::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::LayerWeight;

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
    /// - `Ok(Tensor)` - The gradient tensor to be passed to the previous layer
    /// - `Err(ModelError::ProcessingError(String))` - If the layer encountered an error during processing`
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
    /// The count of parameters as an usize
    fn param_count(&self) -> usize;

    /// Updates the layer parameters using Stochastic Gradient Descent.
    ///
    /// # Parameters
    ///
    /// * `_lr` - Learning rate for parameter updates
    fn update_parameters_sgd(&mut self, _lr: f32);

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
    );

    /// Updates the layer parameters using RMSprop optimizer.
    ///
    /// # Parameters
    ///
    /// - `_lr` - Learning rate for parameter updates
    /// - `_rho` - Decay rate for moving average of squared gradients
    /// - `_epsilon` - Small constant for numerical stability
    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32);

    /// Returns a map of all weights in the layer.
    ///
    /// This method provides access to all weight matrices and bias vectors used by the LSTM layer.
    /// The weights are organized by gate (input, forget, cell, output) and by their role
    /// (kernel, recurrent_kernel, bias) within each gate.
    ///
    /// # Returns
    ///
    /// * A `LayerWeight` enum containing:
    ///   - `LayerWeight::Dense` for Dense layers with weight and bias
    ///   - `LayerWeight::SimpleRNN` for SimpleRNN layers with kernel, recurrent_kernel, and bias
    ///   - `LayerWeight::LSTM` for LSTM layers with weights for input, forget, cell, and output gates
    fn get_weights(&self) -> LayerWeight<'_>;
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
