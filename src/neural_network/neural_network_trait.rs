use crate::error::{IoError, ModelError};
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;

/// Defines the interface for neural network layers.
///
/// This trait provides the core functionality that all neural network layers must implement,
/// including forward and backward propagation, as well as parameter updates for different
/// optimization algorithms
pub trait Layer: std::any::Any + Send + Sync {
    /// Performs forward propagation through the layer.
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    ///
    /// # Returns
    ///
    /// - `Tensor` - The output tensor after forward computation
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError>;

    /// Performs backward propagation through the layer.
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    ///
    /// # Returns
    ///
    /// - `Ok(Tensor)` - The gradient tensor to be passed to the previous layer
    /// - `Err(ModelError)` - If the layer encountered an error during processing
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError>;

    /// Returns the type name of the layer (e.g. "Dense").
    ///
    /// # Returns
    ///
    /// * `&str` - A string slice representing the layer type
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Returns a description of the output shape of the layer.
    ///
    /// # Returns
    ///
    /// - `String` - A string describing the output dimensions
    fn output_shape(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the total number of trainable parameters in the layer.
    ///
    /// # Returns
    ///
    /// - `TrainingParameters` - The count of parameters as an enum variant
    fn param_count(&self) -> TrainingParameters;

    /// Updates the layer parameters using Stochastic Gradient Descent.
    ///
    /// # Parameters
    ///
    /// - `_lr` - Learning rate for parameter updates
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

    /// Updates the layer parameters using AdaGrad optimizer.
    ///
    /// # Parameters
    ///
    /// - `_lr` - Learning rate for parameter updates
    /// - `_epsilon` - Small constant for numerical stability
    fn update_parameters_ada_grad(&mut self, _lr: f32, _epsilon: f32);

    /// Returns a reference to all weights in the layer.
    ///
    /// This method provides access to all weight matrices and bias vectors used by the layer.
    /// The weights are organized by layer type and contain references to the actual weight data.
    ///
    /// # Returns
    ///
    /// - `LayerWeight<'_>` - An enum containing references to layer weights:
    ///     - `LayerWeight::Dense` for Dense layers with weight and bias
    ///     - `LayerWeight::SimpleRNN` for SimpleRNN layers with kernel, recurrent_kernel, and bias
    ///     - `LayerWeight::LSTM` for LSTM layers with weights for input, forget, cell, and output gates
    ///     - `LayerWeight::Conv1D`, `LayerWeight::Conv2D`, `LayerWeight::Conv3D` for convolutional layers
    ///     - `LayerWeight::Empty` for layers with no trainable parameters
    fn get_weights(&self) -> LayerWeight<'_>;

    /// Sets the training mode if the layer is mode-dependent.
    ///
    /// This method allows layers that behave differently during training and inference
    /// to switch between modes. Layers that don't depend on training mode (like Dense,
    /// Activation, Pooling layers) can use the default no-op implementation.
    ///
    /// Layers implementing `ModeDependentLayer` trait should override this method
    /// to call their `set_training()` method.
    ///
    /// # Parameters
    ///
    /// - `_is_training` - `true` for training mode, `false` for inference mode
    fn set_training_if_mode_dependent(&mut self, _is_training: bool) {
        // Default implementation: do nothing
        // Only mode-dependent layers need to override this
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
    /// - `f32` - The scalar loss value
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
    /// - `Tensor` - Tensor containing the gradient of the loss with respect to predictions
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
    /// - `layer` - The layer whose parameters should be updated
    fn update(&mut self, layer: &mut dyn Layer);
}

/// Trait for applying serialized weights to a specific layer type.
///
/// This trait is implemented by serializable weight structures to apply
/// their contained weights to the corresponding layer type. It provides
/// a uniform interface for weight deserialization and application across
/// all layer types.
///
/// # Type Parameters
///
/// - `L` - The layer type that these weights can be applied to
pub trait ApplyWeights<L> {
    /// Applies the serialized weights to a layer instance.
    ///
    /// # Parameters
    ///
    /// - `layer` - Mutable reference to the layer that will receive the weights
    ///
    /// # Returns
    ///
    /// - `Ok(())` - Weights were successfully applied
    /// - `Err(IoError)` - Weight shape mismatch or conversion error
    fn apply_to_layer(&self, layer: &mut L) -> Result<(), IoError>;
}

/// A marker trait for activation layers in neural networks.
///
/// This trait extends the base `Layer` trait to specifically mark layers that provide
/// activation functions. Activation layers are special types of neural network layers
/// that apply non-linear transformations to their inputs, enabling neural networks
/// to learn complex patterns and relationships.
///
/// # Purpose
///
/// The `ActivationLayer` trait serves as a type constraint and marker for layers that:
/// - Apply element-wise non-linear transformations to input data
/// - Don't have trainable parameters (weights or biases)
/// - Preserve the input tensor shape in their output
/// - Can be used as activation functions in other layers (e.g., Dense, Convolutional layers)
pub trait ActivationLayer: Layer {}
