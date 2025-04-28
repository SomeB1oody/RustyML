use crate::ModelError;
use crate::machine_learning::RegularizationType;
use crate::neural_network::Tensor;

pub trait RegressorCommonGetterFunctions {
    /// Gets the current setting for fitting the intercept term
    ///
    /// # Returns
    ///
    /// * `bool` - Returns `true` if the model includes an intercept term, `false` otherwise
    fn get_fit_intercept(&self) -> bool;

    /// Gets the current learning rate
    ///
    /// The learning rate controls the step size in each iteration of gradient descent.
    ///
    /// # Returns
    ///
    /// * `f64` - The current learning rate value
    fn get_learning_rate(&self) -> f64;

    /// Gets the maximum number of iterations
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum number of iterations for the gradient descent algorithm
    fn get_max_iterations(&self) -> usize;

    /// Gets the convergence tolerance threshold
    ///
    /// The convergence tolerance is used to determine when to stop the training process.
    /// Training stops when the change in the loss function between consecutive iterations
    /// is less than this value.
    ///
    /// # Returns
    ///
    /// * `f64` - The current convergence tolerance value
    fn get_tolerance(&self) -> f64;

    /// Returns the actual number of actual iterations performed during the last model fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)` - The number of iterations if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    fn get_actual_iterations(&self) -> Result<usize, ModelError>;

    /// Returns a reference to the regularization type of the model
    ///
    /// This method provides access to the regularization configuration of the model,
    /// which can be None (no regularization), L1 (LASSO), or L2 (Ridge).
    ///
    /// # Returns
    ///
    /// * `&Option<RegularizationType>` - A reference to the regularization type, which will be None if no regularization is applied
    fn get_regularization_type(&self) -> &Option<RegularizationType>;
}

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
    fn param_count(&self) -> usize {
        0
    }

    /// Updates the layer parameters during training.
    ///
    /// This method provides a generic parameter update mechanism that can be
    /// overridden by specific layer implementations. By default, it does nothing.
    ///
    /// # Parameters
    ///
    /// * `_lr` - Learning rate for parameter updates
    fn update_parameters(&mut self, _lr: f32) {}

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
