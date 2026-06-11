//! Core traits for the neural network module: layers, losses, optimizers, weight
//! application, and the flat parameter/gradient view shared between them

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;

/// A single trainable parameter tensor paired with its gradient, exposed as flat slices
///
/// Layers yield their trainable tensors (weights, biases, kernels, gamma/beta, ...) as
/// `ParamGrad`s so that optimizers can update any parameter shape with one flat-slice kernel,
/// instead of every layer/optimizer pair re-implementing the update. `value` and `grad` always
/// have the same length and the same element ordering
pub struct ParamGrad<'a> {
    /// Mutable view of the parameter's contiguous data, updated in place by the optimizer
    pub value: &'a mut [f32],
    /// The corresponding gradient data (same length and ordering as `value`)
    pub grad: &'a [f32],
}

/// Defines the interface for neural network layers
///
/// This trait provides the core functionality that all neural network layers must implement,
/// including forward and backward propagation, plus exposing trainable parameters and their
/// gradients to the optimizer via [`parameters`](Layer::parameters)
pub trait Layer: std::any::Any + Send + Sync {
    /// Performs forward propagation through the layer
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    ///
    /// # Returns
    ///
    /// - `Tensor` - The output tensor after forward computation
    ///
    /// # Errors
    ///
    /// - `Error` - If the forward computation fails (e.g. shape mismatch)
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error>;

    /// Runs the forward pass in inference (eval) mode, taking `&self`
    ///
    /// Unlike [`forward`](Layer::forward), this does **not** record any state for
    /// backpropagation (it writes no caches) and mode-dependent layers (dropout, batch norm, ...)
    /// always use their inference behavior. Because it borrows `&self`, a model can be shared for
    /// concurrent inference. Use it for prediction/serving where no backward pass follows; use
    /// [`forward`](Layer::forward) during training
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    ///
    /// # Returns
    ///
    /// - `Tensor` - The output tensor, identical to what `forward` produces in inference mode
    ///
    /// # Errors
    ///
    /// - `Error` - If the inference computation fails (e.g. shape mismatch)
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error>;

    /// Performs backward propagation through the layer
    ///
    /// # Numerical policy
    ///
    /// Backward is pure math: it does **not** sanitize NaN/Inf (no zeroing, no element-wise
    /// clamping). Such values are propagated, not masked, and surface loudly at the
    /// next forward pass (which rejects non-finite input) or as a NaN loss. To tame large-but-finite
    /// gradients, enable clip-by-global-norm on the optimizer
    /// ([`Optimizer::clip_norm`](crate::neural_network::traits::Optimizer::clip_norm)) rather than
    /// clamping inside a layer, since global-norm scaling preserves gradient direction
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    ///
    /// # Returns
    ///
    /// - `Tensor` - The gradient tensor to be passed to the previous layer
    ///
    /// # Errors
    ///
    /// - `Error` - If the layer encountered an error during processing (e.g. shape mismatch or a
    ///   missing forward-pass cache)
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error>;

    /// Returns the type name of the layer (e.g. "Dense")
    ///
    /// # Returns
    ///
    /// - `&str` - A string slice representing the layer type
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Returns a description of the output shape of the layer
    ///
    /// # Returns
    ///
    /// - `String` - A string describing the output dimensions
    fn output_shape(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the total number of trainable parameters in the layer
    ///
    /// # Returns
    ///
    /// - `TrainingParameters` - The count of parameters as an enum variant
    fn param_count(&self) -> TrainingParameters;

    /// Exposes the layer's trainable parameters and their gradients to the optimizer
    ///
    /// Each returned [`ParamGrad`] pairs a parameter tensor's flat data with its gradient
    /// Layers without trainable parameters (or before a backward pass has produced gradients)
    /// return an empty vector - the default implementation. The order of the returned entries
    /// must be stable across calls, because step-based optimizers key their per-parameter state
    /// by position
    ///
    /// # Returns
    ///
    /// - `Vec<ParamGrad<'_>>` - One entry per trainable tensor that currently has a gradient
    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        Vec::new()
    }

    /// Returns a reference to all weights in the layer
    ///
    /// This method provides access to all weight matrices and bias vectors used by the layer
    /// The weights are organized by layer type and contain references to the actual weight data
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

    /// Sets the training mode if the layer is mode-dependent
    ///
    /// This method allows layers that behave differently during training and inference
    /// to switch between modes. Layers that don't depend on training mode (like Dense,
    /// Activation, Pooling layers) can use the default no-op implementation
    ///
    /// Mode-dependent layers (Dropout, BatchNormalization, etc.) override this method to
    /// forward `is_training` to their own `set_training()`. In this crate that override is
    /// generated by the `mode_dependent_layer_trait!` macro (see the `regularization`
    /// module), so layers do not implement it by hand
    ///
    /// # Parameters
    ///
    /// - `_is_training` - `true` for training mode, `false` for inference mode
    fn set_training_if_mode_dependent(&mut self, _is_training: bool) {
        // No-op by default; only mode-dependent layers override this
    }
}

/// Defines the interface for loss functions used in neural network training
///
/// This trait provides methods to compute both the loss value and its gradient
/// with respect to the predicted values
///
/// # Averaging convention
///
/// Each loss normalizes by what is natural for its family, so the conventions differ on purpose:
/// `compute_grad` is always exactly the gradient of `compute_loss`, but switching loss families
/// rescales the gradient magnitude (and thus the effective learning rate):
///
/// - [`MeanSquaredError`](crate::neural_network::losses::MeanSquaredError),
///   [`MeanAbsoluteError`](crate::neural_network::losses::MeanAbsoluteError) and
///   [`BinaryCrossEntropy`](crate::neural_network::losses::BinaryCrossEntropy) average over
///   **every element** (`y.len()`), treating each output as an independent target
/// - [`CategoricalCrossEntropy`](crate::neural_network::losses::CategoricalCrossEntropy) and
///   [`SparseCategoricalCrossEntropy`](crate::neural_network::losses::SparseCategoricalCrossEntropy)
///   sum over the class axis and average over the **batch** (`y.shape()[0]`), matching the standard
///   per-sample categorical cross-entropy
pub trait Loss {
    /// Computes the loss between true and predicted values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// - `f32` - The scalar loss value
    ///
    /// # Errors
    ///
    /// - `Error` - If the inputs are inconsistent (e.g. mismatched shapes or, for the
    ///   sparse loss, out-of-range labels)
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error>;

    /// Computes the gradient of the loss with respect to the predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor containing the ground truth values
    /// - `y_pred` - Tensor containing the predicted values
    ///
    /// # Returns
    ///
    /// - `Tensor` - Tensor containing the gradient of the loss with respect to predictions
    ///
    /// # Errors
    ///
    /// - `Error` - If the inputs are inconsistent (see [`compute_loss`](Loss::compute_loss))
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error>;
}

/// Defines the interface for optimization algorithms
///
/// This trait provides methods to update layer parameters during
/// the training process
pub trait Optimizer {
    /// Advances the optimizer's global training step
    ///
    /// Called exactly once per batch (before the per-layer [`update`](Optimizer::update) calls)
    /// Step-dependent optimizers such as Adam use this to advance their bias-correction timestep
    /// once per training step rather than once per layer. The default implementation is a no-op,
    /// which is correct for step-independent optimizers (SGD, RMSprop, AdaGrad)
    fn step(&mut self) {}

    /// The global gradient-norm clip threshold, or `None` (the default) to disable clipping
    ///
    /// When `Some(max_norm)`, the training loop computes the global L2 norm across **all** of the
    /// model's gradients and, if it exceeds `max_norm`, scales every gradient by
    /// `max_norm / global_norm` before [`update`](Optimizer::update). This single uniform factor
    /// preserves gradient direction (unlike per-element clamping). A non-finite global norm is left
    /// unscaled so divergence still surfaces rather than being masked
    fn clip_norm(&self) -> Option<f32> {
        None
    }

    /// Updates the parameters of a layer according to the optimization algorithm
    ///
    /// # Parameters
    ///
    /// - `layer` - The layer whose parameters should be updated
    /// - `grad_scale` - Uniform factor applied to every gradient before the update, supplied by the
    ///   training loop to implement clip-by-global-norm. Pass `1.0` for an unscaled update
    fn update(&mut self, layer: &mut dyn Layer, grad_scale: f32);
}

/// Trait for applying serialized weights to a specific layer type
///
/// This trait is implemented by serializable weight structures to apply
/// their contained weights to the corresponding layer type. It provides
/// a uniform interface for weight deserialization and application across
/// all layer types
///
/// # Type Parameters
///
/// - `L` - The layer type that these weights can be applied to
pub trait ApplyWeights<L> {
    /// Applies the serialized weights to a layer instance
    ///
    /// # Parameters
    ///
    /// - `layer` - Mutable reference to the layer that will receive the weights
    ///
    /// # Errors
    ///
    /// - `Error` - Weight shape mismatch or conversion error
    fn apply_to_layer(&self, layer: &mut L) -> Result<(), Error>;
}
