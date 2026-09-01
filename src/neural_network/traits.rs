//! Core traits for the neural network module: layers, losses, optimizers, weight
//! application, and the flat parameter/gradient view shared between them

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::layer_weight::LayerWeight;

/// The stable address of 1 parameter tensor inside a model
///
/// A parameter is identified by the layer that holds it and by the name that the layer gives
/// it. Neither half moves while the model trains, so an optimizer can key its per-parameter
/// state on the pair and reach the same buffer on every step
///
/// The scope is the position of the layer in the model that drives the update.
/// [`Sequential`](crate::neural_network::sequential::Sequential) passes the index of the layer,
/// counted from the input. A caller that drives 1 layer directly passes any value it likes,
/// as long as it passes the same value on every step for that layer
///
/// The name is the `&'static str` that [`Layer::parameters`] puts in the
/// [`ParamGrad`]. It follows the layer, so a layer that stops yielding 1 of its
/// tensors, or that starts yielding a new one, moves no other tensor's address
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId {
    /// Position of the owning layer in the model, counted from the input
    pub scope: usize,
    /// Name the layer gives the tensor, such as `"kernel"`, `"bias"`, or `"gamma"`
    pub name: &'static str,
}

impl ParamId {
    /// Builds the address of the named parameter of the layer at the given position
    ///
    /// # Parameters
    ///
    /// - `scope` - Position of the owning layer in the model, counted from the input
    /// - `name` - Name the layer gives the tensor
    ///
    /// # Returns
    ///
    /// - `ParamId` - The parameter address
    #[inline]
    pub const fn new(scope: usize, name: &'static str) -> Self {
        Self { scope, name }
    }
}

/// A single trainable parameter tensor paired with its gradient, exposed as flat slices
///
/// Layers yield their trainable tensors (weights, biases, kernels, gamma/beta, ...) as
/// `ParamGrad`s. This lets optimizers update any parameter shape with 1 flat-slice kernel,
/// instead of every layer/optimizer pair re-implementing the update. `value` and `grad` always
/// have the same length and the same element ordering
///
/// Construct one with [`ParamGrad::weight`] for a tensor that decoupled weight decay applies to
/// (weight matrices, conv/recurrent kernels). Use [`ParamGrad::no_decay`] for a tensor it skips
/// (biases and normalization scale/shift `gamma`/`beta`). The `decays` flag tells the optimizer
/// which rule applies, so it never has to guess
///
/// The `name` is the half of the parameter address that the layer owns. It uses the Keras 3
/// name of the tensor, such as `kernel`, `recurrent_kernel`, `depthwise_kernel`, `bias`,
/// `embeddings`, `alpha`, `gamma`, or `beta`. A layer must give the same name to the same
/// storage on every call, and must give 2 different tensors 2 different names
pub struct ParamGrad<'a> {
    /// Name the layer gives this tensor. See [`ParamId`]
    pub name: &'static str,
    /// Mutable view of the parameter's contiguous data that the optimizer updates in place
    pub value: &'a mut [f32],
    /// The corresponding gradient data (same length and ordering as `value`)
    pub grad: &'a [f32],
    /// Whether decoupled (AdamW/SGDW-style) weight decay applies to this tensor. `true` for
    /// weight matrices and conv/recurrent kernels, `false` for biases and normalization
    /// scale/shift (`gamma`/`beta`)
    pub decays: bool,
}

impl<'a> ParamGrad<'a> {
    /// A weight tensor that decoupled weight decay applies to (dense/conv/recurrent kernels)
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the tensor, such as `"kernel"`
    /// - `value` - Mutable view of the parameter's contiguous data
    /// - `grad` - The gradient data, of the same length and ordering as `value`
    ///
    /// # Returns
    ///
    /// - `ParamGrad` - The entry, with `decays` set to `true`
    #[inline]
    pub fn weight(name: &'static str, value: &'a mut [f32], grad: &'a [f32]) -> Self {
        Self {
            name,
            value,
            grad,
            decays: true,
        }
    }

    /// A bias or normalization scale/shift (`gamma`/`beta`) tensor that weight decay skips
    ///
    /// # Parameters
    ///
    /// - `name` - Name the layer gives the tensor, such as `"bias"`
    /// - `value` - Mutable view of the parameter's contiguous data
    /// - `grad` - The gradient data, of the same length and ordering as `value`
    ///
    /// # Returns
    ///
    /// - `ParamGrad` - The entry, with `decays` set to `false`
    #[inline]
    pub fn no_decay(name: &'static str, value: &'a mut [f32], grad: &'a [f32]) -> Self {
        Self {
            name,
            value,
            grad,
            decays: false,
        }
    }
}

/// Defines the interface for neural network layers
///
/// Covers the core functionality every neural network layer must implement: forward and
/// backward propagation. It also exposes trainable parameters and their gradients to the
/// optimizer via [`parameters`](Layer::parameters)
pub trait Layer: std::any::Any + Send + Sync {
    /// Runs the forward pass through the layer
    ///
    /// # Parameters
    ///
    /// - `input` - The input tensor to the layer
    ///
    /// # Returns
    ///
    /// - `Tensor` - The layer's output
    ///
    /// # Errors
    ///
    /// - `Error` - If the forward pass fails (e.g. shape mismatch)
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error>;

    /// Runs the forward pass in inference (eval) mode, taking `&self`
    ///
    /// Unlike [`forward`](Layer::forward), this does **not** record any state for
    /// backpropagation and writes no caches. Mode-dependent layers (dropout, batch norm, ...)
    /// always use their inference behavior. Because it borrows `&self`, a model can be shared for
    /// concurrent inference. Use it for prediction or serving, where no backward pass follows.
    /// Use [`forward`](Layer::forward) during training
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
    /// - `Error` - If the inference pass fails (e.g. shape mismatch)
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error>;

    /// Runs the backward pass through the layer
    ///
    /// # Parameters
    ///
    /// - `grad_output` - The gradient tensor from the next layer
    ///
    /// # Returns
    ///
    /// - `Tensor` - The gradient to pass to the previous layer
    ///
    /// # Numerical policy
    ///
    /// Backward is pure math: it does **not** sanitize NaN/Inf (no zeroing, no element-wise
    /// clamping). The backward pass propagates such values instead of masking them. The forward
    /// pass masks nothing either: it validates the rank, the shape, and the layer parameters,
    /// and it never reads the input values to reject them. A NaN or an infinity therefore stays
    /// in the tensor, moves on through every later layer, and shows itself in the output and in
    /// a non-finite loss. [`Embedding`](crate::neural_network::layers::Embedding) is the 1
    /// exception, because it reads its input as a table of row indices and rejects a non-finite
    /// index with `Error::InvalidInput`. To tame large-but-finite gradients, enable
    /// clip-by-global-norm on the optimizer
    /// ([`Optimizer::global_clipnorm`]) instead of clamping inside a layer. Global-norm scaling
    /// preserves gradient direction, unlike per-element clamping
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

    /// Returns how many parameters the layer holds, split by whether training updates them
    ///
    /// # Returns
    ///
    /// - `ParamCounts` - The trainable and the non-trainable element counts
    fn param_count(&self) -> ParamCounts;

    /// Exposes the layer's trainable parameters and their gradients to the optimizer
    ///
    /// Each returned [`ParamGrad`] pairs a parameter tensor's flat data with its gradient, and
    /// names the tensor. Layers without trainable parameters return the empty vector that the
    /// default implementation gives. A layer yields 1 entry per tensor that currently holds a
    /// gradient, so the count can differ from step to step. A tensor with no gradient yet is
    /// simply absent, and it holds back none of the other tensors of the same layer
    ///
    /// The name is the identity of the tensor, and the optimizer keys its per-parameter state
    /// on it (see [`ParamId`]). The same storage must therefore always come back under the same
    /// name, and 2 tensors of 1 layer must never share a name. The order of the entries is free
    ///
    /// # Returns
    ///
    /// - `Vec<ParamGrad<'_>>` - 1 named entry per trainable tensor that currently has a gradient
    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        Vec::new()
    }

    /// Returns a borrowing view of all weights in the layer
    ///
    /// Exposes all weight matrices and bias vectors the layer uses. Each array comes back as a
    /// `Cow::Borrowed` over the layer's live data, so no weights are cloned. The same enum
    /// doubles as the on-disk weight format (owned when deserialized)
    ///
    /// # Returns
    ///
    /// - `LayerWeight<'_>` - An enum borrowing the layer's weights:
    ///     - `LayerWeight::Dense` for Dense layers with weight and bias
    ///     - `LayerWeight::Embedding` for Embedding layers with the lookup table
    ///     - `LayerWeight::SimpleRNN` for SimpleRNN layers with kernel, recurrent_kernel, and bias
    ///     - `LayerWeight::LSTM` / `LayerWeight::GRU` for recurrent layers with fused kernel,
    ///       recurrent_kernel, and bias (gate column blocks `[i | f | g | o]` / `[z | r | h]`)
    ///     - `LayerWeight::Conv1D`, `LayerWeight::Conv2D`, `LayerWeight::Conv3D` for
    ///       convolutional layers
    ///     - `LayerWeight::Conv1DTranspose`, `LayerWeight::Conv2DTranspose`, and
    ///       `LayerWeight::Conv3DTranspose` for the transposed convolutional layers, whose kernel
    ///       carries its filter axis before its input-channel axis
    ///     - `LayerWeight::SeparableConv1D` and `LayerWeight::SeparableConv2D` for the
    ///       separable convolutions, each with a depthwise kernel, a pointwise kernel, and a bias
    ///     - `LayerWeight::DepthwiseConv1D` and `LayerWeight::DepthwiseConv2D` for the depthwise
    ///       convolutions, each with a kernel and a bias
    ///     - `LayerWeight::BatchNormalization` for batch normalization, with gamma, beta, and
    ///       the running mean and variance
    ///     - `LayerWeight::LayerNormalization`, `LayerWeight::InstanceNormalization`, and
    ///       `LayerWeight::GroupNormalization` for their respective layers, each with gamma and
    ///       beta
    ///     - `LayerWeight::PReLU` for the PReLU layer, with its negative-side slopes, whose
    ///       rank follows the input rank
    ///     - `LayerWeight::Empty` for layers with no trainable parameters
    fn get_weights(&self) -> LayerWeight<'_>;

    /// Sets the training mode if the layer is mode-dependent
    ///
    /// Lets layers that behave differently during training and inference switch between
    /// modes. Layers that do not depend on training mode (Dense, Activation, Pooling) can
    /// use the default no-op implementation
    ///
    /// Mode-dependent layers (Dropout, BatchNormalization, etc.) override this method to
    /// forward `is_training` to their own `set_training()`. In this crate, the
    /// `mode_dependent_layer_trait!` macro generates that override (see the `regularization`
    /// module), so layers do not implement it by hand
    ///
    /// # Parameters
    ///
    /// - `_is_training` - `true` for training mode, `false` for inference mode
    fn set_training_if_mode_dependent(&mut self, _is_training: bool) {
        // No-op by default. Only mode-dependent layers override this
    }
}

/// Defines the interface for loss functions used in neural network training
///
/// An implementation computes both the loss value and its gradient with respect to
/// the predicted values
///
/// # Averaging convention
///
/// Each loss normalizes by what is natural for its family, so the conventions differ on
/// purpose. `compute_grad` is always exactly the gradient of `compute_loss`. Switching loss
/// families rescales the gradient magnitude, and thus the effective learning rate:
///
/// - [`MeanSquaredError`](crate::neural_network::losses::MeanSquaredError),
///   [`MeanAbsoluteError`](crate::neural_network::losses::MeanAbsoluteError) and
///   [`BinaryCrossEntropy`](crate::neural_network::losses::BinaryCrossEntropy) average over
///   **every element** (`y.len()`), treating each output as an independent target
/// - [`CategoricalCrossEntropy`](crate::neural_network::losses::CategoricalCrossEntropy) sums
///   over the trailing **class** axis and averages over every **prediction site** (the product
///   of all leading axes). For a `[batch, classes]` target, that divisor is the batch. For the
///   `[batch, height, width, classes]` output of a channels-last softmax conv head, the divisor
///   is `batch * height * width`, 1 site per pixel
/// - [`SparseCategoricalCrossEntropy`](crate::neural_network::losses::SparseCategoricalCrossEntropy)
///   is the same per-sample categorical cross-entropy, but accepts only rank-2
///   `[batch, classes]` predictions, so its divisor is always the batch
///
/// The 2 categorical losses also renormalize `y_pred` along the class axis before
/// clipping when `from_logits` is off, as Keras does. That leaves the loss value alone for an
/// already-normalized head but contributes a row-constant term to the gradient, which a softmax
/// backward annihilates
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
/// An implementation updates layer parameters during training
pub trait Optimizer {
    /// Advances the optimizer's global training step
    ///
    /// Called exactly once per batch, before the per-layer [`update`](Optimizer::update) calls.
    /// The method now carries 1 duty only: a step-dependent optimizer advances the counter that
    /// its own math reads. [`Adam`](crate::neural_network::optimizers::Adam) and
    /// [`AdamW`](crate::neural_network::optimizers::AdamW) advance the bias-correction timestep
    /// here, so the correction moves once per batch rather than once per layer. SGD, RMSprop,
    /// and AdaGrad hold no such counter and keep the no-op default
    ///
    /// The method no longer rewinds anything. Per-parameter state is keyed by
    /// [`ParamId`], which is the position of the layer plus the name the layer gives the
    /// tensor, so nothing walks a cursor that a rewind could leave in the wrong place
    fn step(&mut self) {}

    /// The global gradient-norm clip threshold, or `None` (the default) to disable clipping
    ///
    /// When `Some(max_norm)`, the training loop computes the global L2 norm across **all** of
    /// the model's gradients. If the norm exceeds `max_norm`, it scales every gradient by
    /// `max_norm / global_norm` before [`update`](Optimizer::update). This single uniform factor
    /// preserves gradient direction (unlike per-element clamping). When the global norm is
    /// non-finite, the clip leaves gradients unscaled, so divergence still surfaces instead of
    /// being masked
    ///
    /// The name is Keras' `global_clipnorm`, and it is deliberate. Keras also has a `clipnorm`,
    /// which renormalizes each variable's gradient *independently* against the threshold. The 2
    /// give different directions whenever more than 1 tensor is over the limit. A
    /// threshold tuned for one is therefore not a threshold for the other. Only the global form
    /// exists here
    fn global_clipnorm(&self) -> Option<f32> {
        None
    }

    /// Updates the parameters of a layer according to the optimization algorithm
    ///
    /// The optimizer builds a [`ParamId`] from `scope` and the name of each
    /// [`ParamGrad`], and keys its per-parameter state on that address. The caller must
    /// therefore give the same layer the same `scope` on every step
    ///
    /// # Parameters
    ///
    /// - `scope` - Position of this layer in the model, counted from the input. It is the
    ///   layer half of the parameter address
    /// - `layer` - The layer whose parameters should be updated
    /// - `grad_scale` - Uniform factor that the training loop applies to every gradient before
    ///   the update, to implement clip-by-global-norm. Pass `1.0` for an unscaled update
    fn update(&mut self, scope: usize, layer: &mut dyn Layer, grad_scale: f32);

    /// The current learning rate
    ///
    /// The read half of the scheduling pair. A schedule (exponential decay, cosine annealing,
    /// warmup restarts) derives its next step size from the current one. It needs this method
    /// for that, rather than a copy of the rate kept alongside the model. A separate copy would
    /// drift the moment anything else retunes the optimizer. Reports whatever was last set.
    /// Unlike the constructors, [`set_learning_rate`](Optimizer::set_learning_rate) does not
    /// validate, so a rate set to 0 or to a negative value comes back unchanged
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate, the hook for external learning-rate scheduling
    ///
    /// Call this between batches or epochs, for step decay or warmup, to retune the step size
    /// without rebuilding the optimizer. The optimizer keeps its accumulated state
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - The new learning rate to use for subsequent updates
    fn set_learning_rate(&mut self, learning_rate: f32);
}

/// Trait for applying serialized weights to a specific layer type
///
/// A serializable weight structure implements this to apply its contained weights to the
/// corresponding layer type. Gives every layer type the same interface for deserializing and
/// applying weights
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
