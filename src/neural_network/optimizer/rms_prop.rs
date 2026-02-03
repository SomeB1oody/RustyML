use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::neural_network_trait::{Layer, Optimizer};
use crate::neural_network::optimizer::input_validation_function::{
    validate_decay_rate, validate_epsilon, validate_learning_rate,
};
use ndarray::{Array2, Array3, Array4, Array5};

/// Threshold for switching between sequential and parallel computation.
/// For arrays smaller than this threshold, sequential computation is used
/// to avoid parallelization overhead.
const RMS_PROP_PARALLEL_THRESHOLD: usize = 1024;

/// RMSprop (Root Mean Square Propagation) optimizer.
///
/// Adapts per-parameter learning rates using a moving average of squared gradients.
///
/// # Fields
///
/// - `learning_rate` - Learning rate controlling the size of parameter updates
/// - `rho` - Decay rate for the moving average of squared gradients
/// - `epsilon` - Small constant added for numerical stability
pub struct RMSprop {
    learning_rate: f32,
    rho: f32,
    epsilon: f32,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer with the specified parameters.
    ///
    /// Validates hyperparameters and initializes the optimizer.
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `rho` - Decay rate for moving average of squared gradients (typically 0.9)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new RMSprop optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If any hyperparameter is out of range
    pub fn new(learning_rate: f32, rho: f32, epsilon: f32) -> Result<Self, ModelError> {
        // input validation
        validate_learning_rate(learning_rate)?;
        validate_decay_rate(rho, "rho")?;
        validate_epsilon(epsilon)?;

        Ok(Self {
            learning_rate,
            rho,
            epsilon,
        })
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, layer: &mut dyn Layer) {
        layer.update_parameters_rmsprop(self.learning_rate, self.rho, self.epsilon);
    }
}

/// RMSprop cache for dense or recurrent layers.
///
/// RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization
/// algorithm that maintains a moving average of squared gradients for each parameter.
/// This cache structure stores these moving averages for weights, recurrent weights (if present),
/// and biases during training.
///
/// # Fields
///
/// - `cache` - Moving average of squared gradients for main weight parameters
/// - `cache_recurrent` - Optional moving average of squared gradients for recurrent weight parameters
/// - `bias` - Moving average of squared gradients for bias parameters
#[derive(Debug, Clone, Default)]
pub struct RMSpropCache {
    pub cache: Array2<f32>,
    pub cache_recurrent: Option<Array2<f32>>,
    pub bias: Array2<f32>,
}

impl RMSpropCache {
    /// Creates a new RMSprop cache instance.
    ///
    /// This constructor initializes the cache arrays required by the RMSprop optimizer
    /// for storing running averages of squared gradients. These caches are used during
    /// optimization to compute adaptive learning rates.
    ///
    /// # Parameters
    ///
    /// - `dims` - Dimensions (rows, columns) of the main parameter matrix, used to initialize the `cache` array
    /// - `recurrent_dims` - Optional dimensions (rows, columns) of the recurrent parameter matrix. If `Some`, initializes `cache_recurrent` array; if `None`, `cache_recurrent` remains `None`
    /// - `bias_dims` - Dimensions (rows, columns) of the bias parameter matrix, used to initialize the `bias` array
    ///
    /// # Returns
    ///
    /// - `Self` - A new RMSpropCache instance with all caches initialized to zero matrices
    pub fn new(
        dims: (usize, usize),
        recurrent_dims: Option<(usize, usize)>,
        bias_dims: (usize, usize),
    ) -> Self {
        Self {
            cache: Array2::<f32>::zeros(dims),
            cache_recurrent: recurrent_dims.map(|dims| Array2::<f32>::zeros(dims)),
            bias: Array2::<f32>::zeros(bias_dims),
        }
    }

    /// Updates a parameter and its RMSprop cache.
    ///
    /// # Parameters
    ///
    /// - `param` - Parameter to be updated
    /// - `grad` - Gradient corresponding to the parameter
    /// - `cache` - Cache corresponding to the parameter
    /// - `rho` - Decay rate
    /// - `lr` - Learning rate
    /// - `epsilon` - Small constant for numerical stability
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the parameter length is at least `RMS_PROP_PARALLEL_THRESHOLD`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rustyml::neural_network::optimizer::rms_prop::RMSpropCache;
    ///
    /// let mut cache = RMSpropCache::new((1, 2), None, (1, 2));
    /// let mut param = array![[1.0, 2.0]];
    /// let grad = array![[0.1, 0.1]];
    /// RMSpropCache::update_param(&mut param, &grad, &mut cache.cache, 0.9, 0.001, 1e-8);
    /// ```
    pub fn update_param(
        param: &mut Array2<f32>,
        grad: &Array2<f32>,
        cache: &mut Array2<f32>,
        rho: f32,
        lr: f32,
        epsilon: f32,
    ) {
        let use_parallel = param.len() >= RMS_PROP_PARALLEL_THRESHOLD;

        if use_parallel {
            // Parallel computation for large arrays
            let (new_cache, new_param) = rayon::join(
                || cache.mapv(|x| x * rho) + &(grad.mapv(|x| x * x) * (1.0 - rho)),
                || {
                    let temp_cache =
                        cache.mapv(|x| x * rho) + &(grad.mapv(|x| x * x) * (1.0 - rho));
                    &*param - &(lr * grad / &(temp_cache.mapv(f32::sqrt) + epsilon))
                },
            );
            *cache = new_cache;
            *param = new_param;
        } else {
            // Sequential computation for small arrays
            *cache = cache.mapv(|x| x * rho) + &(grad.mapv(|x| x * x) * (1.0 - rho));
            *param = &*param - &(lr * grad / &(cache.mapv(f32::sqrt) + epsilon));
        }
    }

    /// Updates all parameters using RMSprop optimization algorithm.
    ///
    /// This method updates the weights, recurrent weights (if present), and bias parameters
    /// based on their respective gradients using the RMSprop optimization technique.
    /// RMSprop maintains a moving average of squared gradients to adapt the learning rate
    /// for each parameter.
    ///
    /// # Parameters
    ///
    /// - `param` - Main weight parameters to be updated
    /// - `recurrent_param` - Optional recurrent weight parameters to be updated
    /// - `bias_param` - Bias parameters to be updated
    /// - `grad` - Gradients corresponding to the main weight parameters
    /// - `recurrent_grad` - Optional gradients corresponding to the recurrent weight parameters
    /// - `bias_grad` - Gradients corresponding to the bias parameters
    /// - `rho` - Decay rate for the moving average of squared gradients (typically 0.9)
    /// - `lr` - Learning rate controlling the size of parameter updates
    /// - `epsilon` - Small constant added for numerical stability (typically 1e-8)
    ///
    /// # Performance
    ///
    /// Uses parallel computation when any parameter length is at least `RMS_PROP_PARALLEL_THRESHOLD`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rustyml::neural_network::optimizer::rms_prop::RMSpropCache;
    ///
    /// let mut cache = RMSpropCache::new((1, 2), None, (1, 2));
    /// let mut param = array![[1.0, 2.0]];
    /// let mut bias = array![[0.0, 0.0]];
    /// let grad = array![[0.1, 0.1]];
    /// let bias_grad = array![[0.01, 0.01]];
    /// cache.update_parameters(
    ///     &mut param,
    ///     None,
    ///     &mut bias,
    ///     &grad,
    ///     None,
    ///     &bias_grad,
    ///     0.9,
    ///     0.001,
    ///     1e-8,
    /// );
    /// ```
    pub fn update_parameters(
        &mut self,
        param: &mut Array2<f32>,
        recurrent_param: Option<&mut Array2<f32>>,
        bias_param: &mut Array2<f32>,
        grad: &Array2<f32>,
        recurrent_grad: Option<&Array2<f32>>,
        bias_grad: &Array2<f32>,
        rho: f32,
        lr: f32,
        epsilon: f32,
    ) {
        // Update main parameters
        Self::update_param(param, grad, &mut self.cache, rho, lr, epsilon);

        // Update recurrent parameters (if present)
        if let (Some(rec_param), Some(rec_grad), Some(ref mut rec_cache)) = (
            recurrent_param,
            recurrent_grad,
            self.cache_recurrent.as_mut(),
        ) {
            Self::update_param(rec_param, rec_grad, rec_cache, rho, lr, epsilon);
        }

        // Update bias parameters
        Self::update_param(bias_param, bias_grad, &mut self.bias, rho, lr, epsilon);
    }
}

/// RMSprop cache for Conv1D layers.
///
/// This specialized cache is designed for one-dimensional convolutional neural networks
/// that process sequential data such as time series or text. It maintains moving averages
/// of squared gradients for 3D weight parameters typically used in 1D convolutional layers
/// (batch_size, channels, kernel_size).
///
/// # Fields
///
/// - `cache` - Optional moving average of squared gradients for 3D weight parameters
///   Stores values for 1D convolutional kernels used in sequential feature extraction.
///   None when the layer has no weight parameters.
///
/// - `bias` - Optional moving average of squared gradients for bias parameters Remains 2D even in convolutional contexts but can be None when bias is disabled.
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv1D {
    pub cache: Option<Array3<f32>>,
    pub bias: Option<Array2<f32>>,
}

/// RMSprop cache for Conv2D layers.
///
/// This specialized cache is designed for convolutional neural networks and other architectures
/// that use multi-dimensional tensors for feature extraction. Unlike the standard RMSprop cache,
/// this variant works with 4D arrays typically used in convolutional layers (batch_size, channels,
/// height, width).
///
/// # Fields
///
/// - `cache` - Moving average of squared gradients for 4D weight parameters Typically stores values for convolutional kernels or other multi-dimensional feature extraction parameters.
///
/// - `bias` - Moving average of squared gradients for bias parameters, which remain 2D even in convolutional contexts.
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv2D {
    pub cache: Array4<f32>,
    pub bias: Array2<f32>,
}

/// RMSprop cache for 3D convolutional layers.
///
/// This structure stores the exponentially decaying averages of squared gradients
/// required by the RMSprop (Root Mean Square Propagation) optimizer for updating
/// parameters in 3D convolutional neural network layers. RMSprop adapts the learning
/// rate for each parameter individually by dividing the gradient by a running average
/// of the magnitudes of recent gradients for that parameter.
///
/// # Fields
///
/// - `cache` - 5D array storing the exponentially decaying average of squared gradients
///   for convolution weights with shape (output_channels, input_channels, kernel_depth,
///   kernel_height, kernel_width)
///
/// - `bias` - 3D array storing the exponentially decaying average of squared gradients
///   for bias parameters with shape (1, output_channels)
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv3D {
    pub cache: Array5<f32>,
    pub bias: Array2<f32>,
}

/// RMSprop cache for normalization layers.
///
/// This specialized cache is designed for normalization layers (e.g., BatchNormalization,
/// LayerNormalization) that have gamma (scale) and beta (shift) parameters. It maintains
/// moving averages of squared gradients for these parameters.
///
/// # Fields
///
/// - `cache_gamma` - Moving average of squared gradients for gamma (scale) parameter
/// - `cache_beta` - Moving average of squared gradients for beta (shift) parameter
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheNormalizationLayer {
    pub cache_gamma: Tensor,
    pub cache_beta: Tensor,
}
