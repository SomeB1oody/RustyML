use super::super::{Layer, Optimizer};
use ndarray::{Array2, Array3, Array4, Array5};

/// RMSprop optimizer implementation.
///
/// An optimization algorithm that adapts the learning rate for each parameter
/// using a moving average of squared gradients.
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
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `rho` - Decay rate for moving average of squared gradients (typically 0.9)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// * `RMSprop` - A new RMSprop optimizer instance
    pub fn new(learning_rate: f32, rho: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon,
        }
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, layer: &mut dyn Layer) {
        layer.update_parameters_rmsprop(self.learning_rate, self.rho, self.epsilon);
    }
}

/// Cache structure for the RMSprop optimization algorithm.
///
/// RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization
/// algorithm that maintains a moving average of squared gradients for each parameter.
/// This cache structure stores these moving averages for weights, recurrent weights (if present),
/// and biases during training.
///
/// # Fields
///
/// - `cache`: Moving average of squared gradients for main weight parameters.
/// - `cache_recurrent`: Optional moving average of squared gradients for recurrent weight parameters. This is used in recurrent neural networks like RNN, LSTM, and GRU.
/// - `bias`: Moving average of squared gradients for bias parameters.
#[derive(Debug, Clone, Default)]
pub struct RMSpropCache {
    pub cache: Array2<f32>,
    pub cache_recurrent: Option<Array2<f32>>,
    pub bias: Array2<f32>,
}

impl RMSpropCache {
    /// Creates a new RMSprop cache instance
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
    /// Returns a new `RMSpropCache` instance containing:
    /// - `cache`: A zero-initialized 2D array with `dims` dimensions for storing running average of squared gradients for main parameters
    /// - `cache_recurrent`: If `recurrent_dims` is `Some`, a zero-initialized 2D array with corresponding dimensions; otherwise `None`
    /// - `bias`: A zero-initialized 2D array with `bias_dims` dimensions for storing running average of squared gradients for bias parameters
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

    /// Updates parameters and their corresponding cache
    ///
    /// # Parameters
    ///
    /// - `param` - Parameter to be updated
    /// - `grad` - Gradient corresponding to the parameter
    /// - `cache` - Cache corresponding to the parameter
    /// - `rho` - Decay rate
    /// - `lr` - Learning rate
    /// - `epsilon` - Small constant for numerical stability
    pub fn update_param(
        param: &mut Array2<f32>,
        grad: &Array2<f32>,
        cache: &mut Array2<f32>,
        rho: f32,
        lr: f32,
        epsilon: f32,
    ) {
        // Update cache
        *cache = cache.mapv(|x| x * rho) + &(grad.mapv(|x| x * x) * (1.0 - rho));

        // Apply parameter update
        *param = &*param - &(lr * grad / &(cache.mapv(f32::sqrt) + epsilon));
    }

    /// Updates all parameters using RMSprop optimization algorithm
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

/// Cache structure for the RMSprop optimization algorithm for Conv1D layer.
///
/// This specialized cache is designed for one-dimensional convolutional neural networks
/// that process sequential data such as time series or text. It maintains moving averages
/// of squared gradients for 3D weight parameters typically used in 1D convolutional layers
/// (batch_size, channels, kernel_size).
///
/// # Fields
///
/// - `cache`: Optional moving average of squared gradients for 3D weight parameters.
///   Stores values for 1D convolutional kernels used in sequential feature extraction.
///   None when the layer has no weight parameters.
///
/// - `bias`: Optional moving average of squared gradients for bias parameters.
///   Remains 2D even in convolutional contexts but can be None when bias is disabled.
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv1D {
    pub cache: Option<Array3<f32>>,
    pub bias: Option<Array2<f32>>,
}

/// Cache structure for the RMSprop optimization algorithm for Conv2D layer.
///
/// This specialized cache is designed for convolutional neural networks and other architectures
/// that use multi-dimensional tensors for feature extraction. Unlike the standard RMSprop cache,
/// this variant works with 4D arrays typically used in convolutional layers (batch_size, channels,
/// height, width).
///
/// # Fields
///
/// - `cache`: Moving average of squared gradients for 4D weight parameters. Typically stores values for
///   convolutional kernels or other multi-dimensional feature extraction parameters.
///
/// - `bias`: Moving average of squared gradients for bias parameters, which remain 2D even in
///   convolutional contexts.
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv2D {
    pub cache: Array4<f32>,
    pub bias: Array2<f32>,
}

/// RMSprop optimizer cache for 3D convolutional layers
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
