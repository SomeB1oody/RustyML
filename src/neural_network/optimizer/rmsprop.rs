use crate::neural_network::{Layer, Optimizer};
use ndarray::{Array2, Array3, Array4};

/// RMSprop optimizer implementation.
///
/// An optimization algorithm that adapts the learning rate for each parameter
/// using a moving average of squared gradients.
pub struct RMSprop {
    /// Learning rate controlling the size of parameter updates.
    learning_rate: f32,
    /// Decay rate for the moving average of squared gradients.
    rho: f32,
    /// Small constant added for numerical stability.
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
    /// A new RMSprop optimizer instance
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
    /// Creates a new RMSprop cache
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
/// - `bias`: Moving average of squared gradients for bias parameters, which remain 2D even in
///   convolutional contexts.
#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv2D {
    pub cache: Array4<f32>,
    pub bias: Array2<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct RMSpropCacheConv1D {
    pub cache: Option<Array3<f32>>,
    pub bias: Option<Array2<f32>>,
}
