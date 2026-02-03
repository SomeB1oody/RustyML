pub use ada_grad::AdaGrad;
pub use adam::Adam;
pub use rms_prop::RMSprop;
pub use sgd::SGD;

use ada_grad::{
    AdaGradStates, AdaGradStatesConv1D, AdaGradStatesConv2D, AdaGradStatesConv3D,
    AdaGradStatesNormalizationLayer,
};
use adam::{
    AdamStates, AdamStatesConv1D, AdamStatesConv2D, AdamStatesConv3D, AdamStatesNormalizationLayer,
};
use rms_prop::{
    RMSpropCache, RMSpropCacheConv1D, RMSpropCacheConv2D, RMSpropCacheConv3D,
    RMSpropCacheNormalizationLayer,
};

/// AdaGrad (Adaptive Gradient Algorithm) optimizer
pub mod ada_grad;
/// Adam (Adaptive Moment Estimation) optimizer
pub mod adam;
/// Input validation functions for optimizers
mod input_validation_function;
/// RMSprop (Root Mean Square Propagation) optimizer
pub mod rms_prop;
/// SGD (Stochastic Gradient Descent) optimizer
pub mod sgd;

/// Optimizer cache for dense layers.
///
/// Stores optional per-optimizer state between iterations for dense layers.
///
/// # Fields
///
/// - `adam_states` - Optional cache storage for Adam optimizer states (momentum and velocity terms)
/// - `rmsprop_cache` - Optional cache storage for RMSprop optimizer running averages
/// - `ada_grad_cache` - Optional cache storage for AdaGrad optimizer accumulated squared gradients
#[derive(Debug, Clone, Default)]
pub struct OptimizerCache {
    pub adam_states: Option<AdamStates>,
    pub rmsprop_cache: Option<RMSpropCache>,
    pub ada_grad_cache: Option<AdaGradStates>,
}

/// Optimizer cache for 1D convolutional layers.
///
/// Stores optional per-optimizer state between iterations for Conv1D layers.
///
/// # Fields
///
/// - `adam_states` - Optional cache storage for Adam optimizer states including first and second moment estimates for both weights and biases used in 1D convolution
/// - `rmsprop_cache` - Optional cache storage for RMSprop optimizer state including exponentially decaying averages of squared gradients for weights and biases
/// - `ada_grad_cache` - Optional cache storage for AdaGrad optimizer accumulated squared gradients
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheConv1D {
    pub adam_states: Option<AdamStatesConv1D>,
    pub rmsprop_cache: Option<RMSpropCacheConv1D>,
    pub ada_grad_cache: Option<AdaGradStatesConv1D>,
}

/// Optimizer cache for 2D convolutional layers.
///
/// Stores optional per-optimizer state between iterations for Conv2D layers.
///
/// # Fields
///
/// - `adam_states` - Optional cache storage for Adam optimizer states (momentum and velocity terms) for feature extraction layers
/// - `rmsprop_cache` - Optional cache storage for RMSprop optimizer running averages for feature extraction layers
/// - `ada_grad_cache` - Optional cache storage for AdaGrad optimizer accumulated squared gradients
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheConv2D {
    pub adam_states: Option<AdamStatesConv2D>,
    pub rmsprop_cache: Option<RMSpropCacheConv2D>,
    pub ada_grad_cache: Option<AdaGradStatesConv2D>,
}

/// Optimizer cache for 3D convolutional layers.
///
/// Stores optional per-optimizer state between iterations for Conv3D layers.
///
/// # Fields
///
/// - `adam_states` - Optional cache for Adam optimizer state variables including first and second moment estimates for both weights and biases
/// - `rmsprop_cache` - Optional cache for RMSprop optimizer state variables including exponentially decaying averages of squared gradients for weights and biases
/// - `ada_grad_cache` - Optional cache storage for AdaGrad optimizer accumulated squared gradients
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheConv3D {
    pub adam_states: Option<AdamStatesConv3D>,
    pub rmsprop_cache: Option<RMSpropCacheConv3D>,
    pub ada_grad_cache: Option<AdaGradStatesConv3D>,
}

/// Optimizer cache for normalization layers.
///
/// Stores optional per-optimizer state between iterations for normalization layers.
///
/// # Fields
///
/// - `adam_states` - Optional cache for Adam optimizer state variables including first and second moment estimates for gamma and beta parameters
/// - `rmsprop_cache` - Optional cache for RMSprop optimizer state variables including exponentially decaying averages of squared gradients for gamma and beta parameters
/// - `ada_grad_cache` - Optional cache storage for AdaGrad optimizer accumulated squared gradients for gamma and beta parameters
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheNormalizationLayer {
    pub adam_states: Option<AdamStatesNormalizationLayer>,
    pub rmsprop_cache: Option<RMSpropCacheNormalizationLayer>,
    pub ada_grad_cache: Option<AdaGradStatesNormalizationLayer>,
}
