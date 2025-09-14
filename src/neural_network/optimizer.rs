/// Adam optimizer implementation
pub mod adam;
/// RMSprop optimizer implementation
pub mod rmsprop;
/// Stochastic Gradient Descent (SGD) optimizer
pub mod sgd;

pub use adam::*;
pub use rmsprop::*;
pub use sgd::*;

/// Cache structure for storing optimization algorithm states.
///
/// This structure maintains the state information required by different optimization algorithms
/// between iterations. It can store states for Adam optimizer and RMSprop optimizer simultaneously.
///
/// # Fields
///
/// - `adam_states` - Optional cache storage for Adam optimizer states (momentum and velocity terms)
/// - `rmsprop_cache` - Optional cache storage for RMSprop optimizer running averages
#[derive(Debug, Clone, Default)]
pub struct OptimizerCache {
    pub adam_states: Option<AdamStates>,
    pub rmsprop_cache: Option<RMSpropCache>,
}

/// Optimizer cache for 1D convolutional layer
///
/// Stores optimizer-specific state for Adam and RMSprop optimizers.
/// This includes momentum terms and running averages for both weights and bias.
///
/// # Fields
///
/// - `adam_states` - Optional cache storage for Adam optimizer states including first
///   and second moment estimates for both weights and biases used in 1D convolution
///
/// - `rmsprop_cache` - Optional cache storage for RMSprop optimizer state including
///   exponentially decaying averages of squared gradients for weights and biases
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheConv1D {
    pub adam_states: Option<AdamStatesConv1D>,
    pub rmsprop_cache: Option<RMSpropCacheConv1D>,
}

/// Cache structure for storing optimization algorithm states for Conv2D layer.
///
/// This structure maintains the state information required by different optimization algorithms
/// between iterations specifically for Conv2D layers. It can store states for
/// Adam optimizer and RMSprop optimizer simultaneously.
///
/// # Fields
///
/// - `adam_states` - Optional cache storage for Adam optimizer states (momentum and velocity terms) for feature extraction layers
/// - `rmsprop_cache` - Optional cache storage for RMSprop optimizer running averages for feature extraction layers
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheConv2D {
    pub adam_states: Option<AdamStatesConv2D>,
    pub rmsprop_cache: Option<RMSpropCacheConv2D>,
}

/// Optimizer cache for 3D convolutional layers
///
/// This structure serves as a unified cache container for different optimization algorithms
/// used with 3D convolutional neural network layers. It stores algorithm-specific state
/// variables that are required to maintain optimization momentum and adaptive learning
/// rates across training iterations.
///
/// The cache supports multiple optimization algorithms and only stores the state for
/// the currently active optimizer, helping to manage memory efficiently while providing
/// the flexibility to switch between different optimization strategies.
///
/// # Fields
///
/// - `adam_states` - Optional cache for Adam optimizer state variables including first
///   and second moment estimates for both weights and biases
///
/// - `rmsprop_cache` - Optional cache for RMSprop optimizer state variables including
///   exponentially decaying averages of squared gradients for weights and biases
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheConv3D {
    pub adam_states: Option<AdamStatesConv3D>,
    pub rmsprop_cache: Option<RMSpropCacheConv3D>,
}
