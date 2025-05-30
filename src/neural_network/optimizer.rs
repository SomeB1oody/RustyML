/// Adam optimizer implementation
pub mod adam;
/// RMSprop optimizer implementation
pub mod rmsprop;
/// Stochastic Gradient Descent (SGD) optimizer.
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
