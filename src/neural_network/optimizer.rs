pub mod adam;
pub mod rmsprop;
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

/// Cache structure for storing optimization algorithm states(For Feature Extraction Layers)
#[derive(Debug, Clone, Default)]
pub struct OptimizerCacheFEX {
    pub adam_states: Option<AdamStatesFEX>,
    pub rmsprop_cache: Option<RMSpropCacheFEX>,
}
