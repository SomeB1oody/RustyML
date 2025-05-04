pub mod adam;
pub mod rmsprop;
pub mod sgd;

pub use adam::*;
pub use rmsprop::*;
pub use sgd::*;

pub struct OptimizerCache {
    pub adam_states: Option<AdamStates>,
    pub rmsprop_cache: Option<RMSpropCache>,
}
