/// Prelude module for machine learning functionality
#[cfg(feature = "machine_learning")]
pub mod machine_learning;
/// Prelude module for metric functions
#[cfg(feature = "metrics")]
pub mod metrics;
/// Prelude module for neural network framework
#[cfg(feature = "neural_network")]
pub mod neural_network;
/// Prelude module for utility functions.
#[cfg(feature = "utils")]
pub mod utils;

// Flatten every category into the prelude root
#[cfg(feature = "machine_learning")]
pub use self::machine_learning::*;
#[cfg(feature = "metrics")]
pub use self::metrics::*;
#[cfg(feature = "neural_network")]
pub use self::neural_network::*;
#[cfg(feature = "utils")]
pub use self::utils::*;
