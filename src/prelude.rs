//! Common public types and functions grouped by feature.
//!
//! Import the root to use every enabled category, or select a category submodule.
//! The root resolves the shared `Average` name to the metrics enum when metrics are enabled.
//!
//! # Examples
//!
//! ```rust
//! // Bring every enabled category's items into scope at once:
//! use rustyml::prelude::*;
//!
//! // Or import a single category:
//! // `use rustyml::prelude::machine_learning::*;` for the machine learning models.
//! // `use rustyml::prelude::utils::*;`            for the utility functions.
//! // `use rustyml::prelude::metrics::*;`          for the metric functions.
//! ```

/// Prelude re-exports for the machine learning estimators, traits, and shared enums
#[cfg(feature = "machine_learning")]
pub mod machine_learning;
/// Prelude re-exports for evaluation metrics
#[cfg(feature = "metrics")]
pub mod metrics;
/// Prelude re-exports for the neural network framework: tensors, layers, losses, optimizers,
/// and models
#[cfg(feature = "neural_network")]
pub mod neural_network;
/// Prelude re-exports for the utils module: preprocessing, dataset splitting, and shared
/// estimator traits
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

// `Average` names 2 different items: the metrics averaging mode and the neural network merge
// layer. Both globs above export it, so this explicit re-export breaks the tie and keeps the
// averaging mode at the root. The layer stays reachable as `neural_network::layers::Average`
#[cfg(feature = "metrics")]
pub use crate::metrics::Average;
