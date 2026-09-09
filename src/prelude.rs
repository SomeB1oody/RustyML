//! Prelude that re-exports the crate's machine learning, metrics, neural network, and utility items
//!
//! A single import point for the crate's most commonly used types, traits, and functions. `use
//! rustyml::prelude::*` pulls in every enabled category at once. Alternatively, import 1
//! category at a time through the [`machine_learning`], [`metrics`], [`neural_network`], and
//! [`utils`] submodules. Each category is feature-gated, so the prelude exposes only the items
//! whose feature is enabled
//!
//! # Available components
//!
//! - **Machine learning**: classification (KNN, DecisionTree, LogisticRegression, SVC, LinearSVC,
//!   LDA), regression (LinearRegression), clustering (KMeans, DBSCAN, MeanShift), dimensionality
//!   reduction (PCA, KernelPCA, t-SNE), and anomaly detection (IsolationForest)
//! - **Utilities**: preprocessing (standardize, StandardScaler, normalize, label encoding) and
//!   dataset splitting (train_test_split)
//! - **Metrics**: regression, classification, and clustering evaluation metrics
//! - **Neural network**: layers, optimizers, loss functions, and the `SequentialBuilder` that
//!   makes a `Sequential` model
//!
//! # Examples
//!
//! ```rust
//! // Bring every enabled category's items into scope at once:
//! use rustyml::prelude::*;
//!
//! // Or import a single category:
//! // `use rustyml::prelude::machine_learning::*;` for the machine learning models
//! // `use rustyml::prelude::utils::*;`            for the utility functions
//! // `use rustyml::prelude::metrics::*;`          for the metric functions
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
