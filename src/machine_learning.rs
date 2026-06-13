//! Machine learning models for clustering, classification, regression, and anomaly detection
//!
//! Models are grouped by algorithm family into submodules — [`clustering`](crate::machine_learning::clustering),
//! [`linear_model`](crate::machine_learning::linear_model), [`svm`](crate::machine_learning::svm),
//! [`tree`](crate::machine_learning::tree), [`neighbors`](crate::machine_learning::neighbors),
//! [`discriminant_analysis`](crate::machine_learning::discriminant_analysis), and
//! [`ensemble`](crate::machine_learning::ensemble) — and every estimator is also re-exported here
//! so it is reachable directly as `machine_learning::<Model>`.
//! All estimators implement the shared [`Fit`](crate::machine_learning::traits::Fit) /
//! [`Predict`](crate::machine_learning::traits::Predict) traits.

pub use crate::types::{DistanceCalculationMetric, KernelType, RegularizationType};

/// Clustering estimators: DBSCAN, K-means, and Mean Shift
pub mod clustering;
/// Discriminant analysis: Linear Discriminant Analysis (LDA)
pub mod discriminant_analysis;
/// Ensemble models: Isolation Forest
pub mod ensemble;
/// Linear models: linear and logistic regression
pub mod linear_model;
/// Nearest-neighbor models: K-Nearest Neighbors
pub mod neighbors;
/// Support vector machines: SVC and Linear SVC
pub mod svm;
/// Tree models: decision trees
pub mod tree;

/// Common `Fit` / `Predict` traits implemented by every estimator
pub mod traits;

/// Internal shared helpers for parallel/sequential dispatch across models
mod parallel;
/// Internal kd-tree spatial index for fixed-radius and k-nearest-neighbor queries
pub(crate) mod spatial;
/// Internal shared input-validation helpers used by every model
mod validation;

pub use clustering::{DBSCAN, KMeans, MeanShift, estimate_bandwidth};
pub use discriminant_analysis::{LDA, Shrinkage, Solver};
pub use ensemble::{IsolationForest, IsolationTree};
pub use linear_model::{LinearRegression, LogisticRegression, generate_polynomial_features};
pub use neighbors::{KNN, WeightingStrategy};
pub use svm::{LinearSVC, SVC};
pub use traits::{Fit, Predict};
pub use tree::{Algorithm, DecisionTree, DecisionTreeParams, Node, NodeType};
