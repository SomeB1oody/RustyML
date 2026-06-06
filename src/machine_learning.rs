// The shared estimator configuration enums live in the crate-level `types` module
// (because `KernelType` is also used by the `utility` kernel-PCA code). They are
// re-exported here so they remain reachable as `machine_learning::*` and via each
// model submodule (e.g. `machine_learning::dbscan::DistanceCalculationMetric`).
pub use crate::types::{DistanceCalculationMetric, KernelType, RegularizationType};

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm implementation
pub mod dbscan;
/// Decision Tree implementation for classification and regression task
pub mod decision_tree;
/// Isolation Forest algorithm implementation for anomaly detection
pub mod isolation_forest;
/// K-means clustering implementation for unsupervised learning
pub mod kmeans;
/// K-Nearest Neighbors (KNN) implementation for classification and regression
pub mod knn;
/// Linear regression module implementing the ordinary least squares method
pub mod linear_regression;
/// This module contains the implementation of Linear Support Vector Classification (Linear SVC)
pub mod linear_svc;
/// Logistic regression module for binary classification problems
pub mod logistic_regression;
/// Mean Shift clustering algorithm implementation
pub mod meanshift;
/// Internal shared helpers for parallel/sequential dispatch across models
mod parallel;
/// This module provides an implementation of Support Vector Classification
pub mod svc;
/// Common `Fit` / `Predict` traits implemented by every estimator
pub mod traits;
/// Internal shared input-validation helpers used by every model
mod validation;

// Explicit re-exports keep the flat `machine_learning::` API surface intentional and
// stable: adding a `pub` item to a submodule no longer silently changes this namespace.
pub use dbscan::DBSCAN;
pub use decision_tree::{Algorithm, DecisionTree, DecisionTreeParams, Node, NodeType};
pub use isolation_forest::{IsolationForest, IsolationTree};
pub use kmeans::KMeans;
pub use knn::{KNN, WeightingStrategy};
pub use linear_regression::LinearRegression;
pub use linear_svc::LinearSVC;
pub use logistic_regression::{LogisticRegression, generate_polynomial_features};
pub use meanshift::{MeanShift, estimate_bandwidth};
pub use svc::SVC;
pub use traits::{Fit, Predict};
