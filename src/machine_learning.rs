use crate::{Deserialize, Serialize};

/// Represents different distance calculation methods used in various machine learning algorithms.
///
/// This enum defines common distance metrics that can be used in clustering algorithms,
/// nearest neighbor searches, and other applications where distance between points is relevant.
///
/// # Variants
///
/// - `Euclidean` - Euclidean distance (L2 norm), calculated as the square root of the sum of squared differences between corresponding coordinates.
/// - `Manhattan` - Manhattan distance (L1 norm), calculated as the sum of absolute differences between corresponding coordinates.
/// - `Minkowski` - A generalized metric that includes both Euclidean and Manhattan distances as special cases. Requires an additional parameter p (f64).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum DistanceCalculationMetric {
    Euclidean,
    Manhattan,
    Minkowski(f64),
}

/// Represents different types of regularization techniques used in machine learning models.
///
/// Regularization helps prevent overfitting by adding a penalty term to the model's loss function
/// during training. This enum defines common regularization approaches that can be applied to
/// various learning algorithms.
///
/// # Variants
///
/// - `L1` - L1 regularization (Lasso) that adds the sum of absolute values of parameters
///   multiplied by the specified coefficient. Promotes sparse solutions by driving some
///   parameters to exactly zero.
/// - `L2` - L2 regularization (Ridge) that adds the sum of squared parameter values
///   multiplied by the specified coefficient. Discourages large parameter values but
///   typically does not produce sparse solutions.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum RegularizationType {
    L1(f64),
    L2(f64),
}

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm implementation
pub mod dbscan;
/// Decision Tree implementation for classification and regression task
pub mod decision_tree;
/// This module provides helper functions for machine learning models
mod helper_function;
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
/// This module provides an implementation of Support Vector Classification
pub mod svc;

pub use dbscan::*;
pub use decision_tree::*;
pub use isolation_forest::*;
pub use kmeans::*;
pub use knn::*;
pub use linear_regression::*;
pub use linear_svc::*;
pub use logistic_regression::*;
pub use meanshift::*;
pub use svc::*;
