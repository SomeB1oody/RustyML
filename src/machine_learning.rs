//! Machine learning models for clustering, classification, regression, and anomaly detection
//!
//! Provides DBSCAN, K-means, and Mean Shift clustering; K-Nearest Neighbors, decision trees,
//! Linear SVC, SVC, logistic regression, and Linear Discriminant Analysis for classification;
//! linear regression for regression; Isolation Forest for anomaly detection; and the shared
//! [`Fit`](crate::machine_learning::traits::Fit) / [`Predict`](crate::machine_learning::traits::Predict) traits implemented by every estimator

pub use crate::types::{DistanceCalculationMetric, KernelType, RegularizationType};

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN) clustering
pub mod dbscan;
/// Decision tree for classification and regression
pub mod decision_tree;
/// Isolation Forest for anomaly detection
pub mod isolation_forest;
/// K-means clustering for unsupervised learning
pub mod kmeans;
/// K-Nearest Neighbors (KNN) classification
pub mod knn;
/// Linear Discriminant Analysis for classification and supervised dimensionality reduction
pub mod lda;
/// Linear regression via gradient descent
pub mod linear_regression;
/// Linear Support Vector Classification (Linear SVC)
pub mod linear_svc;
/// Logistic regression for binary classification
pub mod logistic_regression;
/// Mean Shift clustering
pub mod mean_shift;
/// Internal shared helpers for parallel/sequential dispatch across models
mod parallel;
/// Support Vector Classification (SVC)
pub mod svc;
/// Common `Fit` / `Predict` traits implemented by every estimator
pub mod traits;
/// Internal shared input-validation helpers used by every model
mod validation;

pub use dbscan::DBSCAN;
pub use decision_tree::{Algorithm, DecisionTree, DecisionTreeParams, Node, NodeType};
pub use isolation_forest::{IsolationForest, IsolationTree};
pub use kmeans::KMeans;
pub use knn::{KNN, WeightingStrategy};
pub use lda::{LDA, Shrinkage, Solver};
pub use linear_regression::LinearRegression;
pub use linear_svc::LinearSVC;
pub use logistic_regression::{LogisticRegression, generate_polynomial_features};
pub use mean_shift::{MeanShift, estimate_bandwidth};
pub use svc::SVC;
pub use traits::{Fit, Predict};
