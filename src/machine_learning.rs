use crate::ModelError;
use crate::math::*;
use ahash::{AHashMap, AHashSet};
use ndarray::prelude::*;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceCalculationMetric {
    Euclidean,
    Manhattan,
    Minkowski(f64),
}

/// Performs validation checks on the input data matrices.
///
/// This function validates that:
/// - The input data matrix is not empty
/// - The input data does not contain NaN or infinite values
/// - When a target vector is provided:
///   - The target vector is not empty
///   - The target vector length matches the number of rows in the input data
///
/// # Parameters
///
/// * `x` - A 2D array of feature values where rows represent samples and columns represent features
/// * `y` - An optional 1D array representing the target variables or labels corresponding to each sample
///
/// # Returns
///
/// - `Ok(())` - If all validation checks pass
/// - `Err(ModelError::InputValidationError)` - If any validation check fails, with an informative error message
fn preliminary_check(x: ArrayView2<f64>, y: Option<ArrayView1<f64>>) -> Result<(), ModelError> {
    if x.nrows() == 0 {
        return Err(ModelError::InputValidationError(
            "Input data is empty".to_string(),
        ));
    }

    for (i, row) in x.outer_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(ModelError::InputValidationError(format!(
                    "Input data contains NaN or infinite value at position [{}][{}]",
                    i, j
                )));
            }
        }
    }

    if let Some(y) = y {
        if y.len() == 0 {
            return Err(ModelError::InputValidationError(
                "Target vector is empty".to_string(),
            ));
        }

        if y.len() != x.nrows() {
            return Err(ModelError::InputValidationError(format!(
                "Input data and target vector have different lengths, x columns: {}, y length: {}",
                x.nrows(),
                y.len()
            )));
        }
    }
    Ok(())
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso) with the specified regularization strength coefficient
    L1(f64),
    /// L2 regularization (Ridge) with the specified regularization strength coefficient
    L2(f64),
}

/// Linear regression module implementing the ordinary least squares method
pub mod linear_regression;

/// Logistic regression module for binary classification problems
pub mod logistic_regression;

/// K-means clustering implementation for unsupervised learning
pub mod kmeans;

/// K-Nearest Neighbors (KNN) implementation for classification and regression
pub mod knn;

/// Mean Shift clustering algorithm implementation
pub mod meanshift;

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm implementation
pub mod dbscan;

/// Decision Tree implementation for classification and regression task
pub mod decision_tree;

/// Isolation Forest algorithm implementation for anomaly detection
pub mod isolation_forest;

/// This module provides an implementation of Support Vector Classification
pub mod svc;

/// This module contains the implementation of Linear Support Vector Classification (Linear SVC)
pub mod linear_svc;

/// This module provides an implementation of Linear Discriminant Analysis
pub mod linear_discriminant_analysis;

pub use crate::utility::KernelType;
pub use dbscan::*;
pub use decision_tree::*;
pub use isolation_forest::*;
pub use kmeans::*;
pub use knn::*;
pub use linear_discriminant_analysis::*;
pub use linear_regression::*;
pub use linear_svc::*;
pub use logistic_regression::*;
pub use meanshift::*;
pub use svc::*;
