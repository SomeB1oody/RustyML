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
/// - `Minkowski` - A generalized metric that includes both Euclidean and Manhattan distances as special cases. Requires an additional parameter p (not implemented in this enum).
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceCalculationMetric {
    Euclidean,
    Manhattan,
    Minkowski,
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
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso) with the specified regularization strength coefficient
    L1(f64),
    /// L2 regularization (Ridge) with the specified regularization strength coefficient
    L2(f64),
}

/// A macro that generates a getter method for the `fit_intercept` field.
///
/// This macro expands to a public method that returns whether the model
/// should fit an intercept term during training.
///
/// # Generated Method
///
/// * `get_fit_intercept(&self) -> bool` - Returns the current intercept fitting setting
///
/// # Usage
///
/// This macro should be used inside structs that have a `fit_intercept: bool` field.
/// When called, it will generate the corresponding getter method.
macro_rules! get_fit_intercept {
    () => {
        /// Gets the current setting for fitting the intercept term
        ///
        /// # Returns
        ///
        /// * `bool` - Returns `true` if the model includes an intercept term, `false` otherwise
        pub fn get_fit_intercept(&self) -> bool {
            self.fit_intercept
        }
    };
}

/// A macro that generates a getter method for the `learning_rate` field.
///
/// This macro expands to a public method that returns the learning rate
/// value used in gradient-based optimization algorithms.
///
/// # Generated Method
///
/// * `get_learning_rate(&self) -> f64` - Returns the current learning rate value
///
/// # Usage
///
/// This macro should be used inside structs that have a `learning_rate: f64` field.
/// The learning rate controls the step size in gradient descent optimization.
macro_rules! get_learning_rate {
    () => {
        /// Gets the current learning rate
        ///
        /// The learning rate controls the step size in each iteration of gradient descent.
        ///
        /// # Returns
        ///
        /// * `f64` - The current learning rate value
        pub fn get_learning_rate(&self) -> f64 {
            self.learning_rate
        }
    };
}

/// A macro that generates a getter method for the `max_iter` field.
///
/// This macro expands to a public method that returns the maximum number
/// of iterations allowed for iterative algorithms.
///
/// # Generated Method
///
/// * `get_max_iterations(&self) -> usize` - Returns the maximum iteration limit
///
/// # Usage
///
/// This macro should be used inside structs that have a `max_iter: usize` field.
/// This value determines when to stop training if convergence is not achieved.
macro_rules! get_max_iterations {
    () => {
        /// Gets the maximum number of iterations
        ///
        /// # Returns
        ///
        /// * `usize` - The maximum number of iterations for the gradient descent algorithm
        pub fn get_max_iterations(&self) -> usize {
            self.max_iter
        }
    };
}

/// A macro that generates a getter method for the `tol` field.
///
/// This macro expands to a public method that returns the convergence
/// tolerance threshold used to determine when training should stop.
///
/// # Generated Method
///
/// * `get_tolerance(&self) -> f64` - Returns the current tolerance value
///
/// # Usage
///
/// This macro should be used inside structs that have a `tol: f64` field.
/// Training stops when the improvement between iterations falls below this threshold.
macro_rules! get_tolerance {
    () => {
        /// Gets the convergence tolerance threshold
        ///
        /// The convergence tolerance is used to determine when to stop the training process.
        /// Training stops when the change in the loss function between consecutive iterations
        /// is less than this value.
        ///
        /// # Returns
        ///
        /// * `f64` - The current convergence tolerance value
        pub fn get_tolerance(&self) -> f64 {
            self.tol
        }
    };
}

/// A macro that generates a getter method for the `n_iter` field.
///
/// This macro expands to a public method that returns the actual number
/// of iterations performed during the last training session.
///
/// # Generated Method
///
/// * `get_actual_iterations(&self) -> Result<usize, ModelError>` - Returns the iteration count or an error if not fitted
///
/// # Usage
///
/// This macro should be used inside structs that have a `n_iter: Option<usize>` field.
/// Returns an error if the model has not been trained yet.
macro_rules! get_actual_iterations {
    () => {
        /// Returns the actual number of actual iterations performed during the last model fitting.
        ///
        /// # Returns
        ///
        /// - `Ok(usize)` - The number of iterations if the model has been fitted
        /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
        pub fn get_actual_iterations(&self) -> Result<usize, ModelError> {
            match &self.n_iter {
                Some(n_iter) => Ok(*n_iter),
                None => Err(ModelError::NotFitted),
            }
        }
    };
}

/// A macro that generates a getter method for the `regularization_type` field.
///
/// This macro expands to a public method that returns a reference to the
/// regularization configuration of the model.
///
/// # Generated Method
///
/// * `get_regularization_type(&self) -> &Option<RegularizationType>` - Returns a reference to the regularization type
///
/// # Usage
///
/// This macro should be used inside structs that have a `regularization_type: Option<RegularizationType>` field.
/// The regularization type can be None, L1 (LASSO), or L2 (Ridge).
macro_rules! get_regularization_type {
    () => {
        /// Returns a reference to the regularization type of the model
        ///
        /// This method provides access to the regularization configuration of the model,
        /// which can be None (no regularization), L1 (LASSO), or L2 (Ridge).
        ///
        /// # Returns
        ///
        /// * `&Option<RegularizationType>` - A reference to the regularization type, which will be None if no regularization is applied
        pub fn get_regularization_type(&self) -> &Option<RegularizationType> {
            &self.regularization_type
        }
    };
}

/// Generates a getter method for accessing the distance metric used by the instance.
///
/// This macro creates a standard getter method that returns a reference to the distance
/// calculation metric stored in the `metric` field. The metric determines how distances
/// between data points are calculated in various machine learning algorithms.
///
/// # Generated Method
///
/// * `get_metric(&self) -> &DistanceCalculationMetric` - Returns a reference to the DistanceCalculationMetric
macro_rules! get_metric {
    () => {
        pub fn get_metric(&self) -> &DistanceCalculationMetric {
            &self.metric
        }
    };
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
