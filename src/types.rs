//! Shared configuration types used across estimators and utilities.
//!
//! These enums live here, in one place, because they are shared by more than one
//! module:
//! - [`RegularizationType`](crate::types::RegularizationType) and [`DistanceCalculationMetric`](crate::types::DistanceCalculationMetric) — used by the machine
//!   learning models.
//! - [`KernelType`] — used by both [`SVC`](crate::machine_learning::svc::SVC) and
//!   [`KernelPCA`](crate::utility::kernel_pca::KernelPCA).
//!
//! Each type also carries the behavior that belongs to it (distance / kernel
//! evaluation) as inherent methods, so consumers share a single implementation
//! instead of re-matching the enum in every model.

use crate::{Deserialize, Serialize};
use ndarray::ArrayView1;

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

impl DistanceCalculationMetric {
    /// Computes the distance between two vectors under this metric.
    ///
    /// This is the single source of truth for metric dispatch; models such as KNN
    /// and DBSCAN call it instead of re-implementing the `match` over variants.
    ///
    /// # Parameters
    ///
    /// - `a` - First vector
    /// - `b` - Second vector
    ///
    /// # Returns
    ///
    /// - `f64` - The distance between `a` and `b` under this metric
    pub fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        use crate::math::{
            manhattan_distance_row, minkowski_distance_row, squared_euclidean_distance_row,
        };
        match *self {
            DistanceCalculationMetric::Euclidean => squared_euclidean_distance_row(&a, &b).sqrt(),
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(&a, &b),
            DistanceCalculationMetric::Minkowski(p) => minkowski_distance_row(&a, &b, p),
        }
    }
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

/// Kernel function types for Support Vector Machine and Kernel PCA.
///
/// # Variants
/// - `Linear` - Linear kernel: K(x, y) = x·y
/// - `Poly` - Polynomial kernel: K(x, y) = (gamma·x·y + coef0)^degree
/// - `RBF` - Radial Basis Function kernel: K(x, y) = exp(-gamma·|x-y|^2)
/// - `Sigmoid` - Sigmoid kernel: K(x, y) = tanh(gamma·x·y + coef0)
/// - `Cosine` - Cosine kernel: K(x, y) = (x dot y) / (||x|| * ||y||)
#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum KernelType {
    Linear,
    Poly { degree: u32, gamma: f64, coef0: f64 },
    RBF { gamma: f64 },
    Sigmoid { gamma: f64, coef0: f64 },
    Cosine,
}

impl KernelType {
    /// Computes the kernel function value between two vectors.
    ///
    /// This is the single source of truth for kernel dispatch, shared by SVC and
    /// Kernel PCA instead of each re-implementing the `match` over variants.
    ///
    /// # Parameters
    ///
    /// - `x1` - First input vector
    /// - `x2` - Second input vector
    ///
    /// # Returns
    ///
    /// - `f64` - The kernel function value between `x1` and `x2`
    pub fn compute(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        match *self {
            // K(x, y) = x·y
            KernelType::Linear => x1.dot(&x2),
            // K(x, y) = (gamma·x·y + coef0)^degree
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => (gamma * x1.dot(&x2) + coef0).powf(degree as f64),
            // K(x, y) = exp(-gamma·|x-y|^2)
            KernelType::RBF { gamma } => {
                let diff = &x1 - &x2;
                let squared_norm = diff.dot(&diff);
                (-gamma * squared_norm).exp()
            }
            // K(x, y) = tanh(gamma·x·y + coef0)
            KernelType::Sigmoid { gamma, coef0 } => (gamma * x1.dot(&x2) + coef0).tanh(),
            // K(x, y) = (x·y) / (||x|| * ||y||)
            KernelType::Cosine => {
                let norm_product = (x1.dot(&x1) * x2.dot(&x2)).sqrt();
                if norm_product <= f64::EPSILON {
                    0.0
                } else {
                    x1.dot(&x2) / norm_product
                }
            }
        }
    }
}
