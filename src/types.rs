//! Shared configuration types used across estimators and utilities.
//!
//! These enums live here, in one place, because they are shared by more than one
//! module:
//! - [`RegularizationType`](crate::types::RegularizationType) and [`DistanceCalculationMetric`](crate::types::DistanceCalculationMetric) — used by the machine
//!   learning models.
//! - [`KernelType`] — used by both [`SVC`](crate::machine_learning::svc::SVC) and
//!   [`KernelPCA`](crate::utils::kernel_pca::KernelPCA).
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ── KernelType::compute ──────────────────────────────────────────────────

    // Linear kernel: K(x,y) = x·y
    // [1,0]·[0,1] = 1*0 + 0*1 = 0
    #[test]
    fn kernel_linear_orthogonal_vectors() {
        let k = KernelType::Linear;
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-6);
    }

    // [1,2]·[3,4] = 3 + 8 = 11
    #[test]
    fn kernel_linear_general_vectors() {
        let k = KernelType::Linear;
        let x1 = array![1.0_f64, 2.0];
        let x2 = array![3.0_f64, 4.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 11.0, epsilon = 1e-6);
    }

    // RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
    // Identical vectors → diff=[0,0], ||diff||^2=0, exp(0)=1
    #[test]
    fn kernel_rbf_identical_vectors() {
        let k = KernelType::RBF { gamma: 1.0 };
        let x = array![1.0_f64, 0.0];
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-6);
    }

    // diff=[1,-1], ||diff||^2=2, exp(-1*2)=exp(-2)≈0.135335283
    #[test]
    fn kernel_rbf_orthogonal_unit_vectors() {
        let k = KernelType::RBF { gamma: 1.0 };
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        let expected = (-2.0_f64).exp(); // ≈ 0.13533528323661
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), expected, epsilon = 1e-6);
    }

    // Polynomial kernel: K(x,y) = (gamma*x·y + coef0)^degree
    // degree=2, gamma=1, coef0=0; [1,0]·[0,1]=0 → (0)^2 = 0
    #[test]
    fn kernel_poly_degree2_orthogonal() {
        let k = KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 0.0,
        };
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-6);
    }

    // degree=3, gamma=2, coef0=1; [1,1]·[1,1]=2 → (2*2+1)^3=5^3=125
    #[test]
    fn kernel_poly_degree3_general() {
        let k = KernelType::Poly {
            degree: 3,
            gamma: 2.0,
            coef0: 1.0,
        };
        let x1 = array![1.0_f64, 1.0];
        let x2 = array![1.0_f64, 1.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 125.0, epsilon = 1e-6);
    }

    // Sigmoid kernel: K(x,y) = tanh(gamma*x·y + coef0)
    // gamma=1, coef0=0; [1,0]·[1,0]=1 → tanh(1)≈0.76159415595577
    #[test]
    fn kernel_sigmoid_unit_vector() {
        let k = KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        let x = array![1.0_f64, 0.0];
        let expected = 1.0_f64.tanh(); // ≈ 0.76159415595577
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), expected, epsilon = 1e-6);
    }

    // Cosine kernel: K(x,y) = (x·y) / (||x|| * ||y||)
    // zero vector → norm_product=0, epsilon guard → 0.0
    #[test]
    fn kernel_cosine_zero_vector() {
        let k = KernelType::Cosine;
        let zero = array![0.0_f64, 0.0];
        let other = array![1.0_f64, 2.0];
        assert_abs_diff_eq!(k.compute(zero.view(), other.view()), 0.0, epsilon = 1e-6);
    }

    // identical unit vectors → cos(0°)=1.0
    #[test]
    fn kernel_cosine_identical_vectors() {
        let k = KernelType::Cosine;
        let x = array![1.0_f64, 0.0];
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-6);
    }

    // orthogonal vectors → dot=0, cos(90°)=0.0
    #[test]
    fn kernel_cosine_orthogonal_vectors() {
        let k = KernelType::Cosine;
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-6);
    }

    // ── DistanceCalculationMetric::distance ─────────────────────────────────

    // Euclidean: sqrt((3-0)^2 + (4-0)^2) = sqrt(9+16) = 5
    #[test]
    fn distance_euclidean_345_triangle() {
        let metric = DistanceCalculationMetric::Euclidean;
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 5.0, epsilon = 1e-6);
    }

    // Manhattan: |3-0| + |4-0| = 7
    #[test]
    fn distance_manhattan_345() {
        let metric = DistanceCalculationMetric::Manhattan;
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 7.0, epsilon = 1e-6);
    }

    // Minkowski(3): (|3|^3 + |4|^3)^(1/3) = (27+64)^(1/3) = 91^(1/3) ≈ 4.497941
    #[test]
    fn distance_minkowski_p3() {
        let metric = DistanceCalculationMetric::Minkowski(3.0);
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        let expected = 91.0_f64.powf(1.0 / 3.0); // ≈ 4.497941445275415
        assert_abs_diff_eq!(
            metric.distance(a.view(), b.view()),
            expected,
            epsilon = 1e-6
        );
    }

    // Euclidean is symmetric: distance(a,b) == distance(b,a)
    #[test]
    fn distance_euclidean_symmetry() {
        let metric = DistanceCalculationMetric::Euclidean;
        let a = array![1.0_f64, 2.0, 3.0];
        let b = array![4.0_f64, 6.0, 8.0];
        assert_abs_diff_eq!(
            metric.distance(a.view(), b.view()),
            metric.distance(b.view(), a.view()),
            epsilon = 1e-10
        );
    }

    // Zero distance: identical vectors → 0
    #[test]
    fn distance_euclidean_identical_vectors() {
        let metric = DistanceCalculationMetric::Euclidean;
        let a = array![1.0_f64, 2.0];
        assert_abs_diff_eq!(metric.distance(a.view(), a.view()), 0.0, epsilon = 1e-6);
    }

    // Manhattan is symmetric
    #[test]
    fn distance_manhattan_symmetry() {
        let metric = DistanceCalculationMetric::Manhattan;
        let a = array![1.0_f64, 5.0];
        let b = array![3.0_f64, 2.0];
        assert_abs_diff_eq!(
            metric.distance(a.view(), b.view()),
            metric.distance(b.view(), a.view()),
            epsilon = 1e-10
        );
    }
}
