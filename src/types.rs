//! Shared configuration types used across estimators and utilities
//!
//! These enums live in one place because they are shared by more than one module:
//! - [`RegularizationType`](crate::types::RegularizationType) and
//!   [`DistanceCalculationMetric`](crate::types::DistanceCalculationMetric) are used by the
//!   machine learning models
//! - [`KernelType`](crate::types::KernelType) is used by both
//!   [`SVC`](crate::machine_learning::svm::svc::SVC) and
//!   [`KernelPCA`](crate::utils::kernel_pca::KernelPCA)
//!
//! Each type carries its own behavior (distance / kernel evaluation) as inherent
//! methods, so consumers share a single implementation instead of re-matching the
//! enum in every model

// Serde derives are only needed where models serialize (machine_learning / utils)
#[cfg(any(feature = "machine_learning", feature = "utils"))]
use crate::{Deserialize, Serialize};
// The batched kernel-matrix path is only compiled for its callers: SVC and KernelPCA
#[cfg(any(feature = "machine_learning", feature = "utils"))]
use crate::math::matmul::gemm_internal;
#[cfg(any(feature = "machine_learning", feature = "utils"))]
use crate::parallel_gates::{CHEAP_MAP_F64_PARALLEL_THRESHOLD, EXP_MAP_F64_PARALLEL_THRESHOLD};
use ndarray::ArrayView1;
#[cfg(any(feature = "machine_learning", feature = "utils"))]
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, Zip};

/// Distance calculation methods used across machine learning algorithms
///
/// Defines common distance metrics for clustering algorithms, nearest neighbor
/// searches, and other applications where distance between points is relevant
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(
    any(feature = "machine_learning", feature = "utils"),
    derive(Deserialize, Serialize)
)]
pub enum DistanceCalculationMetric {
    /// Euclidean distance (L2 norm): the square root of the sum of squared
    /// differences between corresponding coordinates
    #[default]
    Euclidean,
    /// Manhattan distance (L1 norm): the sum of absolute differences between
    /// corresponding coordinates
    Manhattan,
    /// Generalized metric with Euclidean and Manhattan as special cases; the
    /// `f64` is the order parameter `p`
    Minkowski(f64),
}

impl DistanceCalculationMetric {
    /// Computes the distance between two vectors under this metric
    ///
    /// Single source of truth for metric dispatch; models such as KNN and DBSCAN
    /// call it instead of re-implementing the `match` over variants
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

    /// Returns whether `distance(a, b) <= threshold` under this metric
    pub fn within(&self, a: ArrayView1<f64>, b: ArrayView1<f64>, threshold: f64) -> bool {
        self.comparable_distance(a, b) <= self.comparable_scalar(threshold)
    }

    /// Maps a non-negative scalar (a true distance or a per-axis coordinate gap) into this
    /// metric's order-preserving "comparable" space, where the final root is skipped:
    /// `Euclidean -> t^2`, `Manhattan -> t`, `Minkowski(p) -> t^p`
    ///
    /// Used by spatial indexes so radius thresholds and per-axis pruning bounds can be
    /// compared against [`comparable_distance`](Self::comparable_distance) without repeated
    /// roots. The mapping is monotonic on `t >= 0`, so all ordering decisions are preserved
    pub(crate) fn comparable_scalar(&self, t: f64) -> f64 {
        match *self {
            DistanceCalculationMetric::Euclidean => t * t,
            DistanceCalculationMetric::Manhattan => t,
            DistanceCalculationMetric::Minkowski(p) => t.powf(p),
        }
    }

    /// Distance between two vectors in this metric's comparable space (see
    /// [`comparable_scalar`](Self::comparable_scalar)): the monotonic, root-free form of
    /// [`distance`](Self::distance). Equals `distance(a, b)` raised to the metric's power
    /// (squared for Euclidean, `^p` for Minkowski, unchanged for Manhattan)
    pub(crate) fn comparable_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        use crate::math::{manhattan_distance_row, squared_euclidean_distance_row};
        match *self {
            DistanceCalculationMetric::Euclidean => squared_euclidean_distance_row(&a, &b),
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(&a, &b),
            DistanceCalculationMetric::Minkowski(p) => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs().powf(p))
                .sum(),
        }
    }

    /// Converts a comparable-space distance back to a true distance (inverse of
    /// [`comparable_distance`](Self::comparable_distance)): `Euclidean -> sqrt`,
    /// `Manhattan -> identity`, `Minkowski(p) -> ^(1/p)`
    #[cfg(feature = "machine_learning")]
    pub(crate) fn distance_from_comparable(&self, c: f64) -> f64 {
        match *self {
            DistanceCalculationMetric::Euclidean => c.sqrt(),
            DistanceCalculationMetric::Manhattan => c,
            DistanceCalculationMetric::Minkowski(p) => c.powf(1.0 / p),
        }
    }
}

/// Regularization techniques used in machine learning models
///
/// Regularization helps prevent overfitting by adding a penalty term to the model's
/// loss function during training. The `f64` in each variant is the penalty coefficient
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    any(feature = "machine_learning", feature = "utils"),
    derive(Deserialize, Serialize)
)]
pub enum RegularizationType {
    /// L1 regularization (Lasso): adds the sum of absolute parameter values times the
    /// coefficient. Promotes sparse solutions by driving some parameters to exactly zero
    L1(f64),
    /// L2 regularization (Ridge): adds the sum of squared parameter values times the
    /// coefficient. Discourages large parameter values but typically does not produce
    /// sparse solutions
    L2(f64),
}

/// Kernel function types for Support Vector Machine and Kernel PCA
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(
    any(feature = "machine_learning", feature = "utils"),
    derive(Deserialize, Serialize)
)]
pub enum KernelType {
    /// Linear kernel: `K(x, y) = x*y`
    Linear,
    /// Polynomial kernel: `K(x, y) = (gamma*x*y + coef0)^degree`
    Poly { degree: u32, gamma: f64, coef0: f64 },
    /// Radial Basis Function kernel: `K(x, y) = exp(-gamma*|x-y|^2)`
    RBF { gamma: f64 },
    /// Sigmoid kernel: `K(x, y) = tanh(gamma*x*y + coef0)`
    Sigmoid { gamma: f64, coef0: f64 },
    /// Cosine kernel: `K(x, y) = (x*y) / (||x|| * ||y||)`
    Cosine,
}

impl KernelType {
    /// Computes the kernel function value between two vectors
    ///
    /// Single source of truth for kernel dispatch, shared by SVC and Kernel PCA
    /// instead of each re-implementing the `match` over variants
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
            KernelType::Linear => x1.dot(&x2),
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => (gamma * x1.dot(&x2) + coef0).powi(degree as i32),
            KernelType::RBF { gamma } => {
                let diff = &x1 - &x2;
                let squared_norm = diff.dot(&diff);
                (-gamma * squared_norm).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => (gamma * x1.dot(&x2) + coef0).tanh(),
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

    /// Computes the full kernel matrix `K[i, j] = K(x_i, y_j)` between two sample
    /// sets in one shot, routing the dominant cost through a single parallel GEMM
    ///
    /// Batched counterpart of [`compute`](Self::compute). Every kernel reduces to the
    /// cross-Gram matrix `G = X*Y^T` (one rayon-parallel, cache-blocked matrix
    /// multiply via `gemm_internal`) plus a cheap elementwise transform over the
    /// `[n, m]` result:
    ///
    /// - `Linear`  - `K = G`
    /// - `Poly`    - `K = (gamma*G + coef0)^degree`
    /// - `Sigmoid` - `K = tanh(gamma*G + coef0)`
    /// - `RBF`     - `K = exp(-gamma*D)`, `D[i,j] = ||x_i||^2 + ||y_j||^2 - 2*G[i,j]`
    /// - `Cosine`  - `K = G / (||x_i||*||y_j||)`
    ///
    /// SVC's Gram matrix and Kernel PCA's (cross-)kernel matrix both call this instead
    /// of looping [`compute`](Self::compute) over every pair, turning an `n*m` swarm of
    /// scalar dot products into one GEMM
    ///
    /// The result is numerically equivalent to filling each entry with
    /// [`compute`](Self::compute) up to floating-point rounding; the `RBF` distance is
    /// clamped at zero to absorb the tiny negatives that the `||x||^2 + ||y||^2 - 2x*y`
    /// identity can produce by cancellation
    ///
    /// # Parameters
    ///
    /// - `x` - Left sample set, shape `[n, d]` (one sample per row)
    /// - `y` - Right sample set, shape `[m, d]` (one sample per row)
    ///
    /// # Returns
    ///
    /// - `Array2<f64>` - The `[n, m]` kernel matrix
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub fn compute_matrix<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix2>,
    ) -> Array2<f64>
    where
        S1: Data<Elem = f64> + Sync,
        S2: Data<Elem = f64> + Sync,
    {
        let mut k = gemm_internal(x, &y.t());
        let elems = k.len();

        match *self {
            KernelType::Linear => {}
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => {
                let degree = degree as i32;
                let f = |v: f64| (gamma * v + coef0).powi(degree);
                if elems >= CHEAP_MAP_F64_PARALLEL_THRESHOLD {
                    k.par_mapv_inplace(f);
                } else {
                    k.mapv_inplace(f);
                }
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let f = |v: f64| (gamma * v + coef0).tanh();
                if elems >= EXP_MAP_F64_PARALLEL_THRESHOLD {
                    k.par_mapv_inplace(f);
                } else {
                    k.mapv_inplace(f);
                }
            }
            KernelType::RBF { gamma } => {
                // ||x-y||^2 = ||x||^2 + ||y||^2 - 2x*y; clamp cancellation negatives before exp
                let x_norm_sq = x.map_axis(Axis(1), |row| row.dot(&row));
                let y_norm_sq = y.map_axis(Axis(1), |row| row.dot(&row));
                let transform_row = |mut k_row: ndarray::ArrayViewMut1<f64>, &x_sq: &f64| {
                    Zip::from(&mut k_row).and(&y_norm_sq).for_each(|v, &y_sq| {
                        let dist = (x_sq + y_sq - 2.0 * *v).max(0.0);
                        *v = (-gamma * dist).exp();
                    });
                };
                let zip = Zip::from(k.rows_mut()).and(&x_norm_sq);
                if elems >= EXP_MAP_F64_PARALLEL_THRESHOLD {
                    zip.par_for_each(transform_row);
                } else {
                    zip.for_each(transform_row);
                }
            }
            KernelType::Cosine => {
                let x_norm_sq = x.map_axis(Axis(1), |row| row.dot(&row));
                let y_norm_sq = y.map_axis(Axis(1), |row| row.dot(&row));
                let transform_row = |mut k_row: ndarray::ArrayViewMut1<f64>, &x_sq: &f64| {
                    Zip::from(&mut k_row).and(&y_norm_sq).for_each(|v, &y_sq| {
                        let norm_product = (x_sq * y_sq).sqrt();
                        *v = if norm_product <= f64::EPSILON {
                            0.0
                        } else {
                            *v / norm_product
                        };
                    });
                };
                let zip = Zip::from(k.rows_mut()).and(&x_norm_sq);
                if elems >= CHEAP_MAP_F64_PARALLEL_THRESHOLD {
                    zip.par_for_each(transform_row);
                } else {
                    zip.for_each(transform_row);
                }
            }
        }

        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // KernelType::compute

    // Linear kernel: K(x,y) = x*y
    // [1,0]*[0,1] = 1*0 + 0*1 = 0
    #[test]
    fn kernel_linear_orthogonal_vectors() {
        let k = KernelType::Linear;
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-6);
    }

    // [1,2]*[3,4] = 3 + 8 = 11
    #[test]
    fn kernel_linear_general_vectors() {
        let k = KernelType::Linear;
        let x1 = array![1.0_f64, 2.0];
        let x2 = array![3.0_f64, 4.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 11.0, epsilon = 1e-6);
    }

    // RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
    // Identical vectors -> diff=[0,0], ||diff||^2=0, exp(0)=1
    #[test]
    fn kernel_rbf_identical_vectors() {
        let k = KernelType::RBF { gamma: 1.0 };
        let x = array![1.0_f64, 0.0];
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-6);
    }

    // diff=[1,-1], ||diff||^2=2, exp(-1*2)=exp(-2)~=0.135335283
    #[test]
    fn kernel_rbf_orthogonal_unit_vectors() {
        let k = KernelType::RBF { gamma: 1.0 };
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        let expected = (-2.0_f64).exp(); // ~= 0.13533528323661
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), expected, epsilon = 1e-6);
    }

    // Polynomial kernel: K(x,y) = (gamma*x*y + coef0)^degree
    // degree=2, gamma=1, coef0=0; [1,0]*[0,1]=0 -> (0)^2 = 0
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

    // degree=3, gamma=2, coef0=1; [1,1]*[1,1]=2 -> (2*2+1)^3=5^3=125
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

    // Sigmoid kernel: K(x,y) = tanh(gamma*x*y + coef0)
    // gamma=1, coef0=0; [1,0]*[1,0]=1 -> tanh(1)~=0.76159415595577
    #[test]
    fn kernel_sigmoid_unit_vector() {
        let k = KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        let x = array![1.0_f64, 0.0];
        let expected = 1.0_f64.tanh(); // ~= 0.76159415595577
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), expected, epsilon = 1e-6);
    }

    // Cosine kernel: K(x,y) = (x*y) / (||x|| * ||y||)
    // zero vector -> norm_product=0, epsilon guard -> 0.0
    #[test]
    fn kernel_cosine_zero_vector() {
        let k = KernelType::Cosine;
        let zero = array![0.0_f64, 0.0];
        let other = array![1.0_f64, 2.0];
        assert_abs_diff_eq!(k.compute(zero.view(), other.view()), 0.0, epsilon = 1e-6);
    }

    // identical unit vectors -> cos(0 deg)=1.0
    #[test]
    fn kernel_cosine_identical_vectors() {
        let k = KernelType::Cosine;
        let x = array![1.0_f64, 0.0];
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-6);
    }

    // orthogonal vectors -> dot=0, cos(90 deg)=0.0
    #[test]
    fn kernel_cosine_orthogonal_vectors() {
        let k = KernelType::Cosine;
        let x1 = array![1.0_f64, 0.0];
        let x2 = array![0.0_f64, 1.0];
        assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-6);
    }

    // KernelType::compute_matrix (batched)

    // The batched GEMM path must agree, entry for entry, with looping `compute`
    // over every pair, for every kernel variant, including the asymmetric
    // cross-matrix case `x != y`
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    #[test]
    fn compute_matrix_matches_pairwise() {
        use ndarray::Array2;

        // Deterministic, varied, non-degenerate rows (no all-zero row, so the
        // Cosine guard is not the only thing under test)
        let x = Array2::from_shape_fn((5, 3), |(i, j)| ((i * 3 + j) as f64) * 0.3 - 1.1);
        let y = Array2::from_shape_fn((4, 3), |(i, j)| ((i + 2 * j) as f64) * 0.2 + 0.4);

        let kernels = [
            KernelType::Linear,
            KernelType::Poly {
                degree: 3,
                gamma: 0.5,
                coef0: 1.0,
            },
            KernelType::RBF { gamma: 0.7 },
            KernelType::Sigmoid {
                gamma: 0.3,
                coef0: -0.2,
            },
            KernelType::Cosine,
        ];

        for k in kernels {
            // Symmetric Gram matrix (the SVC / Kernel-PCA-fit case)
            let gram = k.compute_matrix(&x, &x);
            for i in 0..x.nrows() {
                for j in 0..x.nrows() {
                    assert_abs_diff_eq!(
                        gram[[i, j]],
                        k.compute(x.row(i), x.row(j)),
                        epsilon = 1e-9
                    );
                }
            }

            // Cross matrix (the Kernel-PCA-transform case), x != y
            let cross = k.compute_matrix(&x, &y);
            assert_eq!(cross.dim(), (x.nrows(), y.nrows()));
            for i in 0..x.nrows() {
                for j in 0..y.nrows() {
                    assert_abs_diff_eq!(
                        cross[[i, j]],
                        k.compute(x.row(i), y.row(j)),
                        epsilon = 1e-9
                    );
                }
            }
        }
    }

    // RBF diagonal of a Gram matrix must be exactly 1 (||x_i - x_i||^2 clamps to 0)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    #[test]
    fn compute_matrix_rbf_diagonal_is_one() {
        use ndarray::Array2;
        let x = Array2::from_shape_fn((6, 4), |(i, j)| ((i * 4 + j) as f64).sin());
        let gram = KernelType::RBF { gamma: 1.3 }.compute_matrix(&x, &x);
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(gram[[i, i]], 1.0, epsilon = 1e-12);
        }
    }

    // Cosine guard: a zero row yields a full row/column of zeros, matching `compute`
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    #[test]
    fn compute_matrix_cosine_zero_row_guard() {
        use ndarray::Array2;
        let mut x = Array2::from_shape_fn((4, 3), |(i, j)| (i + j) as f64 + 1.0);
        x.row_mut(2).fill(0.0); // force the epsilon guard on row/col 2
        let m = KernelType::Cosine.compute_matrix(&x, &x);
        for j in 0..x.nrows() {
            assert_abs_diff_eq!(m[[2, j]], 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(m[[j, 2]], 0.0, epsilon = 1e-12);
        }
    }

    // DistanceCalculationMetric::distance

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

    // Minkowski(3): (|3|^3 + |4|^3)^(1/3) = (27+64)^(1/3) = 91^(1/3) ~= 4.497941
    #[test]
    fn distance_minkowski_p3() {
        let metric = DistanceCalculationMetric::Minkowski(3.0);
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        let expected = 91.0_f64.powf(1.0 / 3.0); // ~= 4.497941445275415
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

    // Zero distance: identical vectors -> 0
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
