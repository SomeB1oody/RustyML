//! Configuration enums shared across the machine-learning estimators
//!
//! - [`RegularizationType`] is used by the linear models and the linear SVM
//! - [`Gamma`] and [`KernelType`] are used by both
//!   [`SVC`](crate::machine_learning::svm::svc::SVC) and
//!   [`KernelPCA`](crate::machine_learning::decomposition::kernel_pca::KernelPCA)
//!
//! Each kernel carries its own evaluation logic (`compute` / `compute_matrix`) as inherent
//! methods, so consumers share a single implementation instead of re-matching the enum in
//! every model

use crate::math::matmul::gemm_par_auto;
use crate::math::squared_euclidean_distance_row;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, exp_map_f64_parallel_threshold};
use crate::{Deserialize, Serialize};
use ndarray::{Array2, ArrayBase, ArrayView1, Axis, Data, Ix2, Zip};

/// Regularization techniques used in machine learning models
///
/// Regularization helps prevent overfitting by adding a penalty term to the model's
/// loss function during training. The `f64` in each variant is the penalty coefficient
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum RegularizationType {
    /// L1 regularization (Lasso): adds the sum of absolute parameter values times the
    /// coefficient. Promotes sparse solutions by driving some parameters to exactly zero
    L1(f64),
    /// L2 regularization (Ridge): adds the sum of squared parameter values times the
    /// coefficient. Discourages large parameter values but typically does not produce
    /// sparse solutions
    L2(f64),
}

/// Kernel coefficient `gamma`, either an explicit value or a data-dependent rule
///
/// The data-dependent rules are: `Scale` is `1 / (n_features * X.var())` and
/// `Auto` is `1 / n_features`. They are resolved to a concrete value at fit time (when the
/// training data is known) via [`Gamma::resolve`]; kernel evaluation always operates on a
/// resolved [`Gamma::Value`]
#[derive(Debug, Copy, Clone, PartialEq, Deserialize, Serialize)]
pub enum Gamma {
    /// `'scale'`: `1 / (n_features * X.var())`
    Scale,
    /// `'auto'`: `1 / n_features`
    Auto,
    /// An explicit coefficient
    Value(f64),
}

impl Gamma {
    /// Resolves this `gamma` to a concrete coefficient from the training-data statistics
    ///
    /// # Parameters
    ///
    /// - `n_features` - Number of features (columns) in the training data
    /// - `x_variance` - Population variance of all entries of the training matrix (used by `Scale`)
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`](crate::error::Error::InvalidInput) - If `Scale` is requested but
    ///   the data variance is zero (constant features), or `n_features` is zero
    pub fn resolve(self, n_features: usize, x_variance: f64) -> Result<f64, crate::error::Error> {
        if n_features == 0 {
            return Err(crate::error::Error::invalid_input(
                "cannot resolve gamma: data has zero features",
            ));
        }
        let value = match self {
            Gamma::Scale => {
                if x_variance <= 0.0 || !x_variance.is_finite() {
                    return Err(crate::error::Error::invalid_input(
                        "cannot use Gamma::Scale: training data has zero (or non-finite) variance",
                    ));
                }
                1.0 / (n_features as f64 * x_variance)
            }
            Gamma::Auto => 1.0 / n_features as f64,
            Gamma::Value(v) => v,
        };
        Ok(value)
    }

    /// Returns the concrete coefficient of an already-resolved `gamma`
    ///
    /// # Panics
    ///
    /// Panics on [`Gamma::Scale`] / [`Gamma::Auto`], which must be resolved via
    /// [`resolve`](Self::resolve) (e.g. at fit time) before any kernel is evaluated
    fn value(self) -> f64 {
        match self {
            Gamma::Value(v) => v,
            Gamma::Scale | Gamma::Auto => {
                panic!("kernel gamma must be resolved (Gamma::resolve) before kernel evaluation")
            }
        }
    }

    /// Whether an explicit value is finite, or the gamma is an (always-valid) `Scale`/`Auto` rule
    pub(crate) fn explicit_is_finite(self) -> bool {
        match self {
            Gamma::Value(v) => v.is_finite(),
            Gamma::Scale | Gamma::Auto => true,
        }
    }

    /// Whether an explicit value is finite and positive, or an (always-valid) `Scale`/`Auto` rule
    pub(crate) fn explicit_is_positive(self) -> bool {
        match self {
            Gamma::Value(v) => v.is_finite() && v > 0.0,
            Gamma::Scale | Gamma::Auto => true,
        }
    }
}

/// Kernel function types for Support Vector Machine and Kernel PCA
#[derive(Debug, Copy, Clone, PartialEq, Deserialize, Serialize)]
pub enum KernelType {
    /// Linear kernel: `K(x, y) = x*y`
    Linear,
    /// Polynomial kernel: `K(x, y) = (gamma*x*y + coef0)^degree`
    Poly {
        /// Polynomial degree
        degree: u32,
        /// Kernel coefficient (explicit, or `Scale`/`Auto`)
        gamma: Gamma,
        /// Independent term
        coef0: f64,
    },
    /// Radial Basis Function kernel: `K(x, y) = exp(-gamma*|x-y|^2)`
    RBF {
        /// Kernel coefficient (explicit, or `Scale`/`Auto`)
        gamma: Gamma,
    },
    /// Sigmoid kernel: `K(x, y) = tanh(gamma*x*y + coef0)`
    Sigmoid {
        /// Kernel coefficient (explicit, or `Scale`/`Auto`)
        gamma: Gamma,
        /// Independent term
        coef0: f64,
    },
    /// Cosine kernel: `K(x, y) = (x*y) / (||x|| * ||y||)`
    Cosine,
}

impl KernelType {
    /// Returns a copy of this kernel with any `Scale`/`Auto` gamma resolved to a concrete value
    ///
    /// Estimators call this once at fit time (when the training data is known) and store the
    /// resolved kernel, so that both the training Gram matrix and later predictions use the same
    /// fixed coefficient. Kernels without a gamma (`Linear`, `Cosine`) are returned unchanged
    ///
    /// # Parameters
    ///
    /// - `n_features` - Number of features in the training data
    /// - `x_variance` - Population variance of all training-matrix entries (used by `Gamma::Scale`)
    ///
    /// # Errors
    ///
    /// - Propagates [`Gamma::resolve`] errors (zero variance with `Scale`, or zero features)
    pub fn resolve_gamma(
        &self,
        n_features: usize,
        x_variance: f64,
    ) -> Result<KernelType, crate::error::Error> {
        Ok(match *self {
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => KernelType::Poly {
                degree,
                gamma: Gamma::Value(gamma.resolve(n_features, x_variance)?),
                coef0,
            },
            KernelType::RBF { gamma } => KernelType::RBF {
                gamma: Gamma::Value(gamma.resolve(n_features, x_variance)?),
            },
            KernelType::Sigmoid { gamma, coef0 } => KernelType::Sigmoid {
                gamma: Gamma::Value(gamma.resolve(n_features, x_variance)?),
                coef0,
            },
            other => other,
        })
    }
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
    #[inline]
    pub fn compute(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        match *self {
            KernelType::Linear => x1.dot(&x2),
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => (gamma.value() * x1.dot(&x2) + coef0).powi(degree as i32),
            KernelType::RBF { gamma } => {
                // ||x1 - x2||^2 via the shared single-pass, allocation-free helper
                let squared_norm = squared_euclidean_distance_row(&x1, &x2);
                (-gamma.value() * squared_norm).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => (gamma.value() * x1.dot(&x2) + coef0).tanh(),
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
    /// multiply via `gemm_par_auto`) plus a cheap elementwise transform over the
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
    pub fn compute_matrix<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix2>,
    ) -> Array2<f64>
    where
        S1: Data<Elem = f64> + Sync,
        S2: Data<Elem = f64> + Sync,
    {
        let mut k = gemm_par_auto(x, &y.t());
        let elems = k.len();

        match *self {
            KernelType::Linear => {}
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => {
                let degree = degree as i32;
                let gamma = gamma.value();
                let f = |v: f64| (gamma * v + coef0).powi(degree);
                if elems >= cheap_map_f64_parallel_threshold() {
                    k.par_mapv_inplace(f);
                } else {
                    k.mapv_inplace(f);
                }
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let gamma = gamma.value();
                let f = |v: f64| (gamma * v + coef0).tanh();
                if elems >= exp_map_f64_parallel_threshold() {
                    k.par_mapv_inplace(f);
                } else {
                    k.mapv_inplace(f);
                }
            }
            KernelType::RBF { gamma } => {
                let gamma = gamma.value();
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
                if elems >= exp_map_f64_parallel_threshold() {
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
                if elems >= cheap_map_f64_parallel_threshold() {
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

    // Gamma::resolve (data-dependent gamma)

    /// `Gamma::Scale` resolves to 1 / (n_features * X.var())
    #[test]
    fn gamma_scale_resolves_to_inverse_features_times_variance() {
        let n_features = 4;
        let x_var = 2.5;
        let g = Gamma::Scale.resolve(n_features, x_var).unwrap();
        assert_abs_diff_eq!(g, 1.0 / (4.0 * 2.5), epsilon = 1e-12);
    }

    /// `Gamma::Auto` resolves to 1 / n_features
    #[test]
    fn gamma_auto_resolves_to_inverse_features() {
        let g = Gamma::Auto.resolve(5, 999.0).unwrap();
        assert_abs_diff_eq!(g, 1.0 / 5.0, epsilon = 1e-12);
    }

    /// `Gamma::Value` resolves to itself, regardless of the data statistics
    #[test]
    fn gamma_value_resolves_to_itself() {
        let g = Gamma::Value(0.73).resolve(3, 10.0).unwrap();
        assert_abs_diff_eq!(g, 0.73, epsilon = 1e-12);
    }

    /// `Gamma::Scale` errors when the data variance is zero (constant features)
    #[test]
    fn gamma_scale_errors_on_zero_variance() {
        assert!(Gamma::Scale.resolve(3, 0.0).is_err());
        // Auto and Value do not depend on variance and remain valid
        assert!(Gamma::Auto.resolve(3, 0.0).is_ok());
        assert!(Gamma::Value(1.0).resolve(3, 0.0).is_ok());
    }

    /// `resolve_gamma` replaces a kernel's Scale/Auto with the concrete value, leaving
    /// gamma-free kernels (Linear/Cosine) untouched
    #[test]
    fn kernel_resolve_gamma_produces_value_variant() {
        let resolved = KernelType::RBF { gamma: Gamma::Auto }
            .resolve_gamma(8, 1.0)
            .unwrap();
        assert_eq!(
            resolved,
            KernelType::RBF {
                gamma: Gamma::Value(1.0 / 8.0)
            }
        );
        // Linear has no gamma and is returned unchanged
        assert_eq!(
            KernelType::Linear.resolve_gamma(8, 1.0).unwrap(),
            KernelType::Linear
        );
    }

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
        let k = KernelType::RBF {
            gamma: Gamma::Value(1.0),
        };
        let x = array![1.0_f64, 0.0];
        assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-6);
    }

    // diff=[1,-1], ||diff||^2=2, exp(-1*2)=exp(-2)~=0.135335283
    #[test]
    fn kernel_rbf_orthogonal_unit_vectors() {
        let k = KernelType::RBF {
            gamma: Gamma::Value(1.0),
        };
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
            gamma: Gamma::Value(1.0),
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
            gamma: Gamma::Value(2.0),
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
            gamma: Gamma::Value(1.0),
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
                gamma: Gamma::Value(0.5),
                coef0: 1.0,
            },
            KernelType::RBF {
                gamma: Gamma::Value(0.7),
            },
            KernelType::Sigmoid {
                gamma: Gamma::Value(0.3),
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
    #[test]
    fn compute_matrix_rbf_diagonal_is_one() {
        use ndarray::Array2;
        let x = Array2::from_shape_fn((6, 4), |(i, j)| ((i * 4 + j) as f64).sin());
        let gram = KernelType::RBF {
            gamma: Gamma::Value(1.3),
        }
        .compute_matrix(&x, &x);
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(gram[[i, i]], 1.0, epsilon = 1e-12);
        }
    }

    // Cosine guard: a zero row yields a full row/column of zeros, matching `compute`
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
}
