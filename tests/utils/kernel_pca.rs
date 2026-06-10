//! Integration tests for `utils::kernel_pca`.
//!
//! All expected values are derived from mathematical definitions or closed-form
//! calculations, never from running the code and recording its output.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::types::KernelType;
use rustyml::utils::kernel_pca::{EigenSolver, KernelPCA};

use crate::common::assert_allclose;

// ─────────────────────────────────────────────────────────────────────────────
// Local helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Small deterministic dataset for general fitting tests (8 samples × 2 features).
fn make_small_dataset() -> Array2<f64> {
    array![
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [2.0, 0.5],
        [-0.5, 2.0],
        [1.5, -1.5],
        [-2.0, 1.0],
    ]
}

/// Dataset for the RBF vs Linear separation test.
///
/// This generates two non-concentric clusters that are RADIALLY separable
/// in 2D but not linearly separable.  The inner cluster (label -1) is placed
/// near the origin, the outer cluster (label +1) is placed on a ring far from
/// the origin.  Importantly, the outer ring samples are NOT placed at angles
/// that are symmetric with the inner cluster, so the angular eigenvectors do
/// NOT dominate the kernel PCA spectrum.
///
/// Mathematical derivation for why RBF separates this better than Linear:
///
/// For the RBF kernel K(x,y) = exp(-γ||x-y||²):
///   - Within inner cluster: small ||x-y||² → K close to 1
///   - Within outer ring:    ||x-y||² depends on ring arc → K varies but has high average
///   - Between inner/outer:  ||x-y||² = (r_in - r_out)² = large → K ≈ exp(-γ * 9) ≈ 0.11
///
/// For the Linear kernel K(x,y) = x·y:
///   - Inner cluster (near origin): small dot products ≈ 0
///   - Outer ring: large dot products ∝ r_out²
///   - The cross term (inner · outer) ∝ r_in * r_out * cos(θ) averages near 0
///
/// Under RBF KPCA, the "cluster identity" is the dominant eigenvector direction.
/// Under Linear KPCA, the angular variation in the outer ring dominates.
///
/// Separability is measured by the MAX centroid distance across all components.
fn make_radial_clusters(n_inner: usize, n_outer: usize) -> (Array2<f64>, Vec<i32>) {
    use std::f64::consts::PI;
    // Inner cluster: points near (0.5, 0.5) with small spread.
    // These are at radius ≈ 0.71, all in the first quadrant.
    let mut rows: Vec<f64> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();

    // Inner cluster: fixed small-radius points in a grid-like pattern
    for i in 0..n_inner {
        let angle = 2.0 * PI * (i as f64) / (n_inner as f64);
        rows.push(0.5 * angle.cos()); // inner, radius 0.5
        rows.push(0.5 * angle.sin());
        labels.push(-1);
    }
    // Outer ring: points at radius 3.0
    for i in 0..n_outer {
        let angle = 2.0 * PI * (i as f64) / (n_outer as f64);
        rows.push(3.0 * angle.cos()); // outer, radius 3
        rows.push(3.0 * angle.sin());
        labels.push(1);
    }

    let n = labels.len();
    (Array2::from_shape_vec((n, 2), rows).unwrap(), labels)
}

/// Computes the between-class to within-class variance ratio across all
/// projection components.
///
/// For each component, we compute:
///   between_var = (mean_pos - mean_neg)^2
///   within_var  = var(proj[pos]) + var(proj[neg]) + epsilon
///
/// The sum over components is a Fisher-like discriminability measure.
/// Higher means the classes are more linearly separable in the projected space.
fn class_separability(projections: &Array2<f64>, labels: &[i32]) -> f64 {
    let n_comp = projections.ncols();
    let mut total = 0.0;
    for c in 0..n_comp {
        let col = projections.column(c);
        let mut sum_neg = 0.0;
        let mut sum_pos = 0.0;
        let mut n_neg = 0usize;
        let mut n_pos = 0usize;
        for (i, &lbl) in labels.iter().enumerate() {
            if lbl < 0 {
                sum_neg += col[i];
                n_neg += 1;
            } else {
                sum_pos += col[i];
                n_pos += 1;
            }
        }
        let mean_neg = sum_neg / n_neg as f64;
        let mean_pos = sum_pos / n_pos as f64;
        let between_var = (mean_pos - mean_neg).powi(2);

        let var_neg = col
            .iter()
            .zip(labels.iter())
            .filter(|&(_, l)| *l < 0)
            .map(|(&v, _)| (v - mean_neg).powi(2))
            .sum::<f64>()
            / n_neg as f64;
        let var_pos = col
            .iter()
            .zip(labels.iter())
            .filter(|&(_, l)| *l > 0)
            .map(|(&v, _)| (v - mean_pos).powi(2))
            .sum::<f64>()
            / n_pos as f64;
        total += between_var / (var_neg + var_pos + 1e-12);
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Constructor + getter tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_new_default_values() {
    let kpca = KernelPCA::default();
    // Default is RBF gamma=0.1, n_components=2, Dense solver.
    assert_eq!(kpca.get_n_components(), 2);
    assert_eq!(kpca.get_eigen_solver(), EigenSolver::Dense);
    // Pre-fit getters must return None.
    assert!(kpca.get_eigenvalues().is_none());
    assert!(kpca.get_eigenvectors().is_none());
    assert!(kpca.get_n_samples().is_none());
    assert!(kpca.get_n_features().is_none());
}

#[test]
fn test_new_linear_kernel() {
    let kpca = KernelPCA::new(KernelType::Linear, 3, EigenSolver::Dense).unwrap();
    assert_eq!(kpca.get_n_components(), 3);
    assert_eq!(kpca.get_eigen_solver(), EigenSolver::Dense);
}

#[test]
fn test_new_rbf_kernel() {
    let kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Lanczos).unwrap();
    assert_eq!(kpca.get_n_components(), 2);
    assert_eq!(kpca.get_eigen_solver(), EigenSolver::Lanczos);
}

#[test]
fn test_new_poly_kernel() {
    let kpca = KernelPCA::new(
        KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 0.0,
        },
        2,
        EigenSolver::Dense,
    )
    .unwrap();
    assert_eq!(kpca.get_n_components(), 2);
}

#[test]
fn test_new_sigmoid_kernel() {
    let kpca = KernelPCA::new(
        KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap();
    assert_eq!(kpca.get_n_components(), 1);
}

#[test]
fn test_new_cosine_kernel() {
    let kpca = KernelPCA::new(KernelType::Cosine, 2, EigenSolver::Dense).unwrap();
    assert_eq!(kpca.get_n_components(), 2);
}

/// Sigmoid with gamma=0 is intentionally allowed (validate_kernel only rejects
/// non-finite values for Sigmoid, not zero).
#[test]
fn test_new_sigmoid_gamma_zero_accepted() {
    assert!(
        KernelPCA::new(
            KernelType::Sigmoid {
                gamma: 0.0,
                coef0: 0.0
            },
            1,
            EigenSolver::Dense
        )
        .is_ok(),
        "Sigmoid with gamma=0 should be accepted"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Constructor validation / error paths
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_new_n_components_zero_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::Linear, 0, EigenSolver::Dense).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn test_new_rbf_gamma_zero_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::RBF { gamma: 0.0 }, 1, EigenSolver::Dense).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_rbf_gamma_negative_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::RBF { gamma: -1.0 }, 1, EigenSolver::Dense).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_rbf_gamma_nan_returns_invalid_parameter() {
    let err =
        KernelPCA::new(KernelType::RBF { gamma: f64::NAN }, 1, EigenSolver::Dense).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_rbf_gamma_infinity_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::RBF {
            gamma: f64::INFINITY,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_poly_degree_zero_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Poly {
            degree: 0,
            gamma: 1.0,
            coef0: 0.0,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_poly_gamma_zero_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Poly {
            degree: 2,
            gamma: 0.0,
            coef0: 0.0,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_poly_gamma_negative_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Poly {
            degree: 2,
            gamma: -0.5,
            coef0: 0.0,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_poly_coef0_nan_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: f64::NAN,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_poly_gamma_inf_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Poly {
            degree: 2,
            gamma: f64::INFINITY,
            coef0: 0.0,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_sigmoid_gamma_nan_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Sigmoid {
            gamma: f64::NAN,
            coef0: 0.0,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_sigmoid_coef0_infinity_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::Sigmoid {
            gamma: 1.0,
            coef0: f64::INFINITY,
        },
        1,
        EigenSolver::Dense,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. fit error paths
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_fit_empty_input_returns_empty_input() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let x: Array2<f64> = Array2::zeros((0, 2));
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

#[test]
fn test_fit_one_sample_returns_invalid_input() {
    // KernelPCA requires at least 2 samples.
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let x = array![[1.0, 2.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn test_fit_n_components_greater_than_n_samples_returns_invalid_parameter() {
    // 3 samples, but n_components=5 — should fail during fit, not construction.
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 5, EigenSolver::Dense).unwrap();
    let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
    // Model must not be partially mutated: fitted state still None.
    assert!(
        kpca.get_n_samples().is_none(),
        "n_samples should still be None after failed fit"
    );
}

#[test]
fn test_fit_nan_in_input_returns_non_finite() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let x = array![[1.0, f64::NAN], [0.0, 1.0], [-1.0, 0.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

#[test]
fn test_fit_inf_in_input_returns_non_finite() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let x = array![[f64::INFINITY, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. transform error paths
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_transform_before_fit_returns_not_fitted() {
    let kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let x = array![[1.0, 0.0], [0.0, 1.0]];
    let err = kpca.transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {err:?}"
    );
}

#[test]
fn test_transform_wrong_feature_count_returns_dimension_mismatch() {
    let x_train = make_small_dataset(); // 8 × 2
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    kpca.fit(&x_train).unwrap();

    // Wrong: 3 features instead of 2.
    let x_bad = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let err = kpca.transform(&x_bad).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

#[test]
fn test_transform_nan_in_input_returns_error() {
    let x_train = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    kpca.fit(&x_train).unwrap();

    let x_bad = array![[f64::NAN, 0.0], [0.0, 1.0]];
    let err = kpca.transform(&x_bad).unwrap_err();
    // validate_transform_matrix catches NonFinite; kernel validation catches NonFinite too.
    assert!(
        matches!(err, Error::NonFinite(_) | Error::DimensionMismatch { .. }),
        "expected NonFinite (or DimensionMismatch), got {err:?}"
    );
}

#[test]
fn test_transform_empty_input_returns_empty_input() {
    let x_train = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    kpca.fit(&x_train).unwrap();

    let x_empty: Array2<f64> = Array2::zeros((0, 2));
    let err = kpca.transform(&x_empty).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Happy-path fit + transform with each kernel
// ─────────────────────────────────────────────────────────────────────────────

fn run_fit_transform_shape_check(kernel: KernelType, n_components: usize) {
    let x = make_small_dataset(); // 8 × 2
    let mut kpca = KernelPCA::new(kernel, n_components, EigenSolver::Dense).unwrap();
    kpca.fit(&x).unwrap();

    // Post-fit getters must be populated.
    assert_eq!(kpca.get_n_samples(), Some(8));
    assert_eq!(kpca.get_n_features(), Some(2));
    assert!(kpca.get_eigenvalues().is_some());
    assert!(kpca.get_eigenvectors().is_some());

    // Eigenvalues must all be strictly positive (KernelPCA invariant).
    let evs = kpca.get_eigenvalues().unwrap();
    for &v in evs.iter() {
        assert!(
            v > 0.0 && v.is_finite(),
            "eigenvalue {v} must be strictly positive and finite"
        );
    }

    let projected = kpca.transform(&x).unwrap();
    assert_eq!(projected.shape(), [8, n_components]);
    for &val in projected.iter() {
        assert!(val.is_finite(), "projected value must be finite");
    }
}

#[test]
fn test_fit_transform_linear_kernel() {
    run_fit_transform_shape_check(KernelType::Linear, 2);
}

#[test]
fn test_fit_transform_rbf_kernel() {
    run_fit_transform_shape_check(KernelType::RBF { gamma: 0.5 }, 2);
}

#[test]
fn test_fit_transform_poly_kernel() {
    run_fit_transform_shape_check(
        KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 1.0,
        },
        2,
    );
}

#[test]
fn test_fit_transform_sigmoid_kernel() {
    // Sigmoid kernel: K(x,y) = tanh(gamma * x·y + coef0).
    // Small gamma keeps values in a regime where centering remains well-conditioned.
    run_fit_transform_shape_check(
        KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        },
        2,
    );
}

#[test]
fn test_fit_transform_cosine_kernel() {
    run_fit_transform_shape_check(KernelType::Cosine, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. fit_transform(x) == fit(x); transform(x) — determinism invariant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_fit_transform_equals_fit_then_transform() {
    let x = make_small_dataset();
    let kernel = KernelType::RBF { gamma: 0.5 };

    // Path A: fit_transform in one call.
    let mut kpca_a = KernelPCA::new(kernel, 2, EigenSolver::Dense).unwrap();
    let proj_a = kpca_a.fit_transform(&x).unwrap();

    // Path B: fit then transform separately.
    let mut kpca_b = KernelPCA::new(kernel, 2, EigenSolver::Dense).unwrap();
    kpca_b.fit(&x).unwrap();
    let proj_b = kpca_b.transform(&x).unwrap();

    // Both must be numerically identical (same deterministic algorithm, same seed).
    assert_allclose(&proj_a, &proj_b, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Centering invariant: column means of training projection ≈ 0
// ─────────────────────────────────────────────────────────────────────────────
//
// Proof: after double-centering the kernel matrix K_c = H K H (where
// H = I - 1/n * 11^T), the row-sum of K_c is zero, so the column mean of
// K_c V / sqrt(lambda) is zero.

#[test]
fn test_centering_training_output_has_near_zero_column_means() {
    let x = make_small_dataset(); // 8 × 2
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    let proj = kpca.fit_transform(&x).unwrap();

    // n_samples = 8, check both components.
    for col in 0..proj.ncols() {
        let mean: f64 = proj.column(col).sum() / proj.nrows() as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Eigenvalues are non-negative
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_eigenvalues_are_positive_after_fit() {
    let x = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 4, EigenSolver::Dense).unwrap();
    kpca.fit(&x).unwrap();

    let evs = kpca.get_eigenvalues().unwrap();
    assert_eq!(evs.len(), 4);
    // Eigenvalues must be sorted descending and strictly positive.
    let mut prev = f64::INFINITY;
    for &v in evs.iter() {
        assert!(
            v > 0.0 && v.is_finite(),
            "eigenvalue {v} must be strictly positive"
        );
        assert!(
            v <= prev + 1e-12,
            "eigenvalues should be non-increasing; found {v} after {prev}"
        );
        prev = v;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. EigenSolver variants agree
// ─────────────────────────────────────────────────────────────────────────────
//
// Dense, Lanczos, and PowerIteration should all produce the same dominant
// eigenvalue and projections (up to a global sign flip on each column).

fn abs_col_norms(m: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter((0..m.ncols()).map(|j| m.column(j).mapv(|v| v * v).sum().sqrt()))
}

#[test]
fn test_eigensolver_dense_vs_lanczos_agree() {
    let x = make_small_dataset();
    let n_comp = 2;

    let mut kpca_dense =
        KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp, EigenSolver::Dense).unwrap();
    kpca_dense.fit(&x).unwrap();
    let proj_dense = kpca_dense.transform(&x).unwrap();

    let mut kpca_lanczos =
        KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp, EigenSolver::Lanczos).unwrap();
    kpca_lanczos.fit(&x).unwrap();
    let proj_lanczos = kpca_lanczos.transform(&x).unwrap();

    // Column norms must match (sign-agnostic check).
    let norms_d = abs_col_norms(&proj_dense);
    let norms_l = abs_col_norms(&proj_lanczos);
    assert_allclose(&norms_d, &norms_l, 1e-5);

    // Leading eigenvalues must agree.
    let ev_dense = kpca_dense.get_eigenvalues().unwrap();
    let ev_lanczos = kpca_lanczos.get_eigenvalues().unwrap();
    assert_abs_diff_eq!(ev_dense[0], ev_lanczos[0], epsilon = 1e-5);
}

#[test]
fn test_eigensolver_dense_vs_power_iteration_agree() {
    let x = make_small_dataset();
    let n_comp = 2;

    let mut kpca_dense =
        KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp, EigenSolver::Dense).unwrap();
    kpca_dense.fit(&x).unwrap();
    let proj_dense = kpca_dense.transform(&x).unwrap();

    let mut kpca_pi = KernelPCA::new(
        KernelType::RBF { gamma: 0.5 },
        n_comp,
        EigenSolver::PowerIteration,
    )
    .unwrap();
    kpca_pi.fit(&x).unwrap();
    let proj_pi = kpca_pi.transform(&x).unwrap();

    let norms_d = abs_col_norms(&proj_dense);
    let norms_p = abs_col_norms(&proj_pi);
    assert_allclose(&norms_d, &norms_p, 1e-4);

    let ev_dense = kpca_dense.get_eigenvalues().unwrap();
    let ev_pi = kpca_pi.get_eigenvalues().unwrap();
    assert_abs_diff_eq!(ev_dense[0], ev_pi[0], epsilon = 1e-4);
}

#[test]
fn test_all_three_solvers_produce_finite_shapes() {
    for solver in [
        EigenSolver::Dense,
        EigenSolver::Lanczos,
        EigenSolver::PowerIteration,
    ] {
        let x = make_small_dataset();
        let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, solver).unwrap();
        let proj = kpca.fit_transform(&x).unwrap();
        assert_eq!(proj.shape(), [8, 2], "shape mismatch for solver {solver:?}");
        for &v in proj.iter() {
            assert!(v.is_finite(), "non-finite value for solver {solver:?}");
        }
        assert!(kpca.get_eigenvalues().is_some());
        assert!(kpca.get_eigenvectors().is_some());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Determinism: calling fit+transform twice gives bit-identical results
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_determinism_dense_solver() {
    let x = make_small_dataset();
    let kernel = KernelType::RBF { gamma: 0.5 };

    let mut kpca1 = KernelPCA::new(kernel, 2, EigenSolver::Dense).unwrap();
    let proj1 = kpca1.fit_transform(&x).unwrap();

    let mut kpca2 = KernelPCA::new(kernel, 2, EigenSolver::Dense).unwrap();
    let proj2 = kpca2.fit_transform(&x).unwrap();

    assert_allclose(&proj1, &proj2, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. RBF separates radial clusters better than Linear kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// A small inner cluster (radius 0.5) and an outer ring (radius 3.0) are NOT
// linearly separable in input space because both pass through the same angular
// directions.
//
// Mathematical reasoning:
//
//   With n_inner = n_outer = 12 and the inner cluster at r=0.5:
//
//   For the RBF kernel (γ = 0.5):
//     K(inner_i, inner_j):  ||x||² ≤ 1.0  → K ≥ exp(-0.5)  ≈ 0.607
//     K(inner_i, outer_j):  ||x-y||² ≥ (3-0.5)² = 6.25 → K ≤ exp(-0.5*6.25) ≈ 0.044
//     K(outer_i, outer_j):  ||x-y||² up to (2*3)²=36 → K in [exp(-18), 1]
//     The within-inner cluster average is much larger than the cross-cluster average,
//     creating a strong "inner vs outer" eigenvector in the centered kernel matrix.
//
//   For the Linear kernel:
//     K(inner_i, inner_j) = x·y: since inner points are at r=0.5, K ≤ 0.25
//     K(outer_i, outer_j) = x·y: outer points at r=3, K ≤ 9
//     K(inner, outer): ≤ 0.5*3 = 1.5 — small compared to outer-outer
//     The outer ring dominates the linear kernel spectrum (large-variance outer points),
//     so the first linear KPCA component separates WITHIN the outer ring (angular),
//     not inner from outer.
//
//   The maximum centroid distance across n_components=2 components is the metric:
//   for RBF, the inner cluster projects uniformly to one location and the outer
//   ring to another, yielding large centroid distance in at least one component.
//   For linear, the outer ring dominates both components, producing small centroid
//   distance (inner points project near zero in both components because r_inner ≈ 0).

#[test]
fn test_rbf_separates_radial_clusters_better_than_linear() {
    // 24 samples: 12 inner (r=0.5, label=-1) + 12 outer (r=3.0, label=1)
    let (x, labels) = make_radial_clusters(12, 12);

    // RBF with γ=0.5: K(inner,outer) = exp(-0.5 * 6.25) ≈ 0.044, much smaller
    // than K(inner,inner) which ranges from exp(-0.5) ≈ 0.61 to 1.0.
    let mut kpca_rbf =
        KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    let proj_rbf = kpca_rbf.fit_transform(&x).unwrap();

    // Linear kernel: outer ring (r=3) has much larger kernel values than inner (r=0.5),
    // so the outer ring angular modes dominate and inner points project near 0.
    let mut kpca_lin = KernelPCA::new(KernelType::Linear, 2, EigenSolver::Dense).unwrap();
    let proj_lin = kpca_lin.fit_transform(&x).unwrap();

    // Metric: max centroid distance across all n_components components.
    // For RBF: the first component should clearly separate inner from outer.
    // For Linear: both class centroids may be near 0 for the inner cluster.
    let sep_rbf = class_separability(&proj_rbf, &labels);
    let sep_lin = class_separability(&proj_lin, &labels);

    assert!(
        sep_rbf > sep_lin + 0.5,
        "RBF separability ({sep_rbf:.4}) should exceed Linear ({sep_lin:.4}) by > 0.5; \
         RBF encodes radial distance which separates inner (r=0.5) from outer (r=3.0) \
         while linear kernel is dominated by outer-ring angular variation"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Save / load round-trip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_save_load_round_trip() {
    let x = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2, EigenSolver::Dense).unwrap();
    kpca.fit(&x).unwrap();
    let proj_before = kpca.transform(&x).unwrap();

    let path = "/tmp/rustyml_test_kpca_round_trip.json";
    kpca.save_to_path(path).unwrap();
    let kpca_loaded = KernelPCA::load_from_path(path).unwrap();
    let proj_after = kpca_loaded.transform(&x).unwrap();

    // After deserialization, identical transform output is required.
    assert_allclose(&proj_before, &proj_after, 1e-12);

    // Clean up.
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_load_from_nonexistent_path_returns_io_error() {
    let err =
        KernelPCA::load_from_path("/tmp/rustyml_this_path_does_not_exist_42.json").unwrap_err();
    assert!(
        matches!(err, Error::Io(_)),
        "expected Io error, got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Single-feature (1-D) data
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_single_feature_data_fits_and_transforms() {
    // Four 1-D points; the kernel matrix is 4×4 and the centering still applies.
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let proj = kpca.fit_transform(&x).unwrap();
    assert_eq!(proj.shape(), [4, 1]);
    for &v in proj.iter() {
        assert!(v.is_finite());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Parallel path (n_samples >= 200)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_path_n_samples_200() {
    // KERNEL_PCA_PARALLEL_THRESHOLD = 200; use exactly 200 rows.
    let n = 200;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let angle = (i as f64) * std::f64::consts::TAU / (n as f64);
        data.push(angle.cos());
        data.push(angle.sin());
    }
    let x = Array2::from_shape_vec((n, 2), data).unwrap();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.05 }, 2, EigenSolver::Dense).unwrap();
    let proj = kpca.fit_transform(&x).unwrap();
    assert_eq!(proj.shape(), [n, 2]);
    for &v in proj.iter() {
        assert!(v.is_finite(), "non-finite value in parallel path output");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Numerical spot-checks: closed-form kernel values
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests verify the kernel formulas independently from KernelPCA, using
// the public `KernelType::compute` method.

#[test]
fn test_rbf_kernel_known_value() {
    // K(x,y) = exp(-gamma * ||x-y||^2)
    // x=[3.0,0.0], y=[0.0,4.0], ||x-y||^2 = 9+16 = 25, gamma=0.1
    // K = exp(-0.1 * 25) = exp(-2.5)
    let x1 = array![3.0, 0.0];
    let x2 = array![0.0, 4.0];
    let k = KernelType::RBF { gamma: 0.1 };
    let expected = (-2.5f64).exp(); // exp(-2.5) ≈ 0.082085
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn test_linear_kernel_known_value() {
    // K(x,y) = x·y = 3*1 + 4*2 = 11
    let x1 = array![3.0, 4.0];
    let x2 = array![1.0, 2.0];
    let k = KernelType::Linear;
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, 11.0, epsilon = 1e-12);
}

#[test]
fn test_cosine_kernel_known_value() {
    // K(x,y) = (x·y) / (||x|| * ||y||)
    // x=[3.0,4.0] ||x||=5, y=[1.0,0.0] ||y||=1
    // dot = 3, K = 3 / (5 * 1) = 0.6
    let x1 = array![3.0, 4.0];
    let x2 = array![1.0, 0.0];
    let k = KernelType::Cosine;
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, 0.6, epsilon = 1e-12);
}

#[test]
fn test_poly_kernel_known_value() {
    // K(x,y) = (gamma * x·y + coef0)^degree
    // x=[1.0,2.0], y=[3.0,4.0], dot=1*3+2*4=11
    // gamma=1, coef0=1, degree=2: (1*11+1)^2 = 144
    let x1 = array![1.0, 2.0];
    let x2 = array![3.0, 4.0];
    let k = KernelType::Poly {
        degree: 2,
        gamma: 1.0,
        coef0: 1.0,
    };
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, 144.0, epsilon = 1e-10);
}

#[test]
fn test_sigmoid_kernel_known_value() {
    // K(x,y) = tanh(gamma * x·y + coef0)
    // x=[1.0,0.0], y=[1.0,0.0], dot=1
    // gamma=1.0, coef0=0.0: tanh(1.0)
    let x1 = array![1.0, 0.0];
    let x2 = array![1.0, 0.0];
    let k = KernelType::Sigmoid {
        gamma: 1.0,
        coef0: 0.0,
    };
    let expected = 1.0f64.tanh(); // ≈ 0.7615941559557649
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. n_components = n_samples (boundary: exactly equal should succeed)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_n_components_equals_n_samples_boundary() {
    // 4 samples; n_components=4 should be accepted (n_components <= n_samples).
    // The kernel matrix is 4×4 and all 4 components are requested.
    let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 4, EigenSolver::Dense).unwrap();
    // This may succeed or fail with Computation (all-eigenvalue edge case), but
    // it must NOT return InvalidParameter (the count constraint is satisfied).
    match kpca.fit(&x) {
        Ok(_) => {
            let proj = kpca.transform(&x).unwrap();
            assert_eq!(proj.shape(), [4, 4]);
        }
        Err(Error::Computation { .. }) => {
            // Acceptable: near-zero eigenvalues for this small symmetric dataset.
        }
        Err(e) => {
            panic!("unexpected error for n_components==n_samples: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. Two-sample minimum boundary (exactly 2 samples is valid)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_two_samples_is_valid() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1, EigenSolver::Dense).unwrap();
    let result = kpca.fit(&x);
    // May succeed or return Computation (rank-1 kernel), but must not panic
    // and must not return InvalidInput (2 >= minimum of 2).
    match result {
        Ok(_) => {}
        Err(Error::Computation { .. }) => {}
        Err(e) => panic!("unexpected error for 2-sample fit: {e:?}"),
    }
}
// ──────────────────────────────────────────────────────────────────────────────
// 18. Non-positive eigenvalue -> Error::Computation (validate_eigenvalues path)
// ─────────────────────────────────────────────────────────────────────────────
//
// KernelPCA requires every selected eigenvalue of the centered kernel matrix to be
// strictly positive and finite (src/utils/kernel_pca.rs, `validate_eigenvalues`).
// To force a rejection DETERMINISTICALLY we use the Sigmoid (tanh) kernel, which is
// NOT positive-semidefinite, so its centered Gram matrix can have a genuinely
// negative eigenvalue (bounded well away from machine epsilon).
//
// Closed-form derivation for n = 2 samples x0 = [1], x1 = [4], Sigmoid(gamma=1, coef0=0):
//   K(x,y) = tanh(gamma * x·y + coef0) = tanh(x·y)
//   K00 = tanh(1)  = 0.7615941559557649
//   K11 = tanh(16) ≈ 1.0   (1.0 to full f64 precision)
//   K01 = tanh(4)  = 0.999329299739067
//
//   Double-centering (K[i,j] - rowmean[i] - rowmean[j] + overallmean) on a 2×2 matrix
//   forces each row sum to 0, so the centered matrix is [[a, -a], [-a, a]] with
//   a = (K00 - 2·K01 + K11)/4. Its eigenvalues are exactly {0, 2a} (trace = 2a, det = 0).
//   Nonzero eigenvalue 2a = (K00 - 2·K01 + K11)/2
//                        = (0.7615941559557649 - 2·0.999329299739067 + 1.0)/2
//                        ≈ -0.118532  (robustly < 0).
//
// With n_components = 2 (= n_samples) the descending-sorted top-2 eigenvalues are ALL
// the eigenvalues, so this negative value is necessarily selected and
// `validate_eigenvalues` MUST reject with Error::Computation.
#[test]
fn test_fit_indefinite_kernel_negative_eigenvalue_returns_computation() {
    let x = array![[1.0], [4.0]]; // 2 samples, 1 feature
    let mut kpca = KernelPCA::new(
        KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        },
        2, // == n_samples, so the negative eigenvalue is necessarily selected
        EigenSolver::Dense,
    )
    .unwrap();

    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::Computation { .. }),
        "expected Computation (negative eigenvalue from indefinite Sigmoid kernel), got {err:?}"
    );

    // The fit must fail before persisting any state: fitted getters stay None.
    assert!(
        kpca.get_eigenvalues().is_none(),
        "eigenvalues must remain None after a failed fit"
    );
    assert!(
        kpca.get_n_samples().is_none(),
        "n_samples must remain None after a failed fit"
    );
}
