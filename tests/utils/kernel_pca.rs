//! Integration tests for `utils::kernel_pca`: constructor validation, fit/transform
//! error paths, per-kernel happy paths, eigensolver agreement, and closed-form kernel
//! value checks. Expected values come from mathematical definitions, not recorded output

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::types::KernelType;
use rustyml::utils::kernel_pca::{EigenSolver, KernelPCA};

use crate::common::assert_allclose;

// Local helpers

/// Small deterministic dataset for general fitting tests (8 samples x 2 features)
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

/// Build two concentric clusters that are radially but not linearly separable: an inner
/// ring at radius 0.5 (label -1) and an outer ring at radius 3.0 (label +1)
///
/// # Parameters
///
/// - `n_inner` - number of inner-ring samples
/// - `n_outer` - number of outer-ring samples
///
/// # Returns
///
/// - `(Array2<f64>, Vec<i32>)` - the (n_inner + n_outer) x 2 points and their labels
fn make_radial_clusters(n_inner: usize, n_outer: usize) -> (Array2<f64>, Vec<i32>) {
    use std::f64::consts::PI;
    let mut rows: Vec<f64> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();

    for i in 0..n_inner {
        let angle = 2.0 * PI * (i as f64) / (n_inner as f64);
        rows.push(0.5 * angle.cos());
        rows.push(0.5 * angle.sin());
        labels.push(-1);
    }
    for i in 0..n_outer {
        let angle = 2.0 * PI * (i as f64) / (n_outer as f64);
        rows.push(3.0 * angle.cos());
        rows.push(3.0 * angle.sin());
        labels.push(1);
    }

    let n = labels.len();
    (Array2::from_shape_vec((n, 2), rows).unwrap(), labels)
}

/// Fisher-like discriminability: sum over components of (mean_pos - mean_neg)^2 divided by
/// (var_pos + var_neg + epsilon); higher means classes are more separable in the projection
///
/// # Parameters
///
/// - `projections` - projected samples, one row per sample
/// - `labels` - class labels, negative for one class and non-negative for the other
///
/// # Returns
///
/// - `f64` - the summed between-class to within-class variance ratio
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

// Constructor + getter tests

#[test]
fn test_new_default_values() {
    let kpca = KernelPCA::default();
    // Default is RBF gamma=0.1, n_components=2, Dense solver
    assert_eq!(kpca.get_n_components(), 2);
    assert_eq!(kpca.get_eigen_solver(), EigenSolver::Dense);
    // Pre-fit getters must return None
    assert!(kpca.get_eigenvalues().is_none());
    assert!(kpca.get_eigenvectors().is_none());
    assert!(kpca.get_n_samples().is_none());
    assert!(kpca.get_n_features().is_none());
}

#[test]
fn test_new_linear_kernel() {
    let kpca = KernelPCA::new(KernelType::Linear, 3)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    assert_eq!(kpca.get_n_components(), 3);
    assert_eq!(kpca.get_eigen_solver(), EigenSolver::Dense);
}

#[test]
fn test_new_rbf_kernel() {
    let kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Lanczos);
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
    )
    .unwrap()
    .with_eigen_solver(EigenSolver::Dense);
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
    )
    .unwrap()
    .with_eigen_solver(EigenSolver::Dense);
    assert_eq!(kpca.get_n_components(), 1);
}

#[test]
fn test_new_cosine_kernel() {
    let kpca = KernelPCA::new(KernelType::Cosine, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    assert_eq!(kpca.get_n_components(), 2);
}

/// Sigmoid with gamma=0 is accepted, since validation rejects only non-finite values
#[test]
fn test_new_sigmoid_gamma_zero_accepted() {
    assert!(
        KernelPCA::new(
            KernelType::Sigmoid {
                gamma: 0.0,
                coef0: 0.0
            },
            1
        )
        .is_ok(),
        "Sigmoid with gamma=0 should be accepted"
    );
}

// Constructor validation / error paths

#[test]
fn test_new_n_components_zero_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::Linear, 0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn test_new_rbf_gamma_zero_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::RBF { gamma: 0.0 }, 1).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_rbf_gamma_negative_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::RBF { gamma: -1.0 }, 1).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_rbf_gamma_nan_returns_invalid_parameter() {
    let err = KernelPCA::new(KernelType::RBF { gamma: f64::NAN }, 1).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

#[test]
fn test_new_rbf_gamma_infinity_returns_invalid_parameter() {
    let err = KernelPCA::new(
        KernelType::RBF {
            gamma: f64::INFINITY,
        },
        1,
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
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }));
}

// fit error paths

#[test]
fn test_fit_empty_input_returns_empty_input() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let x: Array2<f64> = Array2::zeros((0, 2));
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

#[test]
fn test_fit_one_sample_returns_invalid_input() {
    // KernelPCA requires at least 2 samples
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let x = array![[1.0, 2.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn test_fit_n_components_greater_than_n_samples_returns_invalid_parameter() {
    // 3 samples, but n_components=5 - should fail during fit, not construction
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 5)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
    // Model must not be partially mutated: fitted state still None
    assert!(
        kpca.get_n_samples().is_none(),
        "n_samples should still be None after failed fit"
    );
}

#[test]
fn test_fit_nan_in_input_returns_non_finite() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let x = array![[1.0, f64::NAN], [0.0, 1.0], [-1.0, 0.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

#[test]
fn test_fit_inf_in_input_returns_non_finite() {
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let x = array![[f64::INFINITY, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let err = kpca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

// transform error paths

#[test]
fn test_transform_before_fit_returns_not_fitted() {
    let kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let x = array![[1.0, 0.0], [0.0, 1.0]];
    let err = kpca.transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {err:?}"
    );
}

#[test]
fn test_transform_wrong_feature_count_returns_dimension_mismatch() {
    let x_train = make_small_dataset(); // 8 x 2
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca.fit(&x_train).unwrap();

    // Wrong: 3 features instead of 2
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
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca.fit(&x_train).unwrap();

    let x_bad = array![[f64::NAN, 0.0], [0.0, 1.0]];
    let err = kpca.transform(&x_bad).unwrap_err();
    // Both transform-matrix and kernel validation catch NonFinite
    assert!(
        matches!(err, Error::NonFinite(_) | Error::DimensionMismatch { .. }),
        "expected NonFinite (or DimensionMismatch), got {err:?}"
    );
}

#[test]
fn test_transform_empty_input_returns_empty_input() {
    let x_train = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca.fit(&x_train).unwrap();

    let x_empty: Array2<f64> = Array2::zeros((0, 2));
    let err = kpca.transform(&x_empty).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

// Happy-path fit + transform with each kernel

fn run_fit_transform_shape_check(kernel: KernelType, n_components: usize) {
    let x = make_small_dataset(); // 8 x 2
    let mut kpca = KernelPCA::new(kernel, n_components)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca.fit(&x).unwrap();

    // Post-fit getters must be populated
    assert_eq!(kpca.get_n_samples(), Some(8));
    assert_eq!(kpca.get_n_features(), Some(2));
    assert!(kpca.get_eigenvalues().is_some());
    assert!(kpca.get_eigenvectors().is_some());

    // Eigenvalues must all be strictly positive (KernelPCA invariant)
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
    // K(x,y) = tanh(gamma * x.y + coef0); small gamma keeps centering well-conditioned
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

// fit_transform(x) equals fit(x) then transform(x): determinism invariant

#[test]
fn test_fit_transform_equals_fit_then_transform() {
    let x = make_small_dataset();
    let kernel = KernelType::RBF { gamma: 0.5 };

    // Path A: fit_transform in one call
    let mut kpca_a = KernelPCA::new(kernel, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj_a = kpca_a.fit_transform(&x).unwrap();

    // Path B: fit then transform separately
    let mut kpca_b = KernelPCA::new(kernel, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca_b.fit(&x).unwrap();
    let proj_b = kpca_b.transform(&x).unwrap();

    // Both must be numerically identical (same deterministic algorithm, same seed)
    assert_allclose(&proj_a, &proj_b, 1e-10);
}

// Centering invariant: column means of training projection are ~= 0, since double-centering
// K_c = H K H with H = I - 1/n * 11^T zeroes each row sum, hence each projected column mean

#[test]
fn test_centering_training_output_has_near_zero_column_means() {
    let x = make_small_dataset(); // 8 x 2
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj = kpca.fit_transform(&x).unwrap();

    // n_samples = 8, check both components
    for col in 0..proj.ncols() {
        let mean: f64 = proj.column(col).sum() / proj.nrows() as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-9);
    }
}

// Eigenvalues are positive and sorted descending after fit

#[test]
fn test_eigenvalues_are_positive_after_fit() {
    let x = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 4)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca.fit(&x).unwrap();

    let evs = kpca.get_eigenvalues().unwrap();
    assert_eq!(evs.len(), 4);
    // Eigenvalues must be sorted descending and strictly positive
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

// EigenSolver variants agree: Dense, Lanczos, and PowerIteration yield the same dominant
// eigenvalue and projections, up to a global sign flip on each column

fn abs_col_norms(m: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter((0..m.ncols()).map(|j| m.column(j).mapv(|v| v * v).sum().sqrt()))
}

#[test]
fn test_eigensolver_dense_vs_lanczos_agree() {
    let x = make_small_dataset();
    let n_comp = 2;

    let mut kpca_dense = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca_dense.fit(&x).unwrap();
    let proj_dense = kpca_dense.transform(&x).unwrap();

    let mut kpca_lanczos = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp)
        .unwrap()
        .with_eigen_solver(EigenSolver::Lanczos);
    kpca_lanczos.fit(&x).unwrap();
    let proj_lanczos = kpca_lanczos.transform(&x).unwrap();

    // Column norms must match (sign-agnostic check)
    let norms_d = abs_col_norms(&proj_dense);
    let norms_l = abs_col_norms(&proj_lanczos);
    assert_allclose(&norms_d, &norms_l, 1e-5);

    // Leading eigenvalues must agree
    let ev_dense = kpca_dense.get_eigenvalues().unwrap();
    let ev_lanczos = kpca_lanczos.get_eigenvalues().unwrap();
    assert_abs_diff_eq!(ev_dense[0], ev_lanczos[0], epsilon = 1e-5);
}

#[test]
fn test_eigensolver_dense_vs_power_iteration_agree() {
    let x = make_small_dataset();
    let n_comp = 2;

    let mut kpca_dense = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca_dense.fit(&x).unwrap();
    let proj_dense = kpca_dense.transform(&x).unwrap();

    let mut kpca_pi = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, n_comp)
        .unwrap()
        .with_eigen_solver(EigenSolver::PowerIteration);
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
        let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
            .unwrap()
            .with_eigen_solver(solver);
        let proj = kpca.fit_transform(&x).unwrap();
        assert_eq!(proj.shape(), [8, 2], "shape mismatch for solver {solver:?}");
        for &v in proj.iter() {
            assert!(v.is_finite(), "non-finite value for solver {solver:?}");
        }
        assert!(kpca.get_eigenvalues().is_some());
        assert!(kpca.get_eigenvectors().is_some());
    }
}

// Determinism: calling fit+transform twice gives bit-identical results

#[test]
fn test_determinism_dense_solver() {
    let x = make_small_dataset();
    let kernel = KernelType::RBF { gamma: 0.5 };

    let mut kpca1 = KernelPCA::new(kernel, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj1 = kpca1.fit_transform(&x).unwrap();

    let mut kpca2 = KernelPCA::new(kernel, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj2 = kpca2.fit_transform(&x).unwrap();

    assert_allclose(&proj1, &proj2, 0.0);
}

// RBF separates radial clusters better than Linear: an inner ring (r=0.5) and outer ring
// (r=3.0) are radially but not linearly separable, so RBF separability must exceed Linear

#[test]
fn test_rbf_separates_radial_clusters_better_than_linear() {
    // 24 samples: 12 inner (r=0.5, label=-1) + 12 outer (r=3.0, label=1)
    let (x, labels) = make_radial_clusters(12, 12);

    let mut kpca_rbf = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj_rbf = kpca_rbf.fit_transform(&x).unwrap();

    let mut kpca_lin = KernelPCA::new(KernelType::Linear, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj_lin = kpca_lin.fit_transform(&x).unwrap();

    // Metric: Fisher-like separability summed across both components
    let sep_rbf = class_separability(&proj_rbf, &labels);
    let sep_lin = class_separability(&proj_lin, &labels);

    assert!(
        sep_rbf > sep_lin + 0.5,
        "RBF separability ({sep_rbf:.4}) should exceed Linear ({sep_lin:.4}) by > 0.5; \
         RBF encodes radial distance which separates inner (r=0.5) from outer (r=3.0) \
         while linear kernel is dominated by outer-ring angular variation"
    );
}

// Save / load round-trip

#[test]
fn test_save_load_round_trip() {
    let x = make_small_dataset();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    kpca.fit(&x).unwrap();
    let proj_before = kpca.transform(&x).unwrap();

    let path = "/tmp/rustyml_test_kpca_round_trip.json";
    kpca.save_to_path(path).unwrap();
    let kpca_loaded = KernelPCA::load_from_path(path).unwrap();
    let proj_after = kpca_loaded.transform(&x).unwrap();

    // After deserialization, identical transform output is required
    assert_allclose(&proj_before, &proj_after, 1e-12);

    // Clean up
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

// Single-feature (1-D) data

#[test]
fn test_single_feature_data_fits_and_transforms() {
    // Four 1-D points; the kernel matrix is 4x4 and centering still applies
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj = kpca.fit_transform(&x).unwrap();
    assert_eq!(proj.shape(), [4, 1]);
    for &v in proj.iter() {
        assert!(v.is_finite());
    }
}

// Parallel path (n_samples >= 200)

#[test]
fn test_parallel_path_n_samples_200() {
    // KERNEL_PCA_PARALLEL_THRESHOLD = 200; use exactly 200 rows
    let n = 200;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let angle = (i as f64) * std::f64::consts::TAU / (n as f64);
        data.push(angle.cos());
        data.push(angle.sin());
    }
    let x = Array2::from_shape_vec((n, 2), data).unwrap();
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.05 }, 2)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let proj = kpca.fit_transform(&x).unwrap();
    assert_eq!(proj.shape(), [n, 2]);
    for &v in proj.iter() {
        assert!(v.is_finite(), "non-finite value in parallel path output");
    }
}

// Numerical spot-checks: closed-form kernel values via the public KernelType::compute

#[test]
fn test_rbf_kernel_known_value() {
    // K(x,y) = exp(-gamma * ||x-y||^2); ||x-y||^2 = 25, gamma=0.1, so K = exp(-2.5)
    let x1 = array![3.0, 0.0];
    let x2 = array![0.0, 4.0];
    let k = KernelType::RBF { gamma: 0.1 };
    let expected = (-2.5f64).exp();
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn test_linear_kernel_known_value() {
    // K(x,y) = x.y = 3*1 + 4*2 = 11
    let x1 = array![3.0, 4.0];
    let x2 = array![1.0, 2.0];
    let k = KernelType::Linear;
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, 11.0, epsilon = 1e-12);
}

#[test]
fn test_cosine_kernel_known_value() {
    // K(x,y) = (x.y) / (||x|| * ||y||); dot=3, ||x||=5, ||y||=1, so K = 3/5 = 0.6
    let x1 = array![3.0, 4.0];
    let x2 = array![1.0, 0.0];
    let k = KernelType::Cosine;
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, 0.6, epsilon = 1e-12);
}

#[test]
fn test_poly_kernel_known_value() {
    // K(x,y) = (gamma * x.y + coef0)^degree; dot=11, so (1*11+1)^2 = 144
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
    // K(x,y) = tanh(gamma * x.y + coef0); dot=1, gamma=1, coef0=0, so K = tanh(1.0)
    let x1 = array![1.0, 0.0];
    let x2 = array![1.0, 0.0];
    let k = KernelType::Sigmoid {
        gamma: 1.0,
        coef0: 0.0,
    };
    let expected = 1.0f64.tanh();
    let actual = k.compute(x1.view(), x2.view());
    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}

// n_components == n_samples boundary: exactly equal should succeed

#[test]
fn test_n_components_equals_n_samples_boundary() {
    // 4 samples; n_components=4 is accepted (n_components <= n_samples)
    let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 4)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    // May succeed or fail with Computation, but must not return InvalidParameter
    match kpca.fit(&x) {
        Ok(_) => {
            let proj = kpca.transform(&x).unwrap();
            assert_eq!(proj.shape(), [4, 4]);
        }
        Err(Error::Computation { .. }) => {
            // Acceptable: near-zero eigenvalues for this small symmetric dataset
        }
        Err(e) => {
            panic!("unexpected error for n_components==n_samples: {e:?}");
        }
    }
}

// Two-sample minimum boundary: exactly 2 samples is valid

#[test]
fn test_two_samples_is_valid() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.5 }, 1)
        .unwrap()
        .with_eigen_solver(EigenSolver::Dense);
    let result = kpca.fit(&x);
    // May succeed or return Computation (rank-1 kernel), but must not panic
    // and must not return InvalidInput (2 >= minimum of 2)
    match result {
        Ok(_) => {}
        Err(Error::Computation { .. }) => {}
        Err(e) => panic!("unexpected error for 2-sample fit: {e:?}"),
    }
}
// An indefinite Sigmoid kernel yields a non-positive eigenvalue in its centered Gram matrix;
// Kernel PCA tolerates this (matching scikit-learn): the fit succeeds and the non-positive
// component is zeroed at projection time rather than failing the whole fit
#[test]
fn test_fit_indefinite_kernel_negative_eigenvalue_is_tolerated() {
    let x = array![[1.0], [4.0]]; // 2 samples, 1 feature
    let mut kpca = KernelPCA::new(
        KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        },
        2, // == n_samples, so a non-positive eigenvalue is necessarily selected
    )
    .unwrap();

    // Fit succeeds despite the indefinite kernel, and persists fitted state
    kpca.fit(&x).unwrap();
    let eigenvalues = kpca
        .get_eigenvalues()
        .expect("eigenvalues must be persisted after a successful fit");
    assert_eq!(kpca.get_n_samples(), Some(2));
    assert!(
        eigenvalues.iter().any(|&v| v <= 0.0),
        "the indefinite Sigmoid kernel is expected to yield a non-positive eigenvalue"
    );

    // Projection stays finite, and the non-positive component is zeroed out
    let transformed = kpca.transform(&x).unwrap();
    assert_eq!(transformed.shape(), &[2, 2]);
    assert!(
        transformed.iter().all(|v| v.is_finite()),
        "projection must be finite even when a component has a non-positive eigenvalue"
    );
    let zeroed_columns = (0..transformed.ncols())
        .filter(|&j| transformed.column(j).iter().all(|&v| v == 0.0))
        .count();
    assert!(
        zeroed_columns >= 1,
        "the component with a non-positive eigenvalue must be zeroed in the projection"
    );
}
