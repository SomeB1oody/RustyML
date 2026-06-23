//! Integration tests for rustyml::utils::pca, covering construction/validation,
//! error paths, and numerical results across all three SVD solvers

use ndarray::{Array2, array};
use rustyml::error::Error;
use rustyml::utils::pca::{PCA, SVDSolver};

use crate::common::assert_allclose;

// Helper

/// Collinear data: 5 points on y = 2x, rank-1 with first PC along [1/sqrt(5), 2/sqrt(5)]
/// and sigma_1 = sqrt(50) = 5*sqrt(2)
fn collinear_data() -> Array2<f64> {
    array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]
}

/// Zero-mean data with different variance per axis: PCs are the standard basis
/// with sigma_1 = sqrt(32) > sigma_2 = sqrt(8)
fn anisotropic_data() -> Array2<f64> {
    array![[4.0, 0.0], [0.0, 2.0], [-4.0, 0.0], [0.0, -2.0]]
}

// Construction / validation

/// PCA::new(0) returns Error::InvalidParameter
#[test]
fn test_new_n_components_zero_is_invalid_parameter() {
    let err = PCA::new(0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

/// PCA::new with n_components >= 1 is accepted
#[test]
fn test_new_valid_n_components() {
    PCA::new(1)
        .expect("n_components=1 should be valid")
        .with_svd_solver(SVDSolver::Full);
    PCA::new(2)
        .expect("n_components=2 should be valid")
        .with_svd_solver(SVDSolver::Full);
}

// Error paths: not fitted

/// transform before fit returns Error::NotFitted
#[test]
fn test_transform_before_fit_is_not_fitted() {
    let pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let x = collinear_data();
    let err = pca.transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {:?}",
        err
    );
}

/// inverse_transform before fit returns Error::NotFitted
#[test]
fn test_inverse_transform_before_fit_is_not_fitted() {
    let pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let scores = array![[1.0], [0.0], [0.0], [0.0], [0.0]];
    let err = pca.inverse_transform(&scores).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {:?}",
        err
    );
}

// Error paths: bad inputs at fit time

/// Empty matrix returns EmptyInput
#[test]
fn test_fit_empty_input_is_empty_input() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let x: Array2<f64> = Array2::zeros((0, 2));
    let err = pca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {:?}",
        err
    );
}

/// Single sample returns InvalidInput (check_min_samples requires >= 2)
#[test]
fn test_fit_single_sample_is_invalid_input() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let x = array![[1.0, 2.0]];
    let err = pca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {:?}",
        err
    );
}

/// n_components > n_features returns InvalidParameter (max_components = min(5,2) = 2)
#[test]
fn test_fit_n_components_exceeds_n_features_is_invalid_parameter() {
    let mut pca = PCA::new(3).unwrap().with_svd_solver(SVDSolver::Full);
    let x = collinear_data(); // 5 rows, 2 features
    let err = pca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for n_components > n_features, got {:?}",
        err
    );
}

/// NaN in input returns NonFinite
#[test]
fn test_fit_nan_input_is_non_finite() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let x = array![[1.0, f64::NAN], [2.0, 4.0], [3.0, 6.0]];
    let err = pca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {:?}",
        err
    );
}

/// Inf in input returns NonFinite
#[test]
fn test_fit_inf_input_is_non_finite() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let x = array![[1.0, 2.0], [f64::INFINITY, 4.0], [3.0, 6.0]];
    let err = pca.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {:?}",
        err
    );
}

// Error paths: bad inputs at transform time

/// Wrong feature count on transform returns DimensionMismatch
#[test]
fn test_transform_wrong_feature_count_is_dimension_mismatch() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();
    let x_wrong = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let err = pca.transform(&x_wrong).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// NaN during transform returns NonFinite
#[test]
fn test_transform_nan_is_non_finite() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();
    let x = array![[1.0, f64::NAN], [2.0, 4.0]];
    let err = pca.transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {:?}",
        err
    );
}

// Collinear data: Full SVD - numerical assertions

/// Explained variance ratio is > 0.999 for 1 component on perfectly collinear data
#[test]
fn test_full_collinear_explained_variance_ratio() {
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();

    let ratio = pca
        .get_explained_variance_ratio()
        .expect("model should be fitted; ratio is Some");
    assert_eq!(ratio.len(), 1, "should have 1 variance ratio");
    // explained variance ratio = (50/4) / (50/4) = 1.0 exactly
    assert!(
        ratio[0] > 0.999,
        "explained variance ratio should be ≥ 0.999, got {}",
        ratio[0]
    );
}

/// Singular value for collinear data (Full SVD) is sqrt(50) = 5*sqrt(2)
#[test]
fn test_full_collinear_singular_value() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();

    let sv = pca.get_singular_values().expect("fitted; sv is Some");
    assert_eq!(sv.len(), 1, "should have 1 singular value");
    // sigma = sqrt(50) = 5*sqrt(2)
    let expected_sv = 50.0_f64.sqrt(); // 7.0710678118654755
    assert_abs_diff_eq!(sv[0], expected_sv, epsilon = 1e-6);
}

/// Component row is a unit vector for collinear data (Full SVD)
#[test]
fn test_full_collinear_component_is_unit_norm() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();

    let comp = pca.get_components().expect("fitted; components is Some");
    assert_eq!(comp.nrows(), 1, "1 component row");
    assert_eq!(comp.ncols(), 2, "2 features");

    let norm = (comp[[0, 0]].powi(2) + comp[[0, 1]].powi(2)).sqrt();
    assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
}

/// First principal component direction aligns with +/-[1/sqrt(5), 2/sqrt(5)] for
/// collinear data (SVD is defined up to sign)
#[test]
fn test_full_collinear_component_direction() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();

    let comp = pca.get_components().expect("fitted; components is Some");
    // [1/sqrt(5), 2/sqrt(5)]
    let expected_0 = 1.0_f64 / 5.0_f64.sqrt();
    let expected_1 = 2.0_f64 / 5.0_f64.sqrt();

    // Component may be negated, so compare the absolute dot product with the expected direction
    let dot = comp[[0, 0]] * expected_0 + comp[[0, 1]] * expected_1;
    assert_abs_diff_eq!(dot.abs(), 1.0, epsilon = 1e-8);
}

/// inverse_transform recovers original collinear data within 1e-6 (Full SVD, 1 component)
#[test]
fn test_full_collinear_inverse_transform_reconstruction() {
    let x = collinear_data();
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();
    let projected = pca.transform(&x).unwrap();
    let reconstructed = pca.inverse_transform(&projected).unwrap();

    assert_allclose(&reconstructed, &x, 1e-6);
}

// fit_transform == fit + transform

/// fit_transform(x) produces the same result as fit(x) followed by transform(x)
#[test]
fn test_fit_transform_equals_fit_then_transform_full() {
    let x = collinear_data();

    let mut pca_a = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    let result_a = pca_a.fit_transform(&x).unwrap();

    let mut pca_b = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca_b.fit(&x).unwrap();
    let result_b = pca_b.transform(&x).unwrap();

    assert_allclose(&result_a, &result_b, 1e-10);
}

// Orthonormality of components (anisotropic 2-component case)

/// Component rows are orthonormal for a full 2-component fit
#[test]
fn test_components_orthonormal_full() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();
    let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();

    let comp = pca.get_components().expect("fitted; components is Some");
    assert_eq!(comp.nrows(), 2);
    assert_eq!(comp.ncols(), 2);

    // Each row has unit norm
    for i in 0..2 {
        let norm = (comp[[i, 0]].powi(2) + comp[[i, 1]].powi(2)).sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
    }

    // Rows are orthogonal
    let dot = comp[[0, 0]] * comp[[1, 0]] + comp[[0, 1]] * comp[[1, 1]];
    assert_abs_diff_eq!(dot.abs(), 0.0, epsilon = 1e-8);
}

// Singular values in descending order

/// Singular values are in non-ascending order (anisotropic data: sqrt(32) > sqrt(8))
#[test]
fn test_singular_values_descending_full() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();
    let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();

    let sv = pca.get_singular_values().expect("fitted; sv is Some");
    assert_eq!(sv.len(), 2);
    // sigma_1 = sqrt(32) = 4*sqrt(2)
    let expected_sv1 = 32.0_f64.sqrt(); // 5.6568542495
    // sigma_2 = sqrt(8) = 2*sqrt(2)
    let expected_sv2 = 8.0_f64.sqrt(); // 2.8284271247

    assert_abs_diff_eq!(sv[0], expected_sv1, epsilon = 1e-4);
    assert_abs_diff_eq!(sv[1], expected_sv2, epsilon = 1e-4);
    assert!(
        sv[0] >= sv[1],
        "singular values should be non-ascending: {sv:?}"
    );
}

/// Explained variance ratios sum to ~1.0 for 2 components on 2-feature data
/// (ratios 32/40 = 0.8 and 8/40 = 0.2)
#[test]
fn test_explained_variance_ratio_sums_to_one() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();
    let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();

    let ratio = pca
        .get_explained_variance_ratio()
        .expect("fitted; ratio is Some");
    let sum: f64 = ratio.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

    // Individual ratios: 32/40 = 0.8, 8/40 = 0.2
    assert_abs_diff_eq!(ratio[0].max(ratio[1]), 0.8, epsilon = 1e-4);
    assert_abs_diff_eq!(ratio[0].min(ratio[1]), 0.2, epsilon = 1e-4);
}

// All three SVD solvers - shape agreement and singular-value agreement

/// All three SVDSolver variants produce the correct output shape
#[test]
fn test_all_solvers_produce_correct_output_shape() {
    let x = collinear_data();
    let n_components = 1usize;

    for solver in [
        SVDSolver::Full,
        SVDSolver::Randomized(42),
        SVDSolver::PowerIteration,
    ] {
        let mut pca = PCA::new(n_components).unwrap().with_svd_solver(solver);
        pca.fit(&x).unwrap();
        let projected = pca.transform(&x).unwrap();
        assert_eq!(
            projected.shape(),
            &[5, n_components],
            "solver {:?}: wrong output shape",
            solver
        );
    }
}

/// All three SVDSolver variants agree on the leading singular value (sqrt(50)) for
/// the collinear dataset, with loose tolerance for the approximate solvers
#[test]
fn test_all_solvers_agree_on_singular_value() {
    use approx::assert_abs_diff_eq;

    let x = collinear_data();
    let expected_sv = 50.0_f64.sqrt(); // 7.0710678118654755

    for solver in [
        SVDSolver::Full,
        SVDSolver::Randomized(42),
        SVDSolver::PowerIteration,
    ] {
        let mut pca = PCA::new(1).unwrap().with_svd_solver(solver);
        pca.fit(&x).unwrap();
        let sv = pca.get_singular_values().expect("fitted; sv is Some");
        assert_abs_diff_eq!(
            sv[0],
            expected_sv,
            epsilon = 1e-2,
            // 1e-2 absolute tolerance validates the dominant direction was found;
            // approximate solvers can deviate slightly on a 5-sample problem
        );
    }
}

/// All three solvers agree on component signs: each axis is oriented so its largest-magnitude
/// loading is non-negative, so the solvers produce sign-identical components (svd_flip)
#[test]
fn test_all_solvers_agree_on_component_signs() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();

    let reference = {
        let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
        pca.fit(&x).unwrap();
        pca.get_components()
            .expect("fitted; components is Some")
            .clone()
    };

    // Each component's largest-magnitude loading must be non-negative after the sign fix
    for row in reference.rows() {
        let signed_max = row
            .iter()
            .copied()
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        assert!(
            signed_max >= 0.0,
            "largest-magnitude loading should be non-negative, got {}",
            signed_max
        );
    }

    // The approximate solvers must land on the same orientation, not a flipped one
    for solver in [SVDSolver::Randomized(42), SVDSolver::PowerIteration] {
        let mut pca = PCA::new(2).unwrap().with_svd_solver(solver);
        pca.fit(&x).unwrap();
        let components = pca.get_components().expect("fitted; components is Some");
        // A flipped axis would differ by ~2x the loading; 1e-3 catches it while tolerating the
        // approximate solvers' small numerical drift on the well-separated axes
        for (a, b) in reference.rows().into_iter().zip(components.rows()) {
            for (&x_ref, &x_got) in a.iter().zip(b.iter()) {
                assert_abs_diff_eq!(x_ref, x_got, epsilon = 1e-3);
            }
        }
    }
}

// Randomized solver: determinism

/// Same seed for Randomized produces bit-identical transform output
#[test]
fn test_randomized_determinism_same_seed() {
    let x = anisotropic_data();

    let mut pca_a = PCA::new(2)
        .unwrap()
        .with_svd_solver(SVDSolver::Randomized(42));
    let result_a = pca_a.fit_transform(&x).unwrap();

    let mut pca_b = PCA::new(2)
        .unwrap()
        .with_svd_solver(SVDSolver::Randomized(42));
    let result_b = pca_b.fit_transform(&x).unwrap();

    assert_allclose(&result_a, &result_b, 0.0);
}

// Component orthonormality for Randomized and PowerIteration

/// Randomized solver produces orthonormal component rows on the anisotropic dataset
#[test]
fn test_randomized_components_orthonormal() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();
    let mut pca = PCA::new(2)
        .unwrap()
        .with_svd_solver(SVDSolver::Randomized(42));
    pca.fit(&x).unwrap();

    let comp = pca.get_components().expect("fitted; components is Some");
    for i in 0..2 {
        let norm = (comp[[i, 0]].powi(2) + comp[[i, 1]].powi(2)).sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }
    let dot = comp[[0, 0]] * comp[[1, 0]] + comp[[0, 1]] * comp[[1, 1]];
    assert_abs_diff_eq!(dot.abs(), 0.0, epsilon = 1e-6);
}

/// PowerIteration solver produces unit-norm component rows
#[test]
fn test_power_iteration_component_unit_norm() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();
    let mut pca = PCA::new(2)
        .unwrap()
        .with_svd_solver(SVDSolver::PowerIteration);
    pca.fit(&x).unwrap();

    let comp = pca.get_components().expect("fitted; components is Some");
    for i in 0..2 {
        let norm = (comp[[i, 0]].powi(2) + comp[[i, 1]].powi(2)).sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-5);
    }
}

/// PowerIteration inverse_transform recovers collinear data within 1e-4
#[test]
fn test_power_iteration_inverse_transform_recovers_collinear() {
    let x = collinear_data();
    let mut pca = PCA::new(1)
        .unwrap()
        .with_svd_solver(SVDSolver::PowerIteration);
    pca.fit(&x).unwrap();
    let projected = pca.transform(&x).unwrap();
    let reconstructed = pca.inverse_transform(&projected).unwrap();

    assert_allclose(&reconstructed, &x, 1e-4);
}

// Explained variance ratio > 99.9% on collinear data (all three solvers)

/// Each solver captures > 99.9% of variance in 1 component on collinear data
#[test]
fn test_all_solvers_capture_full_variance_on_collinear_data() {
    let x = collinear_data();
    for solver in [
        SVDSolver::Full,
        SVDSolver::Randomized(42),
        SVDSolver::PowerIteration,
    ] {
        let mut pca = PCA::new(1).unwrap().with_svd_solver(solver);
        pca.fit(&x).unwrap();
        let ratio = pca
            .get_explained_variance_ratio()
            .expect("fitted; ratio is Some");
        assert!(
            ratio[0] > 0.999,
            "solver {:?}: expected ratio > 0.999, got {}",
            solver,
            ratio[0]
        );
    }
}

// Save / load round-trip

/// Saving a fitted PCA to a temp file and reloading it preserves the transform output
#[test]
fn test_save_load_roundtrip_preserves_transform() {
    use std::fs;

    let x = collinear_data();
    let path = "/tmp/rustyml_pca_test_roundtrip.json";

    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();
    let result_before = pca.transform(&x).unwrap();

    pca.save_to_path(path).unwrap();

    let loaded = PCA::load_from_path(path).unwrap();
    let result_after = loaded.transform(&x).unwrap();

    assert_allclose(&result_before, &result_after, 1e-10);

    let _ = fs::remove_file(path);
}

/// load_from_path on a non-existent file returns Err (IoError)
#[test]
fn test_load_nonexistent_path_is_error() {
    let result = PCA::load_from_path("/tmp/rustyml_pca_no_such_file_xyz.json");
    assert!(result.is_err(), "loading a missing file should fail");
}

// Output shape contract

/// transform output shape is (n_samples, n_components)
#[test]
fn test_transform_output_shape() {
    let x = collinear_data(); // 5 rows, 2 cols
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();
    let out = pca.transform(&x).unwrap();
    assert_eq!(out.shape(), &[5, 1]);
}

/// inverse_transform output shape is (n_samples, n_features)
#[test]
fn test_inverse_transform_output_shape() {
    let x = collinear_data(); // 5 rows, 2 cols
    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();
    let projected = pca.transform(&x).unwrap();
    let reconstructed = pca.inverse_transform(&projected).unwrap();
    assert_eq!(reconstructed.shape(), &[5, 2]);
}

// Projection values for anisotropic data (numerical assertion)

/// Projections for anisotropic data match the closed-form values, checked as
/// absolute values since PC sign is ambiguous
#[test]
fn test_anisotropic_projection_values_full() {
    use approx::assert_abs_diff_eq;

    let x = anisotropic_data();
    let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&x).unwrap();
    let projected = pca.transform(&x).unwrap();

    // First column (first PC, larger variance) has absolute values 4,0,4,0
    let col0_abs: Vec<f64> = (0..4).map(|i| projected[[i, 0]].abs()).collect();
    assert_abs_diff_eq!(col0_abs[0], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(col0_abs[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(col0_abs[2], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(col0_abs[3], 0.0, epsilon = 1e-6);

    // First PC column: rows 0 and 2 have opposite signs (+/-4)
    let s0 = projected[[0, 0]];
    let s2 = projected[[2, 0]];
    assert_abs_diff_eq!(s0, -s2, epsilon = 1e-6);

    // Second column (second PC, smaller variance) has absolute values 0,2,0,2
    let col1_abs: Vec<f64> = (0..4).map(|i| projected[[i, 1]].abs()).collect();
    assert_abs_diff_eq!(col1_abs[0], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(col1_abs[1], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(col1_abs[2], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(col1_abs[3], 2.0, epsilon = 1e-6);
}
// get_mean and get_explained_variance (non-ratio) getters

/// get_mean returns the per-feature centering mean, exactly [3, 6] for collinear_data
#[test]
fn test_full_collinear_get_mean() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();

    let mean = pca.get_mean().expect("fitted; mean is Some");
    assert_eq!(mean.len(), 2, "mean has one entry per feature");
    // Closed form: column means of the collinear dataset
    assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-9);
    assert_abs_diff_eq!(mean[1], 6.0, epsilon = 1e-9);
}

/// get_mean is [0, 0] for the zero-centered anisotropic dataset
#[test]
fn test_anisotropic_get_mean_is_zero() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&anisotropic_data()).unwrap();

    let mean = pca.get_mean().expect("fitted; mean is Some");
    assert_eq!(mean.len(), 2);
    assert_abs_diff_eq!(mean[0], 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(mean[1], 0.0, epsilon = 1e-9);
}

/// get_explained_variance (non-ratio variance per component) matches the closed-form
/// sigma^2/(n-1) = [32/3, 8/3] for the anisotropic dataset
#[test]
fn test_anisotropic_get_explained_variance() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&anisotropic_data()).unwrap();

    let ev = pca
        .get_explained_variance()
        .expect("fitted; explained_variance is Some");
    assert_eq!(ev.len(), 2, "one variance per component");
    // sigma_1^2/(n-1) = 32/3, sigma_2^2/(n-1) = 8/3
    assert_abs_diff_eq!(ev[0], 32.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(ev[1], 8.0 / 3.0, epsilon = 1e-6);
}

/// get_explained_variance for the rank-1 collinear dataset is 50/4 = 12.5
#[test]
fn test_full_collinear_get_explained_variance() {
    use approx::assert_abs_diff_eq;

    let mut pca = PCA::new(1).unwrap().with_svd_solver(SVDSolver::Full);
    pca.fit(&collinear_data()).unwrap();

    let ev = pca
        .get_explained_variance()
        .expect("fitted; explained_variance is Some");
    assert_eq!(ev.len(), 1);
    // 50 / (5 - 1) = 12.5
    assert_abs_diff_eq!(ev[0], 12.5, epsilon = 1e-6);
}
