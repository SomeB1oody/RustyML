//! Integration tests for IsolationForest: constructor validation, error paths,
//! score properties, outlier detection, contamination thresholding, determinism,
//! and closed-form checks

use ndarray::{Array2, array, s};
use rustyml::error::Error;
use rustyml::machine_learning::{Contamination, IsolationForest};

// Constructor validation

#[test]
fn test_new_rejects_invalid_scalar_arguments() {
    // Each row supplies 1 invalid scalar, and its InvalidParameter error must carry that name.
    // n_estimators and max_samples go through `new`, max_depth through `with_max_depth`
    type Case = (&'static str, fn() -> Error);
    let cases: [Case; 3] = [
        ("n_estimators", || IsolationForest::new(0, 256).unwrap_err()),
        ("max_samples", || IsolationForest::new(10, 0).unwrap_err()),
        ("max_depth", || {
            IsolationForest::new(10, 256)
                .unwrap()
                .with_max_depth(0)
                .unwrap_err()
        }),
    ];

    for (expected_name, build) in cases {
        let err = build();
        assert!(
            matches!(err, Error::InvalidParameter { ref name, .. } if name == expected_name),
            "expected InvalidParameter for {expected_name}, got: {err:?}"
        );
    }
}

#[test]
fn test_new_valid_explicit_max_depth_succeeds() {
    let model = IsolationForest::new(20, 64)
        .unwrap()
        .with_max_depth(5)
        .unwrap()
        .with_random_state(99);
    assert_eq!(model.get_n_estimators(), 20);
    assert_eq!(model.get_max_samples(), 64);
    assert_eq!(model.get_max_depth(), 5);
    assert_eq!(model.get_random_state(), Some(99));
    assert_eq!(model.get_n_features(), 0);
    assert!(model.get_trees().is_none());
}

#[test]
fn test_new_auto_max_depth_ceil_log2_max_samples() {
    // max_samples = 256: ceil(log2(256)) = 8
    let model = IsolationForest::new(10, 256).unwrap().with_random_state(0);
    assert_eq!(model.get_max_depth(), 8);
}

#[test]
fn test_new_auto_max_depth_ceil_log2_non_power_of_two() {
    // max_samples = 100: ceil(log2(100)) = ceil(6.6439) = 7
    let model = IsolationForest::new(10, 100).unwrap().with_random_state(0);
    assert_eq!(model.get_max_depth(), 7);
}

#[test]
fn test_new_auto_max_depth_ceil_log2_two() {
    // max_samples = 2: ceil(log2(2)) = 1
    let model = IsolationForest::new(10, 2).unwrap().with_random_state(0);
    assert_eq!(model.get_max_depth(), 1);
}

#[test]
fn test_new_auto_max_depth_ceil_log2_one() {
    // max_samples = 1: ceil(log2(1)) = ceil(0.0) = 0, accepted because depth is
    // auto-computed (not an explicit Some(0))
    let model = IsolationForest::new(10, 1).unwrap().with_random_state(0);
    assert_eq!(model.get_max_depth(), 0);
}

#[test]
fn test_default_has_expected_parameter_values() {
    let model = IsolationForest::default();
    assert_eq!(model.get_n_estimators(), 100);
    assert_eq!(model.get_max_samples(), 256);
    assert_eq!(model.get_max_depth(), 8); // ceil(log2(256))
    assert_eq!(model.get_random_state(), None);
    assert_eq!(model.get_n_features(), 0);
    assert!(model.get_trees().is_none());
}

// NotFitted errors

#[test]
fn test_predict_before_fit_returns_not_fitted() {
    let model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let err = model.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("IsolationForest")),
        "expected NotFitted, got: {err:?}"
    );
}

#[test]
fn test_score_sample_before_fit_returns_not_fitted() {
    let model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let err = model.score_sample(&[1.0, 2.0]).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("IsolationForest")),
        "expected NotFitted, got: {err:?}"
    );
}

// fit error paths

#[test]
fn test_fit_empty_data_returns_empty_input() {
    let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let x: Array2<f64> = Array2::zeros((0, 2));
    let err = model.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got: {err:?}"
    );
}

#[test]
fn test_fit_non_finite_returns_non_finite() {
    // The finiteness guard in `fit` must reject every non-finite sentinel.
    for sentinel in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
        let x = array![[1.0, sentinel], [2.0, 3.0]];
        let err = model.fit(&x).unwrap_err();
        assert!(
            matches!(err, Error::NonFinite(_)),
            "expected NonFinite for sentinel={sentinel:?}, got: {err:?}"
        );
    }
}

// predict error paths (after fit)

#[test]
fn test_predict_empty_data_returns_empty_input() {
    let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    model.fit(&train).unwrap();

    let x_empty: Array2<f64> = Array2::zeros((0, 2));
    let err = model.predict(&x_empty).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got: {err:?}"
    );
}

#[test]
fn test_predict_wrong_feature_count_returns_dimension_mismatch() {
    let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    model.fit(&train).unwrap();

    // training had 2 features, predict is given 3
    let x_wrong = array![[1.0, 2.0, 3.0]];
    let err = model.predict(&x_wrong).unwrap_err();
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 2,
                found: 3
            }
        ),
        "expected DimensionMismatch{{expected:2, found:3}}, got: {err:?}"
    );
}

#[test]
fn test_predict_nan_returns_non_finite() {
    let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    model.fit(&train).unwrap();

    let x_nan = array![[f64::NAN, 2.0]];
    let err = model.predict(&x_nan).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got: {err:?}"
    );
}

// score_sample error paths (after fit)

#[test]
fn test_score_sample_wrong_dim_returns_dimension_mismatch() {
    let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(1);
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    model.fit(&train).unwrap();

    // training had 2 features, score is given 3
    let err = model.score_sample(&[1.0, 2.0, 3.0]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 2,
                found: 3
            }
        ),
        "expected DimensionMismatch{{expected:2, found:3}}, got: {err:?}"
    );
}

// Post-fit state

#[test]
fn test_fit_sets_n_features() {
    let mut model = IsolationForest::new(10, 50).unwrap().with_random_state(42);
    let train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    model.fit(&train).unwrap();
    assert_eq!(model.get_n_features(), 3);
}

#[test]
fn test_fit_stores_exactly_n_estimators_trees() {
    let n_estimators = 15_usize;
    let mut model = IsolationForest::new(n_estimators, 50)
        .unwrap()
        .with_random_state(42);
    let train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [2.0, 3.0], [4.0, 5.0]];
    model.fit(&train).unwrap();

    let trees = model.get_trees().expect("trees should be Some after fit");
    assert_eq!(
        trees.len(),
        n_estimators,
        "expected exactly {n_estimators} trees, got {}",
        trees.len()
    );
}

// Scores in [-1, 0)

#[test]
fn test_scores_are_in_negative_unit_interval() {
    // anomaly scores lie in [-1, 0) by design: -(2^(-E/c)) with E, c > 0
    let mut model = IsolationForest::new(50, 64).unwrap().with_random_state(7);
    let train = array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.2, 0.2],
        [100.0, 100.0]
    ];
    model.fit(&train).unwrap();

    let scores = model.score_samples(&train).unwrap();
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (-1.0..0.0).contains(&s),
            "score[{i}] = {s} is outside [0, 1]"
        );
    }
}

#[test]
fn test_score_sample_is_in_negative_unit_interval() {
    let mut model = IsolationForest::new(50, 64).unwrap().with_random_state(7);
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [50.0, 50.0]];
    model.fit(&train).unwrap();

    let s_inlier = model.score_sample(&[0.0, 0.0]).unwrap();
    let s_outlier = model.score_sample(&[50.0, 50.0]).unwrap();

    assert!(
        (-1.0..0.0).contains(&s_inlier),
        "inlier score {s_inlier} not in [0,1]"
    );
    assert!(
        (-1.0..0.0).contains(&s_outlier),
        "outlier score {s_outlier} not in [0,1]"
    );
}

// Outlier detection correctness: anomalous points (far from the bulk) are
// isolated in fewer steps and so get lower (more negative) anomaly scores than inliers

#[test]
fn test_outlier_score_exceeds_all_inlier_scores() {
    // 10 inliers clustered near (0, 0), plus 1 outlier far away at (1000, 1000)
    let inliers = array![
        [0.0, 0.0],
        [0.1, 0.0],
        [-0.1, 0.0],
        [0.0, 0.1],
        [0.0, -0.1],
        [0.1, 0.1],
        [-0.1, 0.1],
        [0.1, -0.1],
        [-0.1, -0.1],
        [0.2, 0.2]
    ];
    let outlier_row = array![[1000.0, 1000.0]];

    let mut train_data: Array2<f64> = Array2::zeros((11, 2));
    train_data.slice_mut(s![..10, ..]).assign(&inliers);
    train_data.slice_mut(s![10..11, ..]).assign(&outlier_row);

    let mut model = IsolationForest::new(100, 64).unwrap().with_random_state(42);
    model.fit(&train_data).unwrap();
    let scores = model.score_samples(&train_data).unwrap();

    let outlier_score = scores[10];
    let min_inlier_score = scores
        .slice(s![..10])
        .fold(f64::INFINITY, |acc, &v| acc.min(v));

    assert!(
        outlier_score < min_inlier_score,
        "outlier score {outlier_score:.4} should fall below all inlier scores (min inlier: {min_inlier_score:.4})"
    );
}

#[test]
fn test_outlier_score_sample_is_below_inlier_via_single_sample_api() {
    // same design as above, verified through the per-sample score_sample API
    let mut train_data: Array2<f64> = Array2::zeros((11, 2));
    let inlier_coords: &[(f64, f64)] = &[
        (0.0, 0.0),
        (0.1, 0.0),
        (-0.1, 0.0),
        (0.0, 0.1),
        (0.0, -0.1),
        (0.1, 0.1),
        (-0.1, 0.1),
        (0.1, -0.1),
        (-0.1, -0.1),
        (0.2, 0.2),
    ];
    for (i, &(x, y)) in inlier_coords.iter().enumerate() {
        train_data[[i, 0]] = x;
        train_data[[i, 1]] = y;
    }
    train_data[[10, 0]] = 1000.0;
    train_data[[10, 1]] = 1000.0;

    let mut model = IsolationForest::new(100, 64).unwrap().with_random_state(42);
    model.fit(&train_data).unwrap();

    let outlier_score = model.score_sample(&[1000.0, 1000.0]).unwrap();
    let min_inlier_score = inlier_coords
        .iter()
        .map(|&(x, y)| model.score_sample(&[x, y]).unwrap())
        .fold(f64::INFINITY, f64::min);

    assert!(
        outlier_score < min_inlier_score,
        "outlier score {outlier_score:.4} should fall below min inlier score {min_inlier_score:.4}"
    );
}

// Identical points get identical scores

#[test]
fn test_identical_points_have_equal_scores() {
    // 2 identical rows must be scored identically under a seeded (deterministic) model
    let data = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [1.0, 2.0], // row 0 and row 2 are identical
        [5.0, 6.0]
    ];
    let mut model = IsolationForest::new(50, 32).unwrap().with_random_state(55);
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();
    assert_eq!(
        scores[0], scores[2],
        "identical inputs must produce identical anomaly scores: {} vs {}",
        scores[0], scores[2]
    );
}

// Determinism: same seed -> identical scores

#[test]
fn test_same_seed_produces_identical_scores() {
    let data = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [50.0, 50.0]];

    let mut model_a = IsolationForest::new(30, 20).unwrap().with_random_state(13);
    model_a.fit(&data).unwrap();
    let scores_a = model_a.score_samples(&data).unwrap();

    let mut model_b = IsolationForest::new(30, 20).unwrap().with_random_state(13);
    model_b.fit(&data).unwrap();
    let scores_b = model_b.score_samples(&data).unwrap();

    assert_eq!(
        scores_a, scores_b,
        "two models with the same seed must produce identical scores"
    );
}

#[test]
fn test_different_seeds_may_produce_different_scores() {
    // distinct seeds with enough trees almost certainly differ on at least 1
    // sample. This guards against the seed being ignored
    let data = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [50.0, 50.0]];

    let mut model_a = IsolationForest::new(50, 32).unwrap().with_random_state(1);
    model_a.fit(&data).unwrap();
    let scores_a = model_a.score_samples(&data).unwrap();

    let mut model_b = IsolationForest::new(50, 32).unwrap().with_random_state(2);
    model_b.fit(&data).unwrap();
    let scores_b = model_b.score_samples(&data).unwrap();

    let any_differ = scores_a
        .iter()
        .zip(scores_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(
        any_differ,
        "different seeds should produce different scores (got identical scores for seeds 1 and 2)"
    );
}

// fit_predict convenience method

#[test]
fn test_fit_predict_matches_fit_then_predict() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];

    let mut model_a = IsolationForest::new(40, 32).unwrap().with_random_state(77);
    model_a.fit(&data).unwrap();
    let labels_a = model_a.predict(&data).unwrap();

    let mut model_b = IsolationForest::new(40, 32).unwrap().with_random_state(77);
    let labels_b = model_b.fit_predict(&data).unwrap();

    assert_eq!(
        labels_a, labels_b,
        "fit_predict must produce the same labels as fit + predict with the same seed"
    );
}

// Single-sample dataset

#[test]
fn test_fit_and_predict_on_single_sample() {
    // 1 training row gives a size-1 leaf with path length 0, so score = -(2^0) = -1.0
    let mut model = IsolationForest::new(5, 10).unwrap().with_random_state(1);
    let data = array![[3.0, 4.0]];
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();
    assert_eq!(scores.len(), 1);
    assert!(
        (scores[0] + 1.0).abs() < 1e-12,
        "single-sample score should be -1.0, got {}",
        scores[0]
    );
}

// n_features is set from data columns

#[test]
fn test_n_features_reflects_training_data_columns() {
    let mut model = IsolationForest::new(10, 20).unwrap().with_random_state(1);
    assert_eq!(model.get_n_features(), 0);

    let data = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
    model.fit(&data).unwrap();
    assert_eq!(model.get_n_features(), 5);
}

// max_samples clamped to nrows

#[test]
fn test_fit_with_fewer_rows_than_max_samples_succeeds() {
    // max_samples=256 but only 5 rows: forest builds on all 5 rows, no panic
    let mut model = IsolationForest::new(10, 256).unwrap().with_random_state(1);
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();
    assert_eq!(scores.len(), 5);
    for &s in scores.iter() {
        assert!((-1.0..0.0).contains(&s), "score {s} not in [0,1]");
    }
}

// Constant-feature column (all equal)

#[test]
fn test_fit_with_constant_feature_column_does_not_panic() {
    // column 1 is constant. The tree builder makes a leaf when max_val - min_val < 1e-10
    let mut model = IsolationForest::new(10, 32).unwrap().with_random_state(42);
    let data = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0]];
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();
    assert_eq!(scores.len(), 5);
}

// 1-D data (single feature)

#[test]
fn test_fit_and_predict_with_single_feature() {
    let mut model = IsolationForest::new(20, 32).unwrap().with_random_state(42);
    // values clustered near 0 except 1 far outlier at 999
    let data = array![[0.0], [0.1], [-0.1], [0.2], [999.0]];
    model.fit(&data).unwrap();
    assert_eq!(model.get_n_features(), 1);
    let scores = model.score_samples(&data).unwrap();
    assert_eq!(scores.len(), 5);
    // outlier (last point) should score below every inlier
    let outlier_score = scores[4];
    let min_inlier = scores.slice(s![..4]).fold(f64::INFINITY, |a, &v| a.min(v));
    assert!(
        outlier_score < min_inlier,
        "1-D outlier score {outlier_score:.4} should fall below inlier scores (min={min_inlier:.4})"
    );
}

// High-dimensional data

#[test]
fn test_fit_and_predict_high_dimensional() {
    // 5 features: scores stay in [-1,0) and the outlier is detected
    let mut model = IsolationForest::new(50, 32).unwrap().with_random_state(42);
    let mut data: Array2<f64> = Array2::zeros((7, 5));
    // 6 inliers near origin
    for i in 0..6 {
        for j in 0..5 {
            data[[i, j]] = 0.1 * i as f64;
        }
    }
    // 1 outlier at (100, 100, 100, 100, 100)
    for j in 0..5 {
        data[[6, j]] = 100.0;
    }
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();
    for &s in scores.iter() {
        assert!((-1.0..0.0).contains(&s));
    }
    let outlier_score = scores[6];
    let min_inlier = scores.slice(s![..6]).fold(f64::INFINITY, |a, &v| a.min(v));
    assert!(
        outlier_score < min_inlier,
        "high-dim outlier {outlier_score:.4} must fall below inlier min {min_inlier:.4}"
    );
}

// Save / load round-trip

#[test]
fn test_save_load_roundtrip_yields_identical_predictions() {
    let data = array![[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [50.0, 50.0]];

    let mut model = IsolationForest::new(30, 20).unwrap().with_random_state(99);
    model.fit(&data).unwrap();
    let scores_before = model.predict(&data).unwrap();

    let path = "/tmp/rustyml_isolation_forest_test.bin";
    model.save_to_path(path).unwrap();

    let loaded = IsolationForest::load_from_path(path).unwrap();
    let scores_after = loaded.predict(&data).unwrap();

    assert_eq!(
        scores_before, scores_after,
        "predictions must be identical before and after save/load"
    );

    // getters are preserved
    assert_eq!(loaded.get_n_estimators(), model.get_n_estimators());
    assert_eq!(loaded.get_max_samples(), model.get_max_samples());
    assert_eq!(loaded.get_max_depth(), model.get_max_depth());
    assert_eq!(loaded.get_n_features(), model.get_n_features());

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_load_from_nonexistent_path_returns_io_error() {
    let err =
        IsolationForest::load_from_path("/tmp/this_file_does_not_exist_rustyml.bin").unwrap_err();
    assert!(
        matches!(err, Error::Io(_)),
        "expected Io error when loading from missing file, got: {err:?}"
    );
}
// Closed-form anomaly score on identical points: every tree is 1 leaf, and the
// path-length correction cancels to score = -0.5, using c(n) = 2*H_{n-1} - 2(n-1)/n

/// max_samples == n_rows gives score = -2^(-c(max_samples)/c(max_samples)) = -0.5, exactly
#[test]
fn test_identical_points_score_equals_one_half_when_sample_size_equals_max_samples() {
    // 4 identical rows, max_samples = 4 == n_rows, so sample_size = min(4,4) = 4
    let data = array![[2.0, 7.0], [2.0, 7.0], [2.0, 7.0], [2.0, 7.0]];
    let mut model = IsolationForest::new(20, 4).unwrap().with_random_state(123);
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();

    // leaf size = max_samples = 4, so score = -2^(-c(4)/c(4)) = -0.5 for every row
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (s + 0.5).abs() < 1e-12,
            "row {i}: expected exactly -0.5, got {s}"
        );
    }
}

/// max_samples > n_rows makes sample_size = n_rows, so normalization must use
/// c(sample_size), not c(max_samples), giving score = -2^(-c(n_rows)/c(n_rows)) = -0.5
#[test]
fn test_identical_points_score_matches_closed_form_when_sample_size_below_max_samples() {
    // 4 identical rows, max_samples = 8 > 4, so sample_size = min(8,4) = 4 (leaf size = 4)
    let data = array![[1.0, -3.0], [1.0, -3.0], [1.0, -3.0], [1.0, -3.0]];
    let mut model = IsolationForest::new(25, 8).unwrap().with_random_state(7);
    model.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();

    // Path length c(4) is normalized by c(sample_size) = c(4): score = -2^(-c(4)/c(4)) = -0.5.
    // Normalizing by c(max_samples) = c(8) instead would wrongly yield about -0.6459, making
    // identical points look more anomalous than they are
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (s + 0.5).abs() < 1e-12,
            "row {i}: expected -0.5 (normalization uses c(sample_size), not c(max_samples)), got {s}"
        );
    }
}

/// A sample landing exactly on the cutoff is an inlier, matching scikit-learn's
/// `decision < 0 -> -1` rule
#[test]
fn samples_exactly_on_the_cutoff_are_inliers() {
    let data = array![[2.0, 7.0], [2.0, 7.0], [2.0, 7.0], [2.0, 7.0]];
    let mut model = IsolationForest::new(20, 4).unwrap().with_random_state(123);
    model.fit(&data).unwrap();

    // These identical rows score exactly -0.5 (see the closed-form tests above), which
    // equals the Auto cutoff, so every decision value is 0.0
    let decision = model.decision_function(&data).unwrap();
    for (i, &d) in decision.iter().enumerate() {
        assert!(d.abs() < 1e-12, "row {i}: decision should be 0.0, got {d}");
    }

    let labels = model.predict(&data).unwrap();
    for (i, &l) in labels.iter().enumerate() {
        assert_eq!(l, 1, "row {i}: a sample on the cutoff is an inlier");
    }
}

/// `predict` must accept inputs whose rows are not contiguous in memory (e.g. a
/// transposed view) without panicking
#[test]
fn predict_handles_non_contiguous_input() {
    // Train on ordinary contiguous data
    let train = array![[0.0, 0.0], [0.1, 0.1], [0.2, -0.1], [5.0, 5.0], [-4.0, 4.0]];
    let mut model = IsolationForest::new(20, 8).unwrap().with_random_state(1);
    model.fit(&train).unwrap();

    // A (features, samples) array transposed to (samples, features) has non-contiguous rows
    let ft = array![[0.0, 0.1, 5.0], [0.0, 0.1, 5.0]]; // shape (2, 3)
    let x = ft.t(); // shape (3, 2), rows stride across memory, so they are not contiguous
    assert!(
        x.row(0).as_slice().is_none(),
        "test setup: transposed rows must be non-contiguous to exercise the bug"
    );

    let result = model.predict(&x);
    assert!(
        result.is_ok(),
        "predict must handle non-contiguous input without panicking, got {result:?}"
    );
    assert_eq!(result.unwrap().len(), 3, "one score per input row");
}

// predict: contamination-based {-1, +1} outlier classification

/// A single obvious outlier among a tight inlier cluster is flagged -1, inliers +1.
/// With contamination 0.1 on 10 training samples, exactly ceil(1.0)=1 sample is flagged.
#[test]
fn predict_flags_obvious_outlier() {
    let x = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.05, 0.15],
        [0.15, 0.05],
        [0.1, 0.1],
        [10.0, 10.0] // obvious outlier
    ];
    let mut model = IsolationForest::new(100, 256)
        .unwrap()
        .with_contamination(Contamination::Fraction(0.1))
        .unwrap()
        .with_random_state(42);
    model.fit(&x).unwrap();

    let labels = model.predict(&x).unwrap();
    let n_out = labels.iter().filter(|&&l| l == -1).count();
    assert_eq!(
        n_out, 1,
        "contamination 0.1 on 10 samples flags exactly 1 outlier"
    );
    assert_eq!(labels[9], -1, "the far point must be the flagged outlier");
    for i in 0..9 {
        assert_eq!(labels[i], 1, "inlier {i} must be labelled +1");
    }
}

/// The number of flagged training samples equals ceil(contamination * n_train) when scores
/// are well-separated (continuous isolation scores make ties negligible)
#[test]
fn predict_count_matches_contamination() {
    let mut flat = Vec::new();
    for i in 0..20 {
        let v = i as f64;
        flat.push(v.sin());
        flat.push(v.cos() * 3.0);
    }
    let x = Array2::from_shape_vec((20, 2), flat).unwrap();
    let mut model = IsolationForest::new(100, 256)
        .unwrap()
        .with_contamination(Contamination::Fraction(0.25))
        .unwrap()
        .with_random_state(7);
    model.fit(&x).unwrap();

    let labels = model.predict(&x).unwrap();
    let n_out = labels.iter().filter(|&&l| l == -1).count();
    assert_eq!(
        n_out, 5,
        "ceil(0.25 * 20) = 5 outliers expected, got {n_out}"
    );
    // every label is exactly -1 or +1
    for &l in labels.iter() {
        assert!(l == -1 || l == 1, "label {l} not in {{-1, +1}}");
    }
}

/// The builder rejects a contamination fraction outside (0.0, 0.5] (or non-finite),
/// before any fitting happens
#[test]
fn with_contamination_rejects_invalid_fraction() {
    for bad in [0.0, -0.1, 0.51, 1.0, f64::NAN, f64::INFINITY] {
        let result = IsolationForest::new(10, 8)
            .unwrap()
            .with_contamination(Contamination::Fraction(bad));
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "contamination={bad} must be rejected"
        );
    }
    assert!(
        IsolationForest::new(10, 8)
            .unwrap()
            .with_contamination(Contamination::Fraction(0.25))
            .is_ok()
    );
}

/// The decision threshold is fitted model state. So a sample gets the same label whether
/// it is predicted alone, in a slice, or in the whole batch
#[test]
fn predict_label_is_independent_of_batching() {
    let x = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.05, 0.15],
        [0.15, 0.05],
        [0.1, 0.1],
        [10.0, 10.0]
    ];
    let mut model = IsolationForest::new(100, 256)
        .unwrap()
        .with_contamination(Contamination::Fraction(0.1))
        .unwrap()
        .with_random_state(42);
    model.fit(&x).unwrap();

    let whole = model.predict(&x).unwrap();

    // 1 row at a time must agree with the full-batch labeling
    for i in 0..x.nrows() {
        let row = x.slice(s![i..i + 1, ..]).to_owned();
        let single = model.predict(&row).unwrap();
        assert_eq!(
            single[0], whole[i],
            "sample {i} labelled {} alone but {} in the full batch",
            single[0], whole[i]
        );
    }

    // And so must an arbitrary split into 2 halves
    let first = model.predict(&x.slice(s![..4, ..]).to_owned()).unwrap();
    let second = model.predict(&x.slice(s![4.., ..]).to_owned()).unwrap();
    let halves: Vec<i32> = first.iter().chain(second.iter()).copied().collect();
    assert_eq!(
        halves,
        whole.to_vec(),
        "splitting the batch must not change any label"
    );
}

/// A lone inlier-looking sample is not automatically labeled an outlier
#[test]
fn predict_on_single_sample_is_not_forced_to_outlier() {
    let x = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.05, 0.15],
        [0.15, 0.05],
        [0.1, 0.1],
        [10.0, 10.0]
    ];
    let mut model = IsolationForest::new(100, 256)
        .unwrap()
        .with_contamination(Contamination::Fraction(0.1))
        .unwrap()
        .with_random_state(42);
    model.fit(&x).unwrap();

    let inlier = array![[0.1, 0.1]];
    assert_eq!(
        model.predict(&inlier).unwrap()[0],
        1,
        "a sample sitting in the middle of the training cluster must score as an inlier"
    );

    let outlier = array![[10.0, 10.0]];
    assert_eq!(
        model.predict(&outlier).unwrap()[0],
        -1,
        "the far point must still be an outlier on its own"
    );
}

// Contamination::Auto

/// Auto is the paper's fixed -0.5 cutoff, recorded on the model at fit time
#[test]
fn contamination_auto_uses_paper_threshold() {
    let x = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [10.0, 10.0]];
    let mut model = IsolationForest::new(50, 32).unwrap().with_random_state(3);
    assert_eq!(model.get_contamination(), Contamination::Auto);
    assert!(model.get_offset().is_none(), "offset is unset before fit");

    model.fit(&x).unwrap();
    let offset = model.get_offset().expect("offset is set by fit");
    assert!(
        (offset + 0.5).abs() < 1e-12,
        "Auto must resolve to the -0.5 cutoff, got {offset}"
    );

    // Labels are exactly the sign of the decision value, and a sample sitting on the cutoff
    // counts as an inlier
    let scores = model.score_samples(&x).unwrap();
    let decision = model.decision_function(&x).unwrap();
    let labels = model.predict(&x).unwrap();
    for (i, (&s, (&d, &l))) in scores
        .iter()
        .zip(decision.iter().zip(labels.iter()))
        .enumerate()
    {
        assert!(
            (d - (s - offset)).abs() < 1e-12,
            "sample {i}: decision {d} should be score {s} minus offset {offset}"
        );
        let expected = if d < 0.0 { -1 } else { 1 };
        assert_eq!(l, expected, "sample {i}: decision {d}");
    }
}

/// A Fraction offset is the NumPy-style percentile of the training scores, matching
/// scikit-learn's `offset_`
#[test]
fn contamination_fraction_offset_is_the_training_score_percentile() {
    let x = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.05, 0.15],
        [0.15, 0.05],
        [0.1, 0.1],
        [10.0, 10.0]
    ];
    let mut model = IsolationForest::new(100, 256)
        .unwrap()
        .with_contamination(Contamination::Fraction(0.2))
        .unwrap()
        .with_random_state(11);
    model.fit(&x).unwrap();

    let offset = model.get_offset().expect("offset is set by fit");
    let mut sorted = model.score_samples(&x).unwrap().to_vec();
    sorted.sort_unstable_by(f64::total_cmp);

    // n=10, c=0.2 gives position (10 - 1) * 0.2 = 1.8, a fractional index that interpolates
    // between the 2nd and 3rd smallest scores (indices 1 and 2)
    let expected = sorted[1] + 0.8 * (sorted[2] - sorted[1]);
    assert!(
        (offset - expected).abs() < 1e-12,
        "offset {offset} should be the interpolated 20th percentile {expected}"
    );
    assert!(
        offset > sorted[1] && offset < sorted[2],
        "offset {offset} should sit strictly between {} and {}",
        sorted[1],
        sorted[2]
    );
}

/// The offset survives a save/load round-trip, so a reloaded model labels identically
#[test]
fn offset_survives_save_load_round_trip() {
    let x = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.05, 0.15],
        [0.15, 0.05],
        [0.1, 0.1],
        [10.0, 10.0]
    ];
    let mut model = IsolationForest::new(60, 32)
        .unwrap()
        .with_contamination(Contamination::Fraction(0.3))
        .unwrap()
        .with_random_state(21);
    model.fit(&x).unwrap();

    let path = "/tmp/rustyml_test_isolation_forest_offset.bin";
    model.save_to_path(path).expect("save should succeed");
    let loaded = IsolationForest::load_from_path(path).expect("load should succeed");

    assert_eq!(loaded.get_contamination(), Contamination::Fraction(0.3));
    assert_eq!(loaded.get_offset(), model.get_offset());
    assert_eq!(
        loaded.predict(&x).unwrap(),
        model.predict(&x).unwrap(),
        "a reloaded model must label identically"
    );
}
