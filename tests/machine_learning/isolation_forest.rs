//! Integration tests for IsolationForest: constructor validation, error paths,
//! score properties, outlier detection, determinism, and closed-form checks

use ndarray::{Array2, array, s};
use rustyml::error::Error;
use rustyml::machine_learning::IsolationForest;

// Constructor validation

#[test]
fn test_new_n_estimators_zero_returns_invalid_parameter() {
    let err = IsolationForest::new(0, 256, None, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { ref name, .. } if name == "n_estimators"),
        "expected InvalidParameter for n_estimators=0, got: {err:?}"
    );
}

#[test]
fn test_new_max_samples_zero_returns_invalid_parameter() {
    let err = IsolationForest::new(10, 0, None, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { ref name, .. } if name == "max_samples"),
        "expected InvalidParameter for max_samples=0, got: {err:?}"
    );
}

#[test]
fn test_new_max_depth_zero_returns_invalid_parameter() {
    let err = IsolationForest::new(10, 256, Some(0), Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { ref name, .. } if name == "max_depth"),
        "expected InvalidParameter for max_depth=Some(0), got: {err:?}"
    );
}

#[test]
fn test_new_valid_explicit_max_depth_succeeds() {
    let model = IsolationForest::new(20, 64, Some(5), Some(99)).unwrap();
    assert_eq!(model.get_n_estimators(), 20);
    assert_eq!(model.get_max_samples(), 64);
    assert_eq!(model.get_max_depth(), 5);
    assert_eq!(model.get_random_state(), Some(99));
    // not yet fitted
    assert_eq!(model.get_n_features(), 0);
    assert!(model.get_trees().is_none());
}

#[test]
fn test_new_auto_max_depth_ceil_log2_max_samples() {
    // max_samples = 256: ceil(log2(256)) = 8
    let model = IsolationForest::new(10, 256, None, Some(0)).unwrap();
    assert_eq!(model.get_max_depth(), 8);
}

#[test]
fn test_new_auto_max_depth_ceil_log2_non_power_of_two() {
    // max_samples = 100: ceil(log2(100)) = ceil(6.6439) = 7
    let model = IsolationForest::new(10, 100, None, Some(0)).unwrap();
    assert_eq!(model.get_max_depth(), 7);
}

#[test]
fn test_new_auto_max_depth_ceil_log2_two() {
    // max_samples = 2: ceil(log2(2)) = 1
    let model = IsolationForest::new(10, 2, None, Some(0)).unwrap();
    assert_eq!(model.get_max_depth(), 1);
}

#[test]
fn test_new_auto_max_depth_ceil_log2_one() {
    // max_samples = 1: ceil(log2(1)) = ceil(0.0) = 0, accepted because depth is
    // auto-computed (not an explicit Some(0))
    let model = IsolationForest::new(10, 1, None, Some(0)).unwrap();
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
    let model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let err = model.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("IsolationForest")),
        "expected NotFitted, got: {err:?}"
    );
}

#[test]
fn test_anomaly_score_before_fit_returns_not_fitted() {
    let model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let err = model.anomaly_score(&[1.0, 2.0]).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("IsolationForest")),
        "expected NotFitted, got: {err:?}"
    );
}

// fit error paths

#[test]
fn test_fit_empty_data_returns_empty_input() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let x: Array2<f64> = Array2::zeros((0, 2)); // 0 rows
    let err = model.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got: {err:?}"
    );
}

#[test]
fn test_fit_nan_returns_non_finite() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let x = array![[1.0, f64::NAN], [2.0, 3.0]];
    let err = model.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite for NaN, got: {err:?}"
    );
}

#[test]
fn test_fit_inf_returns_non_finite() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let x = array![[1.0, f64::INFINITY], [2.0, 3.0]];
    let err = model.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite for Inf, got: {err:?}"
    );
}

#[test]
fn test_fit_neg_inf_returns_non_finite() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let x = array![[1.0, f64::NEG_INFINITY], [2.0, 3.0]];
    let err = model.fit(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite for -Inf, got: {err:?}"
    );
}

// predict error paths (after fit)

#[test]
fn test_predict_empty_data_returns_empty_input() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    model.fit(&train).unwrap();

    let x_empty: Array2<f64> = Array2::zeros((0, 2)); // 0 rows
    let err = model.predict(&x_empty).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got: {err:?}"
    );
}

#[test]
fn test_predict_wrong_feature_count_returns_dimension_mismatch() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    model.fit(&train).unwrap();

    // training had 2 features; predict with 3
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
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    model.fit(&train).unwrap();

    let x_nan = array![[f64::NAN, 2.0]];
    let err = model.predict(&x_nan).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got: {err:?}"
    );
}

// anomaly_score error paths (after fit)

#[test]
fn test_anomaly_score_wrong_dim_returns_dimension_mismatch() {
    let mut model = IsolationForest::new(10, 50, None, Some(1)).unwrap();
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    model.fit(&train).unwrap();

    // training had 2 features; score with 3
    let err = model.anomaly_score(&[1.0, 2.0, 3.0]).unwrap_err();
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
    let mut model = IsolationForest::new(10, 50, None, Some(42)).unwrap();
    let train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    model.fit(&train).unwrap();
    assert_eq!(model.get_n_features(), 3);
}

#[test]
fn test_fit_stores_exactly_n_estimators_trees() {
    let n_estimators = 15_usize;
    let mut model = IsolationForest::new(n_estimators, 50, None, Some(42)).unwrap();
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

// Scores in [0, 1]

#[test]
fn test_predict_scores_are_in_unit_interval() {
    // anomaly scores must lie in [0, 1] by design: 2^(-E/c) with E, c > 0
    let mut model = IsolationForest::new(50, 64, None, Some(7)).unwrap();
    let train = array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.2, 0.2],
        [100.0, 100.0]
    ];
    model.fit(&train).unwrap();

    let scores = model.predict(&train).unwrap();
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&s),
            "score[{i}] = {s} is outside [0, 1]"
        );
    }
}

#[test]
fn test_anomaly_score_is_in_unit_interval() {
    let mut model = IsolationForest::new(50, 64, None, Some(7)).unwrap();
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [50.0, 50.0]];
    model.fit(&train).unwrap();

    let s_inlier = model.anomaly_score(&[0.0, 0.0]).unwrap();
    let s_outlier = model.anomaly_score(&[50.0, 50.0]).unwrap();

    assert!(
        (0.0..=1.0).contains(&s_inlier),
        "inlier score {s_inlier} not in [0,1]"
    );
    assert!(
        (0.0..=1.0).contains(&s_outlier),
        "outlier score {s_outlier} not in [0,1]"
    );
}

// Outlier detection correctness: anomalous points (far from the bulk) are
// isolated in fewer steps and so get higher anomaly scores than inliers

#[test]
fn test_outlier_score_exceeds_all_inlier_scores() {
    // 10 inliers clustered near (0, 0); 1 outlier far away at (1000, 1000)
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

    let mut model = IsolationForest::new(100, 64, None, Some(42)).unwrap();
    model.fit(&train_data).unwrap();
    let scores = model.predict(&train_data).unwrap();

    let outlier_score = scores[10];
    let max_inlier_score = scores
        .slice(s![..10])
        .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));

    assert!(
        outlier_score > max_inlier_score,
        "outlier score {outlier_score:.4} should exceed all inlier scores (max inlier: {max_inlier_score:.4})"
    );
}

#[test]
fn test_outlier_anomaly_score_exceeds_inlier_via_single_sample_api() {
    // same design as above, verified through the per-sample anomaly_score API
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

    let mut model = IsolationForest::new(100, 64, None, Some(42)).unwrap();
    model.fit(&train_data).unwrap();

    let outlier_score = model.anomaly_score(&[1000.0, 1000.0]).unwrap();
    let max_inlier_score = inlier_coords
        .iter()
        .map(|&(x, y)| model.anomaly_score(&[x, y]).unwrap())
        .fold(f64::NEG_INFINITY, f64::max);

    assert!(
        outlier_score > max_inlier_score,
        "outlier score {outlier_score:.4} should exceed max inlier score {max_inlier_score:.4}"
    );
}

// Identical points get identical scores

#[test]
fn test_identical_points_have_equal_scores() {
    // two identical rows must be scored identically under a seeded (deterministic) model
    let data = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [1.0, 2.0], // row 0 and row 2 are identical
        [5.0, 6.0]
    ];
    let mut model = IsolationForest::new(50, 32, None, Some(55)).unwrap();
    model.fit(&data).unwrap();
    let scores = model.predict(&data).unwrap();
    assert_eq!(
        scores[0], scores[2],
        "identical inputs must produce identical anomaly scores: {} vs {}",
        scores[0], scores[2]
    );
}

// Determinism: same seed => identical scores

#[test]
fn test_same_seed_produces_identical_scores() {
    let data = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [50.0, 50.0]];

    let mut model_a = IsolationForest::new(30, 20, None, Some(13)).unwrap();
    model_a.fit(&data).unwrap();
    let scores_a = model_a.predict(&data).unwrap();

    let mut model_b = IsolationForest::new(30, 20, None, Some(13)).unwrap();
    model_b.fit(&data).unwrap();
    let scores_b = model_b.predict(&data).unwrap();

    assert_eq!(
        scores_a, scores_b,
        "two models with the same seed must produce identical scores"
    );
}

#[test]
fn test_different_seeds_may_produce_different_scores() {
    // distinct seeds with enough trees almost certainly differ on at least one
    // sample; guards against the seed being ignored
    let data = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [50.0, 50.0]];

    let mut model_a = IsolationForest::new(50, 32, None, Some(1)).unwrap();
    model_a.fit(&data).unwrap();
    let scores_a = model_a.predict(&data).unwrap();

    let mut model_b = IsolationForest::new(50, 32, None, Some(2)).unwrap();
    model_b.fit(&data).unwrap();
    let scores_b = model_b.predict(&data).unwrap();

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

    let mut model_a = IsolationForest::new(40, 32, None, Some(77)).unwrap();
    model_a.fit(&data).unwrap();
    let scores_a = model_a.predict(&data).unwrap();

    let mut model_b = IsolationForest::new(40, 32, None, Some(77)).unwrap();
    let scores_b = model_b.fit_predict(&data).unwrap();

    assert_eq!(
        scores_a, scores_b,
        "fit_predict must produce the same scores as fit + predict with the same seed"
    );
}

// Single-sample dataset

#[test]
fn test_fit_and_predict_on_single_sample() {
    // one training row gives a size-1 leaf with path length 0, so score = 2^0 = 1.0
    let mut model = IsolationForest::new(5, 10, None, Some(1)).unwrap();
    let data = array![[3.0, 4.0]];
    model.fit(&data).unwrap();
    let scores = model.predict(&data).unwrap();
    assert_eq!(scores.len(), 1);
    assert!(
        (scores[0] - 1.0).abs() < 1e-12,
        "single-sample score should be 1.0, got {}",
        scores[0]
    );
}

// n_features is set from data columns

#[test]
fn test_n_features_reflects_training_data_columns() {
    let mut model = IsolationForest::new(10, 20, None, Some(1)).unwrap();
    assert_eq!(model.get_n_features(), 0); // before fit

    let data = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
    model.fit(&data).unwrap();
    assert_eq!(model.get_n_features(), 5); // after fit
}

// max_samples clamped to nrows

#[test]
fn test_fit_with_fewer_rows_than_max_samples_succeeds() {
    // max_samples=256 but only 5 rows: forest builds on all 5 rows, no panic
    let mut model = IsolationForest::new(10, 256, None, Some(1)).unwrap();
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    model.fit(&data).unwrap();
    let scores = model.predict(&data).unwrap();
    assert_eq!(scores.len(), 5);
    for &s in scores.iter() {
        assert!((0.0..=1.0).contains(&s), "score {s} not in [0,1]");
    }
}

// Constant-feature column (all equal)

#[test]
fn test_fit_with_constant_feature_column_does_not_panic() {
    // column 1 is constant; the tree builder makes a leaf when max_val - min_val < 1e-10
    let mut model = IsolationForest::new(10, 32, None, Some(42)).unwrap();
    let data = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0]];
    model.fit(&data).unwrap();
    let scores = model.predict(&data).unwrap();
    assert_eq!(scores.len(), 5);
}

// 1-D data (single feature)

#[test]
fn test_fit_and_predict_with_single_feature() {
    let mut model = IsolationForest::new(20, 32, None, Some(42)).unwrap();
    // values clustered near 0 except one far outlier at 999
    let data = array![[0.0], [0.1], [-0.1], [0.2], [999.0]];
    model.fit(&data).unwrap();
    assert_eq!(model.get_n_features(), 1);
    let scores = model.predict(&data).unwrap();
    assert_eq!(scores.len(), 5);
    // outlier (last point) should outscore the inliers
    let outlier_score = scores[4];
    let max_inlier = scores
        .slice(s![..4])
        .fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    assert!(
        outlier_score > max_inlier,
        "1-D outlier score {outlier_score:.4} should exceed inlier scores (max={max_inlier:.4})"
    );
}

// High-dimensional data

#[test]
fn test_fit_and_predict_high_dimensional() {
    // 5 features: scores stay in [0,1] and the outlier is detected
    let mut model = IsolationForest::new(50, 32, None, Some(42)).unwrap();
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
    let scores = model.predict(&data).unwrap();
    for &s in scores.iter() {
        assert!((0.0..=1.0).contains(&s));
    }
    let outlier_score = scores[6];
    let max_inlier = scores
        .slice(s![..6])
        .fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    assert!(
        outlier_score > max_inlier,
        "high-dim outlier {outlier_score:.4} must exceed inlier max {max_inlier:.4}"
    );
}

// Save / load round-trip

#[test]
fn test_save_load_roundtrip_yields_identical_predictions() {
    let data = array![[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [50.0, 50.0]];

    let mut model = IsolationForest::new(30, 20, None, Some(99)).unwrap();
    model.fit(&data).unwrap();
    let scores_before = model.predict(&data).unwrap();

    let path = "/tmp/rustyml_isolation_forest_test.json";
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

    // clean up
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_load_from_nonexistent_path_returns_io_error() {
    let err =
        IsolationForest::load_from_path("/tmp/this_file_does_not_exist_rustyml.json").unwrap_err();
    assert!(
        matches!(err, Error::Io(_)),
        "expected Io error when loading from missing file, got: {err:?}"
    );
}
// Closed-form anomaly score on identical points: every tree is one leaf, so the
// score is 2^(-c(sample_size)/c(max_samples)) with c(n) = 2*H_{n-1} - 2(n-1)/n

/// max_samples == n_rows gives score = 2^(-c(max_samples)/c(max_samples)) = 0.5, exactly
#[test]
fn test_identical_points_score_equals_one_half_when_sample_size_equals_max_samples() {
    // 4 identical rows; max_samples = 4 == n_rows, so sample_size = min(4,4) = 4
    let data = array![[2.0, 7.0], [2.0, 7.0], [2.0, 7.0], [2.0, 7.0]];
    let mut model = IsolationForest::new(20, 4, None, Some(123)).unwrap();
    model.fit(&data).unwrap();
    let scores = model.predict(&data).unwrap();

    // leaf size = max_samples = 4, so score = 2^(-c(4)/c(4)) = 0.5 for every row
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (s - 0.5).abs() < 1e-12,
            "row {i}: expected exactly 0.5, got {s}"
        );
    }
}

/// max_samples > n_rows: each tree is built on sample_size = n_rows points, so the score
/// normalization must use c(sample_size), NOT c(max_samples) (Liu et al. normalise by the
/// actual sub-sampling size). For identical points the leaf size equals sample_size, hence
/// score = 2^(-c(n_rows)/c(n_rows)) = 0.5, so identical points are not anomalies
#[test]
fn test_identical_points_score_matches_closed_form_when_sample_size_below_max_samples() {
    // 4 identical rows; max_samples = 8 > 4, so sample_size = min(8,4) = 4 (leaf size = 4)
    let data = array![[1.0, -3.0], [1.0, -3.0], [1.0, -3.0], [1.0, -3.0]];
    let mut model = IsolationForest::new(25, 8, None, Some(7)).unwrap();
    model.fit(&data).unwrap();
    let scores = model.predict(&data).unwrap();

    // Path length c(4) is normalised by c(sample_size) = c(4): score = 2^(-c(4)/c(4)) = 0.5.
    // Normalising by c(max_samples) = c(8) would wrongly yield ~0.6459, flagging identical
    // points as anomalous
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (s - 0.5).abs() < 1e-12,
            "row {i}: expected 0.5 (normalization uses c(sample_size), not c(max_samples)), got {s}"
        );
    }
}
