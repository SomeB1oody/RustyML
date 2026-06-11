//! Integration tests for [`rustyml::machine_learning::dbscan::DBSCAN`], covering
//! constructor validation, fit/predict error paths, cluster correctness across
//! metrics, edge cases, serialization, and the large-dataset parallel branches

use ndarray::{Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::DistanceCalculationMetric;
use rustyml::machine_learning::dbscan::DBSCAN;

// Helpers

/// Build the canonical 2-blob + 1-noise dataset used throughout these tests
///
/// Rows 0..=3 are blob A near (0,0), rows 4..=7 are blob B near (10,10), and row
/// 8 is isolated noise at (5,5). With eps=0.5 and min_samples=2, BFS top-to-bottom
/// assigns blob A cluster 0, blob B cluster 1, and the isolated point -1
fn two_blobs_noise() -> Array2<f64> {
    Array2::from_shape_vec(
        (9, 2),
        vec![
            // blob A
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, // blob B
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, // noise
            5.0, 5.0,
        ],
    )
    .unwrap()
}

// Constructor validation

/// eps = 0 is rejected
#[test]
fn constructor_rejects_eps_zero() {
    let result = DBSCAN::new(0.0, 2, DistanceCalculationMetric::Euclidean);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for eps=0, got: {:?}",
        result
    );
}

/// eps < 0 is rejected
#[test]
fn constructor_rejects_eps_negative() {
    let result = DBSCAN::new(-1.0, 2, DistanceCalculationMetric::Euclidean);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for eps=-1, got: {:?}",
        result
    );
}

/// eps = NaN is rejected
#[test]
fn constructor_rejects_eps_nan() {
    let result = DBSCAN::new(f64::NAN, 2, DistanceCalculationMetric::Euclidean);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for eps=NaN, got: {:?}",
        result
    );
}

/// eps = +inf is rejected
#[test]
fn constructor_rejects_eps_inf() {
    let result = DBSCAN::new(f64::INFINITY, 2, DistanceCalculationMetric::Euclidean);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for eps=inf, got: {:?}",
        result
    );
}

/// min_samples = 0 is rejected
#[test]
fn constructor_rejects_min_samples_zero() {
    let result = DBSCAN::new(0.5, 0, DistanceCalculationMetric::Euclidean);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for min_samples=0, got: {:?}",
        result
    );
}

/// Minkowski(0) is rejected (p must be > 0)
#[test]
fn constructor_rejects_minkowski_p_zero() {
    let result = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(0.0));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for Minkowski(0), got: {:?}",
        result
    );
}

/// Minkowski(-1) is rejected
#[test]
fn constructor_rejects_minkowski_p_negative() {
    let result = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(-1.0));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for Minkowski(-1), got: {:?}",
        result
    );
}

/// Minkowski(NaN) is rejected
#[test]
fn constructor_rejects_minkowski_p_nan() {
    let result = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(f64::NAN));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for Minkowski(NaN), got: {:?}",
        result
    );
}

/// Minkowski(+inf) is rejected
#[test]
fn constructor_rejects_minkowski_p_inf() {
    let result = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(f64::INFINITY));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for Minkowski(inf), got: {:?}",
        result
    );
}

/// Valid parameters are stored and readable via getters
#[test]
fn constructor_valid_stores_parameters() {
    let m = DBSCAN::new(0.75, 3, DistanceCalculationMetric::Manhattan).unwrap();
    approx::assert_abs_diff_eq!(m.get_epsilon(), 0.75, epsilon = 1e-12);
    assert_eq!(m.get_min_samples(), 3);
    assert_eq!(m.get_metric(), DistanceCalculationMetric::Manhattan);
}

/// Default constructor produces documented defaults (eps=0.5, min_samples=5, Euclidean)
#[test]
fn constructor_default_values() {
    let m = DBSCAN::default();
    approx::assert_abs_diff_eq!(m.get_epsilon(), 0.5, epsilon = 1e-12);
    assert_eq!(m.get_min_samples(), 5);
    assert_eq!(m.get_metric(), DistanceCalculationMetric::Euclidean);
    // No fitted state yet
    assert!(m.get_labels().is_none());
    assert!(m.get_core_sample_indices().is_none());
}

// fit() error paths

/// fit on a 0-row matrix returns EmptyInput
#[test]
fn fit_rejects_empty_data() {
    let data: Array2<f64> = Array2::zeros((0, 2));
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    assert!(
        matches!(m.fit(&data), Err(Error::EmptyInput(_))),
        "expected EmptyInput for 0-row data"
    );
}

/// fit on data containing NaN returns NonFinite
#[test]
fn fit_rejects_nan_in_data() {
    let data = array![[1.0f64, 2.0], [f64::NAN, 3.0]];
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    assert!(
        matches!(m.fit(&data), Err(Error::NonFinite(_))),
        "expected NonFinite for NaN in data"
    );
}

/// fit on data containing infinity returns NonFinite
#[test]
fn fit_rejects_inf_in_data() {
    let data = array![[1.0f64, 2.0], [f64::INFINITY, 3.0]];
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    assert!(
        matches!(m.fit(&data), Err(Error::NonFinite(_))),
        "expected NonFinite for infinity in data"
    );
}

// predict() error paths (before / without fit)

/// predict before fit returns NotFitted
#[test]
fn predict_before_fit_returns_not_fitted() {
    let m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    let data = array![[1.0f64, 2.0]];
    assert!(
        matches!(m.predict(&data), Err(Error::NotFitted(_))),
        "expected NotFitted before fit"
    );
}

/// predict on empty new_data returns Ok with an empty array
#[test]
fn predict_empty_new_data_returns_empty_array() {
    let train = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let empty: Array2<f64> = Array2::zeros((0, 2));
    let preds = m.predict(&empty).expect("expected Ok for empty new_data");
    assert_eq!(preds.len(), 0);
}

/// predict with wrong number of features returns DimensionMismatch
#[test]
fn predict_wrong_feature_count_returns_dimension_mismatch() {
    let train = two_blobs_noise(); // 2 features
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let wrong: Array2<f64> = Array2::zeros((3, 3)); // 3 features
    assert!(
        matches!(m.predict(&wrong), Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch for wrong feature count"
    );
}

/// predict with NaN in new_data returns NonFinite
#[test]
fn predict_nan_in_new_data_returns_non_finite() {
    let train = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let bad = array![[f64::NAN, 1.0f64]];
    assert!(
        matches!(m.predict(&bad), Err(Error::NonFinite(_))),
        "expected NonFinite for NaN in new_data"
    );
}

/// predict with infinity in new_data returns NonFinite
#[test]
fn predict_inf_in_new_data_returns_non_finite() {
    let train = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let bad = array![[f64::INFINITY, 1.0f64]];
    assert!(
        matches!(m.predict(&bad), Err(Error::NonFinite(_))),
        "expected NonFinite for infinity in new_data"
    );
}

// Correctness: fit + cluster structure (Euclidean metric)

/// Fitting the 2-blob+noise dataset labels each blob with a distinct non-negative
/// cluster (blob A -> 0, blob B -> 1) and the isolated point as noise (-1)
#[test]
fn fit_euclidean_correct_labels_two_blobs_noise() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let labels = m.get_labels().unwrap();
    assert_eq!(labels.len(), 9, "label count must equal n_samples");

    // Blob A: rows 0..=3 all share the same non-negative label
    let label_a = labels[0];
    assert!(label_a >= 0, "blob A must not be noise");
    assert_eq!(labels[1], label_a);
    assert_eq!(labels[2], label_a);
    assert_eq!(labels[3], label_a);

    // Blob B: rows 4..=7 all share the same non-negative label
    let label_b = labels[4];
    assert!(label_b >= 0, "blob B must not be noise");
    assert_eq!(labels[5], label_b);
    assert_eq!(labels[6], label_b);
    assert_eq!(labels[7], label_b);

    // Two blobs get distinct labels
    assert_ne!(
        label_a, label_b,
        "the two blobs must have different cluster labels"
    );

    // Noise point is -1
    assert_eq!(labels[8], -1, "isolated noise point must be labelled -1");

    // BFS processes rows in ascending order, so blob A is found first (cluster 0)
    // and blob B second (cluster 1)
    assert_eq!(label_a, 0, "blob A should be cluster 0 (discovered first)");
    assert_eq!(label_b, 1, "blob B should be cluster 1 (discovered second)");
}

/// Above 16 features DBSCAN falls back from the kd-tree to the brute-force scan; the
/// clustering result must remain correct on that path
#[test]
fn fit_high_dimensional_falls_back_to_brute_force() {
    let n_features = 18; // above the kd-tree dimensionality cutoff (16)
    // Two tight blobs (rows 0..=3 near 0, rows 4..=7 near 10) and one far noise point (row 8)
    let mut data = Array2::<f64>::zeros((9, n_features));
    for i in 0..4 {
        for j in 0..n_features {
            data[[i, j]] = 0.1 * i as f64;
        }
    }
    for i in 4..8 {
        for j in 0..n_features {
            data[[i, j]] = 10.0 + 0.1 * i as f64;
        }
    }
    for j in 0..n_features {
        data[[8, j]] = 100.0;
    }

    let mut m = DBSCAN::new(2.0, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();
    let labels = m.get_labels().unwrap();

    let label_a = labels[0];
    let label_b = labels[4];
    assert!(label_a >= 0 && label_b >= 0, "both blobs must be clustered");
    assert_ne!(label_a, label_b, "the two blobs must be distinct clusters");
    for i in 0..4 {
        assert_eq!(labels[i], label_a, "blob A row {i} mislabeled");
    }
    for i in 4..8 {
        assert_eq!(labels[i], label_b, "blob B row {i} mislabeled");
    }
    assert_eq!(labels[8], -1, "isolated point must be noise");
}

/// core_sample_indices are sorted ascending and carry only non-negative labels
#[test]
fn fit_euclidean_core_indices_sorted_and_non_noise() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let core_indices = m.get_core_sample_indices().unwrap();
    let labels = m.get_labels().unwrap();

    // Sorted ascending
    for window in core_indices.windows(2) {
        assert!(
            window[0] < window[1],
            "core_sample_indices must be sorted: {:?}",
            core_indices
        );
    }

    // Every core index has a non-negative label
    for &idx in core_indices.iter() {
        assert!(
            labels[idx] >= 0,
            "core point at index {} has label {} (noise)",
            idx,
            labels[idx]
        );
    }

    // With min_samples=2 on 4-point blobs all 8 blob points are core points;
    // the isolated noise point is not a core point
    assert_eq!(
        core_indices.len(),
        8,
        "all 8 blob points should be core points"
    );
}

// Correctness: predict on training data (self-consistency)

/// predict on the exact training rows of core points returns the same cluster
/// labels that fit stored
#[test]
fn predict_core_training_points_return_fit_labels() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let fit_labels = m.get_labels().unwrap();
    let core_indices = m.get_core_sample_indices().unwrap();

    // Build a subset array containing only the core-point rows
    let n_core = core_indices.len();
    let mut core_data = Array2::<f64>::zeros((n_core, 2));
    for (i, &idx) in core_indices.iter().enumerate() {
        core_data.row_mut(i).assign(&data.row(idx));
    }

    let preds = m.predict(&core_data).unwrap();
    for (i, &idx) in core_indices.iter().enumerate() {
        assert_eq!(
            preds[i], fit_labels[idx],
            "predict on core point {} (row {}) returned {} but fit label is {}",
            i, idx, preds[i], fit_labels[idx]
        );
    }
}

/// A new point inside blob A's territory receives blob A's label (0)
#[test]
fn predict_new_point_near_blob_a_returns_label_0() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    // (0.05, 0.05) is at Euclidean distance 0.07 from (0,0), well inside eps=0.5
    let new_point = array![[0.05f64, 0.05]];
    let preds = m.predict(&new_point).unwrap();
    assert_eq!(
        preds[0], 0,
        "point near blob A should be predicted as cluster 0"
    );
}

/// A new point inside blob B's territory receives blob B's label (1)
#[test]
fn predict_new_point_near_blob_b_returns_label_1() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    // (10.05, 10.05) is at Euclidean distance ~= 0.07 from (10,10), well inside eps=0.5
    let new_point = array![[10.05f64, 10.05]];
    let preds = m.predict(&new_point).unwrap();
    assert_eq!(
        preds[0], 1,
        "point near blob B should be predicted as cluster 1"
    );
}

/// A point farther from all core points than eps is predicted as noise (-1)
#[test]
fn predict_far_point_returns_noise() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    // (5, 5) is far from both blobs; Euclidean distance to nearest core point ~= 7.07
    let far_point = array![[5.0f64, 5.0]];
    let preds = m.predict(&far_point).unwrap();
    assert_eq!(
        preds[0], -1,
        "far-away point should be predicted as noise (-1)"
    );
}

/// A point at exactly eps Euclidean distance from the unique core point is assigned
/// to its cluster (the dist <= eps gate is inclusive)
#[test]
fn predict_point_at_eps_boundary_inclusive() {
    // Single-core dataset so the nearest (and only) core distance is unambiguous
    let train = array![[0.0f64, 0.0]];
    let mut m = DBSCAN::new(0.5, 1, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    // (0.5, 0.0): Euclidean distance to (0,0) = 0.5 exactly = eps; the dist <= eps
    // contract puts this in cluster 0
    let boundary_point = array![[0.5f64, 0.0]];
    let preds = m.predict(&boundary_point).unwrap();
    assert_eq!(
        preds[0], 0,
        "point at exactly eps from core (0,0) must be assigned to cluster 0"
    );
}

/// A point just beyond eps is labelled noise (single core (0,0), eps=0.5, query at 0.65)
#[test]
fn predict_point_just_beyond_eps_is_noise() {
    let train = array![[0.0f64, 0.0]];
    let mut m = DBSCAN::new(0.5, 1, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let beyond_point = array![[0.65f64, 0.0]];
    let preds = m.predict(&beyond_point).unwrap();
    assert_eq!(
        preds[0], -1,
        "point 0.65 away from core, beyond eps=0.5, must be -1"
    );
}

// Correctness: distance metrics - Euclidean, Manhattan, Minkowski(p)

/// Euclidean metric clusters the two blobs into 0, 1, and noise
#[test]
fn clustering_euclidean_metric_two_blobs_noise() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    let labels = m.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 9);
    // All blob A points are cluster 0
    assert!(labels.iter().take(4).all(|&l| l == 0));
    // All blob B points are cluster 1
    assert!(labels.iter().skip(4).take(4).all(|&l| l == 1));
    // Noise
    assert_eq!(labels[8], -1);
}

/// Manhattan metric clusters the two blobs the same way as Euclidean on this data
#[test]
fn clustering_manhattan_metric_two_blobs_noise() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Manhattan).unwrap();
    let labels = m.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 9);

    let label_a = labels[0];
    assert!(label_a >= 0);
    assert!(labels.iter().take(4).all(|&l| l == label_a));

    let label_b = labels[4];
    assert!(label_b >= 0);
    assert!(labels.iter().skip(4).take(4).all(|&l| l == label_b));

    assert_ne!(label_a, label_b);
    assert_eq!(labels[8], -1);
}

/// Minkowski(p=3) metric clusters the two blobs into distinct labels plus noise
#[test]
fn clustering_minkowski_p3_metric_two_blobs_noise() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(3.0)).unwrap();
    let labels = m.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 9);

    let label_a = labels[0];
    assert!(label_a >= 0);
    assert!(labels.iter().take(4).all(|&l| l == label_a));

    let label_b = labels[4];
    assert!(label_b >= 0);
    assert!(labels.iter().skip(4).take(4).all(|&l| l == label_b));

    assert_ne!(label_a, label_b);
    assert_eq!(labels[8], -1);
}

/// Minkowski(2) produces the same label vector as Euclidean (p=2 is L2)
#[test]
fn minkowski_p2_same_structure_as_euclidean() {
    let data = two_blobs_noise();

    let mut m_euc = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    let labels_euc = m_euc.fit_predict(&data).unwrap();

    let mut m_mink2 = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(2.0)).unwrap();
    let labels_mink2 = m_mink2.fit_predict(&data).unwrap();

    // Same noise and same cluster membership
    assert_eq!(
        labels_euc, labels_mink2,
        "Minkowski(2) must match Euclidean on the same data"
    );
}

/// Minkowski(1) produces the same label vector as Manhattan (p=1 is L1)
#[test]
fn minkowski_p1_same_structure_as_manhattan() {
    let data = two_blobs_noise();

    let mut m_man = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Manhattan).unwrap();
    let labels_man = m_man.fit_predict(&data).unwrap();

    let mut m_mink1 = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(1.0)).unwrap();
    let labels_mink1 = m_mink1.fit_predict(&data).unwrap();

    assert_eq!(
        labels_man, labels_mink1,
        "Minkowski(1) must match Manhattan on the same data"
    );
}

// Correctness: fit_predict consistency

/// fit_predict returns the same label array as fit + get_labels
#[test]
fn fit_predict_equals_fit_then_get_labels() {
    let data = two_blobs_noise();

    // Model A: fit_predict
    let mut m_a = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    let labels_fp = m_a.fit_predict(&data).unwrap();

    // Model B: fit, then get_labels
    let mut m_b = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m_b.fit(&data).unwrap();
    let labels_fit = m_b.get_labels().unwrap().clone();

    assert_eq!(
        labels_fp, labels_fit,
        "fit_predict must return the same labels as fit + get_labels"
    );
}

// Edge cases

/// A single-point dataset with min_samples=1 yields one core point and one cluster (0)
#[test]
fn single_point_min_samples_1_is_core_cluster_0() {
    let data = array![[1.0f64, 2.0]];
    let mut m = DBSCAN::new(0.5, 1, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let labels = m.get_labels().unwrap();
    let core_indices = m.get_core_sample_indices().unwrap();

    assert_eq!(labels.len(), 1);
    assert_eq!(
        labels[0], 0,
        "single point with min_samples=1 should be cluster 0"
    );
    assert_eq!(core_indices.len(), 1);
    assert_eq!(core_indices[0], 0);
}

/// When eps is tiny and min_samples is high, all points are noise (-1) and
/// core_sample_indices is empty
#[test]
fn all_noise_when_eps_tiny() {
    // 4 points spread 1 unit apart; with eps=0.01 and min_samples=3 no point
    // has enough neighbours, so all are noise
    let data = array![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    let mut m = DBSCAN::new(0.01, 3, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let labels = m.get_labels().unwrap();
    let core_indices = m.get_core_sample_indices().unwrap();

    assert!(
        labels.iter().all(|&l| l == -1),
        "all points should be noise; got: {:?}",
        labels
    );
    assert_eq!(
        core_indices.len(),
        0,
        "core_sample_indices must be empty when all are noise"
    );
}

/// When eps is large enough that every point neighbours every other, the result
/// is a single cluster (label 0 for all)
#[test]
fn all_connected_single_cluster() {
    // 4 points all within 0.5 of each other; eps=5.0 is more than sufficient
    let data = array![[0.0f64, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]];
    let mut m = DBSCAN::new(5.0, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let labels = m.get_labels().unwrap();
    assert!(
        labels.iter().all(|&l| l == 0),
        "all-connected data should produce a single cluster (label 0); got: {:?}",
        labels
    );
}

/// Default constructor produces a model that can be fitted without error
#[test]
fn default_constructor_model_is_usable() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::default();
    // The blobs have 4 points each, so with default min_samples=5 they miss the
    // core criterion; this only checks that fit succeeds, not the cluster shape
    assert!(
        m.fit(&data).is_ok(),
        "fit with default parameters should not error"
    );
}

// predict: nearest-wins semantics (not any-within-eps)

/// predict assigns the nearest core's label, not an arbitrary within-eps core:
/// with cores A=(0,0) and B=(2,0) both within eps of query (0.6,0), A wins (cluster 0)
#[test]
fn predict_assigns_nearest_core_label_not_arbitrary() {
    // Two isolated single-point clusters
    let train = array![[0.0f64, 0.0], [2.0, 0.0]];
    let mut m = DBSCAN::new(1.5, 1, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let labels = m.get_labels().unwrap();
    // Row 0 is cluster 0 (first discovered), row 1 is cluster 1
    assert_eq!(labels[0], 0);
    assert_eq!(labels[1], 1);

    // Q is nearer to A, so it gets cluster 0
    let query = array![[0.6f64, 0.0]];
    let preds = m.predict(&query).unwrap();
    assert_eq!(
        preds[0], 0,
        "nearest core (A) is cluster 0; Q should be predicted 0"
    );
}

/// predict labels a query beyond eps from the nearest core as noise (-1):
/// core A=(0,0), eps=1.5, query at (2,0) gives dist 2.0 > 1.5
#[test]
fn predict_nearest_core_outside_eps_returns_noise() {
    // Single core at origin; query sits 2 units away, beyond eps=1.5
    let train = array![[0.0f64, 0.0]];
    let mut m = DBSCAN::new(1.5, 1, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&train).unwrap();

    let query = array![[2.0f64, 0.0]];
    let preds = m.predict(&query).unwrap();
    assert_eq!(
        preds[0], -1,
        "nearest core is at dist=2.0 > eps=1.5; predict must return -1"
    );
}

// Serialization round-trip

/// save_to_path then load_from_path preserves all fitted state and produces
/// identical predictions on new data
#[test]
fn save_load_round_trip_preserves_state_and_predictions() {
    let data = two_blobs_noise();
    let mut original = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Manhattan).unwrap();
    original.fit(&data).unwrap();

    let path = "/tmp/rustyml_dbscan_test_roundtrip.json";
    original
        .save_to_path(path)
        .expect("save_to_path must succeed");

    let loaded = DBSCAN::load_from_path(path).expect("load_from_path must succeed");

    // Hyperparameters survive round-trip
    approx::assert_abs_diff_eq!(
        loaded.get_epsilon(),
        original.get_epsilon(),
        epsilon = 1e-12
    );
    assert_eq!(loaded.get_min_samples(), original.get_min_samples());
    assert_eq!(loaded.get_metric(), original.get_metric());

    // Fitted labels survive round-trip
    let orig_labels = original.get_labels().unwrap();
    let loaded_labels = loaded.get_labels().unwrap();
    assert_eq!(
        orig_labels, loaded_labels,
        "labels must survive serialization"
    );

    // Core indices survive round-trip
    let orig_indices = original.get_core_sample_indices().unwrap();
    let loaded_indices = loaded.get_core_sample_indices().unwrap();
    assert_eq!(
        orig_indices, loaded_indices,
        "core_sample_indices must survive serialization"
    );

    // Predictions on new data are identical between original and loaded model
    let new_points = array![[0.05f64, 0.05], [10.05, 10.05], [5.0, 5.0]];
    let preds_orig = original.predict(&new_points).unwrap();
    let preds_loaded = loaded.predict(&new_points).unwrap();
    assert_eq!(
        preds_orig, preds_loaded,
        "predict output must be identical after round-trip"
    );

    // Clean up
    let _ = std::fs::remove_file(path);
}

// Label domain: labels are isize, cluster ids >= 0, noise = -1

/// Every label emitted by fit is either -1 (noise) or a non-negative cluster id
#[test]
fn fit_labels_domain_correct() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let labels = m.get_labels().unwrap();
    for &l in labels.iter() {
        assert!(
            l >= -1,
            "label {} is outside valid domain (must be ≥ -1)",
            l
        );
    }
}

/// predict labels are in the same domain (-1 or non-negative cluster id)
#[test]
fn predict_labels_domain_correct() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let test_points = array![[0.05f64, 0.05], [10.05, 10.05], [5.0, 5.0]];
    let preds = m.predict(&test_points).unwrap();
    for &l in preds.iter() {
        assert!(l >= -1, "predicted label {} is outside valid domain", l);
    }
}

/// predict returns the correct label values for the canonical three-case scenario
#[test]
fn predict_label_values_canonical_three_cases() {
    let data = two_blobs_noise();
    let mut m = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    // Three test points with unambiguous labels: (0.05,0.05) in blob A -> 0,
    // (10.05,10.05) in blob B -> 1, and (5,5) far from all cores -> noise
    let test_points = array![[0.05f64, 0.05], [10.05, 10.05], [5.0, 5.0]];
    let preds = m.predict(&test_points).unwrap();

    assert_eq!(preds.len(), 3);
    assert_eq!(preds[0], 0, "point near blob A → cluster 0");
    assert_eq!(preds[1], 1, "point near blob B → cluster 1");
    assert_eq!(preds[2], -1, "isolated point → noise");
}

/// The metric stored after construction matches the requested variant for all
/// three enum arms (Euclidean, Manhattan, Minkowski)
#[test]
fn all_three_metric_variants_stored_correctly() {
    let m_euc = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
    let m_man = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Manhattan).unwrap();
    let m_mink = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Minkowski(4.0)).unwrap();

    assert_eq!(m_euc.get_metric(), DistanceCalculationMetric::Euclidean);
    assert_eq!(m_man.get_metric(), DistanceCalculationMetric::Manhattan);
    assert_eq!(
        m_mink.get_metric(),
        DistanceCalculationMetric::Minkowski(4.0)
    );
}
// Large-dataset parallel branch coverage (>= 1000 samples)

/// Build 1200 points (3 tight, well-separated blobs of 400 each) with deterministic
/// jitter and no RNG
///
/// Rows are blob-contiguous: 0..=399 around (0,0), 400..=799 around (50,0), and
/// 800..=1199 around (25,50). With eps=1.0 and min_samples=5 the within-blob spread
/// (~0.538) keeps each blob fully connected while the >= 49.6 inter-blob gap prevents
/// any crossing edge, so discovery order fixes blob 0 -> 0, blob 1 -> 1, blob 2 -> 2;
/// 1200 >= DBSCAN_PARALLEL_THRESHOLD (1000), so fit takes the parallel region_query branch
fn three_blobs_1200() -> Array2<f64> {
    let centers = [(0.0_f64, 0.0_f64), (50.0, 0.0), (25.0, 50.0)];
    let mut v = Vec::with_capacity(1200 * 2);
    for (cx, cy) in centers {
        for k in 0..400u32 {
            v.push(cx + ((k * 7) % 11) as f64 * 0.04 - 0.20);
            v.push(cy + ((k * 5) % 13) as f64 * 0.03 - 0.18);
        }
    }
    Array2::from_shape_vec((1200, 2), v).unwrap()
}

/// fit's parallel region-query branch on the 1200-point, 3-blob dataset yields
/// exactly 3 clusters, no noise, and blob i in cluster i
#[test]
fn fit_parallel_branch_three_blobs_1200_correct_structure() {
    let data = three_blobs_1200();
    assert_eq!(data.nrows(), 1200, "dataset must cross the 1000 threshold");

    let mut m = DBSCAN::new(1.0, 5, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    let labels = m.get_labels().unwrap();
    assert_eq!(labels.len(), 1200, "one label per sample");

    // No noise: every point is a core point of a fully-connected blob
    assert!(
        labels.iter().all(|&l| l >= 0),
        "no point should be noise on three dense, separated blobs"
    );

    // Exactly 3 distinct cluster ids
    let mut distinct: Vec<isize> = labels.iter().copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct,
        vec![0, 1, 2],
        "expected exactly clusters {{0,1,2}}, got {distinct:?}"
    );

    // Each blob is internally homogeneous and matches its discovery-order id
    for (blob, expected_id) in [(0usize, 0isize), (1, 1), (2, 2)] {
        let start = blob * 400;
        for i in start..start + 400 {
            assert_eq!(
                labels[i], expected_id,
                "row {i} (blob {blob}) should be cluster {expected_id}, got {}",
                labels[i]
            );
        }
    }

    // All 1200 points are core points (every blob point has >= min_samples neighbours)
    let core = m.get_core_sample_indices().unwrap();
    assert_eq!(core.len(), 1200, "all 1200 points should be core points");
}

/// predict's parallel branch on 1200 held-out points labels each blob-contiguous
/// block 0/1/2 and an isolated query at (100,100) as noise (-1)
#[test]
fn predict_parallel_branch_large_heldout_matches_blobs() {
    let data = three_blobs_1200();
    let mut m = DBSCAN::new(1.0, 5, DistanceCalculationMetric::Euclidean).unwrap();
    m.fit(&data).unwrap();

    // 1200 held-out points, blob-contiguous, distinct jitter formula
    let centers = [(0.0_f64, 0.0_f64), (50.0, 0.0), (25.0, 50.0)];
    let mut v = Vec::with_capacity(1200 * 2);
    for (cx, cy) in centers {
        for k in 0..400u32 {
            v.push(cx + ((k * 3) % 7) as f64 * 0.05 - 0.15);
            v.push(cy + ((k * 4) % 7) as f64 * 0.05 - 0.15);
        }
    }
    let held_out = Array2::from_shape_vec((1200, 2), v).unwrap();
    assert_eq!(
        held_out.nrows(),
        1200,
        "held-out set must cross the 1000 threshold"
    );

    let preds = m.predict(&held_out).unwrap();
    assert_eq!(preds.len(), 1200);
    for (block, expected_id) in [(0usize, 0isize), (1, 1), (2, 2)] {
        let start = block * 400;
        for i in start..start + 400 {
            assert_eq!(
                preds[i], expected_id,
                "held-out row {i} (block {block}) should predict cluster {expected_id}, got {}",
                preds[i]
            );
        }
    }

    // An isolated query far from every core is noise
    let far = array![[100.0f64, 100.0]];
    let far_pred = m.predict(&far).unwrap();
    assert_eq!(far_pred[0], -1, "point at (100,100) must be noise (-1)");
}
