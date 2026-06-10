//! Integration tests for MeanShift clustering and estimate_bandwidth: constructor
//! validation, fit/predict correctness, cluster_all and bin_seeding behavior,
//! reproducibility, save/load, and edge cases

use approx::assert_abs_diff_eq;
use ndarray::{Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::mean_shift::{MeanShift, estimate_bandwidth};

// helpers

/// Two well-separated unit blobs of 5 points each, near (0,0) and (20,20); with
/// bandwidth=2.0 every intra-blob point sits inside the kernel window
fn two_blob_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            // Blob A near (0, 0)
            -0.1, 0.0, 0.1, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, // Blob B near (20, 20)
            19.9, 20.0, 20.1, 20.0, 20.0, 19.9, 20.0, 20.1, 20.0, 20.0,
        ],
    )
    .unwrap()
}

// constructor validation

#[test]
fn test_new_zero_bandwidth_is_invalid_parameter() {
    let result = MeanShift::new(0.0, None, None, None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "bandwidth=0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_negative_bandwidth_is_invalid_parameter() {
    let result = MeanShift::new(-1.0, None, None, None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "bandwidth=-1 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_inf_bandwidth_is_invalid_parameter() {
    let result = MeanShift::new(f64::INFINITY, None, None, None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "bandwidth=inf should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_nan_bandwidth_is_invalid_parameter() {
    let result = MeanShift::new(f64::NAN, None, None, None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "bandwidth=NaN should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_zero_max_iter_is_invalid_parameter() {
    let result = MeanShift::new(1.0, Some(0), None, None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "max_iter=0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_zero_tol_is_invalid_parameter() {
    let result = MeanShift::new(1.0, None, Some(0.0), None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol=0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_negative_tol_is_invalid_parameter() {
    let result = MeanShift::new(1.0, None, Some(-1e-6), None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol<0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_inf_tol_is_invalid_parameter() {
    let result = MeanShift::new(1.0, None, Some(f64::INFINITY), None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol=inf should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_nan_tol_is_invalid_parameter() {
    let result = MeanShift::new(1.0, None, Some(f64::NAN), None, None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol=NaN should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_new_valid_parameters_succeeds() {
    let ms = MeanShift::new(2.0, Some(100), Some(1e-4), Some(false), Some(true), None);
    assert!(ms.is_ok(), "valid parameters should succeed");
    let ms = ms.unwrap();
    assert_abs_diff_eq!(ms.get_bandwidth(), 2.0);
    assert_eq!(ms.get_max_iterations(), 100);
    assert_abs_diff_eq!(ms.get_tolerance(), 1e-4);
    assert!(!ms.get_bin_seeding());
    assert!(ms.get_cluster_all());
}

#[test]
fn test_default_constructor_values() {
    let ms = MeanShift::default();
    assert_abs_diff_eq!(ms.get_bandwidth(), 1.0);
    assert_eq!(ms.get_max_iterations(), 300);
    assert_abs_diff_eq!(ms.get_tolerance(), 1e-3);
    assert!(!ms.get_bin_seeding());
    assert!(ms.get_cluster_all());
}

// pre-fit state

#[test]
fn test_getters_return_none_before_fit() {
    let ms = MeanShift::default();
    assert!(ms.get_cluster_centers().is_none());
    assert!(ms.get_labels().is_none());
    assert!(ms.get_n_samples_per_center().is_none());
    assert!(ms.get_actual_iterations().is_none());
}

// predict errors before fitting

#[test]
fn test_predict_before_fit_returns_not_fitted() {
    let ms = MeanShift::default();
    let x = array![[1.0, 2.0]];
    let result = ms.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "predict before fit should return NotFitted, got {:?}",
        result
    );
}

#[test]
fn test_predict_wrong_feature_dimension_returns_dimension_mismatch() {
    let data = two_blob_data(); // 2 features
    let mut ms = MeanShift::new(2.0, None, None, Some(false), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    // Provide 3-feature points; the model was trained on 2-feature points
    let x_wrong = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 20.0, 20.0, 20.0]).unwrap();
    let result = ms.predict(&x_wrong);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "wrong feature dimension should return DimensionMismatch, got {:?}",
        result
    );
}

// fit / predict correctness on designed data

/// fit on two tight, well-separated blobs discovers exactly 2 centers numerically
/// close to the true blob means (0,0) and (20,20)
#[test]
fn test_fit_produces_two_centers_near_true_means() {
    let data = two_blob_data();
    // bin_seeding=true for deterministic seed selection at a fixed bandwidth
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let centers = ms.get_cluster_centers().unwrap();

    // Must discover exactly 2 clusters
    assert_eq!(
        centers.nrows(),
        2,
        "expected exactly 2 cluster centers, got {}",
        centers.nrows()
    );
    assert_eq!(centers.ncols(), 2, "centers must have 2 features");

    // Each center must be within 0.5 units of one true blob mean, matched by
    // whichever fitted center is closest
    let true_centers = [[0.0_f64, 0.0_f64], [20.0_f64, 20.0_f64]];
    for tc in &true_centers {
        let closest_dist = (0..centers.nrows())
            .map(|i| {
                let dx = centers[[i, 0]] - tc[0];
                let dy = centers[[i, 1]] - tc[1];
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest_dist < 0.5,
            "no fitted center is within 0.5 of true center {:?}; fitted centers:\n{:?}",
            tc,
            centers
        );
    }
}

/// fit labels group the 5 Blob-A samples together and the 5 Blob-B samples
/// together (cluster structure, not absolute label values)
#[test]
fn test_fit_labels_match_known_cluster_structure() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let labels = ms.get_labels().unwrap();
    assert_eq!(labels.len(), 10);

    // Samples 0-4 are Blob A, samples 5-9 are Blob B; same-blob samples share a label
    let label_a = labels[0];
    let label_b = labels[5];
    assert_ne!(label_a, label_b, "blobs must be assigned different labels");

    for i in 1..5 {
        assert_eq!(labels[i], label_a, "blob-A sample {} has wrong label", i);
    }
    for i in 6..10 {
        assert_eq!(labels[i], label_b, "blob-B sample {} has wrong label", i);
    }
}

/// predict assigns new points to the correct known cluster
#[test]
fn test_predict_assigns_points_to_correct_cluster() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let labels_train = ms.get_labels().unwrap().clone();
    let label_a = labels_train[0]; // label for blob-A region
    let label_b = labels_train[5]; // label for blob-B region

    // Two canonical predict points, one deep inside each blob
    let x_new = Array2::from_shape_vec(
        (2, 2),
        vec![
            0.05, 0.05, // clearly inside blob A
            19.95, 19.95, // clearly inside blob B
        ],
    )
    .unwrap();

    let preds = ms.predict(&x_new).unwrap();
    assert_eq!(preds.len(), 2);
    assert_eq!(
        preds[0], label_a,
        "point near (0,0) should get blob-A label"
    );
    assert_eq!(
        preds[1], label_b,
        "point near (20,20) should get blob-B label"
    );
}

/// fit_predict returns the same labels as the stored labels
#[test]
fn test_fit_predict_consistent_with_fit_then_labels() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    let labels_fp = ms.fit_predict(&data).unwrap();
    let labels_stored = ms.get_labels().unwrap();
    assert_eq!(labels_fp.len(), labels_stored.len());
    for (a, b) in labels_fp.iter().zip(labels_stored.iter()) {
        assert_eq!(a, b);
    }
}

// cluster_all=false outlier label

/// With cluster_all=false, a point farther than bandwidth from every fitted center
/// receives the label n_clusters (the documented outlier sentinel)
#[test]
fn test_cluster_all_false_outlier_label_is_n_clusters() {
    let data = two_blob_data(); // blobs at (0,0) and (20,20)
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(false), None).unwrap();
    ms.fit(&data).unwrap();

    let centers = ms.get_cluster_centers().unwrap();
    let n_clusters = centers.nrows(); // documented outlier sentinel

    // (10,10) is ~14 units from both blob centers, well beyond the bandwidth of 2.0
    let x_far = Array2::from_shape_vec((1, 2), vec![10.0, 10.0]).unwrap();
    let preds = ms.predict(&x_far).unwrap();

    assert_eq!(
        preds[0], n_clusters,
        "outlier point should receive label n_clusters={}, got {}",
        n_clusters, preds[0]
    );
}

/// With cluster_all=true (default) predict never produces the outlier sentinel
#[test]
fn test_cluster_all_true_never_produces_outlier_label() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let centers = ms.get_cluster_centers().unwrap();
    let n_clusters = centers.nrows();

    // Even a far-away point must be assigned a real cluster when cluster_all=true
    let x_far = Array2::from_shape_vec((1, 2), vec![10.0, 10.0]).unwrap();
    let preds = ms.predict(&x_far).unwrap();

    assert_ne!(
        preds[0], n_clusters,
        "cluster_all=true should never produce outlier label {}, got {}",
        n_clusters, preds[0]
    );
    // Must be a valid cluster index
    assert!(preds[0] < n_clusters);
}

/// fit labels obey the same invariant: no label equals n_clusters when cluster_all=true
#[test]
fn test_fit_labels_cluster_all_true_all_assigned() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let centers = ms.get_cluster_centers().unwrap();
    let n_clusters = centers.nrows();
    let labels = ms.get_labels().unwrap();

    for &l in labels.iter() {
        assert!(
            l < n_clusters,
            "cluster_all=true: label {} must be < n_clusters={}",
            l,
            n_clusters
        );
    }
}

// max_iter respected

#[test]
fn test_actual_iterations_bounded_by_max_iter() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(1), None, Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();
    let actual = ms.get_actual_iterations().unwrap();
    assert_eq!(actual, 1, "with max_iter=1, actual iterations must equal 1");
}

// n_samples_per_center semantics

/// n_samples_per_center counts converged seeds merged into each center (not points
/// assigned), so the counts sum to at most n_samples = 10
#[test]
fn test_n_samples_per_center_sums_equal_seeds_processed() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let counts = ms.get_n_samples_per_center().unwrap();
    let centers = ms.get_cluster_centers().unwrap();

    // There must be as many count entries as cluster centers
    assert_eq!(counts.len(), centers.nrows());

    // The total number of seed-center assignments must not exceed n_samples (10)
    let total: usize = counts.iter().sum();
    assert!(
        total <= 10,
        "seed count sum {} must be <= n_samples=10",
        total
    );
    assert!(total > 0, "at least one seed must have been processed");
}

// bin_seeding flag

#[test]
fn test_bin_seeding_produces_valid_centers() {
    let data = two_blob_data();
    let mut ms_bin =
        MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms_bin.fit(&data).unwrap();

    let centers = ms_bin.get_cluster_centers().unwrap();
    assert!(centers.nrows() >= 1);
    assert_eq!(centers.ncols(), 2);
    assert!(ms_bin.get_labels().is_some());
}

#[test]
fn test_no_bin_seeding_produces_valid_centers() {
    let data = two_blob_data();
    // bin_seeding=false keeps up to 100 shuffled points as seeds; with only 10 samples
    // the whole set is kept, so this just checks the non-bin_seeding branch is valid
    let mut ms_no_bin =
        MeanShift::new(2.0, Some(300), Some(1e-5), Some(false), Some(true), None).unwrap();
    ms_no_bin.fit(&data).unwrap();

    let centers = ms_no_bin.get_cluster_centers().unwrap();
    assert!(centers.nrows() >= 1);
    assert_eq!(centers.ncols(), 2);
}

/// 120 points in three well-separated blobs (40 each) with deterministic 2-D jitter;
/// above the 100-seed cap, the random_state-seeded shuffle decides which points seed
fn three_blobs_over_100() -> Array2<f64> {
    let centers = [(0.0_f64, 0.0_f64), (12.0, 0.0), (6.0, 11.0)];
    let mut v = Vec::with_capacity(120 * 2);
    for (cx, cy) in centers {
        for k in 0..40u32 {
            v.push(cx + ((k * 7) % 11) as f64 * 0.06 - 0.30);
            v.push(cy + ((k * 5) % 13) as f64 * 0.05 - 0.30);
        }
    }
    Array2::from_shape_vec((120, 2), v).unwrap()
}

/// With >100 samples (random 100-of-120 seed subset), two non-bin_seeding fits with
/// the same random_state produce identical centers and identical labels
#[test]
fn same_random_state_makes_non_bin_seeding_fit_reproducible() {
    let data = three_blobs_over_100();
    let run = |seed: u64| {
        let mut ms = MeanShift::new(
            2.0,
            Some(300),
            Some(1e-5),
            Some(false),
            Some(true),
            Some(seed),
        )
        .unwrap();
        ms.fit(&data).unwrap();
        (
            ms.get_cluster_centers().unwrap().clone(),
            ms.get_labels().unwrap().clone(),
        )
    };
    let (c1, l1) = run(42);
    let (c2, l2) = run(42);
    assert_eq!(
        c1.shape(),
        c2.shape(),
        "same seed must yield the same number of clusters"
    );
    crate::common::assert_allclose(&c1, &c2, 1e-12);
    assert_eq!(l1, l2, "same seed must yield identical labels");
}

// estimate_bandwidth

#[test]
fn test_estimate_bandwidth_quantile_zero_is_invalid() {
    let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
    let result = estimate_bandwidth(&x, Some(0.0), None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "quantile=0.0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_estimate_bandwidth_quantile_one_is_invalid() {
    let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
    let result = estimate_bandwidth(&x, Some(1.0), None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "quantile=1.0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_estimate_bandwidth_quantile_negative_is_invalid() {
    let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
    let result = estimate_bandwidth(&x, Some(-0.1), None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "quantile<0 should return InvalidParameter, got {:?}",
        result
    );
}

#[test]
fn test_estimate_bandwidth_quantile_greater_than_one_is_invalid() {
    let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
    let result = estimate_bandwidth(&x, Some(1.5), None, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "quantile>1 should return InvalidParameter, got {:?}",
        result
    );
}

/// Two-point dataset has a single pair, so estimate_bandwidth returns that pair's
/// distance (5.0) for any quantile
#[test]
fn test_estimate_bandwidth_two_point_known_distance() {
    // Distance between (0,0) and (3,4) = sqrt(9+16) = sqrt(25) = 5.0 (exact)
    let x = Array2::from_shape_vec((2, 2), vec![0.0_f64, 0.0, 3.0, 4.0]).unwrap();

    // Any quantile in (0,1) returns 5.0 because there is only one pair
    let bw = estimate_bandwidth(&x, Some(0.3), None, Some(42)).unwrap();
    assert_abs_diff_eq!(bw, 5.0, epsilon = 1e-10);

    let bw_high = estimate_bandwidth(&x, Some(0.9), None, Some(42)).unwrap();
    assert_abs_diff_eq!(bw_high, 5.0, epsilon = 1e-10);
}

/// estimate_bandwidth returns a positive value for typical data
#[test]
fn test_estimate_bandwidth_returns_positive() {
    let data = two_blob_data();
    let bw = estimate_bandwidth(&data, Some(0.3), None, Some(42)).unwrap();
    assert!(bw > 0.0, "bandwidth estimate must be positive, got {}", bw);
}

/// estimate_bandwidth is deterministic: the same seed gives an identical result
#[test]
fn test_estimate_bandwidth_deterministic_with_seed() {
    let data = two_blob_data();
    let bw1 = estimate_bandwidth(&data, Some(0.5), Some(6), Some(99)).unwrap();
    let bw2 = estimate_bandwidth(&data, Some(0.5), Some(6), Some(99)).unwrap();
    assert_abs_diff_eq!(bw1, bw2, epsilon = 1e-15);
}

/// estimate_bandwidth with n_samples larger than n_rows falls back to all rows without panic
#[test]
fn test_estimate_bandwidth_n_samples_larger_than_rows() {
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.866]).unwrap();
    // Requesting 1000 samples when only 3 exist uses all 3, not a panic
    let result = estimate_bandwidth(&x, Some(0.5), Some(1000), Some(42));
    assert!(result.is_ok());
    assert!(result.unwrap() > 0.0);
}

// fit on minimal (single-point) dataset

/// A single-point dataset produces one cluster center equal to that point
#[test]
fn test_fit_single_point_produces_one_center() {
    let x = Array2::from_shape_vec((1, 2), vec![3.0_f64, 7.0]).unwrap();
    let mut ms = MeanShift::new(1.0, None, None, None, None, None).unwrap();
    ms.fit(&x).unwrap();

    let centers = ms.get_cluster_centers().unwrap();
    assert_eq!(centers.nrows(), 1, "single-point fit must yield one center");
    assert_abs_diff_eq!(centers[[0, 0]], 3.0, epsilon = 1e-9);
    assert_abs_diff_eq!(centers[[0, 1]], 7.0, epsilon = 1e-9);

    let labels = ms.get_labels().unwrap();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0], 0);
}

// fit on empty data

#[test]
fn test_fit_empty_data_returns_error() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let mut ms = MeanShift::default();
    let result = ms.fit(&x);
    assert!(
        result.is_err(),
        "fit on empty data should return Err, got Ok"
    );
    // The error should be EmptyInput or InvalidInput (preliminary_check)
    let is_expected = matches!(
        result,
        Err(Error::EmptyInput(_)) | Err(Error::InvalidInput(_))
    );
    assert!(
        is_expected,
        "expected EmptyInput or InvalidInput, got {:?}",
        result
    );
}

// save / load round-trip

#[test]
fn test_save_load_round_trip_identical_predictions() {
    let data = two_blob_data();
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let path = "/tmp/rustyml_mean_shift_test.json";
    ms.save_to_path(path).expect("save_to_path should succeed");

    let loaded = MeanShift::load_from_path(path).expect("load_from_path should succeed");

    // Both must predict identically on the training data
    let preds_original = ms.predict(&data).unwrap();
    let preds_loaded = loaded.predict(&data).unwrap();

    assert_eq!(
        preds_original.len(),
        preds_loaded.len(),
        "original and loaded model must produce same-length predictions"
    );
    for (i, (a, b)) in preds_original.iter().zip(preds_loaded.iter()).enumerate() {
        assert_eq!(
            a, b,
            "prediction mismatch at sample {}: original={}, loaded={}",
            i, a, b
        );
    }

    // Cluster centers must be identical after round-trip
    let centers_original = ms.get_cluster_centers().unwrap();
    let centers_loaded = loaded.get_cluster_centers().unwrap();
    crate::common::assert_allclose(centers_original, centers_loaded, 1e-15);

    // Clean up temp file
    let _ = std::fs::remove_file(path);
}

// determinism of estimate_bandwidth with fixed seed

/// estimate_bandwidth with the same seed and n_samples returns exactly the same
/// value (random_state wires through to StdRng)
#[test]
fn test_estimate_bandwidth_same_seed_same_result() {
    let data = two_blob_data();
    // n_samples smaller than n_rows to exercise the sampling path
    let bw1 = estimate_bandwidth(&data, Some(0.5), Some(5), Some(77)).unwrap();
    let bw2 = estimate_bandwidth(&data, Some(0.5), Some(5), Some(77)).unwrap();
    assert_abs_diff_eq!(bw1, bw2, epsilon = 1e-15);
    assert!(bw1 > 0.0);
}

/// estimate_bandwidth with different seeds (sampling 5 of 10 rows) yields positive
/// values; a probabilistic sanity check asserting only positivity
#[test]
fn test_estimate_bandwidth_different_seeds_both_positive() {
    let data = two_blob_data();
    let bw_a = estimate_bandwidth(&data, Some(0.5), Some(5), Some(1)).unwrap();
    let bw_b = estimate_bandwidth(&data, Some(0.5), Some(5), Some(2)).unwrap();
    assert!(bw_a > 0.0);
    assert!(bw_b > 0.0);
}

// reproducibility of fit: with bin_seeding disabled, `fit` selects up to 100 random
// seed points, so 150 points exercise random_state on a genuine random subset

/// 150 deterministic points across three well-separated blobs (no RNG in the test)
fn three_blob_data() -> Array2<f64> {
    let n = 150;
    let mut x = Array2::zeros((n, 2));
    for i in 0..n {
        let (cx, cy) = match i % 3 {
            0 => (0.0, 0.0),
            1 => (30.0, 0.0),
            _ => (0.0, 30.0),
        };
        x[[i, 0]] = cx + (i as f64 * 0.011).sin() * 0.5;
        x[[i, 1]] = cy + (i as f64 * 0.017).cos() * 0.5;
    }
    x
}

/// Two fits with the same seed (and bin_seeding = false) produce identical labels
/// and cluster centers
#[test]
fn test_fit_same_seed_is_reproducible() {
    let data = three_blob_data();
    let fit = |seed| {
        let mut ms = MeanShift::new(
            5.0,
            Some(300),
            Some(1e-4),
            Some(false),
            Some(true),
            Some(seed),
        )
        .unwrap();
        ms.fit(&data).unwrap();
        ms
    };
    let a = fit(123);
    let b = fit(123);
    assert_eq!(
        a.get_labels().unwrap(),
        b.get_labels().unwrap(),
        "identical seed must yield identical labels"
    );
    crate::common::assert_allclose(
        a.get_cluster_centers().unwrap(),
        b.get_cluster_centers().unwrap(),
        1e-15,
    );
}

/// With bin_seeding = true (no RNG), two fits agree exactly across runs despite
/// AHashMap iteration order and the per-cell representative choice
#[test]
fn test_bin_seeding_is_deterministic_across_runs() {
    let data = three_blob_data();
    let fit = || {
        let mut ms =
            MeanShift::new(5.0, Some(300), Some(1e-4), Some(true), Some(true), None).unwrap();
        ms.fit(&data).unwrap();
        ms
    };
    let a = fit();
    let b = fit();
    assert_eq!(
        a.get_labels().unwrap(),
        b.get_labels().unwrap(),
        "bin_seeding must be deterministic across runs"
    );
    crate::common::assert_allclose(
        a.get_cluster_centers().unwrap(),
        b.get_cluster_centers().unwrap(),
        1e-15,
    );
}
// large-dataset parallel seed-loop + label branch: `fit` parallelizes only when
// n_samples > MEANSHIFT_PARALLEL_THRESHOLD (1000), so 1200 points hit both branches

/// 1200 points: three tight, well-separated blobs (400 each) with deterministic jitter
/// around centers (0,0), (10,0), (5,10), separation far exceeding intra-blob spread
fn three_blobs_1200() -> Array2<f64> {
    let centers = [(0.0_f64, 0.0_f64), (10.0, 0.0), (5.0, 10.0)];
    let mut v = Vec::with_capacity(1200 * 2);
    for (cx, cy) in centers {
        for k in 0..400u32 {
            v.push(cx + ((k * 7) % 11) as f64 * 0.04 - 0.20);
            v.push(cy + ((k * 5) % 13) as f64 * 0.03 - 0.18);
        }
    }
    Array2::from_shape_vec((1200, 2), v).unwrap()
}

/// fit on 1200 points (> 1000, parallel seed-loop and label branch) recovers exactly
/// 3 centers each within 0.5 of a true mean and groups the three 400-point blocks distinctly
#[test]
fn test_fit_parallel_branch_three_blobs_1200() {
    let data = three_blobs_1200();
    assert_eq!(data.nrows(), 1200, "dataset must exceed the 1000 threshold");

    // bin_seeding=true gives deterministic seeds; bandwidth=2.0 separates the blobs
    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    ms.fit(&data).unwrap();

    let centers = ms.get_cluster_centers().unwrap();
    assert_eq!(
        centers.nrows(),
        3,
        "expected exactly 3 cluster centers, got {}",
        centers.nrows()
    );
    assert_eq!(centers.ncols(), 2);

    // Each true mean must have a fitted center within 0.5
    let true_centers = [[0.0_f64, 0.0_f64], [10.0, 0.0], [5.0, 10.0]];
    for tc in &true_centers {
        let closest = (0..centers.nrows())
            .map(|i| {
                let dx = centers[[i, 0]] - tc[0];
                let dy = centers[[i, 1]] - tc[1];
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest < 0.5,
            "no fitted center within 0.5 of true center {:?}; fitted:\n{:?}",
            tc,
            centers
        );
    }

    // Labels (parallel map_collect at this size) must respect the blob blocks
    let labels = ms.get_labels().unwrap();
    assert_eq!(labels.len(), 1200);
    let block_label = [labels[0], labels[400], labels[800]];
    assert_ne!(block_label[0], block_label[1]);
    assert_ne!(block_label[0], block_label[2]);
    assert_ne!(block_label[1], block_label[2]);
    for (blk, &lab) in block_label.iter().enumerate() {
        let start = blk * 400;
        for i in start..start + 400 {
            assert_eq!(
                labels[i], lab,
                "row {i} (blob {blk}) should share label {lab}, got {}",
                labels[i]
            );
        }
    }
}
