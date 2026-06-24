//! Integration tests for `KMeans` clustering: constructor and fit/predict error
//! paths, correctness on designed blob data, inertia invariants, determinism, and
//! save/load round-trips

use crate::common::assert_allclose;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::KMeans;

// Helpers

/// Build 3 tight, well-separated blobs of 5 points each, centred at (0,0),
/// (10,0), (5,10) with per-coordinate offset in [-0.05, +0.05]
///
/// Inter-blob separation (>= 9.9) far exceeds intra-blob spread (<= 0.14), so any
/// correct k=3 run assigns each blob to one cluster with centroid within 0.1 of the
/// true mean
fn three_blob_data() -> Array2<f64> {
    // 15 rows x 2 columns; deterministic offsets keep blobs tight
    #[rustfmt::skip]
    let data = array![
        // Blob 0: true centre (0.0, 0.0)
        [-0.05,  0.03],
        [ 0.04, -0.02],
        [ 0.01,  0.05],
        [-0.03, -0.04],
        [ 0.02,  0.01],
        // Blob 1: true centre (10.0, 0.0)
        [ 9.95,  0.03],
        [10.04, -0.02],
        [10.01,  0.05],
        [ 9.97, -0.04],
        [10.02,  0.01],
        // Blob 2: true centre (5.0, 10.0)
        [ 4.95, 10.03],
        [ 5.04,  9.98],
        [ 5.01, 10.05],
        [ 4.97,  9.96],
        [ 5.02, 10.01],
    ];
    data
}

/// Verify that all points within each blob share one label and that blobs get
/// distinct labels
fn assert_blob_structure(labels: &Array1<usize>, blob_size: usize) {
    let n_blobs = labels.len() / blob_size;
    assert_eq!(
        labels.len(),
        n_blobs * blob_size,
        "labels length must be a multiple of blob_size"
    );

    // Representative label per blob (label of its first point)
    let blob_labels: Vec<usize> = (0..n_blobs).map(|b| labels[b * blob_size]).collect();

    // All points within each blob must share that blob's label
    for b in 0..n_blobs {
        for i in 0..blob_size {
            assert_eq!(
                labels[b * blob_size + i],
                blob_labels[b],
                "blob {b} point {i} has label {} but expected {}",
                labels[b * blob_size + i],
                blob_labels[b],
            );
        }
    }

    // Each blob must have a distinct label
    for i in 0..n_blobs {
        for j in (i + 1)..n_blobs {
            assert_ne!(
                blob_labels[i], blob_labels[j],
                "blobs {i} and {j} share label {}",
                blob_labels[i],
            );
        }
    }
}

// Constructor validation

/// Each invalid constructor argument must yield `Error::InvalidParameter`. Covers
/// `n_clusters = 0`, `max_iterations = 0`, and `tolerance` rejected for
/// `0.0`, negative, `NaN`, and `+Inf` (every other arg held at a valid value).
#[test]
fn constructor_invalid_params_are_rejected() {
    // (n_clusters, max_iterations, tolerance, label)
    let cases: [(usize, usize, f64, &str); 6] = [
        (0, 100, 1e-4, "n_clusters=0"),
        (3, 0, 1e-4, "max_iterations=0"),
        (3, 100, 0.0, "tolerance=0.0"),
        (3, 100, -1e-4, "tolerance=-1e-4"),
        (3, 100, f64::NAN, "tolerance=NaN"),
        (3, 100, f64::INFINITY, "tolerance=+Inf"),
    ];
    for (n_clusters, max_iter, tol, label) in cases {
        let err = KMeans::new(n_clusters, max_iter, tol).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for {label}, got {err:?}"
        );
    }
}

/// Valid parameters succeed and getters reflect the stored values
#[test]
fn constructor_valid_stores_params() {
    let km = KMeans::new(3, 200, 1e-4).unwrap().with_random_state(42);
    assert_eq!(km.get_n_clusters(), 3);
    assert_eq!(km.get_max_iterations(), 200);
    assert_abs_diff_eq!(km.get_tolerance(), 1e-4, epsilon = 1e-15);
    assert_eq!(km.get_random_state(), Some(42));
    // Before fit, post-fit getters return None
    assert!(km.get_centroids().is_none());
    assert!(km.get_labels().is_none());
    assert!(km.get_inertia().is_none());
    assert!(km.get_actual_iterations().is_none());
}

/// `KMeans::default()` produces a valid instance with documented defaults
#[test]
fn default_constructor_has_documented_defaults() {
    let km = KMeans::default();
    assert_eq!(km.get_n_clusters(), 8);
    assert_eq!(km.get_max_iterations(), 300);
    assert_abs_diff_eq!(km.get_tolerance(), 1e-4, epsilon = 1e-15);
    assert_eq!(km.get_random_state(), None);
}

// fit error paths

/// `fit` on an empty array must yield `Error::EmptyInput`
#[test]
fn fit_empty_data_is_error() {
    let mut km = KMeans::new(1, 100, 1e-4).unwrap().with_random_state(0);
    let data: Array2<f64> = Array2::zeros((0, 2));
    let err = km.fit(&data).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// `fit` on data containing a non-finite sentinel (NaN or `+Inf`) must yield
/// `Error::NonFinite` for each sentinel
#[test]
fn fit_non_finite_data_is_error() {
    for sentinel in [f64::NAN, f64::INFINITY] {
        let mut km = KMeans::new(1, 100, 1e-4).unwrap().with_random_state(0);
        let data = array![[1.0, sentinel], [2.0, 3.0]];
        let err = km.fit(&data).unwrap_err();
        assert!(
            matches!(err, Error::NonFinite(_)),
            "expected NonFinite for sentinel={sentinel:?}, got {err:?}"
        );
    }
}

/// `fit` with `n_samples < n_clusters` must yield `Error::InvalidInput`
#[test]
fn fit_fewer_samples_than_clusters_is_error() {
    let mut km = KMeans::new(5, 100, 1e-4).unwrap().with_random_state(0);
    // 3 samples < 5 clusters
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let err = km.fit(&data).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// predict error paths

/// `predict` before `fit` must yield `Error::NotFitted`
#[test]
fn predict_before_fit_is_not_fitted() {
    let km = KMeans::new(3, 100, 1e-4).unwrap().with_random_state(42);
    let data = array![[1.0, 2.0]];
    let err = km.predict(&data).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {err:?}"
    );
}

/// `predict` with wrong number of features must yield `Error::DimensionMismatch`
#[test]
fn predict_wrong_feature_count_is_dimension_mismatch() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let data = three_blob_data(); // shape (15, 2)
    km.fit(&data).unwrap();

    // Feed a 3-feature matrix to a model trained on 2 features
    let bad = array![[1.0, 2.0, 3.0]];
    let err = km.predict(&bad).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

/// `predict` with an empty matrix (after fit) must yield `Error::EmptyInput`
#[test]
fn predict_empty_input_after_fit_is_empty_input() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km.fit(&three_blob_data()).unwrap();

    let empty: Array2<f64> = Array2::zeros((0, 2));
    let err = km.predict(&empty).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// `predict` with NaN input (after fit) must yield `Error::NonFinite`
#[test]
fn predict_nan_input_is_non_finite() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let bad = array![[f64::NAN, 1.0]];
    let err = km.predict(&bad).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

// Correctness on designed data

/// After fitting k=3 on `three_blob_data()`, each blob's true mean has a matching
/// centroid within tol 0.1 (cluster index ordering is arbitrary)
#[test]
fn fit_k3_centroids_near_true_blob_means() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let centroids = km.get_centroids().unwrap();
    assert_eq!(centroids.nrows(), 3, "should have 3 centroids");
    assert_eq!(centroids.ncols(), 2, "centroids should be 2-dimensional");

    let true_means: [[f64; 2]; 3] = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]];
    let tol = 0.1_f64;

    for &[tx, ty] in &true_means {
        let matched = (0..3).any(|i| {
            let cx = centroids[(i, 0)];
            let cy = centroids[(i, 1)];
            (cx - tx).abs() < tol && (cy - ty).abs() < tol
        });
        assert!(
            matched,
            "no centroid within tol={tol} of true mean ({tx}, {ty}); centroids = {centroids:?}"
        );
    }
}

/// After fitting, `get_labels` respects blob structure: each blob's 5 points share
/// one label and the three blobs get distinct labels
#[test]
fn fit_k3_labels_respect_blob_structure() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let labels = km.get_labels().unwrap();
    assert_eq!(labels.len(), 15);
    assert_blob_structure(labels, 5);
}

/// `predict` on the training points agrees with the training labels from `get_labels`
#[test]
fn predict_on_training_data_matches_fit_labels() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let train_labels = km.get_labels().unwrap().clone();
    let pred_labels = km.predict(&data).unwrap();

    assert_eq!(train_labels.len(), pred_labels.len());
    for (t, p) in train_labels.iter().zip(pred_labels.iter()) {
        assert_eq!(t, p, "training label {t} != predicted label {p}");
    }
}

/// `predict` on fresh points near each blob's true centre assigns them to the same
/// cluster as the corresponding training blob
#[test]
fn predict_new_points_near_blob_centres() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let train_labels = km.get_labels().unwrap();
    // Representative label per blob (first point of each group)
    let label_blob0 = train_labels[0];
    let label_blob1 = train_labels[5];
    let label_blob2 = train_labels[10];

    let new_points = array![
        [0.0, 0.0],  // near blob 0 centre
        [10.0, 0.0], // near blob 1 centre
        [5.0, 10.0], // near blob 2 centre
    ];
    let pred = km.predict(&new_points).unwrap();

    assert_eq!(
        pred[0], label_blob0,
        "point near (0,0) should map to blob 0"
    );
    assert_eq!(
        pred[1], label_blob1,
        "point near (10,0) should map to blob 1"
    );
    assert_eq!(
        pred[2], label_blob2,
        "point near (5,10) should map to blob 2"
    );
}

/// `fit_predict` returns the same labels as `fit` + `get_labels`
#[test]
fn fit_predict_equals_fit_then_get_labels() {
    let data = three_blob_data();

    let mut km1 = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km1.fit(&data).unwrap();
    let labels_via_fit = km1.get_labels().unwrap().clone();

    let mut km2 = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    let labels_via_fp = km2.fit_predict(&data).unwrap();

    assert_eq!(labels_via_fit.len(), labels_via_fp.len());
    for (a, b) in labels_via_fit.iter().zip(labels_via_fp.iter()) {
        assert_eq!(a, b);
    }
}

// Inertia invariants

/// Inertia is strictly positive on non-degenerate data (points are not all identical)
#[test]
fn inertia_is_positive_on_blob_data() {
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km.fit(&three_blob_data()).unwrap();
    let inertia = km.get_inertia().unwrap();
    assert!(inertia > 0.0, "inertia must be positive, got {inertia}");
}

/// Inertia with k=1 must be >= inertia with k=3 on the same data, since more
/// clusters can only reduce or equal the sum of squared distances
#[test]
fn inertia_decreases_as_k_grows() {
    let data = three_blob_data();

    let mut km1 = KMeans::new(1, 300, 1e-4).unwrap().with_random_state(42);
    km1.fit(&data).unwrap();
    let inertia_k1 = km1.get_inertia().unwrap();

    let mut km3 = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km3.fit(&data).unwrap();
    let inertia_k3 = km3.get_inertia().unwrap();

    assert!(
        inertia_k1 >= inertia_k3,
        "k=1 inertia ({inertia_k1}) should be >= k=3 inertia ({inertia_k3})"
    );
}

// Boundary cases: k=1 and k=n_samples

/// With k=1, the single centroid equals the global mean of the data
#[test]
fn k1_centroid_equals_global_mean() {
    let data = three_blob_data();
    let mut km = KMeans::new(1, 300, 1e-4).unwrap().with_random_state(42);
    km.fit(&data).unwrap();

    let centroids = km.get_centroids().unwrap();
    assert_eq!(centroids.nrows(), 1);

    // Compute true global mean from the data
    let n = data.nrows() as f64;
    let true_mean_x: f64 = data.column(0).iter().sum::<f64>() / n;
    let true_mean_y: f64 = data.column(1).iter().sum::<f64>() / n;

    assert_abs_diff_eq!(centroids[(0, 0)], true_mean_x, epsilon = 1e-9);
    assert_abs_diff_eq!(centroids[(0, 1)], true_mean_y, epsilon = 1e-9);
}

/// With k=1, all labels must be 0
#[test]
fn k1_all_labels_are_zero() {
    let data = three_blob_data();
    let mut km = KMeans::new(1, 300, 1e-4).unwrap().with_random_state(42);
    km.fit(&data).unwrap();

    let labels = km.get_labels().unwrap();
    for &l in labels.iter() {
        assert_eq!(l, 0, "k=1: all labels must be 0, got {l}");
    }
}

/// With k = n_samples, centroids are shaped (n x 2), labels lie in [0, n), and
/// inertia is non-negative and <= the inertia from k=3
#[test]
fn k_equals_n_samples_boundary() {
    let data = three_blob_data();
    let n = data.nrows();

    let mut km_n = KMeans::new(n, 300, 1e-4).unwrap().with_random_state(42);
    km_n.fit(&data).unwrap();

    // Centroid matrix must have n rows
    let centroids = km_n.get_centroids().unwrap();
    assert_eq!(centroids.nrows(), n, "k=n_samples: should have n centroids");
    assert_eq!(centroids.ncols(), 2);

    // Labels must be in [0, n)
    let labels = km_n.get_labels().unwrap();
    assert_eq!(labels.len(), n);
    for &l in labels.iter() {
        assert!(l < n, "label {l} out of range [0, {n})");
    }

    // Inertia must be non-negative
    let inertia_n = km_n.get_inertia().unwrap();
    assert!(
        inertia_n >= 0.0,
        "inertia must be non-negative, got {inertia_n}"
    );

    // Inertia with k=n must be <= inertia with k=3
    let mut km3 = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km3.fit(&data).unwrap();
    let inertia_3 = km3.get_inertia().unwrap();

    assert!(
        inertia_n <= inertia_3 + 1e-9,
        "k=n_samples inertia ({inertia_n}) should be <= k=3 inertia ({inertia_3})"
    );
}

// Determinism (seeded reproducibility)

/// Two KMeans instances with the same seed produce identical centroids, labels,
/// inertia, and n_iter after fitting the same data
#[test]
fn same_seed_gives_identical_results() {
    let data = three_blob_data();

    let mut km_a = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km_a.fit(&data).unwrap();

    let mut km_b = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km_b.fit(&data).unwrap();

    // Centroids must be identical
    let ca = km_a.get_centroids().unwrap();
    let cb = km_b.get_centroids().unwrap();
    assert_allclose(ca, cb, 0.0_f64);

    // Labels must be identical
    let la = km_a.get_labels().unwrap();
    let lb = km_b.get_labels().unwrap();
    for (a, b) in la.iter().zip(lb.iter()) {
        assert_eq!(a, b, "label mismatch between identical-seed runs");
    }

    // Inertia must be identical
    let ia = km_a.get_inertia().unwrap();
    let ib = km_b.get_inertia().unwrap();
    assert_abs_diff_eq!(ia, ib, epsilon = 0.0);

    // n_iter must be identical
    let na = km_a.get_actual_iterations().unwrap();
    let nb = km_b.get_actual_iterations().unwrap();
    assert_eq!(na, nb, "n_iter mismatch between identical-seed runs");
}

/// Two KMeans instances with different seeds both converge to a blob-correct
/// partition on well-separated data
#[test]
fn different_seeds_still_converge_to_same_partition_on_blob_data() {
    let data = three_blob_data();

    let mut km_a = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(0);
    km_a.fit(&data).unwrap();

    let mut km_b = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(99);
    km_b.fit(&data).unwrap();

    // Both must produce blob-correct label structures
    assert_blob_structure(km_a.get_labels().unwrap(), 5);
    assert_blob_structure(km_b.get_labels().unwrap(), 5);
}

// n_iter is populated after fit

/// After a successful fit, `get_actual_iterations` returns `Some(n)` with
/// `1 <= n <= max_iter`
#[test]
fn actual_iterations_is_in_valid_range_after_fit() {
    let max_iter = 300;
    let mut km = KMeans::new(3, max_iter, 1e-4)
        .unwrap()
        .with_random_state(42);
    km.fit(&three_blob_data()).unwrap();

    let n_iter = km.get_actual_iterations().unwrap();
    assert!(n_iter >= 1, "n_iter must be >= 1, got {n_iter:?}");
    assert!(
        n_iter <= max_iter,
        "n_iter {n_iter:?} exceeds max_iter {max_iter}"
    );
}

// Save / load round-trip

/// Saving a fitted model to JSON and loading it back yields identical predictions
/// and bit-for-bit equal centroids
#[test]
fn save_load_round_trip_preserves_predictions() {
    let data = three_blob_data();
    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km.fit(&data).unwrap();

    let path = "/tmp/rustyml_test_kmeans_round_trip.json";
    km.save_to_path(path).expect("save_to_path should succeed");

    let km_loaded = KMeans::load_from_path(path).expect("load_from_path should succeed");

    // Predictions on the training data must be identical
    let pred_original = km.predict(&data).unwrap();
    let pred_loaded = km_loaded.predict(&data).unwrap();

    assert_eq!(pred_original.len(), pred_loaded.len());
    for (o, l) in pred_original.iter().zip(pred_loaded.iter()) {
        assert_eq!(o, l, "prediction mismatch after save/load round-trip");
    }

    // Centroids must also be bit-for-bit equal (no lossy serialisation)
    let co = km.get_centroids().unwrap();
    let cl = km_loaded.get_centroids().unwrap();
    assert_allclose(co, cl, 0.0_f64);
}

/// Loading from a nonexistent path must yield an `Error::Io` variant
#[test]
fn load_from_nonexistent_path_is_io_error() {
    let err =
        KMeans::load_from_path("/tmp/this_path_does_not_exist_rustyml_kmeans.json").unwrap_err();
    assert!(
        matches!(err, Error::Io(_)),
        "expected Error::Io, got {err:?}"
    );
}

// Larger-dataset fits: a medium-size correctness check on the parallel assignment branch

/// Build 1200 points (3 tight, well-separated blobs of 400 each) with fully
/// deterministic jitter (no RNG)
fn three_blobs_1200() -> Array2<f64> {
    let centers = [(0.0_f64, 0.0_f64), (10.0, 0.0), (5.0, 10.0)];
    let mut v = Vec::with_capacity(1200 * 2);
    for (cx, cy) in centers {
        for k in 0..400u32 {
            v.push(cx + ((k * 7) % 11) as f64 * 0.01 - 0.05);
            v.push(cy + ((k * 5) % 13) as f64 * 0.008 - 0.048);
        }
    }
    Array2::from_shape_vec((1200, 2), v).unwrap()
}

/// k=3 on the 1200-point dataset exercises the parallel assignment branch and still
/// recovers the 3 known blob means (within tol 0.1) and a blob-respecting partition
#[test]
fn fit_parallel_branch_k3_centroids_near_true_means_1200() {
    let data = three_blobs_1200();
    assert_eq!(data.nrows(), 1200, "dataset must cross the 1000 threshold");

    let mut km = KMeans::new(3, 300, 1e-4).unwrap().with_random_state(42);
    km.fit(&data).unwrap();

    let centroids = km.get_centroids().unwrap();
    assert_eq!(centroids.nrows(), 3, "should have 3 centroids");
    assert_eq!(centroids.ncols(), 2);

    let true_means: [[f64; 2]; 3] = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]];
    let tol = 0.1_f64;
    for &[tx, ty] in &true_means {
        let matched = (0..3).any(|i| {
            let cx = centroids[(i, 0)];
            let cy = centroids[(i, 1)];
            (cx - tx).abs() < tol && (cy - ty).abs() < tol
        });
        assert!(
            matched,
            "no centroid within tol={tol} of true mean ({tx}, {ty}); centroids = {centroids:?}"
        );
    }

    // Partition must be blob-pure: each block of 400 shares one label, blocks differ
    let labels = km.get_labels().unwrap();
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

/// Above the deterministic-fold gate (samples x features >= 262_144) the centroid
/// accumulation runs as a blocked parallel fold. Integer-valued rows make every cluster
/// sum exact in `f64` (well below 2^53), so regardless of summation grouping the stored
/// centroids must equal the per-label means recomputed serially - bit for bit. A wiring
/// bug in the fold (wrong rows, lost blocks, misordered merge) cannot pass this
#[test]
fn fit_parallel_accumulate_matches_serial_means_exactly() {
    let n = 16_384_usize;
    let d = 16_usize;
    let k = 4_usize;
    assert!(
        n * d >= 262_144,
        "dataset must cross the sum-gate work metric"
    );

    // 4 integer-valued blobs at offsets 0/100/200/300, deterministic jitter in 0..13
    let mut v = Vec::with_capacity(n * d);
    for i in 0..n {
        for j in 0..d {
            v.push((((i * 31 + j * 7) % 13) + (i % k) * 100) as f64);
        }
    }
    let data = Array2::from_shape_vec((n, d), v).unwrap();

    // One Lloyd step: labels and centroids then come from the same assignment pass
    let mut km = KMeans::new(k, 1, 1e-12).unwrap().with_random_state(7);
    km.fit(&data).unwrap();

    let labels = km.get_labels().unwrap();
    let centroids = km.get_centroids().unwrap();

    let mut sums = Array2::<f64>::zeros((k, d));
    let mut counts = vec![0usize; k];
    for (i, &lab) in labels.iter().enumerate() {
        counts[lab] += 1;
        for j in 0..d {
            sums[(lab, j)] += data[(i, j)];
        }
    }

    for c in 0..k {
        assert!(
            counts[c] > 0,
            "cluster {c} must be non-empty on this designed dataset (else the reseed \
             path replaces its centroid and the mean identity no longer applies)"
        );
        for j in 0..d {
            let expected = sums[(c, j)] / counts[c] as f64;
            assert_eq!(
                centroids[(c, j)],
                expected,
                "centroid ({c},{j}): parallel accumulate diverged from the exact serial mean"
            );
        }
    }
}

// centroid-shift convergence: a converged solution is a fixed point

/// The centroid-shift convergence criterion stops exactly when the centroids stop moving, so
/// a converged k-means solution must satisfy the fixed-point property: every centroid equals
/// the mean of the points assigned to it. (With a tight tolerance the residual is far below
/// the asserted bound.)
#[test]
fn converged_centroids_equal_their_cluster_means() {
    let mut km = KMeans::new(3, 300, 1e-6).unwrap().with_random_state(42);
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let centroids = km.get_centroids().unwrap();
    let labels = km.get_labels().unwrap();
    let n_features = data.ncols();

    for k in 0..3 {
        let mut sum = vec![0.0_f64; n_features];
        let mut count = 0usize;
        for (i, &lbl) in labels.iter().enumerate() {
            if lbl == k {
                for j in 0..n_features {
                    sum[j] += data[[i, j]];
                }
                count += 1;
            }
        }
        assert!(count > 0, "cluster {k} is empty");
        for j in 0..n_features {
            let mean_j = sum[j] / count as f64;
            assert_abs_diff_eq!(centroids[[k, j]], mean_j, epsilon = 1e-3);
        }
    }
}
