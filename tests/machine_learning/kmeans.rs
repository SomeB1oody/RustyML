//! Integration tests for `KMeans` clustering.
//!
//! Expected values are derived from problem design (three well-separated blobs whose
//! true centroids are known by construction) or from closed-form invariants (k=1
//! centroid = global mean, inertia ≥ 0, inertia non-increasing as k grows).
//!
//! This file is a submodule of `tests/machine_learning/main.rs` — no `fn main`, no
//! `#![cfg(...)]` guards.

use crate::common::assert_allclose;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::kmeans::KMeans;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build three tight, well-separated blobs:
///
///   Blob 0 — 5 points around (0.0, 0.0), offset in [-0.05, +0.05]
///   Blob 1 — 5 points around (10.0, 0.0), offset in [-0.05, +0.05]
///   Blob 2 — 5 points around (5.0, 10.0), offset in [-0.05, +0.05]
///
/// The inter-blob separation (≥ 9.9) vastly exceeds the intra-blob spread (≤ 0.14),
/// so any correct k=3 KMeans run must assign all points of each blob to a single
/// cluster, and the centroid for each cluster must be within 0.1 of the true mean.
fn three_blob_data() -> Array2<f64> {
    // 15 rows × 2 columns.
    // Tiny deterministic offsets keep the blobs tight without hiding the structure.
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

/// Given labels produced for `three_blob_data()`, verify that all points belonging
/// to the same blob share the same label and that the three blobs got different labels.
///
/// `blob_size` is the number of points per blob (5 for `three_blob_data()`).
fn assert_blob_structure(labels: &Array1<usize>, blob_size: usize) {
    let n_blobs = labels.len() / blob_size;
    assert_eq!(
        labels.len(),
        n_blobs * blob_size,
        "labels length must be a multiple of blob_size"
    );

    // Collect the representative label for each blob (label of the first point).
    let blob_labels: Vec<usize> = (0..n_blobs).map(|b| labels[b * blob_size]).collect();

    // All points within each blob must share that blob's label.
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

    // Each blob must have a distinct label.
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

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

/// `n_clusters = 0` must yield `Error::InvalidParameter`.
/// Derivation: the constructor guard `if n_clusters == 0` fires immediately.
#[test]
fn constructor_zero_clusters_is_invalid() {
    let err = KMeans::new(0, 100, 1e-4, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// `max_iterations = 0` must yield `Error::InvalidParameter`.
/// Derivation: `validate_max_iterations(0)` checks `== 0` and returns
/// `Err(Error::InvalidParameter{..})`.
#[test]
fn constructor_zero_max_iter_is_invalid() {
    let err = KMeans::new(3, 0, 1e-4, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// `tolerance = 0.0` must yield `Error::InvalidParameter`.
/// Derivation: `validate_tolerance` requires `tolerance > 0`; 0.0 fails.
#[test]
fn constructor_zero_tolerance_is_invalid() {
    let err = KMeans::new(3, 100, 0.0, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Negative tolerance must yield `Error::InvalidParameter`.
#[test]
fn constructor_negative_tolerance_is_invalid() {
    let err = KMeans::new(3, 100, -1e-4, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// `tolerance = f64::NAN` must yield `Error::InvalidParameter`.
/// Derivation: `validate_tolerance` checks `!tolerance.is_finite()`; NaN is not finite.
#[test]
fn constructor_nan_tolerance_is_invalid() {
    let err = KMeans::new(3, 100, f64::NAN, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// `tolerance = f64::INFINITY` must yield `Error::InvalidParameter`.
#[test]
fn constructor_inf_tolerance_is_invalid() {
    let err = KMeans::new(3, 100, f64::INFINITY, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Valid parameters must succeed and getters must reflect the stored values.
/// Derivation: all guards pass; `KMeans::new` returns `Ok(Self)`.
#[test]
fn constructor_valid_stores_params() {
    let km = KMeans::new(3, 200, 1e-4, Some(42)).unwrap();
    assert_eq!(km.get_n_clusters(), 3);
    assert_eq!(km.get_max_iterations(), 200);
    assert_abs_diff_eq!(km.get_tolerance(), 1e-4, epsilon = 1e-15);
    assert_eq!(km.get_random_state(), Some(42));
    // Before fit, post-fit getters return None.
    assert!(km.get_centroids().is_none());
    assert!(km.get_labels().is_none());
    assert!(km.get_inertia().is_none());
    assert!(km.get_actual_iterations().is_none());
}

/// `KMeans::default()` should produce a valid instance with documented defaults.
/// Derivation: `Default::default()` calls `KMeans::new(8, 300, 1e-4, None)`.
#[test]
fn default_constructor_has_documented_defaults() {
    let km = KMeans::default();
    assert_eq!(km.get_n_clusters(), 8);
    assert_eq!(km.get_max_iterations(), 300);
    assert_abs_diff_eq!(km.get_tolerance(), 1e-4, epsilon = 1e-15);
    assert_eq!(km.get_random_state(), None);
}

// ---------------------------------------------------------------------------
// fit error paths
// ---------------------------------------------------------------------------

/// `fit` on an empty array must yield `Error::EmptyInput`.
/// Derivation: `preliminary_check` checks `x.nrows() == 0`.
#[test]
fn fit_empty_data_is_error() {
    let mut km = KMeans::new(1, 100, 1e-4, Some(0)).unwrap();
    let data: Array2<f64> = Array2::zeros((0, 2));
    let err = km.fit(&data).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// `fit` on data with NaN must yield `Error::NonFinite`.
/// Derivation: `preliminary_check` iterates values and detects `is_nan()`.
#[test]
fn fit_nan_data_is_error() {
    let mut km = KMeans::new(1, 100, 1e-4, Some(0)).unwrap();
    let data = array![[1.0, f64::NAN], [2.0, 3.0]];
    let err = km.fit(&data).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// `fit` on data with Inf must yield `Error::NonFinite`.
#[test]
fn fit_inf_data_is_error() {
    let mut km = KMeans::new(1, 100, 1e-4, Some(0)).unwrap();
    let data = array![[1.0, f64::INFINITY], [2.0, 3.0]];
    let err = km.fit(&data).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// `fit` with `n_samples < n_clusters` must yield `Error::InvalidInput`.
/// Derivation: the guard `if n_samples < self.n_clusters` in `fit` fires.
#[test]
fn fit_fewer_samples_than_clusters_is_error() {
    let mut km = KMeans::new(5, 100, 1e-4, Some(0)).unwrap();
    // 3 samples < 5 clusters
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let err = km.fit(&data).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// predict error paths
// ---------------------------------------------------------------------------

/// `predict` before `fit` must yield `Error::NotFitted`.
/// Derivation: `predict` calls `ok_or_else(|| Error::not_fitted("KMeans"))` on
/// `self.centroids` which is `None` before fitting.
#[test]
fn predict_before_fit_is_not_fitted() {
    let km = KMeans::new(3, 100, 1e-4, Some(42)).unwrap();
    let data = array![[1.0, 2.0]];
    let err = km.predict(&data).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {err:?}"
    );
}

/// `predict` with wrong number of features must yield `Error::DimensionMismatch`.
/// Derivation: `validate_predict_input` checks `x.ncols() != expected_features`.
#[test]
fn predict_wrong_feature_count_is_dimension_mismatch() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    let data = three_blob_data(); // shape (15, 2)
    km.fit(&data).unwrap();

    // Feed a 3-feature matrix to a model trained on 2 features.
    let bad = array![[1.0, 2.0, 3.0]];
    let err = km.predict(&bad).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

/// `predict` with an empty matrix (after fit) must yield `Error::EmptyInput`.
/// Derivation: `validate_predict_input` checks `x.is_empty()` first.
#[test]
fn predict_empty_input_after_fit_is_empty_input() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km.fit(&three_blob_data()).unwrap();

    let empty: Array2<f64> = Array2::zeros((0, 2));
    let err = km.predict(&empty).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// `predict` with NaN input (after fit) must yield `Error::NonFinite`.
/// Derivation: `validate_predict_input` checks `!val.is_finite()`.
#[test]
fn predict_nan_input_is_non_finite() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let bad = array![[f64::NAN, 1.0]];
    let err = km.predict(&bad).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Correctness on designed data
// ---------------------------------------------------------------------------

/// After fitting k=3 on three_blob_data(), the centroids must be within 0.1 of
/// each blob's true mean.
///
/// Derivation: blobs are centred at (0,0), (10,0), (5,10) with offsets ≤ 0.05
/// per coordinate.  The maximum possible centroid error is the mean absolute
/// offset, which is at most 0.05.  We use tol=0.1 to leave generous room.
///
/// We cannot assert which cluster INDEX maps to which blob because K-Means label
/// assignment is arbitrary.  Instead we assert that {set of centroids} ≈ {set of
/// true means} by checking that each true mean has a matching centroid within tol.
#[test]
fn fit_k3_centroids_near_true_blob_means() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
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

/// After fitting, labels assigned by `get_labels` must respect blob structure:
/// all 5 points in blob 0 share a label, all 5 in blob 1 share a different label,
/// all 5 in blob 2 share yet another different label.
///
/// Derivation: the blobs are separated by ≥ 9.9 units; with correct k=3 KMeans
/// every point's nearest centroid must be the one from its own blob.
#[test]
fn fit_k3_labels_respect_blob_structure() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let labels = km.get_labels().unwrap();
    assert_eq!(labels.len(), 15);
    assert_blob_structure(labels, 5);
}

/// `predict` on the training points must agree with the training labels returned
/// by `get_labels`.
///
/// Derivation: `predict` assigns each point to its nearest centroid using the
/// same `closest_centroid` function used internally; agreement with the stored
/// labels is a required contract.
#[test]
fn predict_on_training_data_matches_fit_labels() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let train_labels = km.get_labels().unwrap().clone();
    let pred_labels = km.predict(&data).unwrap();

    assert_eq!(train_labels.len(), pred_labels.len());
    for (t, p) in train_labels.iter().zip(pred_labels.iter()) {
        assert_eq!(t, p, "training label {t} != predicted label {p}");
    }
}

/// `predict` on fresh points near each blob's true centre must assign them to
/// the same cluster as the corresponding training blob.
///
/// Derivation: a point at (0.0, 0.0) is closer to the centroid of blob 0
/// than to any other centroid (distance ≈ 0 vs ≥ 9.9); by the nearest-centroid
/// rule it must receive the same label as the training points of blob 0.
#[test]
fn predict_new_points_near_blob_centres() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    let data = three_blob_data();
    km.fit(&data).unwrap();

    let train_labels = km.get_labels().unwrap();
    // Representative label for each blob (first point of each group).
    let label_blob0 = train_labels[0]; // first point of blob 0
    let label_blob1 = train_labels[5]; // first point of blob 1
    let label_blob2 = train_labels[10]; // first point of blob 2

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

/// `fit_predict` must return the same labels as `fit` + `get_labels`.
///
/// Derivation: the source calls `self.fit(data)?` and then returns
/// `self.labels.clone().unwrap()` — identical to calling `get_labels`.
#[test]
fn fit_predict_equals_fit_then_get_labels() {
    let data = three_blob_data();

    let mut km1 = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km1.fit(&data).unwrap();
    let labels_via_fit = km1.get_labels().unwrap().clone();

    let mut km2 = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    let labels_via_fp = km2.fit_predict(&data).unwrap();

    assert_eq!(labels_via_fit.len(), labels_via_fp.len());
    for (a, b) in labels_via_fit.iter().zip(labels_via_fp.iter()) {
        assert_eq!(a, b);
    }
}

// ---------------------------------------------------------------------------
// Inertia invariants
// ---------------------------------------------------------------------------

/// Inertia must be strictly positive on non-degenerate data (points are not
/// all identical).
///
/// Derivation: at least one point is not at its centroid, so at least one
/// squared distance is > 0.
#[test]
fn inertia_is_positive_on_blob_data() {
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km.fit(&three_blob_data()).unwrap();
    let inertia = km.get_inertia().unwrap();
    assert!(inertia > 0.0, "inertia must be positive, got {inertia}");
}

/// Inertia with k=1 must be ≥ inertia with k=3 on the same data, because
/// adding more clusters can only reduce (or equal) the sum of squared distances.
///
/// Derivation: k=1 forces all 15 points to one centroid (the global mean),
/// producing large distances for points in blobs 1 and 2.  k=3 groups each blob
/// separately, giving near-zero distances.
#[test]
fn inertia_decreases_as_k_grows() {
    let data = three_blob_data();

    let mut km1 = KMeans::new(1, 300, 1e-4, Some(42)).unwrap();
    km1.fit(&data).unwrap();
    let inertia_k1 = km1.get_inertia().unwrap();

    let mut km3 = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km3.fit(&data).unwrap();
    let inertia_k3 = km3.get_inertia().unwrap();

    assert!(
        inertia_k1 >= inertia_k3,
        "k=1 inertia ({inertia_k1}) should be >= k=3 inertia ({inertia_k3})"
    );
}

// ---------------------------------------------------------------------------
// Boundary cases: k=1 and k=n_samples
// ---------------------------------------------------------------------------

/// With k=1, the single centroid must equal the global mean of the data.
///
/// Derivation (closed form): KMeans with k=1 assigns all points to one cluster;
/// the centroid update sets it to the mean of all points.  For our 15-point
/// three-blob dataset:
///   mean_x = (5*0 + 5*10 + 5*5) / 15 + (sum of small offsets)/15
///   mean_y = (5*0 + 5*0  + 5*10)/ 15 + (sum of small offsets)/15
/// The small-offset sum: blob0 x = [-0.05+0.04+0.01-0.03+0.02]=−0.01,
///   blob1 x = [−0.05+0.04+0.01−0.03+0.02]=−0.01 (same pattern, shifted),
///   blob2 x offsets same = −0.01.
/// Total x offset = 3*(−0.01) = −0.03; global mean_x = (0+50+25)/15 − 0.03/15 = 5.0 − 0.002 = 4.998.
/// We check within tol 0.01 to avoid fragile arithmetic.
#[test]
fn k1_centroid_equals_global_mean() {
    let data = three_blob_data();
    let mut km = KMeans::new(1, 300, 1e-4, Some(42)).unwrap();
    km.fit(&data).unwrap();

    let centroids = km.get_centroids().unwrap();
    assert_eq!(centroids.nrows(), 1);

    // Compute true global mean from the data.
    let n = data.nrows() as f64;
    let true_mean_x: f64 = data.column(0).iter().sum::<f64>() / n;
    let true_mean_y: f64 = data.column(1).iter().sum::<f64>() / n;

    assert_abs_diff_eq!(centroids[(0, 0)], true_mean_x, epsilon = 1e-9);
    assert_abs_diff_eq!(centroids[(0, 1)], true_mean_y, epsilon = 1e-9);
}

/// With k=1, all labels must be 0.
/// Derivation: there is only one cluster (index 0); every point is assigned to it.
#[test]
fn k1_all_labels_are_zero() {
    let data = three_blob_data();
    let mut km = KMeans::new(1, 300, 1e-4, Some(42)).unwrap();
    km.fit(&data).unwrap();

    let labels = km.get_labels().unwrap();
    for &l in labels.iter() {
        assert_eq!(l, 0, "k=1: all labels must be 0, got {l}");
    }
}

/// With k = n_samples, inertia must be ≤ the inertia from k=3 (more clusters
/// can only keep inertia the same or reduce it).
///
/// Derivation: k=n_samples is an upper bound on the number of clusters that can
/// meaningfully reduce inertia.  Inertia must also be ≥ 0 by definition.
/// We check both the shape of centroids (n×2) and the inertia bound.
#[test]
fn k_equals_n_samples_boundary() {
    let data = three_blob_data();
    let n = data.nrows();

    let mut km_n = KMeans::new(n, 300, 1e-4, Some(42)).unwrap();
    km_n.fit(&data).unwrap();

    // Centroid matrix must have n rows.
    let centroids = km_n.get_centroids().unwrap();
    assert_eq!(centroids.nrows(), n, "k=n_samples: should have n centroids");
    assert_eq!(centroids.ncols(), 2);

    // Labels must be in [0, n).
    let labels = km_n.get_labels().unwrap();
    assert_eq!(labels.len(), n);
    for &l in labels.iter() {
        assert!(l < n, "label {l} out of range [0, {n})");
    }

    // Inertia must be non-negative.
    let inertia_n = km_n.get_inertia().unwrap();
    assert!(
        inertia_n >= 0.0,
        "inertia must be non-negative, got {inertia_n}"
    );

    // Inertia with k=n must be <= inertia with k=3.
    let mut km3 = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km3.fit(&data).unwrap();
    let inertia_3 = km3.get_inertia().unwrap();

    assert!(
        inertia_n <= inertia_3 + 1e-9,
        "k=n_samples inertia ({inertia_n}) should be <= k=3 inertia ({inertia_3})"
    );
}

// ---------------------------------------------------------------------------
// Determinism (seeded reproducibility)
// ---------------------------------------------------------------------------

/// Two KMeans instances with the same seed must produce identical centroids,
/// labels, inertia, and n_iter after fitting the same data.
///
/// Derivation: the algorithm is fully deterministic given the seed because
/// `StdRng::seed_from_u64` is used everywhere randomness appears.
#[test]
fn same_seed_gives_identical_results() {
    let data = three_blob_data();

    let mut km_a = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km_a.fit(&data).unwrap();

    let mut km_b = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km_b.fit(&data).unwrap();

    // Centroids must be identical.
    let ca = km_a.get_centroids().unwrap();
    let cb = km_b.get_centroids().unwrap();
    assert_allclose(ca, cb, 0.0_f64);

    // Labels must be identical.
    let la = km_a.get_labels().unwrap();
    let lb = km_b.get_labels().unwrap();
    for (a, b) in la.iter().zip(lb.iter()) {
        assert_eq!(a, b, "label mismatch between identical-seed runs");
    }

    // Inertia must be identical.
    let ia = km_a.get_inertia().unwrap();
    let ib = km_b.get_inertia().unwrap();
    assert_abs_diff_eq!(ia, ib, epsilon = 0.0);

    // n_iter must be identical.
    let na = km_a.get_actual_iterations().unwrap();
    let nb = km_b.get_actual_iterations().unwrap();
    assert_eq!(na, nb, "n_iter mismatch between identical-seed runs");
}

/// Two KMeans instances with DIFFERENT seeds are not guaranteed to give different
/// results on well-separated data (both will converge to the same partition), but
/// predict must be consistent: the label assigned to a point must correspond to
/// the nearest centroid in both models.
///
/// (We don't assert label index equality across different seeds because the cluster
/// numbering is arbitrary; we assert structural equivalence instead.)
#[test]
fn different_seeds_still_converge_to_same_partition_on_blob_data() {
    let data = three_blob_data();

    let mut km_a = KMeans::new(3, 300, 1e-4, Some(0)).unwrap();
    km_a.fit(&data).unwrap();

    let mut km_b = KMeans::new(3, 300, 1e-4, Some(99)).unwrap();
    km_b.fit(&data).unwrap();

    // Both must produce blob-correct label structures.
    assert_blob_structure(km_a.get_labels().unwrap(), 5);
    assert_blob_structure(km_b.get_labels().unwrap(), 5);
}

// ---------------------------------------------------------------------------
// n_iter is populated after fit
// ---------------------------------------------------------------------------

/// After a successful fit, `get_actual_iterations` must return `Some(n)` with
/// `n >= 1` (at least one iteration ran) and `n <= max_iter`.
///
/// Derivation: the loop runs at least once before convergence can be detected;
/// it exits early or at `max_iter`.
#[test]
fn actual_iterations_is_in_valid_range_after_fit() {
    let max_iter = 300;
    let mut km = KMeans::new(3, max_iter, 1e-4, Some(42)).unwrap();
    km.fit(&three_blob_data()).unwrap();

    let n_iter = km.get_actual_iterations().unwrap();
    assert!(n_iter >= 1, "n_iter must be >= 1, got {n_iter:?}");
    assert!(
        n_iter <= max_iter,
        "n_iter {n_iter:?} exceeds max_iter {max_iter}"
    );
}

// ---------------------------------------------------------------------------
// Save / load round-trip
// ---------------------------------------------------------------------------

/// Saving a fitted model to a JSON file and loading it back must produce a model
/// that yields identical predictions to the original.
///
/// Derivation: `save_to_path` serialises the full struct (including `centroids`);
/// `load_from_path` deserialises it back.  Because `predict` is a pure function
/// of `centroids`, predictions must agree exactly.
#[test]
fn save_load_round_trip_preserves_predictions() {
    let data = three_blob_data();
    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km.fit(&data).unwrap();

    let path = "/tmp/rustyml_test_kmeans_round_trip.json";
    km.save_to_path(path).expect("save_to_path should succeed");

    let km_loaded = KMeans::load_from_path(path).expect("load_from_path should succeed");

    // Predictions on the training data must be identical.
    let pred_original = km.predict(&data).unwrap();
    let pred_loaded = km_loaded.predict(&data).unwrap();

    assert_eq!(pred_original.len(), pred_loaded.len());
    for (o, l) in pred_original.iter().zip(pred_loaded.iter()) {
        assert_eq!(o, l, "prediction mismatch after save/load round-trip");
    }

    // Centroids must also be bit-for-bit equal (no lossy serialisation).
    let co = km.get_centroids().unwrap();
    let cl = km_loaded.get_centroids().unwrap();
    assert_allclose(co, cl, 0.0_f64);
}

/// Loading from a nonexistent path must yield an `Error::Io` variant.
/// Derivation: `File::open` fails with `std::io::Error`; the macro wraps it
/// as `Error::Io(IoError::Std(...))`.
#[test]
fn load_from_nonexistent_path_is_io_error() {
    let err =
        KMeans::load_from_path("/tmp/this_path_does_not_exist_rustyml_kmeans.json").unwrap_err();
    assert!(
        matches!(err, Error::Io(_)),
        "expected Error::Io, got {err:?}"
    );
}
// ---------------------------------------------------------------------------
// Large-dataset parallel assignment branch (n_samples >= 1000)
// ---------------------------------------------------------------------------

/// Build 1200 points (3 tight, well-separated blobs of 400 each) with fully
/// deterministic jitter (no RNG). 1200 >= KMEANS_PARALLEL_THRESHOLD (1000), so
/// `fit` takes the parallel per-sample assignment branch (src kmeans.rs ~line 374).
///
/// Blob centres: (0,0), (10,0), (5,10). Jitter is dx in [-0.05,+0.05],
/// dy in [-0.048,+0.048], so the intra-blob spread (at most ~0.14) is tiny next to the
/// inter-blob separation of at least 9.9. The per-blob mean of the deterministic offsets is
/// ~0 (x offset mean = 0.0, y offset mean = -8e-5, hand-checked), so each blob's
/// empirical mean sits within ~1e-4 of its nominal centre.
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

/// k=3 on the 1200-point dataset exercises the PARALLEL assignment branch and must
/// still recover the 3 known blob means (within tol 0.1) and a blob-respecting
/// partition. Cluster-index order is arbitrary, so we match each true mean to its
/// nearest centroid (same approach as `fit_k3_centroids_near_true_blob_means`).
#[test]
fn fit_parallel_branch_k3_centroids_near_true_means_1200() {
    let data = three_blobs_1200();
    assert_eq!(data.nrows(), 1200, "dataset must cross the 1000 threshold");

    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
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

    // Partition must be blob-pure: each block of 400 shares one label, blocks differ.
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
