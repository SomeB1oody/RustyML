//! Integration tests for `rustyml::utils::t_sne::TSNE`
//!
//! Covers constructor validation, fit_transform output shape and finiteness,
//! determinism, centered output, neighborhood preservation, error paths, and getters

use approx::assert_abs_diff_eq;
use ndarray::Array2;
use rustyml::error::Error;
use rustyml::utils::t_sne::TSNE;

// Small valid dataset (6 points, 2 features) for general tests; perplexity=2.0 < 6
// passes the n_samples guard
fn small_data() -> Array2<f64> {
    ndarray::array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0]
    ]
}

// Constructor validation - perplexity

/// perplexity = 0.0 is rejected as not positive
#[test]
fn new_perplexity_zero_returns_invalid_parameter() {
    let err = TSNE::new(2, 0.0, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Negative perplexity is rejected as InvalidParameter
#[test]
fn new_perplexity_negative_returns_invalid_parameter() {
    let err = TSNE::new(2, -1.0, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// NaN perplexity is rejected as not finite
#[test]
fn new_perplexity_nan_returns_invalid_parameter() {
    let err = TSNE::new(2, f64::NAN, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// +inf perplexity is rejected as not finite
#[test]
fn new_perplexity_infinity_returns_invalid_parameter() {
    let err = TSNE::new(2, f64::INFINITY, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

// Constructor validation - learning_rate

/// learning_rate = 0.0 is rejected as not positive
#[test]
fn new_learning_rate_zero_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, 0.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Negative learning_rate is rejected as InvalidParameter
#[test]
fn new_learning_rate_negative_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, -100.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// NaN learning_rate is rejected as not finite
#[test]
fn new_learning_rate_nan_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, f64::NAN, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// +inf learning_rate is rejected as not finite
#[test]
fn new_learning_rate_infinity_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, f64::INFINITY, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

// Constructor validation - n_components and n_iter

/// n_components = 0 is rejected as InvalidParameter
#[test]
fn new_n_components_zero_returns_invalid_parameter() {
    let err = TSNE::new(0, 5.0, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// n_iter = 0 is rejected as InvalidParameter
#[test]
fn new_n_iter_zero_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, 200.0, 0, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Valid parameters produce Ok
#[test]
fn new_valid_params_returns_ok() {
    let result = TSNE::new(2, 5.0, 200.0, 250, Some(42));
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// fit_transform - output shape

/// fit_transform on an (n, d) matrix returns an (n, n_components) matrix
#[test]
fn fit_transform_output_shape_matches_n_samples_n_components() {
    let tsne = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    assert_eq!(
        embedding.shape(),
        &[6, 2],
        "expected shape [6, 2], got {:?}",
        embedding.shape()
    );
}

/// n_components = 1 produces an (n, 1) output
#[test]
fn fit_transform_output_shape_n_components_one() {
    let tsne = TSNE::new(1, 2.0, 200.0, 100, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    assert_eq!(
        embedding.shape(),
        &[6, 1],
        "expected shape [6, 1], got {:?}",
        embedding.shape()
    );
}

// fit_transform - all finite

/// Every element in the embedding is finite (no NaN or +/-inf)
#[test]
fn fit_transform_all_elements_finite() {
    let tsne = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    for (idx, &v) in embedding.iter().enumerate() {
        assert!(
            v.is_finite(),
            "embedding element at flat index {idx} is not finite: {v}"
        );
    }
}

// fit_transform - determinism (same seed gives bit-identical result)

/// Two calls with the same random_state produce a bit-for-bit identical embedding
#[test]
fn fit_transform_deterministic_same_seed_gives_identical_results() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // eps = 0.0 requires exact bit-equality
    crate::common::assert_allclose(&emb1, &emb2, 0.0_f64);
}

/// Different seeds produce different embeddings
#[test]
fn fit_transform_different_seeds_produce_different_results() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100, Some(0)).unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100, Some(999)).unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // At least one element must differ because the random initializations differ
    let any_differs = emb1.iter().zip(emb2.iter()).any(|(a, b)| a != b);
    assert!(
        any_differs,
        "expected different embeddings for different seeds, but got identical results"
    );
}

// fit_transform - centered output

/// Each column of the final embedding has mean ~= 0 (tolerance 1e-10)
#[test]
fn fit_transform_column_means_are_zero() {
    let tsne = TSNE::new(2, 2.0, 200.0, 300, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    let n = embedding.nrows() as f64;
    let mean0: f64 = embedding.column(0).iter().sum::<f64>() / n;
    let mean1: f64 = embedding.column(1).iter().sum::<f64>() / n;
    assert_abs_diff_eq!(mean0, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(mean1, 0.0, epsilon = 1e-10);
}

/// n_components=1: the single output column also has mean ~= 0
#[test]
fn fit_transform_column_mean_zero_n_components_one() {
    let tsne = TSNE::new(1, 2.0, 200.0, 300, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    let n = embedding.nrows() as f64;
    let mean0: f64 = embedding.column(0).iter().sum::<f64>() / n;
    assert_abs_diff_eq!(mean0, 0.0, epsilon = 1e-10);
}

// fit_transform - neighborhood preservation

/// Two well-separated high-D clusters remain separated in the 2-D embedding
#[test]
fn fit_transform_well_separated_clusters_remain_separated() {
    use std::f64::consts::PI;
    // 10 points evenly spaced on the unit circle; cluster B shifts every x by +200
    let n_per_cluster = 10_usize;
    let shift = 200.0_f64;
    let mut rows: Vec<[f64; 2]> = Vec::with_capacity(2 * n_per_cluster);
    for k in 0..n_per_cluster {
        let theta = 2.0 * PI * k as f64 / n_per_cluster as f64;
        rows.push([theta.cos(), theta.sin()]); // Cluster A
    }
    for k in 0..n_per_cluster {
        let theta = 2.0 * PI * k as f64 / n_per_cluster as f64;
        rows.push([shift + theta.cos(), theta.sin()]); // Cluster B
    }
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let x = Array2::from_shape_vec((2 * n_per_cluster, 2), flat).unwrap();

    // lr=100 stays in the stable range; the default lr=200 overshoots on a dataset this tiny
    // (no per-parameter adaptive gains), letting the embedding expand without settling
    let tsne = TSNE::new(2, 5.0, 100.0, 1000, Some(42)).unwrap();
    let emb = tsne.fit_transform(&x).unwrap();

    // Euclidean distance between two 2-D embedding points
    let dist2d = |i: usize, j: usize| -> f64 {
        let dy0 = emb[[i, 0]] - emb[[j, 0]];
        let dy1 = emb[[i, 1]] - emb[[j, 1]];
        (dy0 * dy0 + dy1 * dy1).sqrt()
    };

    // Centroids of the two clusters in the embedding
    let cen_a: [f64; 2] = [
        (0..n_per_cluster).map(|i| emb[[i, 0]]).sum::<f64>() / n_per_cluster as f64,
        (0..n_per_cluster).map(|i| emb[[i, 1]]).sum::<f64>() / n_per_cluster as f64,
    ];
    let cen_b: [f64; 2] = [
        (n_per_cluster..2 * n_per_cluster)
            .map(|i| emb[[i, 0]])
            .sum::<f64>()
            / n_per_cluster as f64,
        (n_per_cluster..2 * n_per_cluster)
            .map(|i| emb[[i, 1]])
            .sum::<f64>()
            / n_per_cluster as f64,
    ];
    let inter_centroid = {
        let dx = cen_a[0] - cen_b[0];
        let dy = cen_a[1] - cen_b[1];
        (dx * dx + dy * dy).sqrt()
    };

    // Max intra-cluster spread
    let mut max_intra: f64 = 0.0;
    for i in 0..n_per_cluster {
        for j in (i + 1)..n_per_cluster {
            max_intra = max_intra.max(dist2d(i, j));
        }
    }
    for i in n_per_cluster..(2 * n_per_cluster) {
        for j in (i + 1)..(2 * n_per_cluster) {
            max_intra = max_intra.max(dist2d(i, j));
        }
    }

    // The two cluster centroids must be at least 5x the max intra-cluster spread apart:
    // a conservative check given the 100:1 high-D separation ratio and 1000 iterations
    assert!(
        inter_centroid > 5.0 * max_intra,
        "cluster separation weak: inter_centroid={inter_centroid:.6}, \
         max_intra={max_intra:.6}, ratio={:.3}",
        inter_centroid / max_intra.max(1e-10)
    );
}

// Error paths - fit_transform

/// perplexity == n_samples is rejected as InvalidParameter
#[test]
fn fit_transform_perplexity_equals_n_samples_returns_invalid_parameter() {
    // 5 samples, perplexity = 5.0 (= n_samples)
    let x: Array2<f64> =
        ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    let tsne = TSNE::new(2, 5.0, 200.0, 100, Some(42)).unwrap();
    let err = tsne.fit_transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// perplexity > n_samples is rejected as InvalidParameter
#[test]
fn fit_transform_perplexity_exceeds_n_samples_returns_invalid_parameter() {
    let x: Array2<f64> =
        ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    let tsne = TSNE::new(2, 10.0, 200.0, 100, Some(42)).unwrap();
    let err = tsne.fit_transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// A single sample is rejected as InvalidInput (fewer than 2 samples)
#[test]
fn fit_transform_single_sample_returns_invalid_input() {
    let x: Array2<f64> = ndarray::array![[1.0, 2.0, 3.0]];
    let tsne = TSNE::new(2, 0.5, 200.0, 50, Some(42)).unwrap();
    let err = tsne.fit_transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// Empty input (0 rows) is rejected as EmptyInput
#[test]
fn fit_transform_empty_input_returns_empty_input() {
    let x: Array2<f64> = Array2::zeros((0, 3));
    let tsne = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let err = tsne.fit_transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// NaN in input data is rejected as NonFinite
#[test]
fn fit_transform_nan_input_returns_non_finite() {
    let x = ndarray::array![[1.0, f64::NAN], [2.0, 3.0], [4.0, 5.0]];
    let tsne = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let err = tsne.fit_transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// +inf in input data is rejected as NonFinite
#[test]
fn fit_transform_infinity_input_returns_non_finite() {
    let x = ndarray::array![[f64::INFINITY, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let tsne = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let err = tsne.fit_transform(&x).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

// Getter roundtrips

/// Getters return exactly the values passed to new()
#[test]
fn getters_return_construction_values() {
    let tsne = TSNE::new(3, 15.0, 100.0, 500, Some(77)).unwrap();
    assert_eq!(tsne.get_n_components(), 3);
    assert_abs_diff_eq!(tsne.get_perplexity(), 15.0, epsilon = 0.0);
    assert_abs_diff_eq!(tsne.get_learning_rate(), 100.0, epsilon = 0.0);
    assert_eq!(tsne.get_n_iter(), 500);
    assert_eq!(tsne.get_random_state(), Some(77));
}

/// When random_state is None the getter also returns None
#[test]
fn getter_random_state_none() {
    let tsne = TSNE::new(2, 5.0, 200.0, 100, None).unwrap();
    assert_eq!(tsne.get_random_state(), None);
}
