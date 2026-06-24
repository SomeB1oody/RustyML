//! Integration tests for `rustyml::machine_learning::manifold::t_sne::TSNE`
//!
//! Covers constructor validation, fit_transform output shape and finiteness,
//! determinism, centered output, neighborhood preservation, error paths, and getters

use approx::assert_abs_diff_eq;
use ndarray::Array2;
use rustyml::error::Error;
use rustyml::machine_learning::manifold::t_sne::{Init, TSNE, TSNEMethod};

// Small valid dataset (6 points, 2 features); perplexity=2.0 < 6 passes the n_samples guard
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

/// Non-positive (0.0, negative) and non-finite (NaN, +inf) perplexity values are all
/// rejected as InvalidParameter
#[test]
fn new_perplexity_invalid_returns_invalid_parameter() {
    for bad_perplexity in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
        let err = TSNE::new(2, bad_perplexity, 200.0, 250).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for perplexity={bad_perplexity}, got {err:?}"
        );
    }
}

// Constructor validation - learning_rate

/// Non-positive (0.0, negative) and non-finite (NaN, +inf) learning_rate values are all
/// rejected as InvalidParameter
#[test]
fn new_learning_rate_invalid_returns_invalid_parameter() {
    for bad_learning_rate in [0.0_f64, -100.0, f64::NAN, f64::INFINITY] {
        let err = TSNE::new(2, 5.0, bad_learning_rate, 250).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for learning_rate={bad_learning_rate}, got {err:?}"
        );
    }
}

// Constructor validation - n_components and n_iter

/// n_components = 0 and n_iter = 0 are each rejected as InvalidParameter
#[test]
fn new_zero_integer_param_returns_invalid_parameter() {
    // (n_components, n_iter): one argument is the zero offender per row
    for (n_components, n_iter) in [(0, 250), (2, 0)] {
        let err = TSNE::new(n_components, 5.0, 200.0, n_iter).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for (n_components={n_components}, n_iter={n_iter}), got {err:?}"
        );
    }
}

/// Valid parameters produce Ok
#[test]
fn new_valid_params_returns_ok() {
    let result = TSNE::new(2, 5.0, 200.0, 250);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// fit_transform - output shape

/// fit_transform on an (n, d) matrix returns an (n, n_components) matrix
#[test]
fn fit_transform_output_shape_matches_n_samples_n_components() {
    let tsne = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(1, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // eps = 0.0 requires exact bit-equality
    crate::common::assert_allclose(&emb1, &emb2, 0.0_f64);
}

/// Different seeds produce different embeddings under random initialization
///
/// The seed only feeds the random initialization path; PCA initialization is deterministic
/// and would ignore the seed, so this test pins [`Init::Random`]
#[test]
fn fit_transform_different_seeds_produce_different_results() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(0)
        .with_init(Init::Random)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(999)
        .with_init(Init::Random)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // At least one element must differ because the random initializations differ
    let any_differs = emb1.iter().zip(emb2.iter()).any(|(a, b)| a != b);
    assert!(
        any_differs,
        "expected different embeddings for different seeds, but got identical results"
    );
}

/// The default PCA initialization is deterministic, so different seeds give identical results
#[test]
fn fit_transform_pca_init_is_seed_independent() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(0)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(999)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // PCA initialization ignores the seed, and the optimization is otherwise deterministic
    crate::common::assert_allclose(&emb1, &emb2, 0.0_f64);
}

// fit_transform - centered output

/// Each column of the final embedding has mean ~= 0 (tolerance 1e-10)
#[test]
fn fit_transform_column_means_are_zero() {
    let tsne = TSNE::new(2, 2.0, 200.0, 300)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(1, 2.0, 200.0, 300)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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

    // On a set this tiny (20 points) the default lr=200 over-expands the absolute
    // scale; lr=100 keeps the step sized to the data
    let tsne = TSNE::new(2, 5.0, 100.0, 1000)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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

/// perplexity == n_samples (5.0) and perplexity > n_samples (10.0) are both rejected as
/// InvalidParameter (the dataset has 5 samples)
#[test]
fn fit_transform_perplexity_not_below_n_samples_returns_invalid_parameter() {
    let x: Array2<f64> =
        ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    for bad_perplexity in [5.0_f64, 10.0] {
        let tsne = TSNE::new(2, bad_perplexity, 200.0, 100)
            .unwrap()
            .with_random_state(42)
            .with_init(Init::PCA)
            .with_method(TSNEMethod::Exact)
            .unwrap();
        let err = tsne.fit_transform(&x).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for perplexity={bad_perplexity}, got {err:?}"
        );
    }
}

/// A single sample is rejected as InvalidInput (fewer than 2 samples)
#[test]
fn fit_transform_single_sample_returns_invalid_input() {
    let x: Array2<f64> = ndarray::array![[1.0, 2.0, 3.0]];
    let tsne = TSNE::new(2, 0.5, 200.0, 50)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
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
    let tsne = TSNE::new(3, 15.0, 100.0, 500)
        .unwrap()
        .with_random_state(77)
        .with_init(Init::Random)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    assert_eq!(tsne.get_n_components(), 3);
    assert_abs_diff_eq!(tsne.get_perplexity(), 15.0, epsilon = 0.0);
    assert_abs_diff_eq!(tsne.get_learning_rate(), 100.0, epsilon = 0.0);
    assert_eq!(tsne.get_n_iter(), 500);
    assert_eq!(tsne.get_random_state(), Some(77));
    assert_eq!(tsne.get_init(), Init::Random);
    assert_eq!(tsne.get_method(), TSNEMethod::Exact);
}

/// min_grad_norm defaults to 1e-7 and the builder overrides it
#[test]
fn min_grad_norm_default_and_builder() {
    let default_tsne = TSNE::new(2, 5.0, 200.0, 100).unwrap();
    assert_abs_diff_eq!(default_tsne.get_min_grad_norm(), 1e-7, epsilon = 0.0);

    let tuned = default_tsne.with_min_grad_norm(1e-3);
    assert_abs_diff_eq!(tuned.get_min_grad_norm(), 1e-3, epsilon = 0.0);
}

/// Early stopping (a large threshold stops right after early exaggeration) and disabling it
/// (threshold 0 runs the full n_iter) both produce a valid, correctly shaped embedding
#[test]
fn min_grad_norm_early_stop_and_disable_produce_valid_output() {
    let x = small_data();

    // A huge threshold makes the post-exaggeration gradient check trip immediately
    let early = TSNE::new(2, 2.0, 200.0, 1000)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap()
        .with_min_grad_norm(1e9);
    let early_embedding = early.fit_transform(&x).unwrap();
    assert_eq!(early_embedding.shape(), &[6, 2]);
    assert!(early_embedding.iter().all(|v| v.is_finite()));

    // Disabling early stopping still runs to completion with finite output
    let full = TSNE::new(2, 2.0, 200.0, 300)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap()
        .with_min_grad_norm(0.0);
    let full_embedding = full.fit_transform(&x).unwrap();
    assert_eq!(full_embedding.shape(), &[6, 2]);
    assert!(full_embedding.iter().all(|v| v.is_finite()));
}

/// When random_state is None the getter also returns None
#[test]
fn getter_random_state_none() {
    let tsne = TSNE::new(2, 5.0, 200.0, 100)
        .unwrap()
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap();
    assert_eq!(tsne.get_random_state(), None);
}

// Barnes-Hut method

/// Barnes-Hut rejects embeddings with more than 3 components
#[test]
fn new_barnes_hut_rejects_more_than_three_components() {
    let err = TSNE::new(4, 5.0, 200.0, 100)
        .unwrap()
        .with_method(TSNEMethod::BarnesHut { angle: 0.5 })
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for n_components=4 with Barnes-Hut, got {err:?}"
    );
}

/// Barnes-Hut accepts exactly 3 components
#[test]
fn new_barnes_hut_accepts_three_components() {
    let result = TSNE::new(3, 5.0, 200.0, 100)
        .unwrap()
        .with_method(TSNEMethod::BarnesHut { angle: 0.5 });
    assert!(
        result.is_ok(),
        "expected Ok for 3 components, got {result:?}"
    );
}

/// An angle outside [0, 1) is rejected
#[test]
fn new_barnes_hut_rejects_angle_out_of_range() {
    for bad_angle in [1.0_f64, -0.1, f64::NAN] {
        let err = TSNE::new(2, 5.0, 200.0, 100)
            .unwrap()
            .with_method(TSNEMethod::BarnesHut { angle: bad_angle })
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for angle={bad_angle}, got {err:?}"
        );
    }
}

/// The Default instance uses Barnes-Hut with angle 0.5
#[test]
fn default_tsne_uses_barnes_hut() {
    let tsne = TSNE::default();
    assert_eq!(tsne.get_method(), TSNEMethod::BarnesHut { angle: 0.5 });
    assert_eq!(tsne.get_init(), Init::PCA);
}

/// Barnes-Hut produces an embedding of the right shape with only finite values
#[test]
fn fit_transform_barnes_hut_shape_and_finite() {
    let x = small_data();
    let tsne = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::BarnesHut { angle: 0.5 })
        .unwrap();
    let emb = tsne.fit_transform(&x).unwrap();
    assert_eq!(emb.shape(), &[6, 2]);
    for &v in emb.iter() {
        assert!(v.is_finite(), "Barnes-Hut embedding has a non-finite value");
    }
}

/// Barnes-Hut is deterministic for a fixed seed
#[test]
fn fit_transform_barnes_hut_deterministic_same_seed() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(7)
        .with_init(Init::Random)
        .with_method(TSNEMethod::BarnesHut { angle: 0.5 })
        .unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(7)
        .with_init(Init::Random)
        .with_method(TSNEMethod::BarnesHut { angle: 0.5 })
        .unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // The tree build and the sequential Z reduction keep Barnes-Hut bit-reproducible
    crate::common::assert_allclose(&emb1, &emb2, 0.0_f64);
}

/// Barnes-Hut keeps two well-separated high-D clusters apart in the embedding
#[test]
fn fit_transform_barnes_hut_well_separated_clusters_remain_separated() {
    use std::f64::consts::PI;
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

    let tsne = TSNE::new(2, 5.0, 100.0, 1000)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::BarnesHut { angle: 0.5 })
        .unwrap();
    let emb = tsne.fit_transform(&x).unwrap();

    let dist2d = |i: usize, j: usize| -> f64 {
        let dy0 = emb[[i, 0]] - emb[[j, 0]];
        let dy1 = emb[[i, 1]] - emb[[j, 1]];
        (dy0 * dy0 + dy1 * dy1).sqrt()
    };

    let centroid = |start: usize, end: usize| -> [f64; 2] {
        let count = (end - start) as f64;
        [
            (start..end).map(|i| emb[[i, 0]]).sum::<f64>() / count,
            (start..end).map(|i| emb[[i, 1]]).sum::<f64>() / count,
        ]
    };
    let cen_a = centroid(0, n_per_cluster);
    let cen_b = centroid(n_per_cluster, 2 * n_per_cluster);
    let inter_centroid = {
        let dx = cen_a[0] - cen_b[0];
        let dy = cen_a[1] - cen_b[1];
        (dx * dx + dy * dy).sqrt()
    };

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

    // The approximate forces still keep the clusters clearly apart
    assert!(
        inter_centroid > 4.0 * max_intra,
        "Barnes-Hut cluster separation weak: inter_centroid={inter_centroid:.6}, \
         max_intra={max_intra:.6}, ratio={:.3}",
        inter_centroid / max_intra.max(1e-10)
    );
}
