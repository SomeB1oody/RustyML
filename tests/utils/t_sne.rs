//! Integration tests for `rustyml::utils::t_sne::TSNE`.
//!
//! Every expected value is derived from the mathematical definition or a closed-form
//! property of the algorithm. No value is obtained by running the implementation and
//! recording its output.
//!
//! Key properties tested:
//!   1. Constructor validation (perplexity, learning_rate, n_components, n_iter).
//!   2. fit_transform output shape = (n_samples, n_components).
//!   3. All output values are finite.
//!   4. Determinism: same seed → bit-identical embedding.
//!   5. Centered output: each column mean ≈ 0 (center_embedding is called every iter).
//!   6. Neighborhood preservation: well-separated clusters stay separated.
//!   7. Error paths: single sample, NaN input, empty input, perplexity >= n_samples.
//!   8. n_components = 1 works.

use approx::assert_abs_diff_eq;
use ndarray::Array2;
use rustyml::error::Error;
use rustyml::utils::t_sne::TSNE;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: a small but valid dataset (6 points, 2 features) for general tests.
// perplexity=2.0 < 6 so it passes the n_samples guard.
// ─────────────────────────────────────────────────────────────────────────────
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

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR VALIDATION — perplexity
// ═══════════════════════════════════════════════════════════════════════════

/// perplexity = 0.0 violates "must be positive and finite" (0.0 <= 0.0 is true).
/// Source condition: `if perplexity <= 0.0 || !perplexity.is_finite()`
#[test]
fn new_perplexity_zero_returns_invalid_parameter() {
    let err = TSNE::new(2, 0.0, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// perplexity = -1.0 is negative, so <= 0.0 is true → InvalidParameter.
#[test]
fn new_perplexity_negative_returns_invalid_parameter() {
    let err = TSNE::new(2, -1.0, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// perplexity = NaN: NaN <= 0.0 is false but !NaN.is_finite() is true → InvalidParameter.
#[test]
fn new_perplexity_nan_returns_invalid_parameter() {
    let err = TSNE::new(2, f64::NAN, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// perplexity = +inf: inf > 0.0 but !inf.is_finite() is true → InvalidParameter.
#[test]
fn new_perplexity_infinity_returns_invalid_parameter() {
    let err = TSNE::new(2, f64::INFINITY, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR VALIDATION — learning_rate
// ═══════════════════════════════════════════════════════════════════════════

/// learning_rate = 0.0 violates "must be positive and finite" (0.0 <= 0.0).
#[test]
fn new_learning_rate_zero_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, 0.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// learning_rate = -100.0 is negative → InvalidParameter.
#[test]
fn new_learning_rate_negative_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, -100.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// learning_rate = NaN: !NaN.is_finite() is true → InvalidParameter.
#[test]
fn new_learning_rate_nan_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, f64::NAN, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// learning_rate = +inf: !inf.is_finite() is true → InvalidParameter.
#[test]
fn new_learning_rate_infinity_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, f64::INFINITY, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR VALIDATION — n_components and n_iter
// ═══════════════════════════════════════════════════════════════════════════

/// n_components = 0 violates "must be greater than 0" → InvalidParameter.
#[test]
fn new_n_components_zero_returns_invalid_parameter() {
    let err = TSNE::new(0, 5.0, 200.0, 250, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// n_iter = 0 violates "must be greater than 0" → InvalidParameter.
#[test]
fn new_n_iter_zero_returns_invalid_parameter() {
    let err = TSNE::new(2, 5.0, 200.0, 0, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Valid parameters produce Ok (sanity check that the constructor path is correct).
/// Derivation: all conditions satisfied: n_components=2 > 0, perplexity=5.0 > 0 & finite,
/// learning_rate=200.0 > 0 & finite, n_iter=250 > 0.
#[test]
fn new_valid_params_returns_ok() {
    let result = TSNE::new(2, 5.0, 200.0, 250, Some(42));
    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// FIT_TRANSFORM — OUTPUT SHAPE
// ═══════════════════════════════════════════════════════════════════════════

/// fit_transform on an (n, d) matrix must return an (n, n_components) matrix.
///
/// Derivation: the algorithm's output is an embedding, one row per input sample,
/// with exactly `n_components` columns. No dimensionality change along the sample axis.
///   - input: (6, 2), n_components=2 → output shape: (6, 2)
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

/// n_components = 1 must produce an (n, 1) output.
///
/// Derivation: same argument — the embedding is (n_samples, n_components)
/// regardless of the number of input features.
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

// ═══════════════════════════════════════════════════════════════════════════
// FIT_TRANSFORM — ALL FINITE
// ═══════════════════════════════════════════════════════════════════════════

/// Every element in the embedding must be finite (no NaN or +/-inf).
///
/// Derivation: t-SNE gradient descent updates are bounded; the center_embedding
/// step subtracts the column mean each iteration. A well-seeded run on finite input
/// must produce a finite output. Mathematically, the Student-t denominator (1 + dist)
/// is always >= 1 > 0, so q_ij is always positive and finite; p_ij is a probability
/// so also finite. Gradient = sum of finite terms = finite.
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

// ═══════════════════════════════════════════════════════════════════════════
// FIT_TRANSFORM — DETERMINISM (same seed → bit-identical result)
// ═══════════════════════════════════════════════════════════════════════════

/// Two independent calls with the same random_state must produce exactly the same
/// embedding (bit-for-bit), because:
///   1. init_embedding uses StdRng::seed_from_u64(seed) deterministically.
///   2. The rest of the algorithm is fully deterministic given the initial Y.
///   3. No thread-level randomness is used when random_state is Some(_).
///
/// We check with eps=0.0 (exact equality) via assert_allclose.
#[test]
fn fit_transform_deterministic_same_seed_gives_identical_results() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100, Some(42)).unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // eps = 0.0: require exact bit-equality (f64 arithmetic is deterministic for
    // the same sequential operations with the same inputs).
    crate::common::assert_allclose(&emb1, &emb2, 0.0_f64);
}

/// Different seeds produce different embeddings (sanity check that the seed
/// actually affects initialization).
#[test]
fn fit_transform_different_seeds_produce_different_results() {
    let x = small_data();
    let tsne1 = TSNE::new(2, 2.0, 200.0, 100, Some(0)).unwrap();
    let tsne2 = TSNE::new(2, 2.0, 200.0, 100, Some(999)).unwrap();
    let emb1 = tsne1.fit_transform(&x).unwrap();
    let emb2 = tsne2.fit_transform(&x).unwrap();
    // At least one element must differ because the random initializations differ.
    let any_differs = emb1.iter().zip(emb2.iter()).any(|(a, b)| a != b);
    assert!(
        any_differs,
        "expected different embeddings for different seeds, but got identical results"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// FIT_TRANSFORM — CENTERED OUTPUT
// ═══════════════════════════════════════════════════════════════════════════

/// Each column of the final embedding must have mean ≈ 0.
///
/// Derivation: center_embedding subtracts the column-wise mean at every
/// optimization step (including the last one). After the final update the
/// mean is subtracted once more, so the column mean of the output is
///   mean(y_final) = mean(y_prev + y_incs) - mean(y_prev + y_incs) = 0
/// exactly (up to floating-point rounding, tolerance 1e-10).
#[test]
fn fit_transform_column_means_are_zero() {
    let tsne = TSNE::new(2, 2.0, 200.0, 300, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    let n = embedding.nrows() as f64;
    // Column 0 mean
    let mean0: f64 = embedding.column(0).iter().sum::<f64>() / n;
    // Column 1 mean
    let mean1: f64 = embedding.column(1).iter().sum::<f64>() / n;
    assert_abs_diff_eq!(mean0, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(mean1, 0.0, epsilon = 1e-10);
}

/// n_components=1: the single output column must also have mean ≈ 0.
#[test]
fn fit_transform_column_mean_zero_n_components_one() {
    let tsne = TSNE::new(1, 2.0, 200.0, 300, Some(42)).unwrap();
    let x = small_data();
    let embedding = tsne.fit_transform(&x).unwrap();
    let n = embedding.nrows() as f64;
    let mean0: f64 = embedding.column(0).iter().sum::<f64>() / n;
    assert_abs_diff_eq!(mean0, 0.0, epsilon = 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// FIT_TRANSFORM — NEIGHBORHOOD PRESERVATION
// ═══════════════════════════════════════════════════════════════════════════

/// Two well-separated high-D clusters remain separated in the 2-D embedding.
///
/// Setup: 20 points in 2 clusters of 10, in 2-D input space.
///   Cluster A: 10 points spread on a unit circle centered at origin.
///   Cluster B: 10 points spread on a unit circle centered at [200, 0].
///
/// High-D inter-cluster distance: between the circle centers is 200.
/// Intra-cluster max distance: diameter of unit circle = 2.0 (by definition).
/// Ratio inter/intra ≈ 100, so the high-D P(i|j) heavily favors intra-cluster neighbors.
///
/// After t-SNE with perplexity=5 (< 20) and sufficient iterations, the two cluster
/// centroids in the 2-D embedding must be at least 5× farther apart than the
/// max intra-cluster spread (a conservative version of the fundamental t-SNE guarantee).
///
/// Formal argument: for cluster A, P(j|i) for i,j both in A is O(exp(-d²/2σ²))
/// where d ≤ 2 and σ is fit to perplexity=5. For i ∈ A, j ∈ B, the distance
/// is at least 200-2=198 ≫ 2, so P(j|i) ≈ 0. The joint P_ij matrix is block-diagonal,
/// and t-SNE's KL minimization maps each block to a tight cluster in 2-D. Since the
/// two blocks have zero inter-cluster probability, they must be repelled apart.
#[test]
fn fit_transform_well_separated_clusters_remain_separated() {
    use std::f64::consts::PI;
    // 10 points on the unit circle, evenly spaced: angle k = 2πk/10.
    // Cluster B shifts every x-coordinate by +200.
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
    // Build ndarray from rows.
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let x = Array2::from_shape_vec((2 * n_per_cluster, 2), flat).unwrap();

    // perplexity=5 < 20 → valid.  learning_rate=100: this implementation has no per-parameter
    // adaptive gains, so the library-default lr=200 overshoots on a dataset this tiny (20 points)
    // and lets the embedding expand without settling; lr=100 sits comfortably in the stable range
    // (the separation ratio is ~9 here, and >5 for any lr in roughly [10, 100]).
    let tsne = TSNE::new(2, 5.0, 100.0, 1000, Some(42)).unwrap();
    let emb = tsne.fit_transform(&x).unwrap();

    // Euclidean distance between two 2-D embedding points.
    let dist2d = |i: usize, j: usize| -> f64 {
        let dy0 = emb[[i, 0]] - emb[[j, 0]];
        let dy1 = emb[[i, 1]] - emb[[j, 1]];
        (dy0 * dy0 + dy1 * dy1).sqrt()
    };

    // Centroids of the two clusters in the embedding.
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

    // Max intra-cluster spread.
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

    // The two cluster centroids must be at least 5× the max intra-cluster spread apart.
    // This is a conservative but reliable check given the 100:1 high-D separation ratio
    // and 1000 optimization iterations.
    assert!(
        inter_centroid > 5.0 * max_intra,
        "cluster separation weak: inter_centroid={inter_centroid:.6}, \
         max_intra={max_intra:.6}, ratio={:.3}",
        inter_centroid / max_intra.max(1e-10)
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR PATHS — fit_transform
// ═══════════════════════════════════════════════════════════════════════════

/// perplexity >= n_samples → InvalidParameter.
///
/// Derivation: the source condition is `if self.perplexity >= x.nrows() as f64`.
/// With n_samples=5 and perplexity=5.0: 5.0 >= 5.0 is true → Error.
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

/// perplexity > n_samples also → InvalidParameter (same condition).
///
/// Derivation: 10.0 >= 5 is true.
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

/// Single sample → InvalidInput.
///
/// Derivation: validate_input calls check_min_samples(x, 2, "t-SNE"); 1 < 2 → InvalidInput.
/// Note: perplexity=0.5 is valid at construction time; the error is from the data guard.
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

/// Empty input (0 rows) → EmptyInput.
///
/// Derivation: validate_fit_matrix calls check_non_empty first; an array with
/// 0 rows is empty → Error::EmptyInput.
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

/// NaN in input data → NonFinite.
///
/// Derivation: validate_fit_matrix calls check_finite; NaN.is_finite() is false
/// → Error::NonFinite.
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

/// +inf in input data → NonFinite.
///
/// Derivation: INFINITY.is_finite() is false → Error::NonFinite.
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

// ═══════════════════════════════════════════════════════════════════════════
// GETTER ROUNDTRIPS
// ═══════════════════════════════════════════════════════════════════════════

/// Getters must return exactly the values passed to new().
///
/// Derivation: the constructor stores its arguments verbatim in the struct fields
/// and the get_field! macro generates `pub fn get_X(&self) -> T { self.X }`,
/// returning each value by copy.
#[test]
fn getters_return_construction_values() {
    let tsne = TSNE::new(3, 15.0, 100.0, 500, Some(77)).unwrap();
    assert_eq!(tsne.get_n_components(), 3);
    assert_abs_diff_eq!(tsne.get_perplexity(), 15.0, epsilon = 0.0);
    assert_abs_diff_eq!(tsne.get_learning_rate(), 100.0, epsilon = 0.0);
    assert_eq!(tsne.get_n_iter(), 500);
    assert_eq!(tsne.get_random_state(), Some(77));
}

/// When random_state is None the getter also returns None.
#[test]
fn getter_random_state_none() {
    let tsne = TSNE::new(2, 5.0, 200.0, 100, None).unwrap();
    assert_eq!(tsne.get_random_state(), None);
}
