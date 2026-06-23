//! Clustering evaluation metrics
//!
//! Provides extrinsic metrics that compare a clustering against ground-truth labels (NMI, AMI, ARI,
//! homogeneity, completeness, V-measure, Fowlkes-Mallows) and intrinsic metrics that score a
//! clustering from the feature geometry alone (silhouette, Davies-Bouldin, Calinski-Harabasz)

use ahash::AHashMap;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::validate_pair;
use crate::math::squared_euclidean_distance_row;
pub use crate::types::DistanceCalculationMetric;

/// Denominator magnitude below which an AMI/ARI normaliser is treated as a degenerate (perfect)
/// clustering and the score is defined to be `1.0`
const DEGENERATE_DENOM: f64 = 1e-10;

tunable_gate! {
    /// Total scanned-element work (`n * n * d`) at or above which [`silhouette_score`] fills its
    /// pairwise-distance matrix in parallel; below it the serial path is used
    ///
    /// The fill is n tasks of an O(n * d) distance-row scan each, the same cost class as the crate's
    /// calibrated f64 row-scan gate (crossover bracket 65K-262K scanned elements). The constant is
    /// restated here rather than imported because `metrics` stays a lightweight leaf module
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) SILHOUETTE_PARALLEL_MIN_ELEMS => silhouette_parallel_min_elems / set_silhouette_parallel_min_elems = 262_144
}

/// Maps each distinct label to a dense index in `0..k` in order of first appearance
fn label_index(labels: &[usize]) -> AHashMap<usize, usize> {
    let mut index = AHashMap::new();
    for &label in labels {
        let next = index.len();
        index.entry(label).or_insert(next);
    }
    index
}

/// Builds the contingency matrix of two label assignments together with its row and column sums
/// (the cluster sizes in `labels_true` and `labels_pred` respectively)
fn contingency_matrix(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> (Array2<usize>, Vec<usize>, Vec<usize>) {
    let index_true = label_index(labels_true);
    let index_pred = label_index(labels_pred);

    let mut matrix = Array2::<usize>::zeros((index_true.len(), index_pred.len()));
    for (&lt, &lp) in labels_true.iter().zip(labels_pred.iter()) {
        matrix[[index_true[&lt], index_pred[&lp]]] += 1;
    }

    let row_sums = matrix.sum_axis(Axis(1)).to_vec();
    let col_sums = matrix.sum_axis(Axis(0)).to_vec();
    (matrix, row_sums, col_sums)
}

/// Computes the mutual information (in nats): `MI = sum_ij (n_ij/n) * ln(n*n_ij / (a_i*b_j))`
fn mutual_information(
    contingency: &Array2<usize>,
    n: usize,
    row_sums: &[usize],
    col_sums: &[usize],
) -> f64 {
    let n_f = n as f64;
    let mut mi = 0.0;
    for ((i, j), &n_ij) in contingency.indexed_iter() {
        if n_ij > 0 {
            let n_ij_f = n_ij as f64;
            let a = row_sums[i] as f64;
            let b = col_sums[j] as f64;
            mi += (n_ij_f / n_f) * ((n_f * n_ij_f) / (a * b)).ln();
        }
    }
    mi
}

/// Computes the entropy (in nats) of a partition from its cluster sizes: `H = -sum_i p_i * ln(p_i)`
fn entropy_nats(counts: &[usize], n: usize) -> f64 {
    let n_f = n as f64;
    let mut h = 0.0;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / n_f;
            h -= p * p.ln();
        }
    }
    h
}

/// Builds a table of natural log-factorials, `table[i] = ln(i!)`, for `i` in `0..=n_max`
///
/// Computed by cumulative summation so that each log-binomial coefficient needed by the expected
/// mutual information becomes an `O(1)` lookup rather than an `O(k)` sum
fn ln_factorial_table(n_max: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(n_max + 1);
    table.push(0.0); // ln(0!) = ln(1) = 0
    let mut acc = 0.0;
    for i in 1..=n_max {
        acc += (i as f64).ln();
        table.push(acc);
    }
    table
}

/// Computes the expected mutual information (EMI) under the hypergeometric model of random
/// clusterings with the given cluster sizes
///
/// For each pair of clusters `(a_i, b_j)` the overlap `n_ij` follows a hypergeometric distribution;
/// EMI sums `P(n_ij = k) * (k/n) * ln(n*k / (a_i*b_j))` over its support. All binomial coefficients
/// are evaluated in log space from a shared log-factorial table, so the whole computation is
/// `O(R*C*range)` with `O(1)` inner work
fn expected_mutual_information(row_sums: &[usize], col_sums: &[usize], n: usize) -> f64 {
    let n_f = n as f64;
    let ln_fact = ln_factorial_table(n);
    // log C(a, b) = ln(a!) - ln(b!) - ln((a-b)!); valid for 0 <= b <= a <= n
    let log_binom = |a: usize, b: usize| ln_fact[a] - ln_fact[b] - ln_fact[a - b];

    let mut emi = 0.0;
    for &a_i in row_sums {
        for &b_j in col_sums {
            // Hypergeometric support max(0, a_i + b_j - n) <= k <= min(a_i, b_j); k = 0 adds nothing
            // to MI, so start from 1
            let lower = (a_i + b_j).saturating_sub(n).max(1);
            let upper = a_i.min(b_j);

            // log C(n, b_j) is constant across k for this pair; hoist it out of the loop
            let log_c_n_bj = log_binom(n, b_j);
            for k in lower..=upper {
                // ln P(k) = log C(a_i, k) + log C(n - a_i, b_j - k) - log C(n, b_j)
                let log_p = log_binom(a_i, k) + log_binom(n - a_i, b_j - k) - log_c_n_bj;
                let term = (k as f64 / n_f) * ((n_f * k as f64) / (a_i as f64 * b_j as f64)).ln();
                emi += log_p.exp() * term;
            }
        }
    }
    emi
}

/// Collects a `usize` label array into a contiguous `Vec`, tolerating non-contiguous views
fn to_label_vec<S>(labels: &ArrayBase<S, Ix1>) -> Vec<usize>
where
    S: Data<Elem = usize>,
{
    labels.iter().copied().collect()
}

/// Calculates the Normalized Mutual Information (NMI) between two cluster assignments
///
/// NMI normalizes the mutual information by the arithmetic mean of the two clusterings' entropies,
/// giving `0.0` for independent assignments and `1.0` for identical ones. If either clustering has
/// zero entropy (a single cluster), NMI is defined as `0.0`
///
/// # Parameters
///
/// - `labels_true` - Ground-truth cluster assignment for each sample
/// - `labels_pred` - Predicted cluster assignment for each sample
///
/// # Returns
///
/// - `f64` - Normalized mutual information in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::normalized_mutual_info;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 1, 2, 1, 2];
/// let nmi = normalized_mutual_info(&labels_true, &labels_pred);
/// println!("Normalized Mutual Information: {:.4}", nmi);
/// ```
pub fn normalized_mutual_info<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );

    let n = labels_true.len();
    let labels_true = to_label_vec(labels_true);
    let labels_pred = to_label_vec(labels_pred);

    let (contingency, row_sums, col_sums) = contingency_matrix(&labels_true, &labels_pred);
    let mi = mutual_information(&contingency, n, &row_sums, &col_sums);
    let h_true = entropy_nats(&row_sums, n);
    let h_pred = entropy_nats(&col_sums, n);

    // Arithmetic mean of the entropies
    let denominator = (h_true + h_pred) / 2.0;
    if denominator == 0.0 {
        0.0
    } else {
        mi / denominator
    }
}

/// Calculates the Adjusted Mutual Information (AMI) between two cluster assignments
///
/// AMI corrects the mutual information for the agreement expected by chance, scoring `1.0` for
/// identical clusterings and about `0.0` for independent ones (it can be slightly negative). When
/// the normaliser is degenerate (e.g. both clusterings put every sample in one cluster) the score
/// is defined as `1.0`
///
/// # Parameters
///
/// - `labels_true` - Ground-truth cluster assignment for each sample
/// - `labels_pred` - Predicted cluster assignment for each sample
///
/// # Returns
///
/// - `f64` - Adjusted mutual information (typically in `[-1.0, 1.0]`)
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::adjusted_mutual_info;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 1, 2, 1, 2];
/// let ami = adjusted_mutual_info(&labels_true, &labels_pred);
/// println!("Adjusted Mutual Information: {:.4}", ami);
/// ```
pub fn adjusted_mutual_info<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );

    let n = labels_true.len();
    let labels_true = to_label_vec(labels_true);
    let labels_pred = to_label_vec(labels_pred);

    let (contingency, row_sums, col_sums) = contingency_matrix(&labels_true, &labels_pred);
    let mi = mutual_information(&contingency, n, &row_sums, &col_sums);
    let h_true = entropy_nats(&row_sums, n);
    let h_pred = entropy_nats(&col_sums, n);
    let emi = expected_mutual_information(&row_sums, &col_sums, n);

    let denominator = (h_true + h_pred) / 2.0 - emi;
    if denominator.abs() < DEGENERATE_DENOM {
        1.0
    } else {
        (mi - emi) / denominator
    }
}

/// Calculates the Adjusted Rand Index (ARI) between two cluster assignments
///
/// ARI is the Rand index (the fraction of sample pairs that two clusterings agree on, whether by
/// grouping them together or apart) corrected for chance: `1.0` for identical clusterings, about
/// `0.0` for independent ones, and possibly negative for worse-than-random agreement. When the
/// normaliser is degenerate (e.g. fewer than two samples, or both clusterings trivial) the score
/// is defined as `1.0`
///
/// # Parameters
///
/// - `labels_true` - Ground-truth cluster assignment for each sample
/// - `labels_pred` - Predicted cluster assignment for each sample
///
/// # Returns
///
/// - `f64` - Adjusted Rand Index (typically in `[-0.5, 1.0]`)
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::adjusted_rand_index;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 1, 2, 1, 2];
/// let ari = adjusted_rand_index(&labels_true, &labels_pred);
/// println!("Adjusted Rand Index: {:.4}", ari);
/// ```
pub fn adjusted_rand_index<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );

    let n = labels_true.len();
    let labels_true = to_label_vec(labels_true);
    let labels_pred = to_label_vec(labels_pred);

    let (contingency, row_sums, col_sums) = contingency_matrix(&labels_true, &labels_pred);

    // Number of unordered pairs within a group of `size` elements: C(size, 2)
    let comb2 = |size: usize| {
        let size = size as f64;
        size * (size - 1.0) / 2.0
    };

    let sum_comb_cells: f64 = contingency.iter().map(|&n_ij| comb2(n_ij)).sum();
    let sum_comb_true: f64 = row_sums.iter().map(|&a| comb2(a)).sum();
    let sum_comb_pred: f64 = col_sums.iter().map(|&b| comb2(b)).sum();
    let comb_n = comb2(n);

    if comb_n == 0.0 {
        return 1.0; // fewer than two samples: no pairs to disagree on
    }

    let expected = sum_comb_true * sum_comb_pred / comb_n;
    let max_index = 0.5 * (sum_comb_true + sum_comb_pred);
    let denominator = max_index - expected;

    if denominator.abs() < DEGENERATE_DENOM {
        1.0
    } else {
        (sum_comb_cells - expected) / denominator
    }
}

/// Accumulates the symmetric pairwise-distance contributions of the upper-triangle rows in `rows`
/// into `acc`: for each `i` in `rows` and each `j > i`, the single distance `d(i, j)` is added to
/// both `acc[[i, cluster[j]]]` and `acc[[j, cluster[i]]]`
///
/// Visiting only `j > i` (and relying on `d` being symmetric) computes each unordered pair once.
/// The skipped diagonal term `d(i, i) = 0` is a no-op for the running sum, so the result is
/// identical to a full `n x n` scan
fn accumulate_upper_triangle<S>(
    acc: &mut Array2<f64>,
    rows: impl Iterator<Item = usize>,
    x: &ArrayBase<S, Ix2>,
    cluster: &[usize],
    metric: DistanceCalculationMetric,
) where
    S: Data<Elem = f64>,
{
    let n = x.nrows();
    for i in rows {
        let xi = x.row(i);
        for j in (i + 1)..n {
            let d = metric.distance(xi, x.row(j));
            acc[[i, cluster[j]]] += d;
            acc[[j, cluster[i]]] += d;
        }
    }
}

/// Builds `dist_to_cluster[[i, c]]` = total distance from sample `i` to every sample in cluster
/// `c`, exploiting distance symmetry so each unordered pair is evaluated once (halving the metric
/// calls versus a full `n x n` scan)
///
/// Below the scanned-element gate the upper triangle is folded serially, giving a result **bitwise
/// identical** to the old full-scan fill (the only difference, the `d(i, i) = 0` self term, adds
/// nothing to a sum of non-negative distances). At or above the gate the upper-triangle rows are
/// dealt round-robin into `current_num_threads()` buckets - row `i` does `n - 1 - i` pair
/// evaluations, so contiguous splits would be lopsided while interleaving balances them - and each
/// bucket folds into its own private `(n, k)` accumulator before they are summed in bucket order.
/// That fixed grouping keeps the parallel result **run-to-run deterministic at a fixed thread
/// count** (numerically equal to the serial fill, though not bitwise identical across thread
/// counts, since the per-cell summation order changes with the bucket count - the same tradeoff
/// the `gemm` row-split path makes)
fn pairwise_cluster_distances<S>(
    x: &ArrayBase<S, Ix2>,
    cluster: &[usize],
    k: usize,
    metric: DistanceCalculationMetric,
) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let n = x.nrows();
    let scan_work = n.saturating_mul(n).saturating_mul(x.ncols());

    if scan_work < silhouette_parallel_min_elems() {
        let mut dist = Array2::<f64>::zeros((n, k));
        accumulate_upper_triangle(&mut dist, 0..n, x, cluster, metric);
        return dist;
    }

    // Round-robin the rows into per-thread buckets
    let chunks = rayon::current_num_threads().max(1).min(n);
    let partials: Vec<Array2<f64>> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let mut acc = Array2::<f64>::zeros((n, k));
            accumulate_upper_triangle(&mut acc, (c..n).step_by(chunks), x, cluster, metric);
            acc
        })
        .collect();
    partials
        .into_iter()
        .reduce(|mut sum, p| {
            sum += &p;
            sum
        })
        .unwrap_or_else(|| Array2::<f64>::zeros((n, k)))
}

/// Calculates the mean Silhouette Coefficient over all samples under the given distance metric
///
/// For each sample, the silhouette `s = (b - a) / max(a, b)` compares the mean intra-cluster
/// distance `a` with the mean distance `b` to the nearest other cluster; it ranges from `-1`
/// (likely misassigned) through `0` (on a cluster boundary) to `+1` (well clustered). Samples that
/// are the sole member of their cluster contribute `0`. The returned score is the mean over all
/// samples
///
/// Pairwise distances go through [`DistanceCalculationMetric`], the same dispatch point used by the
/// estimators, so any of `Euclidean`, `Manhattan`, or `Minkowski(p)` works. Pass
/// `DistanceCalculationMetric::Euclidean` for the conventional silhouette
///
/// # Parameters
///
/// - `x` - Feature matrix with one sample per row (`n_samples x n_features`)
/// - `labels` - Cluster assignment for each sample
/// - `metric` - Distance metric used for every pairwise distance
///
/// # Returns
///
/// - `f64` - Mean silhouette coefficient in `[-1.0, 1.0]`
///
/// # Panics
///
/// - Panics if the number of rows in `x` differs from the length of `labels`
/// - Panics if the inputs are empty
/// - Panics if the number of distinct clusters is not in `2..=n_samples - 1`
/// - Panics if `metric` is `Minkowski(p)` with `p < 1`
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::silhouette_score;
/// use rustyml::types::DistanceCalculationMetric;
///
/// let x = array![[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]];
/// let labels = array![0, 0, 1, 1];
/// let score = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
/// assert!(score > 0.8);
/// ```
pub fn silhouette_score<S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, Ix1>,
    metric: DistanceCalculationMetric,
) -> f64
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = usize>,
{
    let n = x.nrows();
    let labels = to_label_vec(labels);
    let (cluster, k) = validate_clustering_inputs(n, &labels);

    let mut sizes = vec![0usize; k];
    for &c in &cluster {
        sizes[c] += 1;
    }

    // dist_to_cluster[[i, c]] accumulates the total distance from sample i to every sample in
    // cluster c, computing each unordered pair only once (distances are symmetric)
    let dist_to_cluster = pairwise_cluster_distances(x, &cluster, k, metric);

    let mut total = 0.0;
    for i in 0..n {
        let own = cluster[i];
        if sizes[own] <= 1 {
            continue; // lone sample contributes a silhouette of 0
        }

        let a = dist_to_cluster[[i, own]] / (sizes[own] - 1) as f64;
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c != own {
                let mean_dist = dist_to_cluster[[i, c]] / sizes[c] as f64;
                if mean_dist < b {
                    b = mean_dist;
                }
            }
        }

        let denominator = a.max(b);
        if denominator > 0.0 {
            total += (b - a) / denominator;
        }
    }

    total / n as f64
}

/// Densifies labels to `0..k` and validates them against an `x` of `n_rows` rows for an internal
/// clustering metric: equal length, non-empty, and `2..=n_rows - 1` distinct clusters. Returns the
/// dense cluster index of each sample and the cluster count `k`
fn validate_clustering_inputs(n_rows: usize, labels: &[usize]) -> (Vec<usize>, usize) {
    if n_rows != labels.len() {
        panic!(
            "dimension mismatch: expected {n_rows}, found {}",
            labels.len()
        );
    }
    if n_rows == 0 {
        panic!("input is empty: x and labels");
    }
    let index = label_index(labels);
    let k = index.len();
    if k < 2 || k >= n_rows {
        panic!(
            "invalid input: number of clusters is {k}, valid range is 2 to n_samples - 1 ({})",
            n_rows - 1
        );
    }
    let cluster = labels.iter().map(|label| index[label]).collect();
    (cluster, k)
}

/// Computes each cluster's centroid (mean of its points) and size, given dense cluster indices
fn cluster_centroids<S>(
    x: &ArrayBase<S, Ix2>,
    cluster: &[usize],
    k: usize,
) -> (Array2<f64>, Vec<usize>)
where
    S: Data<Elem = f64>,
{
    let mut centroids = Array2::<f64>::zeros((k, x.ncols()));
    let mut sizes = vec![0usize; k];
    for (i, &c) in cluster.iter().enumerate() {
        sizes[c] += 1;
        let mut centroid = centroids.row_mut(c);
        centroid += &x.row(i);
    }
    for (mut centroid, &size) in centroids.axis_iter_mut(Axis(0)).zip(sizes.iter()) {
        if size > 0 {
            centroid /= size as f64;
        }
    }
    (centroids, sizes)
}

/// Returns `(homogeneity, completeness)` for two label assignments
///
/// Both reuse the mutual information and entropies already defined above: with `C` the classes
/// (`labels_true`) and `K` the clusters (`labels_pred`), homogeneity is `MI / H(C)` and
/// completeness is `MI / H(K)`. A zero entropy (single cluster) makes its score 1.0
fn homogeneity_completeness(labels_true: &[usize], labels_pred: &[usize], n: usize) -> (f64, f64) {
    let (contingency, row_sums, col_sums) = contingency_matrix(labels_true, labels_pred);
    let mi = mutual_information(&contingency, n, &row_sums, &col_sums);
    let h_classes = entropy_nats(&row_sums, n);
    let h_clusters = entropy_nats(&col_sums, n);

    let homogeneity = if h_classes == 0.0 {
        1.0
    } else {
        mi / h_classes
    };
    let completeness = if h_clusters == 0.0 {
        1.0
    } else {
        mi / h_clusters
    };
    (homogeneity, completeness)
}

/// Calculates the homogeneity of a clustering: the degree to which each cluster contains only
/// members of a single ground-truth class
///
/// Scores range from 0.0 to 1.0, with 1.0 for perfectly homogeneous clusters
///
/// # Parameters
///
/// - `labels_true` - Ground-truth class of each sample
/// - `labels_pred` - Predicted cluster of each sample
///
/// # Returns
///
/// - `f64` - Homogeneity score in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::homogeneity_score;
///
/// let labels_true = array![0, 0, 1, 1];
/// let labels_pred = array![0, 0, 1, 1];
/// assert!((homogeneity_score(&labels_true, &labels_pred) - 1.0).abs() < 1e-12);
/// ```
pub fn homogeneity_score<S>(labels_true: &ArrayBase<S, Ix1>, labels_pred: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );
    let n = labels_true.len();
    homogeneity_completeness(&to_label_vec(labels_true), &to_label_vec(labels_pred), n).0
}

/// Calculates the completeness of a clustering: the degree to which all members of a given
/// ground-truth class are assigned to the same cluster
///
/// Scores range from 0.0 to 1.0, with 1.0 for perfectly complete clusters. Completeness is the
/// dual of [`homogeneity_score`] (swapping the roles of the two labelings)
///
/// # Parameters
///
/// - `labels_true` - Ground-truth class of each sample
/// - `labels_pred` - Predicted cluster of each sample
///
/// # Returns
///
/// - `f64` - Completeness score in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::completeness_score;
///
/// let labels_true = array![0, 0, 1, 1];
/// let labels_pred = array![0, 0, 1, 1];
/// assert!((completeness_score(&labels_true, &labels_pred) - 1.0).abs() < 1e-12);
/// ```
pub fn completeness_score<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );
    let n = labels_true.len();
    homogeneity_completeness(&to_label_vec(labels_true), &to_label_vec(labels_pred), n).1
}

/// Calculates the V-measure: the harmonic mean of [`homogeneity_score`] and [`completeness_score`]
///
/// V-measure is symmetric in the two labelings and equals the [`normalized_mutual_info`] computed
/// with arithmetic-mean normalization. Scores range from 0.0 to 1.0
///
/// # Parameters
///
/// - `labels_true` - Ground-truth class of each sample
/// - `labels_pred` - Predicted cluster of each sample
///
/// # Returns
///
/// - `f64` - V-measure score in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::v_measure_score;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 1, 1, 2, 2];
/// assert!((v_measure_score(&labels_true, &labels_pred) - 1.0).abs() < 1e-12);
/// ```
pub fn v_measure_score<S>(labels_true: &ArrayBase<S, Ix1>, labels_pred: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );
    let n = labels_true.len();
    let (homogeneity, completeness) =
        homogeneity_completeness(&to_label_vec(labels_true), &to_label_vec(labels_pred), n);

    if homogeneity + completeness == 0.0 {
        0.0
    } else {
        2.0 * homogeneity * completeness / (homogeneity + completeness)
    }
}

/// Calculates the Fowlkes-Mallows index (FMI) between two cluster assignments
///
/// FMI is the geometric mean of the pairwise precision and recall over sample pairs:
/// `TP / sqrt((TP + FP) * (TP + FN))`, where the counts are over pairs grouped together by each
/// clustering. Scores range from 0.0 to 1.0, with 1.0 for identical clusterings
///
/// # Parameters
///
/// - `labels_true` - Ground-truth cluster assignment of each sample
/// - `labels_pred` - Predicted cluster assignment of each sample
///
/// # Returns
///
/// - `f64` - Fowlkes-Mallows index in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels_true` and `labels_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::fowlkes_mallows_score;
///
/// let labels_true = array![0, 0, 1, 1];
/// let labels_pred = array![0, 0, 1, 1];
/// assert!((fowlkes_mallows_score(&labels_true, &labels_pred) - 1.0).abs() < 1e-12);
/// ```
pub fn fowlkes_mallows_score<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    validate_pair(
        labels_true.len(),
        labels_pred.len(),
        "labels_true and labels_pred",
    );

    let labels_true = to_label_vec(labels_true);
    let labels_pred = to_label_vec(labels_pred);
    let (contingency, row_sums, col_sums) = contingency_matrix(&labels_true, &labels_pred);

    let comb2 = |size: usize| {
        let size = size as f64;
        size * (size - 1.0) / 2.0
    };
    let tk: f64 = contingency.iter().map(|&n_ij| comb2(n_ij)).sum(); // pairs together in both
    let pk: f64 = col_sums.iter().map(|&b| comb2(b)).sum(); // pairs together in pred
    let qk: f64 = row_sums.iter().map(|&a| comb2(a)).sum(); // pairs together in true

    let denominator = (pk * qk).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        tk / denominator
    }
}

/// Calculates the Davies-Bouldin index of a clustering using Euclidean distance
///
/// Each cluster's worst-case similarity to another is `(s_i + s_j) / d(c_i, c_j)`, where `s` is the
/// mean distance of a cluster's points to its centroid `c` and `d` is the centroid distance; the
/// index is the average of these maxima. **Lower is better** (0.0 is ideal), making it a cheaper
/// `O(n * k)` complement to [`silhouette_score`] for evaluating a clustering without ground truth
///
/// # Parameters
///
/// - `x` - Feature matrix with one sample per row (`n_samples x n_features`)
/// - `labels` - Cluster assignment of each sample
///
/// # Returns
///
/// - `f64` - Davies-Bouldin index (>= 0.0; lower is better)
///
/// # Panics
///
/// - Panics if the number of rows in `x` differs from the length of `labels`
/// - Panics if the inputs are empty
/// - Panics if the number of distinct clusters is not in `2..=n_samples - 1`
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::davies_bouldin_score;
///
/// let x = array![[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]];
/// let labels = array![0, 0, 1, 1];
/// assert!(davies_bouldin_score(&x, &labels) < 0.1); // well separated
/// ```
pub fn davies_bouldin_score<S1, S2>(x: &ArrayBase<S1, Ix2>, labels: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = usize>,
{
    let labels = to_label_vec(labels);
    let (cluster, k) = validate_clustering_inputs(x.nrows(), &labels);
    let (centroids, sizes) = cluster_centroids(x, &cluster, k);

    // s[c] = mean distance from cluster c's points to its centroid
    let mut s = vec![0.0_f64; k];
    for (i, &c) in cluster.iter().enumerate() {
        s[c] += squared_euclidean_distance_row(&x.row(i), &centroids.row(c)).sqrt();
    }
    for (distance, &size) in s.iter_mut().zip(sizes.iter()) {
        if size > 0 {
            *distance /= size as f64;
        }
    }

    let mut db = 0.0;
    for i in 0..k {
        let mut max_ratio = 0.0_f64;
        for j in 0..k {
            if i != j {
                let centroid_dist =
                    squared_euclidean_distance_row(&centroids.row(i), &centroids.row(j)).sqrt();
                if centroid_dist > 0.0 {
                    max_ratio = max_ratio.max((s[i] + s[j]) / centroid_dist);
                }
            }
        }
        db += max_ratio;
    }
    db / k as f64
}

/// Calculates the Calinski-Harabasz index (variance ratio criterion) of a clustering
///
/// The ratio of between-cluster dispersion to within-cluster dispersion, scaled by
/// `(n - k) / (k - 1)`. **Higher is better**: well-separated, compact clusters score high. Returns
/// 1.0 in the degenerate case where every point coincides with its centroid (zero within-cluster
/// dispersion)
///
/// # Parameters
///
/// - `x` - Feature matrix with one sample per row (`n_samples x n_features`)
/// - `labels` - Cluster assignment of each sample
///
/// # Returns
///
/// - `f64` - Calinski-Harabasz index (>= 0.0; higher is better)
///
/// # Panics
///
/// - Panics if the number of rows in `x` differs from the length of `labels`
/// - Panics if the inputs are empty
/// - Panics if the number of distinct clusters is not in `2..=n_samples - 1`
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::calinski_harabasz_score;
///
/// let x = array![[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]];
/// let labels = array![0, 0, 1, 1];
/// assert!(calinski_harabasz_score(&x, &labels) > 100.0); // well separated
/// ```
pub fn calinski_harabasz_score<S1, S2>(x: &ArrayBase<S1, Ix2>, labels: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = usize>,
{
    let n = x.nrows();
    let labels = to_label_vec(labels);
    let (cluster, k) = validate_clustering_inputs(n, &labels);
    let (centroids, sizes) = cluster_centroids(x, &cluster, k);
    let overall = x.mean_axis(Axis(0)).unwrap(); // n > 0 is guaranteed

    let mut between = 0.0;
    for (centroid, &size) in centroids.axis_iter(Axis(0)).zip(sizes.iter()) {
        between += size as f64 * squared_euclidean_distance_row(&centroid, &overall);
    }
    let mut within = 0.0;
    for (i, &c) in cluster.iter().enumerate() {
        within += squared_euclidean_distance_row(&x.row(i), &centroids.row(c));
    }

    if within == 0.0 {
        return 1.0;
    }
    (between / within) * ((n - k) as f64 / (k - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // label_index

    /// First-appearance dense remap of distinct labels to first-seen ordinal positions
    #[test]
    fn test_label_index_first_appearance_order() {
        let idx = label_index(&[5, 5, 3, 3, 10]);
        assert_eq!(idx.len(), 3, "three distinct labels expected");
        assert_eq!(idx[&5], 0, "5 appears first, should map to 0");
        assert_eq!(idx[&3], 1, "3 appears second, should map to 1");
        assert_eq!(idx[&10], 2, "10 appears third, should map to 2");
    }

    /// A single repeated label maps to index 0 only
    #[test]
    fn test_label_index_single_label() {
        let idx = label_index(&[7, 7, 7]);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[&7], 0);
    }

    /// All distinct labels each get a unique index in [0, n)
    #[test]
    fn test_label_index_all_distinct() {
        let idx = label_index(&[10, 20, 30]);
        assert_eq!(idx.len(), 3);
        // each appears once in first-seen order: 10->0, 20->1, 30->2
        assert_eq!(idx[&10], 0);
        assert_eq!(idx[&20], 1);
        assert_eq!(idx[&30], 2);
    }

    // contingency_matrix

    /// Every pairing appearing once produces the all-ones 2x2 matrix with row_sums=col_sums=[2,2]
    #[test]
    fn test_contingency_matrix_uniform() {
        let (mat, row_sums, col_sums) = contingency_matrix(&[0, 0, 1, 1], &[0, 1, 0, 1]);
        assert_eq!(mat.shape(), &[2, 2]);
        assert_eq!(mat[[0, 0]], 1);
        assert_eq!(mat[[0, 1]], 1);
        assert_eq!(mat[[1, 0]], 1);
        assert_eq!(mat[[1, 1]], 1);
        assert_eq!(row_sums, vec![2, 2]);
        assert_eq!(col_sums, vec![2, 2]);
    }

    /// Identical labels give a diagonal contingency matrix
    #[test]
    fn test_contingency_matrix_identical_labels() {
        let (mat, row_sums, col_sums) = contingency_matrix(&[0, 0, 1, 1], &[0, 0, 1, 1]);
        assert_eq!(mat.shape(), &[2, 2]);
        assert_eq!(mat[[0, 0]], 2, "class-0 samples all in pred-cluster-0");
        assert_eq!(mat[[0, 1]], 0);
        assert_eq!(mat[[1, 0]], 0);
        assert_eq!(mat[[1, 1]], 2, "class-1 samples all in pred-cluster-1");
        assert_eq!(row_sums, vec![2, 2]);
        assert_eq!(col_sums, vec![2, 2]);
    }

    // entropy_nats

    /// Two equal clusters give H = ln(2)
    #[test]
    fn test_entropy_nats_two_equal_clusters() {
        let h = entropy_nats(&[2, 2], 4);
        assert_abs_diff_eq!(h, std::f64::consts::LN_2, epsilon = 1e-10);
    }

    /// A single cluster covering all samples gives H = 0
    #[test]
    fn test_entropy_nats_single_cluster() {
        let h = entropy_nats(&[4], 4);
        assert_abs_diff_eq!(h, 0.0, epsilon = 1e-10);
    }

    /// Four equal singleton clusters give H = ln(4)
    #[test]
    fn test_entropy_nats_four_equal_clusters() {
        let h = entropy_nats(&[1, 1, 1, 1], 4);
        let expected = (4.0_f64).ln(); // ln(4)
        assert_abs_diff_eq!(h, expected, epsilon = 1e-10);
    }

    /// A zero-count entry contributes nothing to the entropy
    #[test]
    fn test_entropy_nats_zero_count_skipped() {
        let h = entropy_nats(&[0, 4], 4);
        assert_abs_diff_eq!(h, 0.0, epsilon = 1e-10);
    }

    // mutual_information

    /// Independent uniform assignment gives MI = 0
    #[test]
    fn test_mutual_information_independent() {
        let mat = array![[1usize, 1], [1, 1]];
        let mi = mutual_information(&mat, 4, &[2, 2], &[2, 2]);
        assert_abs_diff_eq!(mi, 0.0, epsilon = 1e-10);
    }

    /// Perfect assignment gives MI = H(true) = ln(2)
    #[test]
    fn test_mutual_information_identical() {
        let mat = array![[2usize, 0], [0, 2]];
        let mi = mutual_information(&mat, 4, &[2, 2], &[2, 2]);
        assert_abs_diff_eq!(mi, std::f64::consts::LN_2, epsilon = 1e-10);
    }

    // homogeneity_completeness

    /// Pure clusters (more clusters than classes) give homogeneity 1.0, completeness 0.5, and pin
    /// the tuple ordering (homogeneity first)
    #[test]
    fn test_homogeneity_completeness_pure_clusters() {
        let labels_true = [0usize, 0, 1, 1];
        let labels_pred = [0usize, 1, 2, 3];
        let (h, c) = homogeneity_completeness(&labels_true, &labels_pred, 4);
        assert_abs_diff_eq!(h, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c, 0.5, epsilon = 1e-10);
    }

    /// Identical labels give homogeneity and completeness both 1.0
    #[test]
    fn test_homogeneity_completeness_identical() {
        let labels = [0usize, 0, 1, 1];
        let (h, c) = homogeneity_completeness(&labels, &labels, 4);
        assert_abs_diff_eq!(h, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c, 1.0, epsilon = 1e-10);
    }

    /// Swapping the roles of the pure-clusters case swaps the scores: homogeneity 0.5, completeness 1.0
    #[test]
    fn test_homogeneity_completeness_swapped_roles() {
        let labels_true = [0usize, 1, 2, 3];
        let labels_pred = [0usize, 0, 1, 1];
        let (h, c) = homogeneity_completeness(&labels_true, &labels_pred, 4);
        assert_abs_diff_eq!(h, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(c, 1.0, epsilon = 1e-10);
    }

    // entropy_nats: additional edge cases

    /// Unequal clusters [1,3] match the closed-form entropy
    #[test]
    fn test_entropy_nats_unequal_clusters() {
        let h = entropy_nats(&[1, 3], 4);
        // H = -(1/4)*ln(1/4) - (3/4)*ln(3/4)
        let expected =
            -(1.0_f64 / 4.0) * (1.0_f64 / 4.0).ln() - (3.0_f64 / 4.0) * (3.0_f64 / 4.0).ln();
        assert_abs_diff_eq!(h, expected, epsilon = 1e-10);
    }

    // mutual_information: non-trivial case

    /// Four pure clusters over two true classes give MI = ln(2)
    #[test]
    fn test_mutual_information_pure_clusters() {
        // build contingency directly to avoid depending on contingency_matrix correctness
        let mat = array![[1usize, 1, 0, 0], [0, 0, 1, 1]];
        let mi = mutual_information(&mat, 4, &[2, 2], &[1, 1, 1, 1]);
        assert_abs_diff_eq!(mi, std::f64::consts::LN_2, epsilon = 1e-10);
    }
    // expected_mutual_information (EMI hypergeometric kernel)

    /// EMI for the symmetric 2x2 case (row_sums=[2,2], col_sums=[2,2], n=4) equals ln(2)/3
    #[test]
    fn test_expected_mutual_information_symmetric_2x2() {
        let emi = expected_mutual_information(&[2, 2], &[2, 2], 4);
        let expected = std::f64::consts::LN_2 / 3.0;
        assert_abs_diff_eq!(emi, expected, epsilon = 1e-12);
    }
    // ln_factorial_table

    /// table[i] = ln(i!) with length n_max + 1
    #[test]
    fn test_ln_factorial_table_known_values() {
        let table = ln_factorial_table(5);
        assert_eq!(table.len(), 6, "length should be n_max + 1");
        assert_abs_diff_eq!(table[0], 0.0, epsilon = 1e-12); // ln(0!) = ln(1)
        assert_abs_diff_eq!(table[1], 0.0, epsilon = 1e-12); // ln(1!) = ln(1)
        assert_abs_diff_eq!(table[5], (120.0_f64).ln(), epsilon = 1e-10); // ln(5!) = ln(120)
    }

    // cluster_centroids

    /// Per-cluster mean and size are computed from dense cluster indices
    #[test]
    fn test_cluster_centroids_known_means_and_sizes() {
        let x = array![[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]];
        let cluster = [0usize, 0, 1];
        let (centroids, sizes) = cluster_centroids(&x, &cluster, 2);

        assert_eq!(centroids.shape(), &[2, 2]);
        assert_abs_diff_eq!(centroids[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(centroids[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(centroids[[1, 0]], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(centroids[[1, 1]], 10.0, epsilon = 1e-12);
        assert_eq!(sizes, vec![2, 1]);
    }

    // pairwise_cluster_distances (symmetric fill, serial and parallel paths)

    /// Deterministic pseudo-random feature matrix (hash-based, no rng dependency)
    fn pseudo_random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            let t = (seed as f64) * 0.731 + (i * cols + j) as f64 * 0.618_033_988_7;
            (t.sin() * 43758.5453).fract() - 0.5
        })
    }

    /// Independent full `n x n` scan reference (computes every ordered pair, including the diagonal)
    fn brute_force_dist_to_cluster(
        x: &Array2<f64>,
        cluster: &[usize],
        k: usize,
        metric: DistanceCalculationMetric,
    ) -> Array2<f64> {
        let n = x.nrows();
        let mut dist = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..n {
                dist[[i, cluster[j]]] += metric.distance(x.row(i), x.row(j));
            }
        }
        dist
    }

    /// The serial path (input below the parallel gate) is bitwise identical to the full `n x n`
    /// scan: the only difference, the `d(i, i) = 0` self term, adds nothing to a sum of
    /// non-negative distances
    #[test]
    fn test_pairwise_cluster_distances_serial_matches_full_scan_bitwise() {
        let x = pseudo_random_matrix(12, 5, 1); // 12*12*5 = 720 < gate -> serial
        let cluster: Vec<usize> = (0..12).map(|i| i % 3).collect();
        let got = pairwise_cluster_distances(&x, &cluster, 3, DistanceCalculationMetric::Euclidean);
        let want = brute_force_dist_to_cluster(&x, &cluster, 3, DistanceCalculationMetric::Euclidean);
        assert!(
            got.iter().zip(want.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
            "serial symmetric fill must be bitwise identical to the full scan"
        );
    }

    /// The parallel path (input above the gate) matches the full-scan reference numerically, for
    /// every supported metric
    #[test]
    fn test_pairwise_cluster_distances_parallel_matches_full_scan() {
        let n = 300; // 300*300*4 = 360_000 >= the 262_144 gate -> parallel
        let x = pseudo_random_matrix(n, 4, 2);
        let cluster: Vec<usize> = (0..n).map(|i| i % 7).collect();
        for metric in [
            DistanceCalculationMetric::Euclidean,
            DistanceCalculationMetric::Manhattan,
            DistanceCalculationMetric::Minkowski(3.0),
        ] {
            let got = pairwise_cluster_distances(&x, &cluster, 7, metric);
            let want = brute_force_dist_to_cluster(&x, &cluster, 7, metric);
            assert_eq!(got.shape(), want.shape());
            for (a, b) in got.iter().zip(want.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9,
                    "parallel fill {a} deviates from full-scan reference {b} for {metric:?}"
                );
            }
        }
    }

    /// The parallel path is run-to-run deterministic at a fixed thread count (same bucket grouping
    /// -> bitwise-identical result)
    #[test]
    fn test_pairwise_cluster_distances_parallel_run_to_run_deterministic() {
        let n = 300;
        let x = pseudo_random_matrix(n, 4, 3);
        let cluster: Vec<usize> = (0..n).map(|i| i % 5).collect();
        let a = pairwise_cluster_distances(&x, &cluster, 5, DistanceCalculationMetric::Euclidean);
        let b = pairwise_cluster_distances(&x, &cluster, 5, DistanceCalculationMetric::Euclidean);
        assert!(
            a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
            "repeated parallel fills must be bitwise identical"
        );
    }
}
