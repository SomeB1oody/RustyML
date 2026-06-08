use ahash::AHashMap;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};

use super::validate_pair;
use crate::math::squared_euclidean_distance_row;

/// Denominator magnitude below which an AMI/ARI normaliser is treated as a degenerate (perfect)
/// clustering and the score is defined to be `1.0`.
const DEGENERATE_DENOM: f64 = 1e-10;

/// Maps each distinct label to a dense index in `0..k` in order of first appearance.
fn label_index(labels: &[usize]) -> AHashMap<usize, usize> {
    let mut index = AHashMap::new();
    for &label in labels {
        let next = index.len();
        index.entry(label).or_insert(next);
    }
    index
}

/// Builds the contingency matrix of two label assignments together with its row and column sums
/// (the cluster sizes in `labels_true` and `labels_pred` respectively).
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

/// Computes the mutual information (in nats): `MI = Σ_ij (n_ij/n) · ln(n·n_ij / (a_i·b_j))`.
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

/// Computes the entropy (in nats) of a partition from its cluster sizes: `H = -Σ_i p_i · ln(p_i)`.
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

/// Builds a table of natural log-factorials, `table[i] = ln(i!)`, for `i` in `0..=n_max`.
///
/// Computed by cumulative summation so that each log-binomial coefficient needed by the expected
/// mutual information becomes an `O(1)` lookup rather than an `O(k)` sum.
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
/// clusterings with the given cluster sizes.
///
/// For each pair of clusters `(a_i, b_j)` the overlap `n_ij` follows a hypergeometric distribution;
/// EMI sums `P(n_ij = k) · (k/n) · ln(n·k / (a_i·b_j))` over its support. All binomial coefficients
/// are evaluated in log space from a shared log-factorial table, so the whole computation is
/// `O(R·C·range)` with `O(1)` inner work.
fn expected_mutual_information(row_sums: &[usize], col_sums: &[usize], n: usize) -> f64 {
    let n_f = n as f64;
    let ln_fact = ln_factorial_table(n);
    // log C(a, b) = ln(a!) - ln(b!) - ln((a-b)!); valid for 0 <= b <= a <= n.
    let log_binom = |a: usize, b: usize| ln_fact[a] - ln_fact[b] - ln_fact[a - b];

    let mut emi = 0.0;
    for &a_i in row_sums {
        for &b_j in col_sums {
            // Hypergeometric support: max(0, a_i + b_j - n) <= k <= min(a_i, b_j). k = 0 adds
            // nothing to MI, so start from 1.
            let lower = (a_i + b_j).saturating_sub(n).max(1);
            let upper = a_i.min(b_j);

            // log C(n, b_j) is constant across k for this pair; hoist it out of the loop.
            let log_c_n_bj = log_binom(n, b_j);
            for k in lower..=upper {
                // ln P(k) = log C(a_i, k) + log C(n - a_i, b_j - k) - log C(n, b_j)
                let log_p = log_binom(a_i, k) + log_binom(n - a_i, b_j - k) - log_c_n_bj;
                let term =
                    (k as f64 / n_f) * ((n_f * k as f64) / (a_i as f64 * b_j as f64)).ln();
                emi += log_p.exp() * term;
            }
        }
    }
    emi
}

/// Collects a `usize` label array into a contiguous `Vec`, tolerating non-contiguous views.
fn to_label_vec<S>(labels: &ArrayBase<S, Ix1>) -> Vec<usize>
where
    S: Data<Elem = usize>,
{
    labels.iter().copied().collect()
}

/// Calculates the Normalized Mutual Information (NMI) between two cluster assignments.
///
/// NMI normalizes the mutual information by the geometric mean of the two clusterings' entropies,
/// giving `0.0` for independent assignments and `1.0` for identical ones. If either clustering has
/// zero entropy (a single cluster), NMI is defined as `0.0`.
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
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::normalized_mutual_info;
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
    validate_pair(labels_true.len(), labels_pred.len(), "labels_true and labels_pred");

    let n = labels_true.len();
    let labels_true = to_label_vec(labels_true);
    let labels_pred = to_label_vec(labels_pred);

    let (contingency, row_sums, col_sums) = contingency_matrix(&labels_true, &labels_pred);
    let mi = mutual_information(&contingency, n, &row_sums, &col_sums);
    let h_true = entropy_nats(&row_sums, n);
    let h_pred = entropy_nats(&col_sums, n);

    let denominator = (h_true * h_pred).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        mi / denominator
    }
}

/// Calculates the Adjusted Mutual Information (AMI) between two cluster assignments.
///
/// AMI corrects the mutual information for the agreement expected by chance, scoring `1.0` for
/// identical clusterings and about `0.0` for independent ones (it can be slightly negative). When
/// the normaliser is degenerate (e.g. both clusterings put every sample in one cluster) the score
/// is defined as `1.0`.
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
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::adjusted_mutual_info;
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
    validate_pair(labels_true.len(), labels_pred.len(), "labels_true and labels_pred");

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

/// Calculates the Adjusted Rand Index (ARI) between two cluster assignments.
///
/// ARI is the Rand index (the fraction of sample pairs that two clusterings agree on, whether by
/// grouping them together or apart) corrected for chance: `1.0` for identical clusterings, about
/// `0.0` for independent ones, and possibly negative for worse-than-random agreement. When the
/// normaliser is degenerate (e.g. fewer than two samples, or both clusterings trivial) the score
/// is defined as `1.0`.
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
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::adjusted_rand_index;
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
    validate_pair(labels_true.len(), labels_pred.len(), "labels_true and labels_pred");

    let n = labels_true.len();
    let labels_true = to_label_vec(labels_true);
    let labels_pred = to_label_vec(labels_pred);

    let (contingency, row_sums, col_sums) = contingency_matrix(&labels_true, &labels_pred);

    // Number of unordered pairs within a group of `size` elements: C(size, 2).
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

/// Calculates the mean Silhouette Coefficient over all samples using Euclidean distance.
///
/// For each sample, the silhouette `s = (b - a) / max(a, b)` compares the mean intra-cluster
/// distance `a` with the mean distance `b` to the nearest other cluster; it ranges from `-1`
/// (likely misassigned) through `0` (on a cluster boundary) to `+1` (well clustered). Samples that
/// are the sole member of their cluster contribute `0`. The returned score is the mean over all
/// samples.
///
/// This is an `O(n²·d)` computation in the number of samples `n` and features `d`.
///
/// # Parameters
///
/// - `x` - Feature matrix with one sample per row (`n_samples × n_features`)
/// - `labels` - Cluster assignment for each sample
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
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::silhouette_score;
///
/// let x = array![[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]];
/// let labels = array![0, 0, 1, 1];
/// let score = silhouette_score(&x, &labels);
/// assert!(score > 0.8);
/// ```
pub fn silhouette_score<S1, S2>(x: &ArrayBase<S1, Ix2>, labels: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = usize>,
{
    let n = x.nrows();
    if n != labels.len() {
        panic!("dimension mismatch: expected {n}, found {}", labels.len());
    }
    if n == 0 {
        panic!("input is empty: x and labels");
    }

    // Densify labels to 0..k so cluster sizes and distance sums index contiguously.
    let labels = to_label_vec(labels);
    let index = label_index(&labels);
    let k = index.len();
    if k < 2 || k >= n {
        panic!(
            "invalid input: number of clusters is {k}, valid range is 2 to n_samples - 1 ({})",
            n - 1
        );
    }
    let cluster: Vec<usize> = labels.iter().map(|label| index[label]).collect();

    let mut sizes = vec![0usize; k];
    for &c in &cluster {
        sizes[c] += 1;
    }

    // dist_to_cluster[[i, c]] accumulates the total distance from sample i to every sample in
    // cluster c. Filled once using the symmetry of the distance metric.
    let mut dist_to_cluster = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = squared_euclidean_distance_row(&x.row(i), &x.row(j)).sqrt();
            dist_to_cluster[[i, cluster[j]]] += d;
            dist_to_cluster[[j, cluster[i]]] += d;
        }
    }

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
