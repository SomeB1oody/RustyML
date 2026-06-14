//! Integration tests for `src/metrics/clustering.rs`
//!
//! Expected values come from each metric's closed-form definition. Panic paths use
//! `#[should_panic]`, since the validators panic rather than return `Result`

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::metrics::*;
use rustyml::types::DistanceCalculationMetric;

// adjusted_rand_index

/// Identical 3-class assignment gives ARI = 1.0 exactly
#[test]
fn test_ari_identical_clustering() {
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(adjusted_rand_index(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// Canonical pair true=[0,0,1,1,2,2], pred=[0,0,1,2,1,2] gives ARI = 1/6
#[test]
fn test_ari_canonical_one_sixth() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(adjusted_rand_index(&t, &p), 1.0_f64 / 6.0, epsilon = 1e-12);
}

/// Interleaved partition where all pairs disagree gives ARI = -1/2
#[test]
fn test_ari_completely_different() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(adjusted_rand_index(&t, &p), -0.5_f64, epsilon = 1e-12);
}

/// ARI is symmetric: ARI(t, p) == ARI(p, t)
#[test]
fn test_ari_symmetry() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(
        adjusted_rand_index(&t, &p),
        adjusted_rand_index(&p, &t),
        epsilon = 1e-15,
    );
}

/// ARI with a single sample returns 1.0 (degenerate, no pairs)
#[test]
fn test_ari_single_sample_degenerate() {
    let t = array![0usize];
    let p = array![0usize];
    assert_abs_diff_eq!(adjusted_rand_index(&t, &p), 1.0, epsilon = 1e-12);
}

/// ARI lies in [-0.5, 1.0] for a typical input
#[test]
fn test_ari_value_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 1, 2, 0, 1, 2];
    let ari = adjusted_rand_index(&t, &p);
    assert!((-0.5..=1.0).contains(&ari));
}

/// ARI panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_ari_length_mismatch_panics() {
    let t = array![0usize, 0, 1];
    let p = array![0usize, 1];
    let _ = adjusted_rand_index(&t, &p);
}

/// ARI panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_ari_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = adjusted_rand_index(&t, &p);
}

// normalized_mutual_info

/// Identical clustering gives NMI = 1.0
#[test]
fn test_nmi_identical_clustering() {
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(
        normalized_mutual_info(&labels, &labels),
        1.0,
        epsilon = 1e-12
    );
}

/// Canonical pair [0,0,1,1,2,2] vs [0,0,1,2,1,2] gives NMI = (1/3 + (2/3)*ln(3/2)/ln(3))
#[test]
fn test_nmi_canonical_pair() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    // 1/3 + (2/3)*ln(1.5)/ln(3) ~= 0.5793801642857
    let expected = 1.0 / 3.0 + (2.0 / 3.0) * (1.5_f64).ln() / (3.0_f64).ln();
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), expected, epsilon = 1e-9);
}

/// Independent clusterings give NMI = 0
#[test]
fn test_nmi_independent_clustering() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), 0.0, epsilon = 1e-12);
}

/// NMI = 0 when a partition has a single cluster (zero entropy, zero denominator)
#[test]
fn test_nmi_single_pred_cluster_zero() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 0, 0, 0];
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), 0.0, epsilon = 1e-12);
}

/// NMI is symmetric: NMI(t, p) == NMI(p, t)
#[test]
fn test_nmi_symmetry() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(
        normalized_mutual_info(&t, &p),
        normalized_mutual_info(&p, &t),
        epsilon = 1e-15,
    );
}

/// NMI lies in [0.0, 1.0] for any valid inputs
#[test]
fn test_nmi_range_zero_to_one() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let nmi = normalized_mutual_info(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&nmi));
}

/// Asymmetric case true=[0,0,1,1] pred=[0,1,2,3]: MI=ln(2), H_true=ln(2), H_pred=2ln(2), so with
/// arithmetic-mean normalization NMI = ln(2) / ((ln(2) + 2ln(2)) / 2) = 2/3
#[test]
fn test_nmi_asymmetric_case() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    let expected = 2.0_f64 / 3.0; // arithmetic mean (sklearn >= 0.22)
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), expected, epsilon = 1e-9);
}

/// Arithmetic-mean NMI equals V-measure exactly (both are MI / ((H_true + H_pred) / 2))
#[test]
fn test_nmi_equals_v_measure() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(
        normalized_mutual_info(&t, &p),
        v_measure_score(&t, &p),
        epsilon = 1e-12
    );
}

/// NMI panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_nmi_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = normalized_mutual_info(&t, &p);
}

/// NMI panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_nmi_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = normalized_mutual_info(&t, &p);
}

// adjusted_mutual_info

/// Identical clustering gives AMI = 1.0 exactly
#[test]
fn test_ami_identical_clustering() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(adjusted_mutual_info(&labels, &labels), 1.0, epsilon = 1e-10);
}

/// Independent clusterings [0,0,1,1] vs [0,1,0,1] give chance-corrected AMI = -0.5
#[test]
fn test_ami_independent_clustering() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(adjusted_mutual_info(&t, &p), -0.5_f64, epsilon = 1e-9);
}

/// Canonical pair [0,0,1,1,2,2] vs [0,0,1,2,1,2] gives AMI = 1/6
#[test]
fn test_ami_canonical_pair() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(adjusted_mutual_info(&t, &p), 1.0 / 6.0, epsilon = 1e-7);
}

/// AMI is symmetric: AMI(t, p) == AMI(p, t)
#[test]
fn test_ami_symmetry() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(
        adjusted_mutual_info(&t, &p),
        adjusted_mutual_info(&p, &t),
        epsilon = 1e-12,
    );
}

/// AMI is at most 1.0 for any valid input
#[test]
fn test_ami_at_most_one() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert!(adjusted_mutual_info(&t, &p) <= 1.0 + 1e-12);
}

/// AMI panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_ami_length_mismatch_panics() {
    let t = array![0usize, 0, 1];
    let p = array![0usize, 1];
    let _ = adjusted_mutual_info(&t, &p);
}

/// AMI panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_ami_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = adjusted_mutual_info(&t, &p);
}

// homogeneity_score

/// Each cluster a pure subset of one class gives homogeneity = 1.0
#[test]
fn test_homogeneity_perfect() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    assert_abs_diff_eq!(homogeneity_score(&t, &p), 1.0, epsilon = 1e-12);
}

/// Identical labels give homogeneity = 1.0
#[test]
fn test_homogeneity_identical() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(homogeneity_score(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// Imperfect homogeneity true=[0,1,2,3] pred=[0,0,1,1] gives 0.5
#[test]
fn test_homogeneity_partial() {
    let t = array![0usize, 1, 2, 3];
    let p = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(homogeneity_score(&t, &p), 0.5, epsilon = 1e-12);
}

/// Homogeneity lies in [0.0, 1.0] for valid inputs
#[test]
fn test_homogeneity_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let h = homogeneity_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&h));
}

/// Homogeneity panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_homogeneity_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = homogeneity_score(&t, &p);
}

/// Homogeneity panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_homogeneity_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = homogeneity_score(&t, &p);
}

// completeness_score

/// All members of each class in one cluster gives completeness = 1.0
#[test]
fn test_completeness_perfect() {
    let t = array![0usize, 1, 2, 3];
    let p = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(completeness_score(&t, &p), 1.0, epsilon = 1e-12);
}

/// Identical labels give completeness = 1.0
#[test]
fn test_completeness_identical() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(completeness_score(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// Imperfect completeness true=[0,0,1,1] pred=[0,1,2,3] gives 0.5
#[test]
fn test_completeness_partial() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    assert_abs_diff_eq!(completeness_score(&t, &p), 0.5, epsilon = 1e-12);
}

/// Homogeneity and completeness differ for true=[0,0,1,1] pred=[0,1,2,3] (1.0 vs 0.5)
#[test]
fn test_completeness_asymmetric_differs_from_homogeneity() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    let h = homogeneity_score(&t, &p);
    let c = completeness_score(&t, &p);
    assert_abs_diff_eq!(h, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c, 0.5, epsilon = 1e-12);
    // Confirm they differ - swapping would give (0.5, 1.0) instead of (1.0, 0.5)
    assert!((h - c).abs() > 0.4);
}

/// Completeness lies in [0.0, 1.0]
#[test]
fn test_completeness_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let c = completeness_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&c));
}

/// Completeness panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_completeness_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = completeness_score(&t, &p);
}

/// Completeness panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_completeness_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = completeness_score(&t, &p);
}

// v_measure_score

/// Identical clustering gives V-measure = 1.0
#[test]
fn test_v_measure_identical() {
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(v_measure_score(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// V-measure is the harmonic mean of h and c; h=1.0, c=0.5 give 2/3
#[test]
fn test_v_measure_asymmetric_two_thirds() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    let expected = 2.0_f64 / 3.0;
    assert_abs_diff_eq!(v_measure_score(&t, &p), expected, epsilon = 1e-12);
}

/// V-measure = 0 when MI = 0 (h = c = 0)
#[test]
fn test_v_measure_independent_zero() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(v_measure_score(&t, &p), 0.0, epsilon = 1e-12);
}

/// V-measure is symmetric (the harmonic mean is commutative)
#[test]
fn test_v_measure_symmetry() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(
        v_measure_score(&t, &p),
        v_measure_score(&p, &t),
        epsilon = 1e-15,
    );
}

/// V-measure lies in [0.0, 1.0]
#[test]
fn test_v_measure_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let v = v_measure_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&v));
}

/// V-measure panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_v_measure_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = v_measure_score(&t, &p);
}

/// V-measure panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_v_measure_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = v_measure_score(&t, &p);
}

// fowlkes_mallows_score

/// Identical clustering gives FMI = 1.0
#[test]
fn test_fmi_identical() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(
        fowlkes_mallows_score(&labels, &labels),
        1.0,
        epsilon = 1e-12
    );
}

/// All pairs disagree gives FMI = 0
#[test]
fn test_fmi_all_disagree_zero() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(fowlkes_mallows_score(&t, &p), 0.0, epsilon = 1e-12);
}

/// Canonical partial match [0,0,1,1,2,2] vs [0,0,1,2,1,2] gives FMI = 1/3
#[test]
fn test_fmi_partial_match_one_third() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let expected = 1.0_f64 / 3.0;
    assert_abs_diff_eq!(fowlkes_mallows_score(&t, &p), expected, epsilon = 1e-12);
}

/// FMI = 0 when all clusters are singletons (no pairs in either partition)
#[test]
fn test_fmi_all_singletons_degenerate() {
    let t = array![0usize, 1];
    let p = array![0usize, 1];
    assert_abs_diff_eq!(fowlkes_mallows_score(&t, &p), 0.0, epsilon = 1e-12);
}

/// FMI is symmetric
#[test]
fn test_fmi_symmetry() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(
        fowlkes_mallows_score(&t, &p),
        fowlkes_mallows_score(&p, &t),
        epsilon = 1e-15,
    );
}

/// FMI lies in [0.0, 1.0] for any valid inputs
#[test]
fn test_fmi_range() {
    let t = array![0usize, 0, 0, 1, 1, 1];
    let p = array![0usize, 0, 1, 1, 1, 0];
    let fmi = fowlkes_mallows_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&fmi));
}

/// FMI panics on length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_fmi_length_mismatch_panics() {
    let t = array![0usize, 0, 1];
    let p = array![0usize, 1];
    let _ = fowlkes_mallows_score(&t, &p);
}

/// FMI panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_fmi_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = fowlkes_mallows_score(&t, &p);
}

// silhouette_score

/// Well-separated 1-D clusters give silhouette = 359/399 (~= 0.8997)
#[test]
fn test_silhouette_well_separated() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    let expected = 359.0_f64 / 399.0; // = (19/21 + 17/19) / 2
    assert_abs_diff_eq!(
        silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean),
        expected,
        epsilon = 1e-9
    );
}

/// The metric is honoured: Manhattan distances give a different (closed-form) score than the
/// Euclidean one on the same 2-D points
#[test]
fn test_silhouette_respects_manhattan_metric() {
    let x = array![[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]];
    let labels = array![0usize, 0, 1, 1];
    // Manhattan: each a = 1; b for p0/p3 = 20.5, for p1/p2 = 19.5 -> mean = (39/41 + 37/39) / 2
    let expected = (39.0_f64 / 41.0 + 37.0_f64 / 39.0) / 2.0;
    let manhattan = silhouette_score(&x, &labels, DistanceCalculationMetric::Manhattan);
    assert_abs_diff_eq!(manhattan, expected, epsilon = 1e-9);
    // Differs from the Euclidean score on these points, confirming the metric is actually used
    let euclidean = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
    assert!((manhattan - euclidean).abs() > 1e-3);
}

/// Exercises the parallel path (n >= the internal threshold). Two well-separated clusters of
/// coincident points give a == 0 and b == const for every sample, so the silhouette is exactly 1.0
#[test]
fn test_silhouette_parallel_path_large_n() {
    // 128 samples: 64 at the origin (cluster 0), 64 at (10, 0) (cluster 1)
    let n = 128;
    let x = Array2::from_shape_fn(
        (n, 2),
        |(i, j)| {
            if j == 0 && i >= n / 2 { 10.0 } else { 0.0 }
        },
    );
    let labels = Array1::from_shape_fn(n, |i| if i < n / 2 { 0usize } else { 1 });
    let score = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
    assert_abs_diff_eq!(score, 1.0, epsilon = 1e-12);
}

/// Mislabeled points where distant pairs share a cluster give silhouette = -4/9
#[test]
fn test_silhouette_mislabeled_negative() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 1, 1, 0]; // p0 and p3 in cluster 0, p1 and p2 in cluster 1
    let expected = -4.0_f64 / 9.0;
    assert_abs_diff_eq!(
        silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean),
        expected,
        epsilon = 1e-9
    );
}

/// A lone singleton cluster contributes 0 to the mean; score = 1/6 here
#[test]
fn test_silhouette_lone_cluster_contributes_zero() {
    let x = array![[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]];
    let labels = array![0usize, 1, 1];
    let expected = 1.0_f64 / 6.0;
    assert_abs_diff_eq!(
        silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean),
        expected,
        epsilon = 1e-9
    );
}

/// When all points coincide (a=b=0 everywhere), silhouette returns 0.0
#[test]
fn test_silhouette_all_same_location_zero() {
    let x = array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(
        silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean),
        0.0,
        epsilon = 1e-12
    );
}

/// Silhouette lies in [-1.0, 1.0]
#[test]
fn test_silhouette_range() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    let s = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
    assert!((-1.0 - 1e-12..=1.0 + 1e-12).contains(&s));
}

/// Silhouette panics when the cluster count equals n_samples (all singletons)
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_silhouette_all_singletons_panics() {
    let x = array![[0.0], [1.0], [2.0]];
    let labels = array![0usize, 1, 2]; // k = n = 3, invalid
    let _ = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
}

/// Silhouette panics when only one distinct cluster is present
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_silhouette_single_cluster_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 0, 0];
    let _ = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
}

/// Silhouette panics on row/label length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_silhouette_length_mismatch_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 1];
    let _ = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
}

/// Silhouette panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_silhouette_empty_panics() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let labels: Array1<usize> = array![];
    let _ = silhouette_score(&x, &labels, DistanceCalculationMetric::Euclidean);
}

// davies_bouldin_score

/// Well-separated 1-D clusters give DB = 0.1
#[test]
fn test_db_well_separated_exact() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(davies_bouldin_score(&x, &labels), 0.1, epsilon = 1e-9);
}

/// Points coinciding with their centroids give DB = 0.0
#[test]
fn test_db_perfect_compact_zero() {
    let x = array![[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(davies_bouldin_score(&x, &labels), 0.0, epsilon = 1e-12);
}

/// Three tight 1-D clusters give DB = 0.04 (= 1/25)
#[test]
fn test_db_three_tight_clusters() {
    let x = array![
        [0.0, 0.0],
        [0.2, 0.0],
        [5.0, 0.0],
        [5.2, 0.0],
        [10.0, 0.0],
        [10.2, 0.0]
    ];
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(davies_bouldin_score(&x, &labels), 0.04, epsilon = 1e-9);
}

/// DB is non-negative for any valid inputs
#[test]
fn test_db_nonnegative() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert!(davies_bouldin_score(&x, &labels) >= 0.0);
}

/// DB panics on all-singleton input (k = n)
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_db_all_singletons_panics() {
    let x = array![[0.0], [1.0], [2.0]];
    let labels = array![0usize, 1, 2];
    let _ = davies_bouldin_score(&x, &labels);
}

/// DB panics on single cluster (k = 1)
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_db_single_cluster_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 0, 0];
    let _ = davies_bouldin_score(&x, &labels);
}

/// DB panics on row/label length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_db_length_mismatch_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]];
    let labels = array![0usize, 1];
    let _ = davies_bouldin_score(&x, &labels);
}

/// DB panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_db_empty_panics() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let labels: Array1<usize> = array![];
    let _ = davies_bouldin_score(&x, &labels);
}

// calinski_harabasz_score

/// Well-separated 1-D clusters give CH = 200.0
#[test]
fn test_ch_well_separated_exact() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(calinski_harabasz_score(&x, &labels), 200.0, epsilon = 1e-9);
}

/// Three tight clusters give CH = 2500.0
#[test]
fn test_ch_three_tight_clusters() {
    let x = array![
        [0.0, 0.0],
        [0.2, 0.0],
        [5.0, 0.0],
        [5.2, 0.0],
        [10.0, 0.0],
        [10.2, 0.0]
    ];
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(calinski_harabasz_score(&x, &labels), 2500.0, epsilon = 1e-6);
}

/// When all points sit on their centroids, within-dispersion = 0 and CH returns 1.0
#[test]
fn test_ch_zero_within_dispersion_returns_one() {
    let x = array![[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(calinski_harabasz_score(&x, &labels), 1.0, epsilon = 1e-12);
}

/// CH is non-negative for any valid inputs (a ratio of dispersions >= 0)
#[test]
fn test_ch_nonnegative() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert!(calinski_harabasz_score(&x, &labels) >= 0.0);
}

/// CH increases as clusters become more separated
#[test]
fn test_ch_more_separation_higher_score() {
    let x_close = array![[0.0], [1.0], [2.0], [3.0]];
    let x_far = array![[0.0], [1.0], [10.0], [11.0]];
    let labels = array![0usize, 0, 1, 1];
    let ch_close = calinski_harabasz_score(&x_close, &labels);
    let ch_far = calinski_harabasz_score(&x_far, &labels);
    assert!(
        ch_far > ch_close,
        "more-separated clusters should have higher CH; got close={ch_close}, far={ch_far}"
    );
}

/// CH panics on all-singleton clusters (k = n)
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_ch_all_singletons_panics() {
    let x = array![[0.0], [1.0], [2.0]];
    let labels = array![0usize, 1, 2];
    let _ = calinski_harabasz_score(&x, &labels);
}

/// CH panics on single cluster (k = 1)
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_ch_single_cluster_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 0, 0];
    let _ = calinski_harabasz_score(&x, &labels);
}

/// CH panics on row/label length mismatch
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_ch_length_mismatch_panics() {
    let x = array![[0.0], [1.0], [5.0]];
    let labels = array![0usize, 1];
    let _ = calinski_harabasz_score(&x, &labels);
}

/// CH panics on empty input
#[test]
#[should_panic(expected = "input is empty")]
fn test_ch_empty_panics() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let labels: Array1<usize> = array![];
    let _ = calinski_harabasz_score(&x, &labels);
}
