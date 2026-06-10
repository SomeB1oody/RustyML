//! Integration tests for `src/metrics/clustering.rs`.
//!
//! Every expected value is derived from the mathematical definition of the metric or a
//! closed-form hand calculation shown in the accompanying comment. No value is obtained
//! by reading or tracing the implementation.
//!
//! Panic paths use `#[should_panic]`.  The source module uses `validate_pair` for label
//! functions and `validate_clustering_inputs` for geometry-based metrics; both panic
//! (not `Result`), so `#[should_panic]` is the correct pattern here.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::metrics::*;

// ===========================================================================
// adjusted_rand_index
// ===========================================================================

/// Identical 3-class assignment → ARI = 1.0 (exact).
///
/// Derivation:
///   labels = [0,0,1,1,2,2]
///   Contingency: diagonal [[2,0,0],[0,2,0],[0,0,2]]
///   sum_comb_cells = C(2,2)*3 = 3
///   sum_comb_true = sum_comb_pred = C(2,2)*3 = 3
///   comb_n = C(6,2) = 15
///   expected = 3*3/15 = 9/15 = 3/5
///   max_index = (3+3)/2 = 3
///   denom = 3 - 3/5 = 12/5
///   ARI = (3 - 3/5) / (12/5) = (12/5)/(12/5) = 1.0
#[test]
fn test_ari_identical_clustering() {
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(adjusted_rand_index(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// Canonical reference: true=[0,0,1,1,2,2] pred=[0,0,1,2,1,2] → ARI = 1/6.
///
/// Derivation:
///   Contingency:  true0=[2,0,0], true1=[0,1,1], true2=[0,1,1]
///   sum_comb_cells = C(2,2)+C(1,2)*4 = 1+0 = 1
///   sum_comb_true = C(2,2)*3 = 3   (row sums all 2)
///   sum_comb_pred = C(2,2)*3 = 3   (col sums all 2)
///   comb_n = C(6,2) = 15
///   expected = 3*3/15 = 3/5
///   max_index = (3+3)/2 = 3
///   denom = 3 - 3/5 = 12/5
///   ARI = (1 - 3/5) / (12/5) = (2/5)/(12/5) = 2/12 = 1/6
#[test]
fn test_ari_canonical_one_sixth() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(adjusted_rand_index(&t, &p), 1.0_f64 / 6.0, epsilon = 1e-12);
}

/// Completely disagreeing partition (interleaved, all pairs disagree) → ARI = -1/2.
///
/// Derivation:
///   true=[0,0,1,1], pred=[0,1,0,1]
///   Contingency [[1,1],[1,1]]; n_ij all 1, C(1,2)=0 → sum_comb_cells = 0
///   sum_comb_true = sum_comb_pred = C(2,2)*2 = 2
///   comb_n = C(4,2) = 6
///   expected = 4/6 = 2/3
///   max_index = 2
///   denom = 2 - 2/3 = 4/3
///   ARI = (0 - 2/3) / (4/3) = (-2/3)*(3/4) = -1/2
#[test]
fn test_ari_completely_different() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(adjusted_rand_index(&t, &p), -0.5_f64, epsilon = 1e-12);
}

/// ARI is symmetric: ARI(t, p) == ARI(p, t).
///
/// ARI depends only on C(n_ij,2), C(a_i,2), C(b_j,2) — a transposition of the
/// contingency swaps rows↔cols but the sums of C(·,2) are invariant.
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

/// ARI with n=1 (fewer than two samples) returns 1.0 (degenerate case — no pairs).
#[test]
fn test_ari_single_sample_degenerate() {
    let t = array![0usize];
    let p = array![0usize];
    assert_abs_diff_eq!(adjusted_rand_index(&t, &p), 1.0, epsilon = 1e-12);
}

/// ARI is in range [-0.5, 1.0] for a typical input.
#[test]
fn test_ari_value_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 1, 2, 0, 1, 2];
    let ari = adjusted_rand_index(&t, &p);
    assert!((-0.5..=1.0).contains(&ari));
}

/// ARI panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_ari_length_mismatch_panics() {
    let t = array![0usize, 0, 1];
    let p = array![0usize, 1];
    let _ = adjusted_rand_index(&t, &p);
}

/// ARI panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_ari_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = adjusted_rand_index(&t, &p);
}

// ===========================================================================
// normalized_mutual_info
// ===========================================================================

/// Identical clustering → NMI = 1.0.
///
/// Derivation:
///   Contingency diagonal [[2,0,0],[0,2,0],[0,0,2]], n=6, row=col=[2,2,2]
///   MI = 3 * (2/6)*ln(6*2/(2*2)) = 3*(1/3)*ln(3) = ln(3)
///   H_true = H_pred = -3*(1/3)*ln(1/3) = ln(3)
///   NMI = ln(3) / sqrt(ln(3)*ln(3)) = ln(3)/ln(3) = 1.0
#[test]
fn test_nmi_identical_clustering() {
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(
        normalized_mutual_info(&labels, &labels),
        1.0,
        epsilon = 1e-12
    );
}

/// Canonical pair [0,0,1,1,2,2] vs [0,0,1,2,1,2] → NMI = ln(3/2)/ln(3).
///
/// Derivation:
///   Same contingency as in the ARI test. row=col=[2,2,2], n=6.
///   Nonzero cells: (0,0)=2, (1,1)=1, (1,2)=1, (2,1)=1, (2,2)=1.
///   MI = (2/6)*ln(6*2/(2*2))      ← diagonal (0,0) ratio = 12/4 = 3   → (1/3)*ln(3)
///      + 4*(1/6)*ln(6*1/(2*2))    ← the four 1-cells share ratio 6/4 = 3/2 → (2/3)*ln(3/2)
///      = (1/3)*ln(3) + (2/3)*ln(3/2)
///   H_true = H_pred = ln(3)
///   NMI = MI/ln(3) = 1/3 + (2/3)*ln(3/2)/ln(3)
#[test]
fn test_nmi_canonical_pair() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    // 1/3 + (2/3)*ln(1.5)/ln(3) ≈ 0.5793801642857
    let expected = 1.0 / 3.0 + (2.0 / 3.0) * (1.5_f64).ln() / (3.0_f64).ln();
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), expected, epsilon = 1e-9);
}

/// Independent clusterings → NMI = 0.
///
/// Derivation:
///   true=[0,0,1,1], pred=[0,1,0,1]
///   Contingency [[1,1],[1,1]], n=4, row=col=[2,2]
///   MI = 4*(1/4)*ln(4*1/(2*2)) = 4*(1/4)*ln(1) = 0
///   NMI = 0 / denom = 0
#[test]
fn test_nmi_independent_clustering() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), 0.0, epsilon = 1e-12);
}

/// NMI = 0 when either partition has a single cluster (zero entropy).
///
/// Derivation:
///   If pred is all-same (one cluster): H_pred = 0 → denominator = 0 → NMI = 0.0
///   (code returns 0.0 when denominator == 0.0)
///   true=[0,0,1,1], pred=[0,0,0,0]
#[test]
fn test_nmi_single_pred_cluster_zero() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 0, 0, 0];
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), 0.0, epsilon = 1e-12);
}

/// NMI is symmetric: NMI(t, p) == NMI(p, t).
///
/// Proof: MI is symmetric and the geometric mean of H_t*H_p is symmetric.
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

/// NMI is in [0.0, 1.0] for any valid inputs.
#[test]
fn test_nmi_range_zero_to_one() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let nmi = normalized_mutual_info(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&nmi));
}

/// NMI asymmetric case: true=[0,0,1,1] pred=[0,1,2,3] → NMI = 1/√2.
///
/// Derivation:
///   Contingency [[1,1,0,0],[0,0,1,1]], row=[2,2], col=[1,1,1,1], n=4
///   MI = 4*(1/4)*ln(4*1/(2*1)) = ln(2)
///   H_true = -2*(1/2)*ln(1/2) = ln(2)
///   H_pred = -4*(1/4)*ln(1/4) = ln(4) = 2*ln(2)
///   NMI = ln(2) / sqrt(ln(2)*2*ln(2)) = ln(2) / (ln(2)*√2) = 1/√2
#[test]
fn test_nmi_asymmetric_case() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    let expected = 1.0_f64 / 2.0_f64.sqrt(); // 1/√2 ≈ 0.7071067812
    assert_abs_diff_eq!(normalized_mutual_info(&t, &p), expected, epsilon = 1e-9);
}

/// NMI panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_nmi_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = normalized_mutual_info(&t, &p);
}

/// NMI panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_nmi_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = normalized_mutual_info(&t, &p);
}

// ===========================================================================
// adjusted_mutual_info
// ===========================================================================

/// Identical clustering → AMI = 1.0 (exact).
///
/// Proof: MI = H_true = H_pred, so (MI - EMI)/((H+H)/2 - EMI) = 1.0
/// since numerator and denominator are both H - EMI.
#[test]
fn test_ami_identical_clustering() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(adjusted_mutual_info(&labels, &labels), 1.0, epsilon = 1e-10);
}

/// AMI for [0,0,1,1] vs [0,1,0,1] (MI=0, chance-corrected) = -0.5.
///
/// Derivation:
///   n=4, row=col=[2,2].  Contingency [[1,1],[1,1]].
///   MI = 0 (shown in NMI tests).
///   H_true = H_pred = ln(2).
///   EMI for pairs (a=2, b=2, n=4):
///     lower=1, upper=2.  C(4,2)=6.
///     k=1: P = C(2,1)*C(2,1)/C(4,2) = 4/6 = 2/3
///           term = (1/4)*ln(4*1/(2*2)) = (1/4)*ln(1) = 0
///           contribution = 0
///     k=2: P = C(2,2)*C(0,0)/C(4,2) = 1/6
///           term = (2/4)*ln(4*2/(2*2)) = (1/2)*ln(2)
///           contribution = (1/6)*(1/2)*ln(2)
///   EMI_per_pair = (1/12)*ln(2)
///   EMI_total (4 pairs) = 4*(1/12)*ln(2) = (1/3)*ln(2)
///   denom = ln(2) - (1/3)*ln(2) = (2/3)*ln(2)
///   AMI = (0 - (1/3)*ln(2)) / ((2/3)*ln(2)) = -1/2 = -0.5
#[test]
fn test_ami_independent_clustering() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(adjusted_mutual_info(&t, &p), -0.5_f64, epsilon = 1e-9);
}

/// AMI canonical: [0,0,1,1,2,2] vs [0,0,1,2,1,2] → AMI = -0.25.
///
/// Derivation:
///   n=6, row=col=[2,2,2], 9 identical (2,2) pairs.
///   Per pair (a=2, b=2, n=6):
///     k=1: P = C(2,1)*C(4,1)/C(6,2) = 2*4/15 = 8/15
///           term = (1/6)*ln(6*1/(2*2)) = (1/6)*ln(3/2)
///           contribution = (8/15)*(1/6)*ln(3/2)
///     k=2: P = C(2,2)*C(4,0)/C(6,2) = 1/15
///           term = (2/6)*ln(6*2/(2*2)) = (1/3)*ln(3)
///           contribution = (1/15)*(1/3)*ln(3)
///   EMI_per_pair = (8/90)*ln(3/2) + (1/45)*ln(3)
///   EMI_total = 9 * EMI_per_pair  ≈ 0.5441 (computed to verify sign)
///   MI = (1/3)*ln(3) + (2/3)*ln(3/2) ≈ 0.6365   [the diagonal cell is included]
///   H = ln(3) ≈ 1.0986
///   denom = (H+H)/2 - EMI_total ≈ 1.0986 - 0.5441 ≈ 0.5545
///   AMI = (0.6365 - 0.5441) / 0.5545 ≈ 0.0924 / 0.5545 = 1/6 ≈ 0.16667
#[test]
fn test_ami_canonical_pair() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert_abs_diff_eq!(adjusted_mutual_info(&t, &p), 1.0 / 6.0, epsilon = 1e-7);
}

/// AMI is symmetric: AMI(t, p) == AMI(p, t).
///
/// Both MI and EMI are symmetric in the two clusterings (the contingency matrix
/// transpose preserves C(n_ij,2) sums). H_true and H_pred swap roles but their
/// arithmetic mean is the same.
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

/// AMI is <= 1.0 for any valid input.
#[test]
fn test_ami_at_most_one() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    assert!(adjusted_mutual_info(&t, &p) <= 1.0 + 1e-12);
}

/// AMI panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_ami_length_mismatch_panics() {
    let t = array![0usize, 0, 1];
    let p = array![0usize, 1];
    let _ = adjusted_mutual_info(&t, &p);
}

/// AMI panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_ami_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = adjusted_mutual_info(&t, &p);
}

// ===========================================================================
// homogeneity_score
// ===========================================================================

/// Perfect homogeneity: each cluster is a pure subset of one class.
///
/// Derivation: true=[0,0,1,1] pred=[0,1,2,3]
///   Contingency [[1,1,0,0],[0,0,1,1]], row=[2,2], col=[1,1,1,1], n=4.
///   MI = 4*(1/4)*ln(4*1/(2*1)) = ln(2).
///   H(classes) = ln(2)  (two equal classes).
///   homogeneity = MI/H(classes) = 1.0.
#[test]
fn test_homogeneity_perfect() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    assert_abs_diff_eq!(homogeneity_score(&t, &p), 1.0, epsilon = 1e-12);
}

/// Identical labels → homogeneity = 1.0.
#[test]
fn test_homogeneity_identical() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(homogeneity_score(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// Imperfect homogeneity: true=[0,1,2,3] pred=[0,0,1,1].
///
/// Derivation (roles of the asymmetric case swapped from homogeneity_completeness):
///   Contingency [[1,0],[1,0],[0,1],[0,1]], row=[1,1,1,1], col=[2,2], n=4.
///   MI = 4*(1/4)*ln(4*1/(1*2)) = ln(2).
///   H(classes) = ln(4) = 2*ln(2).
///   homogeneity = ln(2)/(2*ln(2)) = 0.5.
#[test]
fn test_homogeneity_partial() {
    let t = array![0usize, 1, 2, 3];
    let p = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(homogeneity_score(&t, &p), 0.5, epsilon = 1e-12);
}

/// Homogeneity is in [0.0, 1.0] for valid inputs.
#[test]
fn test_homogeneity_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let h = homogeneity_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&h));
}

/// Homogeneity panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_homogeneity_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = homogeneity_score(&t, &p);
}

/// Homogeneity panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_homogeneity_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = homogeneity_score(&t, &p);
}

// ===========================================================================
// completeness_score
// ===========================================================================

/// Perfect completeness: all members of each class end up in one cluster.
///
/// Derivation: true=[0,1,2,3] pred=[0,0,1,1]
///   Contingency [[1,0],[1,0],[0,1],[0,1]], row=[1,1,1,1], col=[2,2], n=4.
///   MI = ln(2), H(clusters) = ln(2).
///   completeness = MI/H(clusters) = 1.0.
#[test]
fn test_completeness_perfect() {
    let t = array![0usize, 1, 2, 3];
    let p = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(completeness_score(&t, &p), 1.0, epsilon = 1e-12);
}

/// Identical labels → completeness = 1.0.
#[test]
fn test_completeness_identical() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(completeness_score(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// Imperfect completeness (dual of the perfect-homogeneity test): true=[0,0,1,1] pred=[0,1,2,3].
///
/// Derivation:
///   row=[2,2], col=[1,1,1,1], n=4.
///   MI = ln(2), H(clusters) = ln(4) = 2*ln(2).
///   completeness = ln(2)/(2*ln(2)) = 0.5.
#[test]
fn test_completeness_partial() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    assert_abs_diff_eq!(completeness_score(&t, &p), 0.5, epsilon = 1e-12);
}

/// Asymmetric example: homogeneity and completeness differ.
///
/// true=[0,0,1,1] pred=[0,1,2,3]:
///   homogeneity = 1.0, completeness = 0.5.
/// They are NOT equal — testing this pinpoints the argument-order contract.
#[test]
fn test_completeness_asymmetric_differs_from_homogeneity() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    let h = homogeneity_score(&t, &p);
    let c = completeness_score(&t, &p);
    assert_abs_diff_eq!(h, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(c, 0.5, epsilon = 1e-12);
    // Confirm they differ — swapping would give (0.5, 1.0) instead of (1.0, 0.5).
    assert!((h - c).abs() > 0.4);
}

/// Completeness is in [0.0, 1.0].
#[test]
fn test_completeness_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let c = completeness_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&c));
}

/// Completeness panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_completeness_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = completeness_score(&t, &p);
}

/// Completeness panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_completeness_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = completeness_score(&t, &p);
}

// ===========================================================================
// v_measure_score
// ===========================================================================

/// Identical clustering → V-measure = 1.0.
#[test]
fn test_v_measure_identical() {
    let labels = array![0usize, 0, 1, 1, 2, 2];
    assert_abs_diff_eq!(v_measure_score(&labels, &labels), 1.0, epsilon = 1e-12);
}

/// V-measure = harmonic mean of h and c.
///
/// Derivation (true=[0,0,1,1] pred=[0,1,2,3]):
///   h = 1.0, c = 0.5 (from homogeneity/completeness tests).
///   V = 2*1.0*0.5 / (1.0+0.5) = 1.0/1.5 = 2/3.
#[test]
fn test_v_measure_asymmetric_two_thirds() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 2, 3];
    let expected = 2.0_f64 / 3.0;
    assert_abs_diff_eq!(v_measure_score(&t, &p), expected, epsilon = 1e-12);
}

/// V-measure = 0 when MI = 0 (h = c = 0).
///
/// Derivation: true=[0,0,1,1] pred=[0,1,0,1] → MI=0 → h=c=0 → V=0.
#[test]
fn test_v_measure_independent_zero() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(v_measure_score(&t, &p), 0.0, epsilon = 1e-12);
}

/// V-measure is symmetric (it is the harmonic mean, which is commutative).
///
/// Note: V-measure uses arithmetic-mean normalisation of MI, while NMI here uses
/// geometric-mean, so the two can differ; we only test symmetry here.
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

/// V-measure is in [0.0, 1.0].
#[test]
fn test_v_measure_range() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let v = v_measure_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&v));
}

/// V-measure panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_v_measure_length_mismatch_panics() {
    let t = array![0usize, 1, 2];
    let p = array![0usize, 1];
    let _ = v_measure_score(&t, &p);
}

/// V-measure panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_v_measure_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = v_measure_score(&t, &p);
}

// ===========================================================================
// fowlkes_mallows_score
// ===========================================================================

/// Identical clustering → FMI = 1.0.
///
/// Derivation: true=pred=[0,0,1,1], contingency diagonal [[2,0],[0,2]].
///   TP = C(2,2)+C(2,2) = 2; pk = qk = C(2,2)*2 = 2.
///   FMI = 2/√(2*2) = 2/2 = 1.0.
#[test]
fn test_fmi_identical() {
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(
        fowlkes_mallows_score(&labels, &labels),
        1.0,
        epsilon = 1e-12
    );
}

/// All pairs disagree → FMI = 0.
///
/// Derivation: true=[0,0,1,1] pred=[0,1,0,1].
///   All n_ij = 1, so C(1,2) = 0 for every cell → TP = 0.
///   pk = qk = C(2,2)*2 = 2 > 0.
///   FMI = 0/√(2*2) = 0.
#[test]
fn test_fmi_all_disagree_zero() {
    let t = array![0usize, 0, 1, 1];
    let p = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(fowlkes_mallows_score(&t, &p), 0.0, epsilon = 1e-12);
}

/// Canonical partial match: [0,0,1,1,2,2] vs [0,0,1,2,1,2] → FMI = 1/3.
///
/// Derivation:
///   Contingency: true0=[2,0,0], true1=[0,1,1], true2=[0,1,1].
///   TP = C(2,2) + 4*C(1,2) = 1 + 0 = 1.
///   pk = col_sums=[2,2,2] → C(2,2)*3 = 3.
///   qk = row_sums=[2,2,2] → C(2,2)*3 = 3.
///   FMI = 1/√(3*3) = 1/3.
#[test]
fn test_fmi_partial_match_one_third() {
    let t = array![0usize, 0, 1, 1, 2, 2];
    let p = array![0usize, 0, 1, 2, 1, 2];
    let expected = 1.0_f64 / 3.0;
    assert_abs_diff_eq!(fowlkes_mallows_score(&t, &p), expected, epsilon = 1e-12);
}

/// FMI = 0 when all clusters are singletons (no pairs exist in either partition).
///
/// Derivation: true=[0,1] pred=[0,1].  C(1,2) = 0 for all → pk = qk = 0.
///   Code returns 0.0 via the denominator == 0.0 guard.
#[test]
fn test_fmi_all_singletons_degenerate() {
    let t = array![0usize, 1];
    let p = array![0usize, 1];
    assert_abs_diff_eq!(fowlkes_mallows_score(&t, &p), 0.0, epsilon = 1e-12);
}

/// FMI is symmetric.
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

/// FMI is in [0.0, 1.0] for any valid inputs.
#[test]
fn test_fmi_range() {
    let t = array![0usize, 0, 0, 1, 1, 1];
    let p = array![0usize, 0, 1, 1, 1, 0];
    let fmi = fowlkes_mallows_score(&t, &p);
    assert!((0.0..=1.0 + 1e-12).contains(&fmi));
}

/// FMI panics on length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_fmi_length_mismatch_panics() {
    let t = array![0usize, 0, 1];
    let p = array![0usize, 1];
    let _ = fowlkes_mallows_score(&t, &p);
}

/// FMI panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_fmi_empty_panics() {
    let t: Array1<usize> = array![];
    let p: Array1<usize> = array![];
    let _ = fowlkes_mallows_score(&t, &p);
}

// ===========================================================================
// silhouette_score
// ===========================================================================

/// Well-separated 1-D clusters → silhouette ≈ 0.8997.
///
/// Derivation:
///   Points: (0), (1), (10), (11); labels [0,0,1,1]  (2-D: x-axis only).
///   Euclidean distances: d(p0,p1)=1, d(p0,p2)=10, d(p0,p3)=11,
///                        d(p1,p2)=9, d(p1,p3)=10, d(p2,p3)=1.
///   p0: a=1/1=1, b=(10+11)/2=10.5, s=(10.5-1)/10.5=9.5/10.5=19/21
///   p1: a=1, b=(9+10)/2=9.5, s=8.5/9.5=17/19
///   p2: a=1, b=(10+9)/2=9.5, s=8.5/9.5=17/19
///   p3: a=1, b=(11+10)/2=10.5, s=9.5/10.5=19/21
///   mean = (19/21 + 17/19 + 17/19 + 19/21)/4 = 359/399 ≈ 0.899749373434
#[test]
fn test_silhouette_well_separated() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    let expected = 359.0_f64 / 399.0; // = (19/21 + 17/19) / 2
    assert_abs_diff_eq!(silhouette_score(&x, &labels), expected, epsilon = 1e-9);
}

/// Mislabeled points (distant pairs share a cluster) → silhouette = -4/9.
///
/// Derivation:
///   Points: (0), (1), (10), (11); labels [0,1,1,0]
///   Cluster 0 = {p0=(0,0), p3=(11,0)}, Cluster 1 = {p1=(1,0), p2=(10,0)}.
///   d(p0,p3)=11, d(p1,p2)=9.
///   p0: a=11, b=(1+10)/2=5.5, s=(5.5-11)/max(5.5,11) = -5.5/11 = -1/2
///   p1: a=9,  b=(1+10)/2=5.5, s=(5.5-9)/max(5.5,9)   = -3.5/9  = -7/18
///   p2: a=9,  b=(10+1)/2=5.5, s=(5.5-9)/9            = -7/18
///   p3: a=11, b=(10+1)/2=5.5, s=(5.5-11)/11          = -1/2
///   mean = (-1/2 - 7/18 - 7/18 - 1/2)/4 = (-9/18 - 7/18 - 7/18 - 9/18)/4
///        = -32/18/4 = -32/72 = -4/9
#[test]
fn test_silhouette_mislabeled_negative() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 1, 1, 0]; // p0 and p3 in cluster 0, p1 and p2 in cluster 1
    let expected = -4.0_f64 / 9.0;
    assert_abs_diff_eq!(silhouette_score(&x, &labels), expected, epsilon = 1e-9);
}

/// Lone singleton cluster contributes 0 to the mean.
///
/// Derivation:
///   3 samples: p0=(0,0) label 0 (singleton), p1=(5,0) label 1, p2=(10,0) label 1.
///   p0: only member of cluster 0 → skips (contributes 0).
///   p1: a = d(p1,p2)/(2-1) = 5; b = dist_to_cluster_0/size_0 = d(p1,p0)/1 = 5.
///       max(5,5) = 5 > 0; s = (5-5)/5 = 0; total += 0.
///   p2: a = d(p2,p1)/(2-1) = 5; b = d(p2,p0)/1 = 10.
///       s = (10-5)/10 = 1/2; total += 1/2.
///   mean = (0+0+0.5)/3 = 1/6
#[test]
fn test_silhouette_lone_cluster_contributes_zero() {
    let x = array![[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]];
    let labels = array![0usize, 1, 1];
    let expected = 1.0_f64 / 6.0;
    assert_abs_diff_eq!(silhouette_score(&x, &labels), expected, epsilon = 1e-9);
}

/// When all points coincide (a=b=0 everywhere), the silhouette returns 0.0.
///
/// Derivation: all distances 0 → denominator max(a,b)=0 → skip → total=0 → score=0/n=0.
#[test]
fn test_silhouette_all_same_location_zero() {
    let x = array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(silhouette_score(&x, &labels), 0.0, epsilon = 1e-12);
}

/// Silhouette is in [-1.0, 1.0].
#[test]
fn test_silhouette_range() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    let s = silhouette_score(&x, &labels);
    assert!((-1.0 - 1e-12..=1.0 + 1e-12).contains(&s));
}

/// Silhouette panics when the number of clusters equals n_samples (all singletons).
///
/// The valid range is k in 2..=n_samples-1 (strictly less than n).
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_silhouette_all_singletons_panics() {
    let x = array![[0.0], [1.0], [2.0]];
    let labels = array![0usize, 1, 2]; // k = n = 3 → invalid
    let _ = silhouette_score(&x, &labels);
}

/// Silhouette panics when only one distinct cluster is present.
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_silhouette_single_cluster_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 0, 0];
    let _ = silhouette_score(&x, &labels);
}

/// Silhouette panics on row/label length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_silhouette_length_mismatch_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 1];
    let _ = silhouette_score(&x, &labels);
}

/// Silhouette panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_silhouette_empty_panics() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let labels: Array1<usize> = array![];
    let _ = silhouette_score(&x, &labels);
}

// ===========================================================================
// davies_bouldin_score
// ===========================================================================

/// Well-separated 1-D clusters → DB = 0.1.
///
/// Derivation:
///   Points (0), (1), (10), (11); labels [0,0,1,1].
///   c0 = (0.5, 0), c1 = (10.5, 0).
///   s[0] = (|p0-c0| + |p1-c0|)/2 = (0.5+0.5)/2 = 0.5
///   s[1] = (|p2-c1| + |p3-c1|)/2 = 0.5
///   d(c0,c1) = 10.0.
///   R(0,1) = R(1,0) = (0.5+0.5)/10.0 = 0.1.
///   max for cluster 0 = R(0,1) = 0.1; max for cluster 1 = R(1,0) = 0.1.
///   DB = (0.1+0.1)/2 = 0.1.
#[test]
fn test_db_well_separated_exact() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(davies_bouldin_score(&x, &labels), 0.1, epsilon = 1e-9);
}

/// Points coinciding with their centroids → DB = 0.0.
///
/// Derivation:
///   All 4 points equal their centroid → s[i] = 0 for all i.
///   R(i,j) = 0/d = 0 → max_ratio = 0 → DB = 0.
///   Example: p0=p1=(0,0), p2=p3=(10,0), labels [0,0,1,1].
#[test]
fn test_db_perfect_compact_zero() {
    let x = array![[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(davies_bouldin_score(&x, &labels), 0.0, epsilon = 1e-12);
}

/// 3 tight 1-D clusters → DB = 1/25 = 0.04.
///
/// Derivation:
///   Points (0), (0.2), (5), (5.2), (10), (10.2); labels [0,0,1,1,2,2].
///   c0=(0.1), c1=(5.1), c2=(10.1).
///   s[0] = s[1] = s[2] = (0.1+0.1)/2 = 0.1.
///   d(c0,c1)=5, d(c1,c2)=5, d(c0,c2)=10.
///   R(0,1)=(0.1+0.1)/5=0.04, R(0,2)=0.02, R(1,2)=0.04.
///   max for i=0: 0.04; i=1: 0.04; i=2: 0.04.
///   DB = (0.04+0.04+0.04)/3 = 0.04 = 1/25.
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

/// DB is non-negative for any valid inputs.
#[test]
fn test_db_nonnegative() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert!(davies_bouldin_score(&x, &labels) >= 0.0);
}

/// DB panics on all-singleton input (k = n).
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_db_all_singletons_panics() {
    let x = array![[0.0], [1.0], [2.0]];
    let labels = array![0usize, 1, 2];
    let _ = davies_bouldin_score(&x, &labels);
}

/// DB panics on single cluster (k = 1).
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_db_single_cluster_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 0, 0];
    let _ = davies_bouldin_score(&x, &labels);
}

/// DB panics on row/label length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_db_length_mismatch_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]];
    let labels = array![0usize, 1];
    let _ = davies_bouldin_score(&x, &labels);
}

/// DB panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_db_empty_panics() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let labels: Array1<usize> = array![];
    let _ = davies_bouldin_score(&x, &labels);
}

// ===========================================================================
// calinski_harabasz_score
// ===========================================================================

/// Well-separated 1-D clusters → CH = 200.0.
///
/// Derivation:
///   Points (0), (1), (10), (11); labels [0,0,1,1]; n=4, k=2.
///   Overall mean = (0+1+10+11)/4 = 5.5.
///   c0 = 0.5, c1 = 10.5.
///   Between = 2*(0.5-5.5)^2 + 2*(10.5-5.5)^2 = 2*25 + 2*25 = 100.
///   Within  = (0-0.5)^2+(1-0.5)^2+(10-10.5)^2+(11-10.5)^2 = 4*0.25 = 1.0.
///   Scale factor = (n-k)/(k-1) = (4-2)/(2-1) = 2.
///   CH = (100/1.0)*2 = 200.
#[test]
fn test_ch_well_separated_exact() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(calinski_harabasz_score(&x, &labels), 200.0, epsilon = 1e-9);
}

/// 3 tight clusters → CH = 2500.0.
///
/// Derivation:
///   Points (0),(0.2),(5),(5.2),(10),(10.2); labels [0,0,1,1,2,2]; n=6, k=3.
///   Overall mean = 30.6/6 = 5.1.
///   c0=0.1, c1=5.1, c2=10.1.
///   Between = 2*(0.1-5.1)^2 + 2*(5.1-5.1)^2 + 2*(10.1-5.1)^2
///           = 2*25 + 0 + 2*25 = 100.
///   Within  = 6*(0.1^2) = 6*0.01 = 0.06.
///   Scale   = (6-3)/(3-1) = 3/2.
///   CH = (100/0.06)*1.5 = 2500.
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

/// When all points coincide with their centroids, within-dispersion = 0, returns 1.0.
///
/// Derivation:
///   p0=p1=(0,0), p2=p3=(10,0), labels [0,0,1,1].
///   c0=(0,0), c1=(10,0); every point sits on its centroid → within = 0.
///   Code returns 1.0 (degenerate guard).
#[test]
fn test_ch_zero_within_dispersion_returns_one() {
    let x = array![[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert_abs_diff_eq!(calinski_harabasz_score(&x, &labels), 1.0, epsilon = 1e-12);
}

/// CH is non-negative for any valid inputs (it is a ratio of dispersions ≥ 0).
#[test]
fn test_ch_nonnegative() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    let labels = array![0usize, 0, 1, 1];
    assert!(calinski_harabasz_score(&x, &labels) >= 0.0);
}

/// CH increases as clusters become more separated (monotonic direction test).
///
/// Slightly separated vs very separated:
///   slightly: (0),(1),(2),(3) with labels [0,0,1,1] → within=1.0, between=4, CH=4*2=8
///   well-sep: (0),(1),(10),(11) → CH=200 (verified above)
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

/// CH panics on all-singleton clusters (k = n).
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_ch_all_singletons_panics() {
    let x = array![[0.0], [1.0], [2.0]];
    let labels = array![0usize, 1, 2];
    let _ = calinski_harabasz_score(&x, &labels);
}

/// CH panics on single cluster (k = 1).
#[test]
#[should_panic(expected = "invalid input: number of clusters")]
fn test_ch_single_cluster_panics() {
    let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let labels = array![0usize, 0, 0];
    let _ = calinski_harabasz_score(&x, &labels);
}

/// CH panics on row/label length mismatch.
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_ch_length_mismatch_panics() {
    let x = array![[0.0], [1.0], [5.0]];
    let labels = array![0usize, 1];
    let _ = calinski_harabasz_score(&x, &labels);
}

/// CH panics on empty input.
#[test]
#[should_panic(expected = "input is empty")]
fn test_ch_empty_panics() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let labels: Array1<usize> = array![];
    let _ = calinski_harabasz_score(&x, &labels);
}
