//! Integration tests for the `math` feature.
//!
//! `math` is a leaf module of pure, stateless, closed-form primitives, so its tests live
//! in this single file (declared as a `[[test]]` target in `Cargo.toml`, `autotests = false`).
//! All assertions are scalar/closed-form — use approx's `assert_abs_diff_eq!`
//! macro directly; no array helper is needed.
//!
//! IMPORTANT: Every expected value is derived from the mathematical definition or a
//! hand calculation — never from reading the implementation source.
#![cfg(feature = "math")]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::array;
use rustyml::math::*;

// ═══════════════════════════════════════════════════════════
// entropy
// ═══════════════════════════════════════════════════════════

/// Perfectly balanced binary dataset: p(0)=0.5, p(1)=0.5.
/// entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = -0.5*(-1) - 0.5*(-1) = 1.0 bit
#[test]
fn test_entropy_balanced_binary() {
    let labels = array![0.0_f64, 1.0, 1.0, 0.0];
    assert_abs_diff_eq!(entropy(&labels), 1.0, epsilon = 1e-9);
}

/// Homogeneous class — only label 0 present.
/// p(0)=1.0; entropy = -1.0*log2(1.0) = 0.0
#[test]
fn test_entropy_all_same_class() {
    let labels = array![0.0_f64, 0.0, 0.0];
    assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-9);
}

/// Empty array: function returns 0.0 (early-return on total_samples == 0.0).
#[test]
fn test_entropy_empty() {
    let labels: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
}

/// Four uniform classes: p_i = 0.25 each.
/// entropy = -4 * (0.25 * log2(0.25)) = -4 * (0.25 * (-2)) = 2.0 bits
#[test]
fn test_entropy_four_uniform_classes() {
    let labels = array![0.0_f64, 1.0, 2.0, 3.0];
    // derivation: H = -4 * 0.25 * log2(0.25) = -4 * 0.25 * (-2) = 2.0
    assert_abs_diff_eq!(entropy(&labels), 2.0, epsilon = 1e-9);
}

/// Single element: only one class; entropy = 0.0
#[test]
fn test_entropy_single_element() {
    let labels = array![42.0_f64];
    assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
}

/// Entropy is non-negative for any valid input.
#[test]
fn test_entropy_non_negative() {
    let cases: &[ndarray::Array1<f64>] = &[
        array![0.0, 0.0, 1.0],
        array![0.0, 1.0, 2.0],
        array![7.0, 7.0, 7.0],
    ];
    for labels in cases {
        assert!(entropy(labels) >= 0.0, "entropy must be non-negative");
    }
}

/// Three classes, counts 2:1:1 → probabilities 0.5, 0.25, 0.25.
/// H = -0.5*log2(0.5) - 0.25*log2(0.25) - 0.25*log2(0.25)
///   = 0.5 + 0.25*2 + 0.25*2 = 0.5 + 0.5 + 0.5 = 1.5
#[test]
fn test_entropy_three_classes_unequal() {
    let labels = array![0.0_f64, 0.0, 1.0, 2.0];
    // p(0)=0.5, p(1)=0.25, p(2)=0.25
    // H = -0.5*log2(0.5) - 0.25*log2(0.25) - 0.25*log2(0.25)
    //   = 0.5 + 0.5 + 0.5 = 1.5
    assert_abs_diff_eq!(entropy(&labels), 1.5, epsilon = 1e-9);
}

// ═══════════════════════════════════════════════════════════
// gini
// ═══════════════════════════════════════════════════════════

/// Balanced binary: p=0.5 each. gini = 1 - (0.5^2 + 0.5^2) = 0.5
#[test]
fn test_gini_balanced_binary() {
    let labels = array![0.0_f64, 0.0, 1.0, 1.0];
    // gini = 1 - (0.25 + 0.25) = 0.5
    assert_abs_diff_eq!(gini(&labels), 0.5, epsilon = 1e-9);
}

/// Homogeneous: only one class. gini = 1 - 1.0^2 = 0.0
#[test]
fn test_gini_all_same_class() {
    let labels = array![3.0_f64, 3.0, 3.0];
    assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-9);
}

/// Empty array returns 0.0 (early-return).
#[test]
fn test_gini_empty() {
    let labels: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
}

/// Four uniform classes: p_i = 0.25 each. gini = 1 - 4*(0.0625) = 0.75
#[test]
fn test_gini_four_uniform_classes() {
    let labels = array![0.0_f64, 1.0, 2.0, 3.0];
    // gini = 1 - 4*(0.25^2) = 1 - 4*0.0625 = 1 - 0.25 = 0.75
    assert_abs_diff_eq!(gini(&labels), 0.75, epsilon = 1e-9);
}

/// Single element: gini = 1 - 1.0^2 = 0.0
#[test]
fn test_gini_single_element() {
    let labels = array![99.0_f64];
    assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
}

/// Gini is in [0.0, 1.0].
#[test]
fn test_gini_value_range() {
    let cases: &[ndarray::Array1<f64>] = &[
        array![0.0, 1.0],
        array![0.0, 1.0, 2.0],
        array![0.0, 0.0, 1.0, 2.0, 3.0],
    ];
    for labels in cases {
        let g = gini(labels);
        assert!((0.0..=1.0).contains(&g), "gini={g} not in [0,1]");
    }
}

/// Three classes, counts 2:1:1 → p(0)=0.5, p(1)=0.25, p(2)=0.25.
/// gini = 1 - (0.25 + 0.0625 + 0.0625) = 1 - 0.375 = 0.625
#[test]
fn test_gini_three_classes_unequal() {
    let labels = array![0.0_f64, 0.0, 1.0, 2.0];
    // gini = 1 - (0.5^2 + 0.25^2 + 0.25^2) = 1 - (0.25 + 0.0625 + 0.0625) = 0.625
    assert_abs_diff_eq!(gini(&labels), 0.625, epsilon = 1e-9);
}

// ═══════════════════════════════════════════════════════════
// squared_euclidean_distance_row
// ═══════════════════════════════════════════════════════════

/// [1,2] vs [4,6]: (4-1)^2 + (6-2)^2 = 9 + 16 = 25
#[test]
fn test_squared_euclidean_basic() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    assert_abs_diff_eq!(
        squared_euclidean_distance_row(&v1, &v2),
        25.0,
        epsilon = 1e-10
    );
}

/// [1,2,3] vs [4,5,6]: (3^2)+(3^2)+(3^2) = 9+9+9 = 27
#[test]
fn test_squared_euclidean_three_dim() {
    let v1 = array![1.0_f64, 2.0, 3.0];
    let v2 = array![4.0_f64, 5.0, 6.0];
    assert_abs_diff_eq!(
        squared_euclidean_distance_row(&v1, &v2),
        27.0,
        epsilon = 1e-10
    );
}

/// Identical vectors: distance = 0
#[test]
fn test_squared_euclidean_identical() {
    let v = array![3.0_f64, 7.0, -1.0];
    assert_abs_diff_eq!(squared_euclidean_distance_row(&v, &v), 0.0, epsilon = 1e-12);
}

/// Symmetry: d(a,b) == d(b,a)
#[test]
fn test_squared_euclidean_symmetry() {
    let a = array![1.0_f64, 5.0];
    let b = array![4.0_f64, 1.0];
    let d_ab = squared_euclidean_distance_row(&a, &b);
    let d_ba = squared_euclidean_distance_row(&b, &a);
    assert_abs_diff_eq!(d_ab, d_ba, epsilon = 1e-12);
}

/// Non-negative: distance >= 0
#[test]
fn test_squared_euclidean_non_negative() {
    let a = array![2.0_f64, -3.0];
    let b = array![-1.0_f64, 4.0];
    assert!(squared_euclidean_distance_row(&a, &b) >= 0.0);
}

// ═══════════════════════════════════════════════════════════
// manhattan_distance_row
// ═══════════════════════════════════════════════════════════

/// [1,2] vs [4,6]: |4-1| + |6-2| = 3 + 4 = 7
#[test]
fn test_manhattan_basic() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    assert_abs_diff_eq!(manhattan_distance_row(&v1, &v2), 7.0, epsilon = 1e-10);
}

/// Identical vectors: 0
#[test]
fn test_manhattan_identical() {
    let v = array![1.0_f64, 2.0, 3.0];
    assert_abs_diff_eq!(manhattan_distance_row(&v, &v), 0.0, epsilon = 1e-12);
}

/// Symmetry: L1(a,b) == L1(b,a)
#[test]
fn test_manhattan_symmetry() {
    let a = array![0.0_f64, 5.0, -2.0];
    let b = array![3.0_f64, 1.0, 4.0];
    assert_abs_diff_eq!(
        manhattan_distance_row(&a, &b),
        manhattan_distance_row(&b, &a),
        epsilon = 1e-12
    );
}

/// [3,4] vs [0,0]: |3| + |4| = 7
#[test]
fn test_manhattan_from_origin() {
    let a = array![3.0_f64, 4.0];
    let b = array![0.0_f64, 0.0];
    assert_abs_diff_eq!(manhattan_distance_row(&a, &b), 7.0, epsilon = 1e-10);
}

// ═══════════════════════════════════════════════════════════
// minkowski_distance_row
// ═══════════════════════════════════════════════════════════

/// p=1 recovers L1 (manhattan): [1,2] vs [4,6] → 7
#[test]
fn test_minkowski_p1_equals_manhattan() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    let manhattan = manhattan_distance_row(&v1, &v2);
    let mink1 = minkowski_distance_row(&v1, &v2, 1.0);
    assert_abs_diff_eq!(mink1, manhattan, epsilon = 1e-10);
}

/// p=2 recovers L2 (Euclidean): [1,2] vs [4,6].
/// squared_euclidean = 25, so euclidean = 5.0
/// minkowski p=2 should equal sqrt(squared_euclidean)
#[test]
fn test_minkowski_p2_equals_euclidean() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    let sq_euc = squared_euclidean_distance_row(&v1, &v2); // 25.0
    let expected = sq_euc.sqrt(); // 5.0
    let mink2 = minkowski_distance_row(&v1, &v2, 2.0);
    assert_abs_diff_eq!(mink2, expected, epsilon = 1e-9);
}

/// p=2, [3,4] vs origin: sqrt(9+16) = 5.0
#[test]
fn test_minkowski_p2_pythagorean() {
    let a = array![3.0_f64, 4.0];
    let b = array![0.0_f64, 0.0];
    // 3-4-5 right triangle: sqrt(9+16) = 5
    assert_abs_diff_eq!(minkowski_distance_row(&a, &b, 2.0), 5.0, epsilon = 1e-10);
}

/// p=3, [1,2] vs [4,6].
/// diffs: 3, 4.  sum = 3^3 + 4^3 = 27 + 64 = 91.  result = 91^(1/3).
/// 91^(1/3): 4.497^3 ≈ 90.94; 4.498^3 ≈ 91.00 → use relative tolerance
#[test]
fn test_minkowski_p3_hand_calc() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    // |4-1|^3 + |6-2|^3 = 27 + 64 = 91; result = 91^(1/3)
    let expected = 91.0_f64.powf(1.0 / 3.0);
    assert_relative_eq!(
        minkowski_distance_row(&v1, &v2, 3.0),
        expected,
        max_relative = 1e-10
    );
}

/// Identical vectors: minkowski distance = 0 for any p
#[test]
fn test_minkowski_identical() {
    let v = array![2.0_f64, 5.0, -1.0];
    assert_abs_diff_eq!(minkowski_distance_row(&v, &v, 2.0), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(minkowski_distance_row(&v, &v, 3.0), 0.0, epsilon = 1e-12);
}

/// Symmetry: minkowski(a,b,p) == minkowski(b,a,p)
#[test]
fn test_minkowski_symmetry() {
    let a = array![1.0_f64, 5.0];
    let b = array![4.0_f64, 1.0];
    assert_abs_diff_eq!(
        minkowski_distance_row(&a, &b, 2.5),
        minkowski_distance_row(&b, &a, 2.5),
        epsilon = 1e-12
    );
}

// ═══════════════════════════════════════════════════════════
// sum_of_square_total
// ═══════════════════════════════════════════════════════════

/// [1,2,3]: mean=2. SST=(1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1 = 2.0
#[test]
fn test_sst_basic() {
    let v = array![1.0_f64, 2.0, 3.0];
    assert_abs_diff_eq!(sum_of_square_total(&v), 2.0, epsilon = 1e-10);
}

/// All-same values: deviations are zero, SST = 0.0
#[test]
fn test_sst_all_same() {
    let v = array![5.0_f64, 5.0, 5.0, 5.0];
    assert_abs_diff_eq!(sum_of_square_total(&v), 0.0, epsilon = 1e-12);
}

/// Empty array returns 0.0 (early-return).
#[test]
fn test_sst_empty() {
    let v: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(sum_of_square_total(&v), 0.0, epsilon = 1e-12);
}

/// Single element: mean = value, deviation = 0, SST = 0.0
#[test]
fn test_sst_single_element() {
    let v = array![7.0_f64];
    assert_abs_diff_eq!(sum_of_square_total(&v), 0.0, epsilon = 1e-12);
}

/// [1,3]: mean=2. SST=(1-2)^2+(3-2)^2 = 1+1 = 2.0
#[test]
fn test_sst_two_elements() {
    let v = array![1.0_f64, 3.0];
    assert_abs_diff_eq!(sum_of_square_total(&v), 2.0, epsilon = 1e-10);
}

/// SST relationship to variance: SST = variance * n
#[test]
fn test_sst_equals_variance_times_n() {
    let v = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let sst = sum_of_square_total(&v);
    let var = variance(&v);
    let n = v.len() as f64;
    // SST = variance * n by definition (both use population formula)
    assert_abs_diff_eq!(sst, var * n, epsilon = 1e-9);
}

// ═══════════════════════════════════════════════════════════
// sum_of_squared_errors
// ═══════════════════════════════════════════════════════════

/// [2,3] predicted vs [1,3] actual: (2-1)^2 + (3-3)^2 = 1+0 = 1.0
#[test]
fn test_sse_basic() {
    let predicted = array![2.0_f64, 3.0];
    let actual = array![1.0_f64, 3.0];
    assert_abs_diff_eq!(
        sum_of_squared_errors(&predicted, &actual),
        1.0,
        epsilon = 1e-10
    );
}

/// Perfect prediction: SSE = 0.0
#[test]
fn test_sse_perfect_prediction() {
    let v = array![1.0_f64, 2.0, 3.0];
    assert_abs_diff_eq!(sum_of_squared_errors(&v, &v), 0.0, epsilon = 1e-12);
}

/// [0,0,0] vs [1,1,1]: 3 * 1 = 3.0
#[test]
fn test_sse_unit_errors() {
    let predicted = array![0.0_f64, 0.0, 0.0];
    let actual = array![1.0_f64, 1.0, 1.0];
    // (0-1)^2 * 3 = 3.0
    assert_abs_diff_eq!(
        sum_of_squared_errors(&predicted, &actual),
        3.0,
        epsilon = 1e-10
    );
}

/// Single element: [5] vs [2] → (5-2)^2 = 9
#[test]
fn test_sse_single_element() {
    let p = array![5.0_f64];
    let a = array![2.0_f64];
    assert_abs_diff_eq!(sum_of_squared_errors(&p, &a), 9.0, epsilon = 1e-10);
}

/// SSE is symmetric (swapping predicted and actual gives same result).
/// (p-a)^2 = (a-p)^2 by definition.
#[test]
fn test_sse_symmetry() {
    let p = array![1.0_f64, 3.0, 5.0];
    let a = array![2.0_f64, 1.0, 4.0];
    assert_abs_diff_eq!(
        sum_of_squared_errors(&p, &a),
        sum_of_squared_errors(&a, &p),
        epsilon = 1e-12
    );
}

// ═══════════════════════════════════════════════════════════
// variance
// ═══════════════════════════════════════════════════════════

/// [1,2,3]: mean=2. var=((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3 ≈ 0.66667
#[test]
fn test_variance_basic() {
    let v = array![1.0_f64, 2.0, 3.0];
    let expected = 2.0_f64 / 3.0; // = 0.666...
    assert_abs_diff_eq!(variance(&v), expected, epsilon = 1e-9);
}

/// [2,4,4,4,5,5,7,9]: mean=5. Deviations: -3,-1,-1,-1,0,0,2,4.
/// Squares: 9,1,1,1,0,0,4,16. Sum=32. var=32/8=4.0
#[test]
fn test_variance_known_textbook() {
    let v = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    // mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5.0
    // SSQ = 9+1+1+1+0+0+4+16 = 32; var = 32/8 = 4.0
    assert_abs_diff_eq!(variance(&v), 4.0, epsilon = 1e-9);
}

/// Single element: variance = 0.0 (no spread)
#[test]
fn test_variance_single_element() {
    let v = array![42.0_f64];
    assert_abs_diff_eq!(variance(&v), 0.0, epsilon = 1e-12);
}

/// Empty array returns 0.0.
#[test]
fn test_variance_empty() {
    let v: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(variance(&v), 0.0, epsilon = 1e-12);
}

/// Constant array: all same value → variance = 0.0
#[test]
fn test_variance_constant() {
    let v = array![3.0_f64, 3.0, 3.0, 3.0];
    assert_abs_diff_eq!(variance(&v), 0.0, epsilon = 1e-12);
}

/// Variance is non-negative.
#[test]
fn test_variance_non_negative() {
    let v = array![1.0_f64, 5.0, 2.0, 8.0, 3.0];
    assert!(variance(&v) >= 0.0);
}

// ═══════════════════════════════════════════════════════════
// standard_deviation
// ═══════════════════════════════════════════════════════════

/// [1,2,3]: variance=2/3; std = sqrt(2/3) ≈ 0.8164965809...
#[test]
fn test_std_basic() {
    let v = array![1.0_f64, 2.0, 3.0];
    let expected = (2.0_f64 / 3.0).sqrt(); // ≈ 0.81649658
    assert_abs_diff_eq!(standard_deviation(&v), expected, epsilon = 1e-9);
}

/// [2,4,4,4,5,5,7,9]: variance=4.0; std=sqrt(4)=2.0
#[test]
fn test_std_known_textbook() {
    let v = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert_abs_diff_eq!(standard_deviation(&v), 2.0, epsilon = 1e-9);
}

/// Empty array: 0.0
#[test]
fn test_std_empty() {
    let v: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(standard_deviation(&v), 0.0, epsilon = 1e-12);
}

/// Cross-check: std = sqrt(variance) for the same array.
#[test]
fn test_std_equals_sqrt_variance() {
    let v = array![1.0_f64, 3.0, 5.0, 7.0, 9.0];
    let var = variance(&v);
    let std = standard_deviation(&v);
    assert_abs_diff_eq!(std, var.sqrt(), epsilon = 1e-10);
}

/// Constant array: std = 0.0
#[test]
fn test_std_constant() {
    let v = array![5.0_f64, 5.0, 5.0];
    assert_abs_diff_eq!(standard_deviation(&v), 0.0, epsilon = 1e-12);
}

/// Standard deviation is non-negative.
#[test]
fn test_std_non_negative() {
    let v = array![-3.0_f64, 1.0, 7.0];
    assert!(standard_deviation(&v) >= 0.0);
}

// ═══════════════════════════════════════════════════════════
// sigmoid
// ═══════════════════════════════════════════════════════════

/// sigmoid(0.0) = 1/(1+e^0) = 1/2 = 0.5 (by definition)
#[test]
fn test_sigmoid_at_zero() {
    assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
}

/// sigmoid(ln 3) = 1/(1+e^{-ln3}) = 1/(1 + 1/3) = 3/4 = 0.75
#[test]
fn test_sigmoid_at_ln3() {
    let x = 3.0_f64.ln();
    // 1/(1 + e^{-ln3}) = 1/(1 + 1/3) = 1/(4/3) = 3/4 = 0.75
    assert_abs_diff_eq!(sigmoid(x), 0.75, epsilon = 1e-10);
}

/// sigmoid(-ln 3) = 1/(1+e^{ln3}) = 1/(1+3) = 0.25 (symmetry: sigmoid(-x)=1-sigmoid(x))
#[test]
fn test_sigmoid_symmetry() {
    let x = 3.0_f64.ln();
    // sigmoid(-x) = 1 - sigmoid(x) = 1 - 0.75 = 0.25
    assert_abs_diff_eq!(sigmoid(-x), 1.0 - sigmoid(x), epsilon = 1e-12);
    assert_abs_diff_eq!(sigmoid(-x), 0.25, epsilon = 1e-10);
}

/// Output always in (0, 1) for finite inputs (clipped to exactly 0 or 1 at extremes).
#[test]
fn test_sigmoid_output_range() {
    for &z in &[-1000.0_f64, -1.0, 0.0, 1.0, 1000.0] {
        let s = sigmoid(z);
        assert!((0.0..=1.0).contains(&s), "sigmoid({z}) = {s} not in [0,1]");
    }
}

/// Very large positive input: implementation clips to 1.0 (z > 500.0 branch).
#[test]
fn test_sigmoid_large_positive() {
    assert_abs_diff_eq!(sigmoid(1000.0), 1.0, epsilon = 1e-12);
}

/// Very large negative input: implementation clips to 0.0 (z < -500.0 branch).
#[test]
fn test_sigmoid_large_negative() {
    assert_abs_diff_eq!(sigmoid(-1000.0), 0.0, epsilon = 1e-12);
}

/// Monotonically increasing: sigmoid(a) < sigmoid(b) when a < b
#[test]
fn test_sigmoid_monotone() {
    assert!(sigmoid(-2.0) < sigmoid(-1.0));
    assert!(sigmoid(-1.0) < sigmoid(0.0));
    assert!(sigmoid(0.0) < sigmoid(1.0));
    assert!(sigmoid(1.0) < sigmoid(2.0));
}

// ═══════════════════════════════════════════════════════════
// logistic_loss
// ═══════════════════════════════════════════════════════════

/// Single sample, logit=0, label=0.
/// Stable formula: max(0,0) - 0*0 + ln(1+exp(0)) = 0 + ln(2) = ln(2) ≈ 0.6931...
#[test]
fn test_logistic_loss_logit_zero_label_zero() {
    let logits = array![0.0_f64];
    let labels = array![0.0_f64];
    let expected = 2.0_f64.ln(); // ln(2), same as for label=1 since x=0
    assert_abs_diff_eq!(logistic_loss(&logits, &labels), expected, epsilon = 1e-9);
}

/// Two perfect predictions at large logit values → loss approaches 0.
/// logit=+20, label=1: max(0,20) - 20*1 + ln(1+exp(-20)) = 0 + tiny ≈ 0
/// logit=-20, label=0: max(0,-20) - (-20)*0 + ln(1+exp(-20)) = 0 + tiny ≈ 0
#[test]
fn test_logistic_loss_near_zero_on_confident_correct() {
    let logits = array![20.0_f64, -20.0];
    let labels = array![1.0_f64, 0.0];
    // Both losses are essentially 0 for very confident correct predictions
    assert_abs_diff_eq!(logistic_loss(&logits, &labels), 0.0, epsilon = 1e-6);
}

/// Loss >= 0 always (it is the average of per-sample non-negative values).
#[test]
fn test_logistic_loss_non_negative() {
    let logits = array![0.5_f64, -1.0, 2.0, -0.3];
    let labels = array![1.0_f64, 0.0, 1.0, 0.0];
    assert!(logistic_loss(&logits, &labels) >= 0.0);
}

/// Three samples: logits=[1.0, -1.0, 0.0], labels=[1.0, 0.0, 1.0].
/// Per stable formula: max(0,x) - x*y + ln(1+exp(-|x|))
///   x=1,y=1:  max(0,1) - 1*1 + ln(1+e^{-1}) = 1 - 1 + ln(1+1/e) = ln((e+1)/e)
///             = ln(e+1) - 1 ≈ 1.313261687 - 1 = 0.313262
///   x=-1,y=0: max(0,-1) - (-1)*0 + ln(1+e^{-1}) = 0 + ln((e+1)/e) ≈ 0.313262
///   x=0,y=1:  max(0,0) - 0*1 + ln(1+e^0) = ln(2) ≈ 0.693147
///   mean = (0.313262 + 0.313262 + 0.693147)/3 = 1.319671/3 ≈ 0.439890
#[test]
fn test_logistic_loss_three_samples() {
    let logits = array![1.0_f64, -1.0, 0.0];
    let labels = array![1.0_f64, 0.0, 1.0];
    // derivation: see above; all three are ln((e+1)/e) or ln(2)
    // ln((e+1)/e) = ln(e+1) - ln(e) = ln(e+1) - 1
    let e = std::f64::consts::E;
    let loss_a = (1.0 + 1.0 / e).ln(); // ln(1 + e^{-1})
    let loss_b = (1.0 + 1.0 / e).ln(); // same
    let loss_c = 2.0_f64.ln(); // ln(2)
    let expected = (loss_a + loss_b + loss_c) / 3.0;
    assert_abs_diff_eq!(logistic_loss(&logits, &labels), expected, epsilon = 1e-9);
}

// ═══════════════════════════════════════════════════════════
// hinge_loss
// ═══════════════════════════════════════════════════════════

/// Perfect classification with large margin: all samples have y*m >= 1.
/// scores=[2.0,-3.0], labels=[1.0,-1.0]:
///   max(0,1-1*2)=0, max(0,1-(-1)*(-3))=max(0,-2)=0 → mean=0
#[test]
fn test_hinge_loss_large_margins_zero() {
    let margins = array![2.0_f64, -3.0];
    let labels = array![1.0_f64, -1.0];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-10);
}

/// All misclassified by margin=1: y*m = -1 for all.
/// Scores=[-1,+1], labels=[1,-1]:
///   max(0,1-1*(-1))=2, max(0,1-(-1)*1)=2 → mean=2.0
#[test]
fn test_hinge_loss_all_wrong() {
    let margins = array![-1.0_f64, 1.0];
    let labels = array![1.0_f64, -1.0];
    // item 0: max(0, 1 - 1*(-1)) = max(0,2) = 2
    // item 1: max(0, 1 - (-1)*1) = max(0,2) = 2
    // mean = 4/2 = 2.0
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 2.0, epsilon = 1e-10);
}

/// Exactly on the decision boundary (margin*label = 1): loss = 0.
/// margin=1, label=1: max(0, 1-1*1) = max(0,0) = 0
#[test]
fn test_hinge_loss_on_boundary() {
    let margins = array![1.0_f64];
    let labels = array![1.0_f64];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-12);
}

/// Mixed: [1.5, 0.5, -0.5], labels=[1,-1,1].
/// item 0: max(0,1-1*1.5) = max(0,-0.5) = 0
/// item 1: max(0,1-(-1)*0.5) = max(0,1.5) = 1.5
/// item 2: max(0,1-1*(-0.5)) = max(0,1.5) = 1.5
/// mean = (0+1.5+1.5)/3 = 3.0/3 = 1.0
#[test]
fn test_hinge_loss_mixed() {
    let margins = array![1.5_f64, 0.5, -0.5];
    let labels = array![1.0_f64, -1.0, 1.0];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 1.0, epsilon = 1e-10);
}

/// Hinge loss is non-negative by construction (max(0,...) guarantees it).
#[test]
fn test_hinge_loss_non_negative() {
    let margins = array![0.3_f64, -0.7, 1.2, 2.5];
    let labels = array![1.0_f64, -1.0, 1.0, -1.0];
    assert!(hinge_loss(&margins, &labels) >= 0.0);
}

// ═══════════════════════════════════════════════════════════
// average_path_length_factor
// ═══════════════════════════════════════════════════════════

/// n=4: uses exact harmonic (n<=50).
/// H_3 = 1 + 1/2 + 1/3 = 11/6
/// c(4) = 2*(11/6) - 2*(4-1)/4 = 11/3 - 6/4 = 22/6 - 9/6 = 13/6 ≈ 2.1666...
#[test]
fn test_average_path_length_n4() {
    // H_3 = 1 + 0.5 + 1/3 = 11/6
    // c(4) = 2*(11/6) - 2*3/4 = 11/3 - 3/2 = 22/6 - 9/6 = 13/6
    let expected = 13.0_f64 / 6.0;
    assert_abs_diff_eq!(average_path_length_factor(4), expected, epsilon = 1e-9);
}

/// n=5: H_4 = 1 + 1/2 + 1/3 + 1/4 = 25/12
/// c(5) = 2*(25/12) - 2*(5-1)/5 = 25/6 - 8/5 = 125/30 - 48/30 = 77/30 ≈ 2.56667
#[test]
fn test_average_path_length_n5() {
    // H_4 = 1 + 1/2 + 1/3 + 1/4 = 12/12+6/12+4/12+3/12 = 25/12
    // c(5) = 2*(25/12) - 2*4/5 = 25/6 - 8/5 = 125/30 - 48/30 = 77/30
    let expected = 77.0_f64 / 30.0;
    assert_abs_diff_eq!(average_path_length_factor(5), expected, epsilon = 1e-9);
}

/// Output is 0.0 for n=0 and n=1; 1.0 for n=2 (special-case returns).
#[test]
fn test_average_path_length_base_cases() {
    assert_abs_diff_eq!(average_path_length_factor(0), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(average_path_length_factor(1), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(average_path_length_factor(2), 1.0, epsilon = 1e-12);
}

/// Factor is positive for n >= 3.
#[test]
fn test_average_path_length_positive_for_n_ge_3() {
    for n in 3..=200 {
        let f = average_path_length_factor(n);
        assert!(f > 0.0, "expected positive factor for n={n}, got {f}");
    }
}

/// At the exact branch boundary (n=50 exact-harmonic, n=51 ln+gamma),
/// the output is continuous (no jump): |c(50) - c(51)| < 0.1
#[test]
fn test_average_path_length_continuous_at_branch_boundary() {
    let f50 = average_path_length_factor(50);
    let f51 = average_path_length_factor(51);
    // The step from n=50 to n=51 should be a small increment, not a discontinuity.
    let delta = (f51 - f50).abs();
    assert!(
        delta < 0.1,
        "discontinuity at branch boundary: c(50)={f50}, c(51)={f51}, |diff|={delta}"
    );
    // And the direction should still be increasing
    assert!(f51 > f50, "expected c(51)={f51} > c(50)={f50}");
}
// ═══════════════════════════════════════════════════════════
// variance / standard_deviation — non-finite (NaN) handling
// ═══════════════════════════════════════════════════════════

/// `variance` skips non-finite entries and computes the variance over the finite subset.
///
/// For `[1, NaN, 3]` the NaN is dropped; the remaining finite values {1, 3} have mean 2 and
/// population variance ((1-2)² + (3-2)²) / 2 = 1.0 (dividing by the finite count, 2). So
/// `variance(&[1, NaN, 3])` must equal `variance(&[1, 3])` == 1.0. An input with no finite
/// values has nothing to summarize, so it returns 0.0 (like the empty case).
#[test]
fn test_variance_skips_non_finite_and_uses_finite_subset() {
    let finite_only = array![1.0_f64, 3.0];
    assert_abs_diff_eq!(variance(&finite_only), 1.0, epsilon = 1e-12);

    // A NaN element is skipped ⇒ same finite-subset variance as {1, 3}.
    let with_nan = array![1.0_f64, f64::NAN, 3.0];
    assert_abs_diff_eq!(variance(&with_nan), 1.0, epsilon = 1e-12);

    // An infinite element is skipped the same way.
    let with_inf = array![1.0_f64, f64::INFINITY, 3.0];
    assert_abs_diff_eq!(variance(&with_inf), 1.0, epsilon = 1e-12);

    // No finite values at all ⇒ 0.0.
    let all_nan = array![f64::NAN, f64::NAN];
    assert_abs_diff_eq!(variance(&all_nan), 0.0, epsilon = 1e-12);
}

/// `standard_deviation` inherits `variance`'s finite-subset contract — it is `sqrt(variance)`,
/// so a NaN/infinite entry is skipped rather than propagated.
///
/// For `[1, NaN, 3]` the finite subset {1, 3} has variance 1.0, so the standard deviation is
/// sqrt(1.0) = 1.0 (NOT NaN).
#[test]
fn test_standard_deviation_skips_non_finite() {
    let with_nan = array![1.0_f64, f64::NAN, 3.0];
    assert_abs_diff_eq!(standard_deviation(&with_nan), 1.0, epsilon = 1e-12);

    // Consistency: standard_deviation == sqrt(variance) over the same finite subset.
    assert_abs_diff_eq!(
        standard_deviation(&with_nan),
        variance(&with_nan).sqrt(),
        epsilon = 1e-12
    );
}
