//! Integration tests for the `math` feature: entropy, gini, distance metrics, variance,
//! standard deviation, sigmoid, loss functions, and the isolation-forest path-length factor
#![cfg(feature = "math")]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::array;
use rustyml::math::*;

// entropy

/// Balanced binary dataset (p=0.5 each) has entropy 1.0 bit
#[test]
fn test_entropy_balanced_binary() {
    let labels = array![0.0_f64, 1.0, 1.0, 0.0];
    assert_abs_diff_eq!(entropy(&labels), 1.0, epsilon = 1e-9);
}

/// Homogeneous class (single label present) has entropy 0.0
#[test]
fn test_entropy_all_same_class() {
    let labels = array![0.0_f64, 0.0, 0.0];
    assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-9);
}

/// Empty array returns 0.0
#[test]
fn test_entropy_empty() {
    let labels: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
}

/// 4 uniform classes (p=0.25 each) have entropy 2.0 bits
#[test]
fn test_entropy_four_uniform_classes() {
    let labels = array![0.0_f64, 1.0, 2.0, 3.0];
    assert_abs_diff_eq!(entropy(&labels), 2.0, epsilon = 1e-9);
}

/// Single element has entropy 0.0
#[test]
fn test_entropy_single_element() {
    let labels = array![42.0_f64];
    assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
}

/// Entropy is non-negative for any valid input
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

/// 3 classes with counts 2:1:1 (p = 0.5, 0.25, 0.25) have entropy 1.5
#[test]
fn test_entropy_three_classes_unequal() {
    let labels = array![0.0_f64, 0.0, 1.0, 2.0];
    assert_abs_diff_eq!(entropy(&labels), 1.5, epsilon = 1e-9);
}

// gini

/// Balanced binary (p=0.5 each) has gini 0.5
#[test]
fn test_gini_balanced_binary() {
    let labels = array![0.0_f64, 0.0, 1.0, 1.0];
    assert_abs_diff_eq!(gini(&labels), 0.5, epsilon = 1e-9);
}

/// Homogeneous (single class) has gini 0.0
#[test]
fn test_gini_all_same_class() {
    let labels = array![3.0_f64, 3.0, 3.0];
    assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-9);
}

/// Empty array returns 0.0
#[test]
fn test_gini_empty() {
    let labels: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
}

/// 4 uniform classes (p=0.25 each) have gini 0.75
#[test]
fn test_gini_four_uniform_classes() {
    let labels = array![0.0_f64, 1.0, 2.0, 3.0];
    assert_abs_diff_eq!(gini(&labels), 0.75, epsilon = 1e-9);
}

/// Single element has gini 0.0
#[test]
fn test_gini_single_element() {
    let labels = array![99.0_f64];
    assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
}

/// Gini is in [0.0, 1.0]
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

/// 3 classes with counts 2:1:1 (p = 0.5, 0.25, 0.25) have gini 0.625
#[test]
fn test_gini_three_classes_unequal() {
    let labels = array![0.0_f64, 0.0, 1.0, 2.0];
    assert_abs_diff_eq!(gini(&labels), 0.625, epsilon = 1e-9);
}

// squared_euclidean_distance_row

/// [1,2] vs [4,6] gives squared distance 25
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

/// [1,2,3] vs [4,5,6] gives squared distance 27
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

/// Identical vectors have distance 0
#[test]
fn test_squared_euclidean_identical() {
    let v = array![3.0_f64, 7.0, -1.0];
    assert_abs_diff_eq!(squared_euclidean_distance_row(&v, &v), 0.0, epsilon = 1e-12);
}

/// Distance is symmetric: d(a,b) == d(b,a)
#[test]
fn test_squared_euclidean_symmetry() {
    let a = array![1.0_f64, 5.0];
    let b = array![4.0_f64, 1.0];
    let d_ab = squared_euclidean_distance_row(&a, &b);
    let d_ba = squared_euclidean_distance_row(&b, &a);
    assert_abs_diff_eq!(d_ab, d_ba, epsilon = 1e-12);
}

/// Distance is non-negative
#[test]
fn test_squared_euclidean_non_negative() {
    let a = array![2.0_f64, -3.0];
    let b = array![-1.0_f64, 4.0];
    assert!(squared_euclidean_distance_row(&a, &b) >= 0.0);
}

// manhattan_distance_row

/// [1,2] vs [4,6] gives L1 distance 7
#[test]
fn test_manhattan_basic() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    assert_abs_diff_eq!(manhattan_distance_row(&v1, &v2), 7.0, epsilon = 1e-10);
}

/// Identical vectors have distance 0
#[test]
fn test_manhattan_identical() {
    let v = array![1.0_f64, 2.0, 3.0];
    assert_abs_diff_eq!(manhattan_distance_row(&v, &v), 0.0, epsilon = 1e-12);
}

/// L1 distance is symmetric: L1(a,b) == L1(b,a)
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

/// [3,4] vs origin gives L1 distance 7
#[test]
fn test_manhattan_from_origin() {
    let a = array![3.0_f64, 4.0];
    let b = array![0.0_f64, 0.0];
    assert_abs_diff_eq!(manhattan_distance_row(&a, &b), 7.0, epsilon = 1e-10);
}

// minkowski_distance_row

/// p=1 recovers the L1 (manhattan) distance
#[test]
fn test_minkowski_p1_equals_manhattan() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    let manhattan = manhattan_distance_row(&v1, &v2);
    let mink1 = minkowski_distance_row(&v1, &v2, 1.0);
    assert_abs_diff_eq!(mink1, manhattan, epsilon = 1e-10);
}

/// p=2 recovers the L2 (Euclidean) distance, equal to sqrt(squared_euclidean)
#[test]
fn test_minkowski_p2_equals_euclidean() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    let sq_euc = squared_euclidean_distance_row(&v1, &v2); // 25.0
    let expected = sq_euc.sqrt(); // 5.0
    let mink2 = minkowski_distance_row(&v1, &v2, 2.0);
    assert_abs_diff_eq!(mink2, expected, epsilon = 1e-9);
}

/// p=2 on [3,4] vs origin gives the 3-4-5 right-triangle distance 5.0
#[test]
fn test_minkowski_p2_pythagorean() {
    let a = array![3.0_f64, 4.0];
    let b = array![0.0_f64, 0.0];
    assert_abs_diff_eq!(minkowski_distance_row(&a, &b, 2.0), 5.0, epsilon = 1e-10);
}

/// p=3 on [1,2] vs [4,6] gives (3^3 + 4^3)^(1/3) = 91^(1/3)
#[test]
fn test_minkowski_p3_hand_calc() {
    let v1 = array![1.0_f64, 2.0];
    let v2 = array![4.0_f64, 6.0];
    let expected = 91.0_f64.powf(1.0 / 3.0);
    assert_relative_eq!(
        minkowski_distance_row(&v1, &v2, 3.0),
        expected,
        max_relative = 1e-10
    );
}

/// Identical vectors have minkowski distance 0 for any p
#[test]
fn test_minkowski_identical() {
    let v = array![2.0_f64, 5.0, -1.0];
    assert_abs_diff_eq!(minkowski_distance_row(&v, &v, 2.0), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(minkowski_distance_row(&v, &v, 3.0), 0.0, epsilon = 1e-12);
}

/// Minkowski distance is symmetric: minkowski(a,b,p) == minkowski(b,a,p)
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

// sum_of_square_total

/// [1,2,3] (mean 2) gives SST 2.0
#[test]
fn test_sst_basic() {
    let v = array![1.0_f64, 2.0, 3.0];
    assert_abs_diff_eq!(sum_of_square_total(&v), 2.0, epsilon = 1e-10);
}

/// All-same values give SST 0.0
#[test]
fn test_sst_all_same() {
    let v = array![5.0_f64, 5.0, 5.0, 5.0];
    assert_abs_diff_eq!(sum_of_square_total(&v), 0.0, epsilon = 1e-12);
}

/// Empty array returns 0.0
#[test]
fn test_sst_empty() {
    let v: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(sum_of_square_total(&v), 0.0, epsilon = 1e-12);
}

/// Single element gives SST 0.0
#[test]
fn test_sst_single_element() {
    let v = array![7.0_f64];
    assert_abs_diff_eq!(sum_of_square_total(&v), 0.0, epsilon = 1e-12);
}

/// [1,3] (mean 2) gives SST 2.0
#[test]
fn test_sst_two_elements() {
    let v = array![1.0_f64, 3.0];
    assert_abs_diff_eq!(sum_of_square_total(&v), 2.0, epsilon = 1e-10);
}

/// SST equals variance * n (both use the population formula)
#[test]
fn test_sst_equals_variance_times_n() {
    let v = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let sst = sum_of_square_total(&v);
    let var = variance(&v);
    let n = v.len() as f64;
    assert_abs_diff_eq!(sst, var * n, epsilon = 1e-9);
}

// sum_of_squared_errors

/// [2,3] predicted vs [1,3] actual gives SSE 1.0
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

/// Perfect prediction gives SSE 0.0
#[test]
fn test_sse_perfect_prediction() {
    let v = array![1.0_f64, 2.0, 3.0];
    assert_abs_diff_eq!(sum_of_squared_errors(&v, &v), 0.0, epsilon = 1e-12);
}

/// [0,0,0] vs [1,1,1] gives SSE 3.0
#[test]
fn test_sse_unit_errors() {
    let predicted = array![0.0_f64, 0.0, 0.0];
    let actual = array![1.0_f64, 1.0, 1.0];
    assert_abs_diff_eq!(
        sum_of_squared_errors(&predicted, &actual),
        3.0,
        epsilon = 1e-10
    );
}

/// Single element [5] vs [2] gives SSE 9
#[test]
fn test_sse_single_element() {
    let p = array![5.0_f64];
    let a = array![2.0_f64];
    assert_abs_diff_eq!(sum_of_squared_errors(&p, &a), 9.0, epsilon = 1e-10);
}

/// SSE is symmetric under swapping predicted and actual, since (p-a)^2 = (a-p)^2
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

// variance

/// [1,2,3] (mean 2) has population variance 2/3
#[test]
fn test_variance_basic() {
    let v = array![1.0_f64, 2.0, 3.0];
    let expected = 2.0_f64 / 3.0;
    assert_abs_diff_eq!(variance(&v), expected, epsilon = 1e-9);
}

/// Textbook sample [2,4,4,4,5,5,7,9] (mean 5) has variance 4.0
#[test]
fn test_variance_known_textbook() {
    let v = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert_abs_diff_eq!(variance(&v), 4.0, epsilon = 1e-9);
}

/// Single element has variance 0.0
#[test]
fn test_variance_single_element() {
    let v = array![42.0_f64];
    assert_abs_diff_eq!(variance(&v), 0.0, epsilon = 1e-12);
}

/// Empty array returns 0.0
#[test]
fn test_variance_empty() {
    let v: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(variance(&v), 0.0, epsilon = 1e-12);
}

/// Constant array has variance 0.0
#[test]
fn test_variance_constant() {
    let v = array![3.0_f64, 3.0, 3.0, 3.0];
    assert_abs_diff_eq!(variance(&v), 0.0, epsilon = 1e-12);
}

/// Variance is non-negative
#[test]
fn test_variance_non_negative() {
    let v = array![1.0_f64, 5.0, 2.0, 8.0, 3.0];
    assert!(variance(&v) >= 0.0);
}

// standard_deviation

/// [1,2,3] (variance 2/3) has std sqrt(2/3)
#[test]
fn test_std_basic() {
    let v = array![1.0_f64, 2.0, 3.0];
    let expected = (2.0_f64 / 3.0).sqrt();
    assert_abs_diff_eq!(standard_deviation(&v), expected, epsilon = 1e-9);
}

/// Textbook sample [2,4,4,4,5,5,7,9] (variance 4.0) has std 2.0
#[test]
fn test_std_known_textbook() {
    let v = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert_abs_diff_eq!(standard_deviation(&v), 2.0, epsilon = 1e-9);
}

/// Empty array returns 0.0
#[test]
fn test_std_empty() {
    let v: ndarray::Array1<f64> = array![];
    assert_abs_diff_eq!(standard_deviation(&v), 0.0, epsilon = 1e-12);
}

/// std equals sqrt(variance) for the same array
#[test]
fn test_std_equals_sqrt_variance() {
    let v = array![1.0_f64, 3.0, 5.0, 7.0, 9.0];
    let var = variance(&v);
    let std = standard_deviation(&v);
    assert_abs_diff_eq!(std, var.sqrt(), epsilon = 1e-10);
}

/// Constant array has std 0.0
#[test]
fn test_std_constant() {
    let v = array![5.0_f64, 5.0, 5.0];
    assert_abs_diff_eq!(standard_deviation(&v), 0.0, epsilon = 1e-12);
}

/// Standard deviation is non-negative
#[test]
fn test_std_non_negative() {
    let v = array![-3.0_f64, 1.0, 7.0];
    assert!(standard_deviation(&v) >= 0.0);
}

// sigmoid

/// sigmoid(0.0) = 0.5
#[test]
fn test_sigmoid_at_zero() {
    assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
}

/// sigmoid(ln 3) = 0.75
#[test]
fn test_sigmoid_at_ln3() {
    let x = 3.0_f64.ln();
    assert_abs_diff_eq!(sigmoid(x), 0.75, epsilon = 1e-10);
}

/// Reflection symmetry sigmoid(-x) = 1 - sigmoid(x); sigmoid(-ln 3) = 0.25
#[test]
fn test_sigmoid_symmetry() {
    let x = 3.0_f64.ln();
    assert_abs_diff_eq!(sigmoid(-x), 1.0 - sigmoid(x), epsilon = 1e-12);
    assert_abs_diff_eq!(sigmoid(-x), 0.25, epsilon = 1e-10);
}

/// Output stays in [0, 1] across a range of inputs
#[test]
fn test_sigmoid_output_range() {
    for &z in &[-1000.0_f64, -1.0, 0.0, 1.0, 1000.0] {
        let s = sigmoid(z);
        assert!((0.0..=1.0).contains(&s), "sigmoid({z}) = {s} not in [0,1]");
    }
}

/// Very large positive input clips to 1.0
#[test]
fn test_sigmoid_large_positive() {
    assert_abs_diff_eq!(sigmoid(1000.0), 1.0, epsilon = 1e-12);
}

/// Very large negative input clips to 0.0
#[test]
fn test_sigmoid_large_negative() {
    assert_abs_diff_eq!(sigmoid(-1000.0), 0.0, epsilon = 1e-12);
}

/// Sigmoid is monotonically increasing
#[test]
fn test_sigmoid_monotone() {
    assert!(sigmoid(-2.0) < sigmoid(-1.0));
    assert!(sigmoid(-1.0) < sigmoid(0.0));
    assert!(sigmoid(0.0) < sigmoid(1.0));
    assert!(sigmoid(1.0) < sigmoid(2.0));
}

// logistic_loss

/// Single sample with logit=0, label=0 has loss ln(2)
#[test]
fn test_logistic_loss_logit_zero_label_zero() {
    let logits = array![0.0_f64];
    let labels = array![0.0_f64];
    let expected = 2.0_f64.ln();
    assert_abs_diff_eq!(logistic_loss(&logits, &labels), expected, epsilon = 1e-9);
}

/// Confident correct predictions at large logits give loss near 0
#[test]
fn test_logistic_loss_near_zero_on_confident_correct() {
    let logits = array![20.0_f64, -20.0];
    let labels = array![1.0_f64, 0.0];
    assert_abs_diff_eq!(logistic_loss(&logits, &labels), 0.0, epsilon = 1e-6);
}

/// Logistic loss is non-negative
#[test]
fn test_logistic_loss_non_negative() {
    let logits = array![0.5_f64, -1.0, 2.0, -0.3];
    let labels = array![1.0_f64, 0.0, 1.0, 0.0];
    assert!(logistic_loss(&logits, &labels) >= 0.0);
}

/// 3 samples average to the mean of two ln(1+e^-1) terms and one ln(2)
#[test]
fn test_logistic_loss_three_samples() {
    let logits = array![1.0_f64, -1.0, 0.0];
    let labels = array![1.0_f64, 0.0, 1.0];
    let e = std::f64::consts::E;
    let loss_a = (1.0 + 1.0 / e).ln();
    let loss_b = (1.0 + 1.0 / e).ln();
    let loss_c = 2.0_f64.ln();
    let expected = (loss_a + loss_b + loss_c) / 3.0;
    assert_abs_diff_eq!(logistic_loss(&logits, &labels), expected, epsilon = 1e-9);
}

// hinge_loss

/// Confident correct classifications (y*m >= 1) give loss 0
#[test]
fn test_hinge_loss_large_margins_zero() {
    let margins = array![2.0_f64, -3.0];
    let labels = array![1.0_f64, -1.0];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-10);
}

/// All samples misclassified by margin 1 (y*m = -1) give loss 2.0
#[test]
fn test_hinge_loss_all_wrong() {
    let margins = array![-1.0_f64, 1.0];
    let labels = array![1.0_f64, -1.0];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 2.0, epsilon = 1e-10);
}

/// Exactly on the decision boundary (margin*label = 1) gives loss 0
#[test]
fn test_hinge_loss_on_boundary() {
    let margins = array![1.0_f64];
    let labels = array![1.0_f64];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-12);
}

/// Mixed margins average to loss 1.0
#[test]
fn test_hinge_loss_mixed() {
    let margins = array![1.5_f64, 0.5, -0.5];
    let labels = array![1.0_f64, -1.0, 1.0];
    assert_abs_diff_eq!(hinge_loss(&margins, &labels), 1.0, epsilon = 1e-10);
}

/// Hinge loss is non-negative by construction
#[test]
fn test_hinge_loss_non_negative() {
    let margins = array![0.3_f64, -0.7, 1.2, 2.5];
    let labels = array![1.0_f64, -1.0, 1.0, -1.0];
    assert!(hinge_loss(&margins, &labels) >= 0.0);
}

// average_path_length_factor

/// n=4 (exact-harmonic branch) gives c(4) = 13/6
#[test]
fn test_average_path_length_n4() {
    let expected = 13.0_f64 / 6.0;
    assert_abs_diff_eq!(average_path_length_factor(4), expected, epsilon = 1e-9);
}

/// n=5 (exact-harmonic branch) gives c(5) = 77/30
#[test]
fn test_average_path_length_n5() {
    let expected = 77.0_f64 / 30.0;
    assert_abs_diff_eq!(average_path_length_factor(5), expected, epsilon = 1e-9);
}

/// Base cases: 0.0 for n=0 and n=1, 1.0 for n=2
#[test]
fn test_average_path_length_base_cases() {
    assert_abs_diff_eq!(average_path_length_factor(0), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(average_path_length_factor(1), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(average_path_length_factor(2), 1.0, epsilon = 1e-12);
}

/// Factor is positive for n >= 3
#[test]
fn test_average_path_length_positive_for_n_ge_3() {
    for n in 3..=200 {
        let f = average_path_length_factor(n);
        assert!(f > 0.0, "expected positive factor for n={n}, got {f}");
    }
}

/// Output is continuous across the exact/approximate branch boundary at n=50/51
#[test]
fn test_average_path_length_continuous_at_branch_boundary() {
    let f50 = average_path_length_factor(50);
    let f51 = average_path_length_factor(51);
    let delta = (f51 - f50).abs();
    assert!(
        delta < 0.1,
        "discontinuity at branch boundary: c(50)={f50}, c(51)={f51}, |diff|={delta}"
    );
    assert!(f51 > f50, "expected c(51)={f51} > c(50)={f50}");
}

// variance / standard_deviation - non-finite (NaN) handling

/// variance skips non-finite entries and computes over the finite subset; [1, NaN, 3] matches
/// [1, 3] (variance 1.0), and an all-non-finite input returns 0.0
#[test]
fn test_variance_skips_non_finite_and_uses_finite_subset() {
    let finite_only = array![1.0_f64, 3.0];
    assert_abs_diff_eq!(variance(&finite_only), 1.0, epsilon = 1e-12);

    // A NaN element is skipped, giving the same finite-subset variance as {1, 3}
    let with_nan = array![1.0_f64, f64::NAN, 3.0];
    assert_abs_diff_eq!(variance(&with_nan), 1.0, epsilon = 1e-12);

    // An infinite element is skipped the same way
    let with_inf = array![1.0_f64, f64::INFINITY, 3.0];
    assert_abs_diff_eq!(variance(&with_inf), 1.0, epsilon = 1e-12);

    // No finite values at all gives 0.0
    let all_nan = array![f64::NAN, f64::NAN];
    assert_abs_diff_eq!(variance(&all_nan), 0.0, epsilon = 1e-12);
}

/// standard_deviation inherits variance's finite-subset contract via sqrt; [1, NaN, 3] gives 1.0
#[test]
fn test_standard_deviation_skips_non_finite() {
    let with_nan = array![1.0_f64, f64::NAN, 3.0];
    assert_abs_diff_eq!(standard_deviation(&with_nan), 1.0, epsilon = 1e-12);

    // standard_deviation equals sqrt(variance) over the same finite subset
    assert_abs_diff_eq!(
        standard_deviation(&with_nan),
        variance(&with_nan).sqrt(),
        epsilon = 1e-12
    );
}
