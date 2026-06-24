//! Integration tests for the public `math` surface: the pairwise distance metrics
//!
//! Everything else that used to live in `math` (variance/std-dev, impurity, losses, sigmoid,
//! SST/SSE, the isolation-forest path-length factor) has been inlined or down-sunk into its
//! consumer and is tested there or via that consumer's public API
#![cfg(feature = "math")]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::array;
use rustyml::math::*;

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
