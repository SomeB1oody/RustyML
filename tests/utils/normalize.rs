//! Integration tests for `src/utils/normalize.rs`.
//!
//! Every expected value is derived from the mathematical definition of the
//! norm and a closed-form hand calculation.  No value is obtained by running
//! the implementation.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array2, Array3, IxDyn, array};
use rustyml::error::Error;
use rustyml::utils::normalize::{NormalizationAxis, NormalizationOrder, normalize};

use crate::common::assert_allclose;

// ───────────────────────────────────────────────────────────────────────────
// L2 / Row
// ───────────────────────────────────────────────────────────────────────────

/// Row [3, 4] under L2: norm = sqrt(3²+4²) = sqrt(25) = 5.
/// Normalised: [3/5, 4/5] = [0.6, 0.8].
#[test]
fn test_l2_row_single_row_3_4() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[0.6, 0.8]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Two-row matrix: row0 = [3,4], row1 = [0,5].
/// row0 L2 = 5  → [0.6, 0.8].
/// row1 L2 = 5  → [0.0, 1.0].
#[test]
fn test_l2_row_two_rows() {
    let data: Array2<f64> = array![[3.0, 4.0], [0.0, 5.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[0.6, 0.8], [0.0, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Zero row stays all-zeros (no NaN/Inf introduced).
/// Row [0, 0]: norm = 0 → lane kept as [0, 0].
#[test]
fn test_l2_row_zero_row_preserved() {
    let data: Array2<f64> = array![[3.0, 4.0], [0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    // row0: [0.6, 0.8];  row1: zero norm → [0.0, 0.0]
    let expected: Array2<f64> = array![[0.6, 0.8], [0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Negative values keep their sign after L2 normalisation.
/// Row [-3, 4]: norm = sqrt(9+16) = 5  → [-0.6, 0.8].
#[test]
fn test_l2_row_negative_values_keep_sign() {
    let data: Array2<f64> = array![[-3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[-0.6, 0.8]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After normalisation each row has unit L2 norm (verified analytically:
/// ‖[0.6, 0.8]‖₂ = sqrt(0.36+0.64) = 1).
#[test]
fn test_l2_row_each_row_has_unit_norm() {
    let data: Array2<f64> = array![[3.0, 4.0], [5.0, 12.0], [8.0, 6.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    for row in result.rows() {
        let norm_sq: f64 = row.iter().map(|x| x * x).sum();
        // Each non-zero row must have unit norm; all input rows are non-zero here.
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// L1 / Row
// ───────────────────────────────────────────────────────────────────────────

/// Row [3, 4] under L1: |3|+|4| = 7  → [3/7, 4/7].
#[test]
fn test_l1_row_single_row_3_4() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    // 3/7 ≈ 0.428571…  4/7 ≈ 0.571428…
    let expected: Array2<f64> = array![[3.0 / 7.0, 4.0 / 7.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Two-row matrix, L1 row:
///   row0 [1,2,3]: |1|+|2|+|3|=6 → [1/6, 1/3, 1/2].
///   row1 [4,0,0]: |4| = 4 → [1.0, 0.0, 0.0].
#[test]
fn test_l1_row_two_rows_three_cols() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], [1.0, 0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L1 normalisation the sum of absolute values of each row equals 1.
#[test]
fn test_l1_row_each_row_unit_l1_norm() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [5.0, 10.0, 15.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    for row in result.rows() {
        let l1: f64 = row.iter().map(|x| x.abs()).sum();
        assert_abs_diff_eq!(l1, 1.0, epsilon = 1e-12);
    }
}

/// L1 zero lane stays all-zeros.
#[test]
fn test_l1_row_zero_row_preserved() {
    let data: Array2<f64> = array![[1.0, 2.0], [0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    // row0: |1|+|2|=3 → [1/3, 2/3];  row1: norm=0 → [0.0, 0.0]
    assert_abs_diff_eq!(result[[0, 0]], 1.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[[0, 1]], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-15);
}

// ───────────────────────────────────────────────────────────────────────────
// Max / Row
// ───────────────────────────────────────────────────────────────────────────

/// Row [3, 4] under Max: max(|3|,|4|) = 4  → [3/4, 1.0] = [0.75, 1.0].
#[test]
fn test_max_row_single_row_3_4() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[0.75, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Max norm with negatives: row [-6, 2] → max absolute = 6 → [-1.0, 2/6] = [-1.0, 1/3].
#[test]
fn test_max_row_negative_values() {
    let data: Array2<f64> = array![[-6.0, 2.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    // -6/6 = -1.0,  2/6 = 1/3
    let expected: Array2<f64> = array![[-1.0, 1.0 / 3.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After Max normalisation the maximum absolute value of each row equals 1.
#[test]
fn test_max_row_each_row_max_is_one() {
    let data: Array2<f64> = array![[3.0, 4.0, 1.0], [10.0, 2.0, 5.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    for row in result.rows() {
        let max_abs = row
            .iter()
            .map(|x| x.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert_abs_diff_eq!(max_abs, 1.0, epsilon = 1e-12);
    }
}

/// Max zero lane stays all-zeros.
#[test]
fn test_max_row_zero_row_preserved() {
    let data: Array2<f64> = array![[2.0, 4.0], [0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    // row0: max=4 → [0.5, 1.0]; row1: max=0 → [0.0, 0.0]
    let expected: Array2<f64> = array![[0.5, 1.0], [0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

// ───────────────────────────────────────────────────────────────────────────
// Lp / Row
// ───────────────────────────────────────────────────────────────────────────

/// Lp(3) of row [2, 4]:
///   norm = (|2|³ + |4|³)^(1/3) = (8 + 64)^(1/3) = 72^(1/3).
///   Normalised: [2/72^(1/3),  4/72^(1/3)].
#[test]
fn test_lp3_row_2_4() {
    let data: Array2<f64> = array![[2.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(3.0)).unwrap();
    let norm = 72_f64.powf(1.0 / 3.0); // 72^(1/3)
    let expected: Array2<f64> = array![[2.0 / norm, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Lp(p=2) is identical to L2 for [3, 4]:
///   (3² + 4²)^(1/2) = 5 → [0.6, 0.8].
#[test]
fn test_lp2_same_as_l2() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let lp2 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(2.0)).unwrap();
    let l2 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    assert_allclose(&lp2, &l2, 1e-12);
}

/// Lp(p=1) is identical to L1 for [3, 4]:
///   (|3|¹ + |4|¹)^1 = 7 → [3/7, 4/7].
#[test]
fn test_lp1_same_as_l1() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let lp1 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(1.0)).unwrap();
    let l1 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    assert_allclose(&lp1, &l1, 1e-12);
}

/// Lp(p=0.5) normalisation of row [1, 4]:
///   norm = (|1|^0.5 + |4|^0.5)^(1/0.5) = (1 + 2)^2 = 9.
///   Normalised: [1/9, 4/9].
#[test]
fn test_lp_half_row_1_4() {
    let data: Array2<f64> = array![[1.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(0.5)).unwrap();
    // (1^0.5 + 4^0.5)^(1/0.5) = (1 + 2)^2 = 9
    let norm = 9.0_f64;
    let expected: Array2<f64> = array![[1.0 / norm, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

// ───────────────────────────────────────────────────────────────────────────
// Column axis
// ───────────────────────────────────────────────────────────────────────────
//
// Column axis uses axis_from_end=2, so for a 2-D array of shape (m,n) it
// operates along Axis(0).  Each "lane" is a column-vector.

/// L2 Column: matrix [[1,2],[3,4]].
///   col0 [1,3]: norm = sqrt(1+9) = sqrt(10).
///   col1 [2,4]: norm = sqrt(4+16) = sqrt(20) = 2*sqrt(5).
///   result col0: [1/sqrt(10), 3/sqrt(10)];
///   result col1: [2/(2*sqrt(5)), 4/(2*sqrt(5))] = [1/sqrt(5), 2/sqrt(5)].
#[test]
fn test_l2_column_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    let sqrt10 = 10_f64.sqrt();
    let sqrt5 = 5_f64.sqrt();
    let expected: Array2<f64> = array![[1.0 / sqrt10, 1.0 / sqrt5], [3.0 / sqrt10, 2.0 / sqrt5]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L2 column normalisation each column has unit norm.
#[test]
fn test_l2_column_each_col_unit_norm() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    for col in result.columns() {
        let norm_sq: f64 = col.iter().map(|x| x * x).sum();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
    }
}

/// L1 Column: matrix [[1,2],[3,4]].
///   col0 [1,3]: |1|+|3|=4 → [1/4, 3/4] = [0.25, 0.75].
///   col1 [2,4]: |2|+|4|=6 → [2/6, 4/6] = [1/3, 2/3].
#[test]
fn test_l1_column_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[0.25, 1.0 / 3.0], [0.75, 2.0 / 3.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L1 column normalisation the sum of absolute values of each column
/// equals 1.
#[test]
fn test_l1_column_each_col_unit_l1_norm() {
    let data: Array2<f64> = array![[1.0, 5.0], [3.0, 15.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L1).unwrap();
    for col in result.columns() {
        let l1: f64 = col.iter().map(|x| x.abs()).sum();
        assert_abs_diff_eq!(l1, 1.0, epsilon = 1e-12);
    }
}

/// Max Column: matrix [[3,2],[1,8]].
///   col0 [3,1]: max(|3|,|1|)=3 → [3/3, 1/3] = [1.0, 1/3].
///   col1 [2,8]: max(|2|,|8|)=8 → [2/8, 8/8] = [0.25, 1.0].
#[test]
fn test_max_column_2x2() {
    let data: Array2<f64> = array![[3.0, 2.0], [1.0, 8.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[1.0, 0.25], [1.0 / 3.0, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After Max column normalisation the maximum absolute value of each column
/// equals 1.
#[test]
fn test_max_column_each_col_max_is_one() {
    let data: Array2<f64> = array![[1.0, 7.0], [5.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::Max).unwrap();
    for col in result.columns() {
        let max_abs = col
            .iter()
            .map(|x| x.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert_abs_diff_eq!(max_abs, 1.0, epsilon = 1e-12);
    }
}

/// Lp(3) Column: matrix [[2,1],[4,3]].
///   col0 [2,4]: norm3 = (8+64)^(1/3) = 72^(1/3) → [2/72^(1/3), 4/72^(1/3)].
///   col1 [1,3]: norm3 = (1+27)^(1/3) = 28^(1/3) → [1/28^(1/3), 3/28^(1/3)].
#[test]
fn test_lp3_column_2x2() {
    let data: Array2<f64> = array![[2.0, 1.0], [4.0, 3.0]];
    let result = normalize(
        &data,
        NormalizationAxis::Column,
        NormalizationOrder::Lp(3.0),
    )
    .unwrap();
    let n0 = 72_f64.powf(1.0 / 3.0); // col0 norm
    let n1 = 28_f64.powf(1.0 / 3.0); // col1 norm
    let expected: Array2<f64> = array![[2.0 / n0, 1.0 / n1], [4.0 / n0, 3.0 / n1]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Zero column stays all-zeros under every norm.
/// Matrix [[1,0],[2,0]]: col1 is all zero; L2 normalisation must not produce NaN.
#[test]
fn test_column_zero_column_preserved() {
    let data: Array2<f64> = array![[1.0, 0.0], [2.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    // col0 [1,2]: norm = sqrt(5) → [1/sqrt(5), 2/sqrt(5)]
    // col1 [0,0]: norm = 0 → [0.0, 0.0]
    let sqrt5 = 5_f64.sqrt();
    let expected: Array2<f64> = array![[1.0 / sqrt5, 0.0], [2.0 / sqrt5, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

// ───────────────────────────────────────────────────────────────────────────
// Global axis
// ───────────────────────────────────────────────────────────────────────────

/// L2 Global: [[1,2],[3,4]].
///   All values: 1,2,3,4; norm = sqrt(1+4+9+16) = sqrt(30).
///   Each element divided by sqrt(30).
#[test]
fn test_l2_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let norm = 30_f64.sqrt(); // sqrt(1+4+9+16) = sqrt(30)
    let expected: Array2<f64> = array![[1.0 / norm, 2.0 / norm], [3.0 / norm, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

/// L2 Global on a 1-D array: [3, 4].
///   norm = 5  → [0.6, 0.8].
#[test]
fn test_l2_global_1d() {
    let data: Array1<f64> = array![3.0, 4.0];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let expected: Array1<f64> = array![0.6, 0.8];
    assert_allclose(&result, &expected, 1e-12);
}

/// L1 Global: [[1,2],[3,4]].
///   Sum of absolutes = 1+2+3+4 = 10.
///   Result: [[0.1, 0.2], [0.3, 0.4]].
#[test]
fn test_l1_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[0.1, 0.2], [0.3, 0.4]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Max Global: [[1,2],[3,8]].
///   max absolute = 8.
///   Result: [[1/8, 2/8], [3/8, 8/8]] = [[0.125, 0.25], [0.375, 1.0]].
#[test]
fn test_max_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 8.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[0.125, 0.25], [0.375, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Lp(3) Global: [[1,2],[0,4]].
///   norm = (1³+2³+0³+4³)^(1/3) = (1+8+0+64)^(1/3) = 73^(1/3).
///   Each element divided by 73^(1/3).
#[test]
fn test_lp3_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [0.0, 4.0]];
    let result = normalize(
        &data,
        NormalizationAxis::Global,
        NormalizationOrder::Lp(3.0),
    )
    .unwrap();
    let norm = 73_f64.powf(1.0 / 3.0); // (1+8+0+64)^(1/3)
    let expected: Array2<f64> = array![[1.0 / norm, 2.0 / norm], [0.0, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

/// L2 Global on an all-zero array: global norm = 0 → all elements remain 0.
#[test]
fn test_l2_global_all_zeros_preserved() {
    let data: Array2<f64> = Array2::zeros((3, 3));
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = Array2::zeros((3, 3));
    assert_allclose(&result, &expected, 1e-15);
}

/// Global L2 norm of the result equals 1 (when the input is non-zero).
/// Verified: ‖result‖₂² = sum((x/‖x‖₂)²) = 1.
#[test]
fn test_l2_global_result_has_unit_norm() {
    let data: Array2<f64> = array![[3.0, 1.0], [4.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let norm_sq: f64 = result.iter().map(|x| x * x).sum();
    assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
}

// ───────────────────────────────────────────────────────────────────────────
// 3-D array (Row normalises along last axis)
// ───────────────────────────────────────────────────────────────────────────

/// For a 3-D array of shape (2,2,2), Row normalises along Axis(2) (last axis).
///   Array:
///     [[[3,4],[0,0]],
///      [[1,0],[2,2]]]
///   Lanes along Axis(2):
///     [3,4]  → norm=5            → [0.6, 0.8]
///     [0,0]  → norm=0            → [0.0, 0.0]
///     [1,0]  → norm=1            → [1.0, 0.0]
///     [2,2]  → norm=sqrt(4+4)=2√2 → [1/√2, 1/√2]
#[test]
fn test_l2_row_3d_normalizes_last_axis() {
    let data: Array3<f64> = array![[[3.0, 4.0], [0.0, 0.0]], [[1.0, 0.0], [2.0, 2.0]]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let inv_sqrt2 = 1.0_f64 / 2_f64.sqrt();
    let expected: Array3<f64> = array![
        [[0.6, 0.8], [0.0, 0.0]],
        [[1.0, 0.0], [inv_sqrt2, inv_sqrt2]]
    ];
    assert_allclose(&result, &expected, 1e-12);
}

/// For a 3-D array, Column normalises along axis N-2 = Axis(1).
/// Shape (2,2,2):
///   Array:
///     [[[1,2],[3,4]],
///      [[5,6],[7,8]]]
///
///   Lanes along Axis(1) (there are 4 such lanes, one per (batch, feature) pair):
///     batch=0, feature=0: [1, 3] L2 norm=sqrt(10)  → [1/√10, 3/√10]
///     batch=0, feature=1: [2, 4] L2 norm=sqrt(20)  → [1/√5,  2/√5]
///     batch=1, feature=0: [5, 7] L2 norm=sqrt(74)  → [5/√74, 7/√74]
///     batch=1, feature=1: [6, 8] L2 norm=sqrt(100)=10 → [0.6, 0.8]
#[test]
fn test_l2_column_3d_normalizes_second_to_last_axis() {
    let data: Array3<f64> = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    let sqrt10 = 10_f64.sqrt();
    let sqrt20 = 20_f64.sqrt(); // = 2*sqrt(5)
    let sqrt74 = 74_f64.sqrt();
    // batch=1, feature=1: sqrt(36+64)=sqrt(100)=10
    let expected: Array3<f64> = array![
        [[1.0 / sqrt10, 2.0 / sqrt20], [3.0 / sqrt10, 4.0 / sqrt20]],
        [[5.0 / sqrt74, 6.0 / 10.0], [7.0 / sqrt74, 8.0 / 10.0]]
    ];
    assert_allclose(&result, &expected, 1e-12);
}

// ───────────────────────────────────────────────────────────────────────────
// Original array is NOT mutated
// ───────────────────────────────────────────────────────────────────────────

/// normalize returns a new array; the original is unchanged.
#[test]
fn test_normalize_does_not_mutate_original() {
    let original: Array2<f64> = array![[3.0, 4.0], [1.0, 2.0]];
    let snapshot = original.clone();
    let _result = normalize(&original, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    // original must still equal the snapshot taken before the call
    assert_allclose(&original, &snapshot, 1e-15);
}

/// Same non-mutation check for Global axis.
#[test]
fn test_normalize_global_does_not_mutate_original() {
    let original: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let snapshot = original.clone();
    let _result = normalize(&original, NormalizationAxis::Global, NormalizationOrder::L1).unwrap();
    assert_allclose(&original, &snapshot, 1e-15);
}

// ───────────────────────────────────────────────────────────────────────────
// Error paths
// ───────────────────────────────────────────────────────────────────────────

/// Empty array → EmptyInput.
#[test]
fn test_empty_array_returns_empty_input() {
    let data: Array2<f64> = Array2::zeros((0, 3));
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// Array containing NaN → NonFinite.
#[test]
fn test_nan_returns_non_finite() {
    let data: Array2<f64> = array![[1.0, f64::NAN]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Array containing +Inf → NonFinite.
#[test]
fn test_pos_inf_returns_non_finite() {
    let data: Array2<f64> = array![[f64::INFINITY, 1.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Array containing -Inf → NonFinite.
#[test]
fn test_neg_inf_returns_non_finite() {
    let data: Array2<f64> = array![[f64::NEG_INFINITY, 1.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Lp with p = 0 → InvalidParameter (p must be positive).
#[test]
fn test_lp_zero_p_returns_invalid_parameter() {
    let data: Array2<f64> = array![[1.0, 2.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(0.0)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Lp with p = -1 → InvalidParameter (p must be positive).
#[test]
fn test_lp_negative_p_returns_invalid_parameter() {
    let data: Array2<f64> = array![[1.0, 2.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(-1.0)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Lp with p = +Inf → InvalidParameter (p must be finite).
#[test]
fn test_lp_inf_p_returns_invalid_parameter() {
    let data: Array2<f64> = array![[1.0, 2.0]];
    let err = normalize(
        &data,
        NormalizationAxis::Row,
        NormalizationOrder::Lp(f64::INFINITY),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Lp with p = NaN → InvalidParameter (p must be finite).
#[test]
fn test_lp_nan_p_returns_invalid_parameter() {
    let data: Array2<f64> = array![[1.0, 2.0]];
    let err = normalize(
        &data,
        NormalizationAxis::Row,
        NormalizationOrder::Lp(f64::NAN),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// 1-D array with Row axis → InvalidInput (requires at least 2 dimensions).
#[test]
fn test_1d_row_axis_returns_invalid_input() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// 1-D array with Column axis → InvalidInput (requires at least 2 dimensions).
#[test]
fn test_1d_column_axis_returns_invalid_input() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0];
    let err = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Dynamic-dimension array (IxDyn)
// ───────────────────────────────────────────────────────────────────────────

/// normalize accepts dynamic arrays (IxDyn).  L2 global on [3, 4] → [0.6, 0.8].
#[test]
fn test_l2_global_ixdyn_1d() {
    let data: Array<f64, IxDyn> = Array::from_vec(vec![3.0, 4.0]).into_dyn();
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let expected: Array<f64, IxDyn> = Array::from_vec(vec![0.6, 0.8]).into_dyn();
    assert_allclose(&result, &expected, 1e-12);
}
