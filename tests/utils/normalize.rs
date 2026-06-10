//! Integration tests for `normalize`: L1/L2/Max/Lp norms across Row, Column,
//! and Global axes, plus zero-lane preservation and error paths

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array2, Array3, IxDyn, array};
use rustyml::error::Error;
use rustyml::utils::normalize::{NormalizationAxis, NormalizationOrder, normalize};

use crate::common::assert_allclose;

// L2 / Row

/// Row [3, 4] under L2 normalizes to [0.6, 0.8]
#[test]
fn test_l2_row_single_row_3_4() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[0.6, 0.8]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Two-row matrix: each row is L2-normalized independently
#[test]
fn test_l2_row_two_rows() {
    let data: Array2<f64> = array![[3.0, 4.0], [0.0, 5.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[0.6, 0.8], [0.0, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// A zero row stays all-zeros under L2 (no NaN/Inf introduced)
#[test]
fn test_l2_row_zero_row_preserved() {
    let data: Array2<f64> = array![[3.0, 4.0], [0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[0.6, 0.8], [0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Negative values keep their sign after L2 normalization
#[test]
fn test_l2_row_negative_values_keep_sign() {
    let data: Array2<f64> = array![[-3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = array![[-0.6, 0.8]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L2 row normalization each non-zero row has unit norm
#[test]
fn test_l2_row_each_row_has_unit_norm() {
    let data: Array2<f64> = array![[3.0, 4.0], [5.0, 12.0], [8.0, 6.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    for row in result.rows() {
        let norm_sq: f64 = row.iter().map(|x| x * x).sum();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
    }
}

// L1 / Row

/// Row [3, 4] under L1 normalizes to [3/7, 4/7]
#[test]
fn test_l1_row_single_row_3_4() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[3.0 / 7.0, 4.0 / 7.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Two-row, three-column matrix: each row is L1-normalized independently
#[test]
fn test_l1_row_two_rows_three_cols() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], [1.0, 0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L1 row normalization the sum of absolute values of each row equals 1
#[test]
fn test_l1_row_each_row_unit_l1_norm() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [5.0, 10.0, 15.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    for row in result.rows() {
        let l1: f64 = row.iter().map(|x| x.abs()).sum();
        assert_abs_diff_eq!(l1, 1.0, epsilon = 1e-12);
    }
}

/// An L1 zero lane stays all-zeros
#[test]
fn test_l1_row_zero_row_preserved() {
    let data: Array2<f64> = array![[1.0, 2.0], [0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    assert_abs_diff_eq!(result[[0, 0]], 1.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[[0, 1]], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-15);
}

// Max / Row

/// Row [3, 4] under Max normalizes to [0.75, 1.0]
#[test]
fn test_max_row_single_row_3_4() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[0.75, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Max norm with negatives: row [-6, 2] normalizes to [-1.0, 1/3]
#[test]
fn test_max_row_negative_values() {
    let data: Array2<f64> = array![[-6.0, 2.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[-1.0, 1.0 / 3.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After Max row normalization the maximum absolute value of each row equals 1
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

/// A Max zero lane stays all-zeros
#[test]
fn test_max_row_zero_row_preserved() {
    let data: Array2<f64> = array![[2.0, 4.0], [0.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[0.5, 1.0], [0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

// Lp / Row

/// Lp(3) of row [2, 4] normalizes by norm = 72^(1/3)
#[test]
fn test_lp3_row_2_4() {
    let data: Array2<f64> = array![[2.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(3.0)).unwrap();
    let norm = 72_f64.powf(1.0 / 3.0);
    let expected: Array2<f64> = array![[2.0 / norm, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Lp(p=2) is identical to L2
#[test]
fn test_lp2_same_as_l2() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let lp2 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(2.0)).unwrap();
    let l2 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    assert_allclose(&lp2, &l2, 1e-12);
}

/// Lp(p=1) is identical to L1
#[test]
fn test_lp1_same_as_l1() {
    let data: Array2<f64> = array![[3.0, 4.0]];
    let lp1 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(1.0)).unwrap();
    let l1 = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L1).unwrap();
    assert_allclose(&lp1, &l1, 1e-12);
}

/// Lp(p=0.5) of row [1, 4] normalizes by norm = 9
#[test]
fn test_lp_half_row_1_4() {
    let data: Array2<f64> = array![[1.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(0.5)).unwrap();
    let norm = 9.0_f64;
    let expected: Array2<f64> = array![[1.0 / norm, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

// Column axis: for a 2-D array of shape (m,n) it operates along Axis(0), so
// each lane is a column-vector

/// L2 Column-normalizes each column of [[1,2],[3,4]] to unit norm
#[test]
fn test_l2_column_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    let sqrt10 = 10_f64.sqrt();
    let sqrt5 = 5_f64.sqrt();
    let expected: Array2<f64> = array![[1.0 / sqrt10, 1.0 / sqrt5], [3.0 / sqrt10, 2.0 / sqrt5]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L2 column normalization each column has unit norm
#[test]
fn test_l2_column_each_col_unit_norm() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    for col in result.columns() {
        let norm_sq: f64 = col.iter().map(|x| x * x).sum();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
    }
}

/// L1 Column-normalizes each column of [[1,2],[3,4]]
#[test]
fn test_l1_column_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[0.25, 1.0 / 3.0], [0.75, 2.0 / 3.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After L1 column normalization the sum of absolute values of each column equals 1
#[test]
fn test_l1_column_each_col_unit_l1_norm() {
    let data: Array2<f64> = array![[1.0, 5.0], [3.0, 15.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L1).unwrap();
    for col in result.columns() {
        let l1: f64 = col.iter().map(|x| x.abs()).sum();
        assert_abs_diff_eq!(l1, 1.0, epsilon = 1e-12);
    }
}

/// Max Column-normalizes each column of [[3,2],[1,8]]
#[test]
fn test_max_column_2x2() {
    let data: Array2<f64> = array![[3.0, 2.0], [1.0, 8.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[1.0, 0.25], [1.0 / 3.0, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// After Max column normalization the maximum absolute value of each column equals 1
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

/// Lp(3) Column-normalizes each column of [[2,1],[4,3]]
#[test]
fn test_lp3_column_2x2() {
    let data: Array2<f64> = array![[2.0, 1.0], [4.0, 3.0]];
    let result = normalize(
        &data,
        NormalizationAxis::Column,
        NormalizationOrder::Lp(3.0),
    )
    .unwrap();
    let n0 = 72_f64.powf(1.0 / 3.0);
    let n1 = 28_f64.powf(1.0 / 3.0);
    let expected: Array2<f64> = array![[2.0 / n0, 1.0 / n1], [4.0 / n0, 3.0 / n1]];
    assert_allclose(&result, &expected, 1e-12);
}

/// A zero column stays all-zeros under L2 (no NaN produced)
#[test]
fn test_column_zero_column_preserved() {
    let data: Array2<f64> = array![[1.0, 0.0], [2.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    let sqrt5 = 5_f64.sqrt();
    let expected: Array2<f64> = array![[1.0 / sqrt5, 0.0], [2.0 / sqrt5, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

// Global axis

/// L2 Global divides every element of [[1,2],[3,4]] by sqrt(30)
#[test]
fn test_l2_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let norm = 30_f64.sqrt();
    let expected: Array2<f64> = array![[1.0 / norm, 2.0 / norm], [3.0 / norm, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

/// L2 Global on the 1-D array [3, 4] normalizes to [0.6, 0.8]
#[test]
fn test_l2_global_1d() {
    let data: Array1<f64> = array![3.0, 4.0];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let expected: Array1<f64> = array![0.6, 0.8];
    assert_allclose(&result, &expected, 1e-12);
}

/// L1 Global on [[1,2],[3,4]] normalizes to [[0.1, 0.2], [0.3, 0.4]]
#[test]
fn test_l1_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L1).unwrap();
    let expected: Array2<f64> = array![[0.1, 0.2], [0.3, 0.4]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Max Global on [[1,2],[3,8]] normalizes to [[0.125, 0.25], [0.375, 1.0]]
#[test]
fn test_max_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 8.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::Max).unwrap();
    let expected: Array2<f64> = array![[0.125, 0.25], [0.375, 1.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Lp(3) Global divides every element of [[1,2],[0,4]] by 73^(1/3)
#[test]
fn test_lp3_global_2x2() {
    let data: Array2<f64> = array![[1.0, 2.0], [0.0, 4.0]];
    let result = normalize(
        &data,
        NormalizationAxis::Global,
        NormalizationOrder::Lp(3.0),
    )
    .unwrap();
    let norm = 73_f64.powf(1.0 / 3.0);
    let expected: Array2<f64> = array![[1.0 / norm, 2.0 / norm], [0.0, 4.0 / norm]];
    assert_allclose(&result, &expected, 1e-12);
}

/// L2 Global on an all-zero array leaves every element at 0
#[test]
fn test_l2_global_all_zeros_preserved() {
    let data: Array2<f64> = Array2::zeros((3, 3));
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let expected: Array2<f64> = Array2::zeros((3, 3));
    assert_allclose(&result, &expected, 1e-15);
}

/// The global L2 norm of the result equals 1 for non-zero input
#[test]
fn test_l2_global_result_has_unit_norm() {
    let data: Array2<f64> = array![[3.0, 1.0], [4.0, 0.0]];
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let norm_sq: f64 = result.iter().map(|x| x * x).sum();
    assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
}

// 3-D array (Row normalizes along the last axis)

/// For a 3-D array of shape (2,2,2), Row L2-normalizes lanes along Axis(2)
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

/// For a 3-D array of shape (2,2,2), Column L2-normalizes lanes along Axis(1)
#[test]
fn test_l2_column_3d_normalizes_second_to_last_axis() {
    let data: Array3<f64> = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let result = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap();
    let sqrt10 = 10_f64.sqrt();
    let sqrt20 = 20_f64.sqrt();
    let sqrt74 = 74_f64.sqrt();
    let expected: Array3<f64> = array![
        [[1.0 / sqrt10, 2.0 / sqrt20], [3.0 / sqrt10, 4.0 / sqrt20]],
        [[5.0 / sqrt74, 6.0 / 10.0], [7.0 / sqrt74, 8.0 / 10.0]]
    ];
    assert_allclose(&result, &expected, 1e-12);
}

// Original array is not mutated

/// normalize returns a new array and leaves the input unchanged
#[test]
fn test_normalize_does_not_mutate_original() {
    let original: Array2<f64> = array![[3.0, 4.0], [1.0, 2.0]];
    let snapshot = original.clone();
    let _result = normalize(&original, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    assert_allclose(&original, &snapshot, 1e-15);
}

/// Same non-mutation check for the Global axis
#[test]
fn test_normalize_global_does_not_mutate_original() {
    let original: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let snapshot = original.clone();
    let _result = normalize(&original, NormalizationAxis::Global, NormalizationOrder::L1).unwrap();
    assert_allclose(&original, &snapshot, 1e-15);
}

// Error paths

/// An empty array returns EmptyInput
#[test]
fn test_empty_array_returns_empty_input() {
    let data: Array2<f64> = Array2::zeros((0, 3));
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// An array containing NaN returns NonFinite
#[test]
fn test_nan_returns_non_finite() {
    let data: Array2<f64> = array![[1.0, f64::NAN]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// An array containing +Inf returns NonFinite
#[test]
fn test_pos_inf_returns_non_finite() {
    let data: Array2<f64> = array![[f64::INFINITY, 1.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// An array containing -Inf returns NonFinite
#[test]
fn test_neg_inf_returns_non_finite() {
    let data: Array2<f64> = array![[f64::NEG_INFINITY, 1.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Lp with p = 0 returns InvalidParameter (p must be positive)
#[test]
fn test_lp_zero_p_returns_invalid_parameter() {
    let data: Array2<f64> = array![[1.0, 2.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(0.0)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Lp with p = -1 returns InvalidParameter (p must be positive)
#[test]
fn test_lp_negative_p_returns_invalid_parameter() {
    let data: Array2<f64> = array![[1.0, 2.0]];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::Lp(-1.0)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Lp with p = +Inf returns InvalidParameter (p must be finite)
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

/// Lp with p = NaN returns InvalidParameter (p must be finite)
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

/// A 1-D array with the Row axis returns InvalidInput (requires at least 2 dimensions)
#[test]
fn test_1d_row_axis_returns_invalid_input() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0];
    let err = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// A 1-D array with the Column axis returns InvalidInput (requires at least 2 dimensions)
#[test]
fn test_1d_column_axis_returns_invalid_input() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0];
    let err = normalize(&data, NormalizationAxis::Column, NormalizationOrder::L2).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// Dynamic-dimension array (IxDyn)

/// normalize accepts dynamic arrays: L2 global on [3, 4] normalizes to [0.6, 0.8]
#[test]
fn test_l2_global_ixdyn_1d() {
    let data: Array<f64, IxDyn> = Array::from_vec(vec![3.0, 4.0]).into_dyn();
    let result = normalize(&data, NormalizationAxis::Global, NormalizationOrder::L2).unwrap();
    let expected: Array<f64, IxDyn> = Array::from_vec(vec![0.6, 0.8]).into_dyn();
    assert_allclose(&result, &expected, 1e-12);
}
