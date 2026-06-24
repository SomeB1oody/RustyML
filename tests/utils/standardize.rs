//! Integration tests for `rustyml::utils::standardize`, covering Global, Column,
//! and Row z-score standardization plus the error paths

use crate::common::assert_allclose;
use ndarray::{Array1, Array2, Array3, array};
use rustyml::error::Error;
use rustyml::utils::standardize::{StandardizationAxis, standardize};

// Global axis

/// Global standardization of a 1-D array yields population z-scores
#[test]
fn test_global_closed_form_1d() {
    let data: Array1<f64> = array![1.0, 3.0];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array1<f64> = array![-1.0, 1.0];
    assert_allclose(&result, &expected, 1e-7);
}

/// Global standardization of a 1x2 matrix preserves shape
#[test]
fn test_global_closed_form_2d() {
    let data: Array2<f64> = array![[1.0, 3.0]];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array2<f64> = array![[-1.0, 1.0]];
    assert_allclose(&result, &expected, 1e-7);
}

/// A healthy feature is divided by the raw sqrt(variance), giving exact z-scores with no
/// stability bias, since a healthy feature is never perturbed
#[test]
fn test_healthy_feature_exact_zscores() {
    let data: Array1<f64> = array![1.0, 3.0];
    // variance = 1.0, so the divisor is exactly 1.0 and the z-scores are exactly +/-1.0
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array1<f64> = array![-1.0, 1.0];
    assert_allclose(&result, &expected, 1e-12);
}

/// Constant detection is machine-precision and variance-based, so a tiny but real
/// variance is still normalized
#[test]
fn test_small_but_real_variance_is_normalized_like_sklearn() {
    let data: Array1<f64> = array![1.0, 1.0 + 1e-8];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array1<f64> = array![-1.0, 1.0];
    assert_allclose(&result, &expected, 1e-4);
}

/// All-identical values give zero variance and all-zero output without NaN
#[test]
fn test_global_zero_variance_all_zeros() {
    let data: Array2<f64> = array![[3.0, 3.0], [3.0, 3.0]];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
    assert!(result.iter().all(|v| v.is_finite()));
}

/// Single-element array standardizes to zero (variance = 0)
#[test]
fn test_global_single_element() {
    let data: Array2<f64> = array![[5.0]];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array2<f64> = array![[0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// Global standardization of [1,2,3,4,5] matches exact sqrt(2)-based z-scores
#[test]
fn test_global_1d_five_elements() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};
    let expected: Array1<f64> = array![-SQRT_2, -FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2, SQRT_2];
    assert_allclose(&result, &expected, 1e-8);
}

/// Global standardization flattens a 3-D array and preserves its shape
#[test]
fn test_global_3d_array() {
    let data: Array3<f64> = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]];
    let result = standardize(&data, StandardizationAxis::Global).unwrap();
    let expected: Array3<f64> = array![[
        [-1.4638501094, -0.8783100656],
        [-0.2927700219, 0.2927700219],
        [0.8783100656, 1.4638501094]
    ]];
    assert_allclose(&result, &expected, 1e-7);
    assert_eq!(result.shape(), data.shape());
}

// Column axis

/// Column standardization computes per-column population z-scores
#[test]
fn test_column_axis_closed_form() {
    let data: Array2<f64> = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Column).unwrap();
    let expected: Array2<f64> = array![
        [-1.2247448714, -1.2247448714],
        [0.0, 0.0],
        [1.2247448714, 1.2247448714]
    ];
    assert_allclose(&result, &expected, 1e-7);
}

/// Each standardized column has mean ~= 0 and population std ~= 1
#[test]
fn test_column_axis_invariants() {
    let data: Array2<f64> = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Column).unwrap();
    let n = result.nrows() as f64;
    for col in result.columns() {
        let mean: f64 = col.sum() / n;
        let variance: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 1e-10, "column mean {mean} not near 0");
        assert!(
            (variance - 1.0).abs() < 1e-7,
            "column variance {variance} not near 1"
        );
    }
}

/// A zero-variance column standardizes to all zeros while other columns are unaffected
#[test]
fn test_column_axis_zero_variance_column() {
    let data: Array2<f64> = array![[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]];
    let result = standardize(&data, StandardizationAxis::Column).unwrap();
    assert!(
        result.iter().all(|v| v.is_finite()),
        "no NaN or Inf expected"
    );
    let col0: Vec<f64> = result.column(0).to_vec();
    for v in &col0 {
        assert!(
            v.abs() < 1e-6,
            "zero-variance column value should be 0, got {v}"
        );
    }
    let expected_col1: Array1<f64> = array![-1.2247448714, 0.0, 1.2247448714];
    let actual_col1: Array1<f64> = result.column(1).to_owned();
    assert_allclose(&actual_col1, &expected_col1, 1e-7);
}

// Row axis

/// Row standardization computes per-row population z-scores
#[test]
fn test_row_axis_closed_form() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Row).unwrap();
    let expected: Array2<f64> = array![
        [-1.2247448714, 0.0, 1.2247448714],
        [-1.2247448714, 0.0, 1.2247448714]
    ];
    assert_allclose(&result, &expected, 1e-7);
}

/// Each standardized row has mean ~= 0 and population std ~= 1
#[test]
fn test_row_axis_invariants() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Row).unwrap();
    let n = result.ncols() as f64;
    for row in result.rows() {
        let mean: f64 = row.sum() / n;
        let variance: f64 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 1e-10, "row mean {mean} not near 0");
        assert!(
            (variance - 1.0).abs() < 1e-7,
            "row variance {variance} not near 1"
        );
    }
}

/// A zero-variance row standardizes to all zeros while other rows are unaffected
#[test]
fn test_row_axis_zero_variance_row() {
    let data: Array2<f64> = array![[3.0, 3.0, 3.0], [1.0, 3.0, 5.0]];
    let result = standardize(&data, StandardizationAxis::Row).unwrap();
    assert!(
        result.iter().all(|v| v.is_finite()),
        "no NaN or Inf expected"
    );
    let row0: Array1<f64> = result.row(0).to_owned();
    let expected_row0: Array1<f64> = array![0.0, 0.0, 0.0];
    assert_allclose(&row0, &expected_row0, 1e-6);
    let row1: Array1<f64> = result.row(1).to_owned();
    let expected_row1: Array1<f64> = array![-1.2247448714, 0.0, 1.2247448714];
    assert_allclose(&row1, &expected_row1, 1e-7);
}

// Immutability

/// standardize does not mutate its input array
#[test]
fn test_original_array_not_mutated() {
    let data: Array2<f64> = array![[1.0, 3.0], [5.0, 7.0]];
    let original = data.clone();
    let _result = standardize(&data, StandardizationAxis::Global).unwrap();
    assert_eq!(
        data, original,
        "standardize must not modify the input array"
    );
}

#[test]
fn test_original_array_not_mutated_column() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let original = data.clone();
    let _result = standardize(&data, StandardizationAxis::Column).unwrap();
    assert_eq!(data, original);
}

// Error paths

/// Empty array returns EmptyInput
#[test]
fn test_error_empty_global() {
    let data: Array2<f64> = Array2::zeros((0, 0));
    let err = standardize(&data, StandardizationAxis::Global).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

#[test]
fn test_error_empty_column() {
    let data: Array2<f64> = Array2::zeros((0, 3));
    let err = standardize(&data, StandardizationAxis::Column).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// Non-finite values (NaN, +Inf) in input return NonFinite
#[test]
fn test_error_non_finite_input() {
    for sentinel in [f64::NAN, f64::INFINITY] {
        let data: Array2<f64> = array![[1.0, sentinel], [3.0, 4.0]];
        let err = standardize(&data, StandardizationAxis::Global).unwrap_err();
        assert!(
            matches!(err, Error::NonFinite(_)),
            "expected NonFinite for sentinel {sentinel}, got {err:?}"
        );
    }
}

/// Row and Column standardization on a 1-D array return InvalidInput (requires >= 2 dims)
#[test]
fn test_error_axis_on_1d_array() {
    for axis in [StandardizationAxis::Row, StandardizationAxis::Column] {
        let data: Array1<f64> = array![1.0, 2.0, 3.0];
        let err = standardize(&data, axis).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput for {axis:?} on 1-D array, got {err:?}"
        );
    }
}
