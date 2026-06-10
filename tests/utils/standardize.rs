// Integration tests for `rustyml::utils::standardize`.
//
// All expected values are derived from the mathematical definitions (population
// z-score) or closed-form arithmetic. No value was produced by running the
// implementation first.

use crate::common::assert_allclose;
use ndarray::{Array1, Array2, Array3, array};
use rustyml::error::Error;
use rustyml::utils::standardize::{StandardizationAxis, standardize};

// ─── Global axis ─────────────────────────────────────────────────────────────

/// CLOSED-FORM: data = [1.0, 3.0].
/// mean = (1+3)/2 = 2.0
/// population variance = ((1-2)^2 + (3-2)^2) / 2 = 1.0
/// std = sqrt(1.0 + (1e-8)^2) ≈ 1.0
/// z-scores = [(1-2)/1, (3-2)/1] = [-1.0, 1.0]
#[test]
fn test_global_closed_form_1d() {
    let data: Array1<f64> = array![1.0, 3.0];
    let result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    let expected: Array1<f64> = array![-1.0, 1.0];
    assert_allclose(&result, &expected, 1e-7);
}

/// Same values as a 1×2 matrix — shape is preserved.
#[test]
fn test_global_closed_form_2d() {
    let data: Array2<f64> = array![[1.0, 3.0]];
    let result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    let expected: Array2<f64> = array![[-1.0, 1.0]];
    assert_allclose(&result, &expected, 1e-7);
}

/// Verify that the epsilon flooring uses `sqrt(variance + eps^2)` and NOT
/// `sqrt(variance) + eps`.
///
/// data = [1.0, 3.0], eps = 0.1
/// variance = 1.0
/// CORRECT:  sqrt(1.0 + 0.01)     = sqrt(1.01)   ≈ 1.00498756…
///           z = ±1/1.00498756    ≈ ±0.99503719…
/// WRONG:    sqrt(1.0) + 0.1      = 1.1
///           z = ±1/1.1           ≈ ±0.90909090…
/// The two approaches differ by ~0.09, well outside the 1e-4 tolerance used here.
#[test]
fn test_epsilon_formula_sqrt_var_plus_eps_squared() {
    let data: Array1<f64> = array![1.0, 3.0];
    // hand-calc: sqrt(1.0 + 0.1^2) = sqrt(1.01) = 1.004987562112...
    // z = (1.0 - 2.0) / 1.004987562112 = -0.995037190209...
    let result = standardize(&data, StandardizationAxis::Global, 0.1).unwrap();
    let expected: Array1<f64> = array![-0.995037190209, 0.995037190209];
    // Verify it matches the eps^2 formula...
    assert_allclose(&result, &expected, 1e-8);
    // ...and does NOT match the sqrt(var)+eps formula (±0.909090...) by a large margin.
    assert!(
        (result[0] - (-0.909090909)).abs() > 0.08,
        "result must not match the incorrect sqrt(var)+eps formula"
    );
}

/// All-identical values: variance = 0 → std = sqrt(0 + eps^2) = eps.
/// z = (x - mean) / eps = 0 for every element. No NaN.
#[test]
fn test_global_zero_variance_all_zeros() {
    let data: Array2<f64> = array![[3.0, 3.0], [3.0, 3.0]];
    let result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    let expected: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
    assert_allclose(&result, &expected, 1e-12);
    // Confirm no NaN leaked through.
    assert!(result.iter().all(|v| v.is_finite()));
}

/// Single-element array: mean = value, variance = 0, std = eps,
/// z = (value - value) / eps = 0.
#[test]
fn test_global_single_element() {
    let data: Array2<f64> = array![[5.0]];
    let result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    let expected: Array2<f64> = array![[0.0]];
    assert_allclose(&result, &expected, 1e-12);
}

/// 1-D array of [1, 2, 3, 4, 5] with Global axis.
/// mean = 3.0
/// population variance = (4+1+0+1+4)/5 = 10/5 = 2.0
/// std = sqrt(2.0 + (1e-8)^2) ≈ 1.41421356…
/// z = [-sqrt(2), -1/sqrt(2), 0, 1/sqrt(2), sqrt(2)]
///   = [-1.41421356…, -0.70710678…, 0, 0.70710678…, 1.41421356…]
#[test]
fn test_global_1d_five_elements() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    // z = [-sqrt(2), -1/sqrt(2), 0, 1/sqrt(2), sqrt(2)] — the exact mathematical constants.
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};
    let expected: Array1<f64> = array![-SQRT_2, -FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2, SQRT_2];
    assert_allclose(&result, &expected, 1e-8);
}

/// 3-D array shaped (1, 3, 2) with Global axis.
/// Flatten: [1,2,3,4,5,6], n=6, mean=3.5
/// population variance = ((1-3.5)^2+(2-3.5)^2+…+(6-3.5)^2)/6
///   = (6.25+2.25+0.25+0.25+2.25+6.25)/6 = 17.5/6 ≈ 2.91666…
/// std = sqrt(2.91666… + (1e-8)^2) ≈ 1.70782512766…
/// z = [-1.46385…, -0.87831…, -0.29277…, 0.29277…, 0.87831…, 1.46385…]
#[test]
fn test_global_3d_array() {
    let data: Array3<f64> = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]];
    let result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    // hand-calc (see block comment above)
    let expected: Array3<f64> = array![[
        [-1.4638501094, -0.8783100656],
        [-0.2927700219, 0.2927700219],
        [0.8783100656, 1.4638501094]
    ]];
    assert_allclose(&result, &expected, 1e-7);
    // shape preserved
    assert_eq!(result.shape(), data.shape());
}

// ─── Column axis ─────────────────────────────────────────────────────────────

/// CLOSED-FORM: data = [[1,4],[2,5],[3,6]] (3 rows, 2 columns).
/// Column 0: values [1,2,3], mean=2, population variance=2/3
///   std = sqrt(2/3 + (1e-8)^2) = sqrt(0.666…) ≈ 0.816496580928
///   z = [(1-2)/0.8165, (2-2)/0.8165, (3-2)/0.8165]
///     = [-1.224744871…, 0, 1.224744871…]
/// Column 1: values [4,5,6], mean=5, same variance/std as col 0
///   z = [-1.224744871…, 0, 1.224744871…]
#[test]
fn test_column_axis_closed_form() {
    let data: Array2<f64> = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap();
    // sqrt(2/3) = 0.8164965809... so z-scores are ±1/sqrt(2/3) = ±sqrt(3/2) = ±1.2247448714
    let expected: Array2<f64> = array![
        [-1.2247448714, -1.2247448714],
        [0.0, 0.0],
        [1.2247448714, 1.2247448714]
    ];
    assert_allclose(&result, &expected, 1e-7);
}

/// Each standardized column must have mean ≈ 0 and population std ≈ 1.
#[test]
fn test_column_axis_invariants() {
    let data: Array2<f64> = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap();
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

/// Zero-variance column: all identical values in a column → all zeros for that column.
/// data = [[3,1],[3,3],[3,5]] — column 0 is all 3.0 (zero variance).
/// Col 0: variance=0, std=eps → z = (3-3)/eps = 0 for all.
/// Col 1: [1,3,5], mean=3, var=8/3, std=sqrt(8/3+eps^2) ≈ 1.63299…
///   z = [(1-3)/1.63299, 0, (5-3)/1.63299] = [-1.2247…, 0, 1.2247…]
#[test]
fn test_column_axis_zero_variance_column() {
    let data: Array2<f64> = array![[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]];
    let result = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap();
    assert!(
        result.iter().all(|v| v.is_finite()),
        "no NaN or Inf expected"
    );
    // Column 0: all zeros
    let col0: Vec<f64> = result.column(0).to_vec();
    for v in &col0 {
        assert!(
            v.abs() < 1e-6,
            "zero-variance column value should be 0, got {v}"
        );
    }
    // Column 1: standard z-score pattern
    let expected_col1: Array1<f64> = array![-1.2247448714, 0.0, 1.2247448714];
    let actual_col1: Array1<f64> = result.column(1).to_owned();
    assert_allclose(&actual_col1, &expected_col1, 1e-7);
}

// ─── Row axis ────────────────────────────────────────────────────────────────

/// CLOSED-FORM: data = [[1,2,3],[4,5,6]] (2 rows, 3 columns).
/// Row 0: values [1,2,3], mean=2, population variance = (1+0+1)/3 = 2/3
///   std = sqrt(2/3 + eps^2) ≈ 0.816496580928
///   z = [(1-2)/0.8165, 0, (3-2)/0.8165] = [-1.2247…, 0, 1.2247…]
/// Row 1: values [4,5,6], mean=5, same variance/std
///   z = [-1.2247…, 0, 1.2247…]
#[test]
fn test_row_axis_closed_form() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Row, 1e-8).unwrap();
    let expected: Array2<f64> = array![
        [-1.2247448714, 0.0, 1.2247448714],
        [-1.2247448714, 0.0, 1.2247448714]
    ];
    assert_allclose(&result, &expected, 1e-7);
}

/// Each standardized row must have mean ≈ 0 and population std ≈ 1.
#[test]
fn test_row_axis_invariants() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Row, 1e-8).unwrap();
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

/// Zero-variance row in a multi-row matrix (Row axis).
/// data = [[3,3,3],[1,3,5]]
/// Row 0: all 3.0 → variance=0, std=eps → z = [0,0,0].
/// Row 1: [1,3,5], mean=3, population variance = (4+0+4)/3 = 8/3
///   std = sqrt(8/3 + eps^2) ≈ 1.632993161855
///   z = [(1-3)/1.6330, 0, (5-3)/1.6330] = [-1.2247…, 0, 1.2247…]
#[test]
fn test_row_axis_zero_variance_row() {
    let data: Array2<f64> = array![[3.0, 3.0, 3.0], [1.0, 3.0, 5.0]];
    let result = standardize(&data, StandardizationAxis::Row, 1e-8).unwrap();
    assert!(
        result.iter().all(|v| v.is_finite()),
        "no NaN or Inf expected"
    );
    // Row 0: all zeros
    let row0: Array1<f64> = result.row(0).to_owned();
    let expected_row0: Array1<f64> = array![0.0, 0.0, 0.0];
    assert_allclose(&row0, &expected_row0, 1e-6);
    // Row 1: standard z-scores
    let row1: Array1<f64> = result.row(1).to_owned();
    let expected_row1: Array1<f64> = array![-1.2247448714, 0.0, 1.2247448714];
    assert_allclose(&row1, &expected_row1, 1e-7);
}

// ─── Immutability ─────────────────────────────────────────────────────────────

/// The original array must NOT be mutated by standardize (it takes &ArrayBase).
#[test]
fn test_original_array_not_mutated() {
    let data: Array2<f64> = array![[1.0, 3.0], [5.0, 7.0]];
    let original = data.clone();
    let _result = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap();
    assert_eq!(
        data, original,
        "standardize must not modify the input array"
    );
}

#[test]
fn test_original_array_not_mutated_column() {
    let data: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let original = data.clone();
    let _result = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap();
    assert_eq!(data, original);
}

// ─── Error paths ──────────────────────────────────────────────────────────────

/// Empty array → EmptyInput.
#[test]
fn test_error_empty_global() {
    let data: Array2<f64> = Array2::zeros((0, 0));
    let err = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

#[test]
fn test_error_empty_column() {
    let data: Array2<f64> = Array2::zeros((0, 3));
    let err = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

/// NaN in input → NonFinite.
#[test]
fn test_error_nan_input() {
    let data: Array2<f64> = array![[1.0, f64::NAN], [3.0, 4.0]];
    let err = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Infinite value in input → NonFinite.
#[test]
fn test_error_inf_input() {
    let data: Array2<f64> = array![[1.0, f64::INFINITY], [3.0, 4.0]];
    let err = standardize(&data, StandardizationAxis::Global, 1e-8).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Negative epsilon → InvalidParameter.
#[test]
fn test_error_negative_epsilon() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let err = standardize(&data, StandardizationAxis::Global, -1e-8).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Zero epsilon → InvalidParameter.
#[test]
fn test_error_zero_epsilon() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let err = standardize(&data, StandardizationAxis::Global, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// NaN epsilon → InvalidParameter (!epsilon.is_finite() branch).
#[test]
fn test_error_nan_epsilon() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let err = standardize(&data, StandardizationAxis::Global, f64::NAN).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Infinite epsilon → InvalidParameter (!epsilon.is_finite() branch).
#[test]
fn test_error_infinite_epsilon() {
    let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let err = standardize(&data, StandardizationAxis::Global, f64::INFINITY).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// Row standardization on a 1-D array → InvalidInput (requires ≥2 dims).
#[test]
fn test_error_row_on_1d_array() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0];
    let err = standardize(&data, StandardizationAxis::Row, 1e-8).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for Row on 1-D array, got {err:?}"
    );
}

/// Column standardization on a 1-D array → InvalidInput (requires ≥2 dims).
#[test]
fn test_error_column_on_1d_array() {
    let data: Array1<f64> = array![1.0, 2.0, 3.0];
    let err = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for Column on 1-D array, got {err:?}"
    );
}
