use super::*;

// Helper function for approximate equality checks
fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

// Helper function for comparing arrays with approximate equality
fn assert_arrays_approx_equal<D>(
    result: &ArrayBase<ndarray::OwnedRepr<f64>, D>,
    expected: &ArrayBase<ndarray::OwnedRepr<f64>, D>,
    epsilon: f64,
) where
    D: Dimension,
{
    assert_eq!(result.shape(), expected.shape());
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!(
            approx_eq(*a, *b, epsilon),
            "Arrays not equal: expected {}, got {}, difference: {}",
            b,
            a,
            (a - b).abs()
        );
    }
}

/// Test global L2 normalization
#[test]
fn test_normalize_global_l2() {
    let data = array![[3.0, 4.0], [1.0, 2.0]];
    let result = normalize(
        data.view(),
        NormalizationAxis::Global,
        NormalizationOrder::L2,
    )
    .unwrap();

    // Calculate global L2 norm: sqrt(3^2 + 4^2 + 1^2 + 2^2) = sqrt(30) ≈ 5.477
    let expected_norm = 30.0f64.sqrt();
    let expected = array![
        [3.0 / expected_norm, 4.0 / expected_norm],
        [1.0 / expected_norm, 2.0 / expected_norm]
    ];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Test row-wise L2 normalization
#[test]
fn test_normalize_row_l2() {
    let data = array![[3.0, 4.0], [5.0, 12.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();

    // First row: norm = sqrt(3^2 + 4^2) = 5
    // Second row: norm = sqrt(5^2 + 12^2) = 13
    let expected = array![[3.0 / 5.0, 4.0 / 5.0], [5.0 / 13.0, 12.0 / 13.0]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);

    // Verify each row has L2 norm of 1
    for row in result.rows() {
        let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }
}

/// Test column-wise L2 normalization
#[test]
fn test_normalize_column_l2() {
    let data = array![[3.0, 4.0], [4.0, 3.0]];
    let result = normalize(
        data.view(),
        NormalizationAxis::Column,
        NormalizationOrder::L2,
    )
    .unwrap();

    // First column: norm = sqrt(3^2 + 4^2) = 5
    // Second column: norm = sqrt(4^2 + 3^2) = 5
    let expected = array![[3.0 / 5.0, 4.0 / 5.0], [4.0 / 5.0, 3.0 / 5.0]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);

    // Verify each column has L2 norm of 1
    for col in result.columns() {
        let norm: f64 = col.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }
}

/// Test L1 normalization
#[test]
fn test_normalize_l1() {
    let data = array![[3.0, 4.0], [1.0, 2.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L1).unwrap();

    // First row: L1 norm = |3| + |4| = 7
    // Second row: L1 norm = |1| + |2| = 3
    let expected = array![[3.0 / 7.0, 4.0 / 7.0], [1.0 / 3.0, 2.0 / 3.0]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);

    // Verify each row has L1 norm of 1
    for row in result.rows() {
        let norm: f64 = row.iter().map(|&x| x.abs()).sum();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }
}

/// Test Max normalization
#[test]
fn test_normalize_max() {
    let data = array![[3.0, -8.0], [6.0, 2.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::Max).unwrap();

    // First row: Max norm = max(|3|, |-8|) = 8
    // Second row: Max norm = max(|6|, |2|) = 6
    let expected = array![[3.0 / 8.0, -8.0 / 8.0], [6.0 / 6.0, 2.0 / 6.0]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);

    // Verify each row has Max norm of 1
    for row in result.rows() {
        let norm: f64 = row
            .iter()
            .map(|&x| x.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(approx_eq(norm, 1.0, 1e-10));
    }
}

/// Test Lp normalization (p=3)
#[test]
fn test_normalize_lp() {
    let data = array![[2.0, 4.0], [1.0, 3.0]];
    let p = 3.0;
    let result = normalize(
        data.view(),
        NormalizationAxis::Row,
        NormalizationOrder::Lp(p),
    )
    .unwrap();

    // First row: L3 norm = (2^3 + 4^3)^(1/3) = (8 + 64)^(1/3) = 72^(1/3)
    // Second row: L3 norm = (1^3 + 3^3)^(1/3) = (1 + 27)^(1/3) = 28^(1/3)
    let norm1 = 72.0_f64.powf(1.0 / 3.0);
    let norm2 = 28.0_f64.powf(1.0 / 3.0);
    let expected = array![[2.0 / norm1, 4.0 / norm1], [1.0 / norm2, 3.0 / norm2]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Test zero vector handling
#[test]
fn test_normalize_zero_vector() {
    let data = array![[0.0, 0.0], [1.0, 2.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();

    // First row is all zeros, should remain zero
    // Second row: norm = sqrt(1^2 + 2^2) = sqrt(5)
    let norm2 = 5.0_f64.sqrt();
    let expected = array![[0.0, 0.0], [1.0 / norm2, 2.0 / norm2]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Test 3D array row normalization
#[test]
fn test_normalize_3d_row() {
    let data = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();

    // Verify result shape
    assert_eq!(result.shape(), data.shape());

    // Verify last axis (rows) are normalized
    for matrix in result.axis_iter(Axis(0)) {
        for row in matrix.rows() {
            let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!(approx_eq(norm, 1.0, 1e-10));
        }
    }
}

/// Test 1D array global normalization
#[test]
fn test_normalize_1d_global() {
    let data = array![3.0, 4.0, 5.0];
    let result = normalize(
        data.view(),
        NormalizationAxis::Global,
        NormalizationOrder::L2,
    )
    .unwrap();

    let norm = (3.0f64 * 3.0 + 4.0f64 * 4.0 + 5.0f64 * 5.0).sqrt(); // sqrt(50)
    let expected = array![3.0 / norm, 4.0 / norm, 5.0 / norm];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Test empty array error
#[test]
fn test_normalize_empty_array() {
    let data: Array2<f64> = array![[]];
    let result = normalize(
        data.view(),
        NormalizationAxis::Global,
        NormalizationOrder::L2,
    );

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

/// Test NaN input
#[test]
fn test_normalize_nan_input() {
    let data = array![[1.0, f64::NAN], [2.0, 3.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

/// Test infinite input
#[test]
fn test_normalize_infinite_input() {
    let data = array![[1.0, f64::INFINITY], [2.0, 3.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

/// Test invalid Lp parameters
#[test]
fn test_normalize_invalid_lp_parameter() {
    let data = array![[1.0, 2.0], [3.0, 4.0]];

    // Test p <= 0
    let result = normalize(
        data.view(),
        NormalizationAxis::Row,
        NormalizationOrder::Lp(0.0),
    );
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));

    let result = normalize(
        data.view(),
        NormalizationAxis::Row,
        NormalizationOrder::Lp(-1.0),
    );
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));

    // Test p as infinity
    let result = normalize(
        data.view(),
        NormalizationAxis::Row,
        NormalizationOrder::Lp(f64::INFINITY),
    );
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));

    // Test p as NaN
    let result = normalize(
        data.view(),
        NormalizationAxis::Row,
        NormalizationOrder::Lp(f64::NAN),
    );
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

/// Test 1D array row/column normalization errors
#[test]
fn test_normalize_1d_row_column_error() {
    let data = array![1.0, 2.0, 3.0];

    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2);
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));

    let result = normalize(
        data.view(),
        NormalizationAxis::Column,
        NormalizationOrder::L2,
    );
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

/// Test negative values handling
#[test]
fn test_normalize_negative_values() {
    let data = array![[-3.0, 4.0], [-1.0, -2.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();

    // First row: norm = sqrt((-3)^2 + 4^2) = 5
    // Second row: norm = sqrt((-1)^2 + (-2)^2) = sqrt(5)
    let norm2 = 5.0_f64.sqrt();
    let expected = array![[-3.0 / 5.0, 4.0 / 5.0], [-1.0 / norm2, -2.0 / norm2]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Test very small values handling
#[test]
fn test_normalize_very_small_values() {
    let data = array![[1e-16, 2e-16], [1.0, 2.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();

    // First row norm is very small, should be set to zero
    // Second row: norm = sqrt(5)
    let norm2 = 5.0_f64.sqrt();
    let expected = array![[0.0, 0.0], [1.0 / norm2, 2.0 / norm2]];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Performance test: large array
#[test]
fn test_normalize_large_array_performance() {
    let size = 1000;
    let data = Array2::from_shape_fn((size, size), |(i, j)| (i + j) as f64);

    let start = std::time::Instant::now();
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2);
    let duration = start.elapsed();

    assert!(result.is_ok());
    println!("Large array normalization took: {:?}", duration);
}

/// Edge case: single element array
#[test]
fn test_normalize_single_element() {
    let data = array![[5.0]];
    let result = normalize(
        data.view(),
        NormalizationAxis::Global,
        NormalizationOrder::L2,
    )
    .unwrap();

    let expected = array![[1.0]];
    assert_arrays_approx_equal(&result, &expected, 1e-10);
}

/// Test consistency across all normalization axes
#[test]
fn test_normalize_axis_consistency() {
    let data = array![[1.0, 2.0], [3.0, 4.0]];

    // For a 2x2 matrix, row and column normalization should produce different results
    let row_result =
        normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
    let col_result = normalize(
        data.view(),
        NormalizationAxis::Column,
        NormalizationOrder::L2,
    )
    .unwrap();
    let global_result = normalize(
        data.view(),
        NormalizationAxis::Global,
        NormalizationOrder::L2,
    )
    .unwrap();

    // Verify they are indeed different
    assert_ne!(row_result, col_result);
    assert_ne!(row_result, global_result);
    assert_ne!(col_result, global_result);

    // Verify shapes remain unchanged
    assert_eq!(row_result.shape(), data.shape());
    assert_eq!(col_result.shape(), data.shape());
    assert_eq!(global_result.shape(), data.shape());
}

/// Test Max norm with all negative values
#[test]
fn test_normalize_max_all_negative() {
    let data = array![[-3.0, -8.0, -2.0], [-6.0, -4.0, -1.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::Max).unwrap();

    // First row: Max norm = max(|-3|, |-8|, |-2|) = 8
    // Second row: Max norm = max(|-6|, |-4|, |-1|) = 6
    let expected = array![
        [-3.0 / 8.0, -8.0 / 8.0, -2.0 / 8.0],
        [-6.0 / 6.0, -4.0 / 6.0, -1.0 / 6.0]
    ];

    assert_arrays_approx_equal(&result, &expected, 1e-10);

    // Verify each row has Max norm of 1
    for row in result.rows() {
        let norm: f64 = row
            .iter()
            .map(|&x| x.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(approx_eq(norm, 1.0, 1e-10));
    }
}

/// Test Max norm with mixed positive and negative values
#[test]
fn test_normalize_max_mixed_signs() {
    let data = array![[-5.0, 3.0, -2.0], [4.0, -7.0, 1.0]];
    let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::Max).unwrap();

    // First row: Max norm = max(|-5|, |3|, |-2|) = 5
    // Second row: Max norm = max(|4|, |-7|, |1|) = 7
    let expected = array![
        [-5.0 / 5.0, 3.0 / 5.0, -2.0 / 5.0],
        [4.0 / 7.0, -7.0 / 7.0, 1.0 / 7.0]
    ];

    assert_arrays_approx_equal(&result, &expected, 1e-10);
}
