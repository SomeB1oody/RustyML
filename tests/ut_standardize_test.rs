#![cfg(feature = "utility")]

use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use rustyml::error::ModelError;
use rustyml::utility::standardize::*;

const EPSILON: f64 = 1e-8;

/// Helper function to calculate mean and standard deviation for validation
fn calculate_mean_and_std(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let values_vec: Vec<f64> = values.collect();
    let mean = values_vec.iter().sum::<f64>() / values_vec.len() as f64;
    let variance =
        values_vec.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values_vec.len() as f64;
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

#[test]
fn test_standardize_global() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    // After global standardization, the entire array should have mean ≈ 0 and std ≈ 1
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    let variance: f64 =
        result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
    let std_dev = variance.sqrt();

    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
}

#[test]
fn test_standardize_by_columns() {
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let data2 = data.clone();
    let result = standardize(&data, StandardizationAxis::Column, EPSILON).unwrap();

    // Check mean and standard deviation for each column
    for col_idx in 0..data2.ncols() {
        let col = result.column(col_idx);
        let (mean, std_dev) = calculate_mean_and_std(col.iter().cloned());

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_standardize_by_rows() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let data2 = data.clone();
    let result = standardize(&data, StandardizationAxis::Row, EPSILON).unwrap();

    // Check mean and standard deviation for each row
    for row_idx in 0..data2.nrows() {
        let row = result.row(row_idx);
        let (mean, std_dev) = calculate_mean_and_std(row.iter().cloned());

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_standardize_single_value() {
    let data = array![[5.0]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    // Single value should be standardized to 0 (since variance is 0, so (x-mean)/epsilon = 0)
    assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);
}

#[test]
fn test_standardize_identical_values() {
    let data = array![[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    // When all values are identical, standardization should result in all zeros
    for &val in result.iter() {
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_standardize_1d_array() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    let variance: f64 =
        result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
    let std_dev = variance.sqrt();

    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
}

#[test]
fn test_standardize_3d_array() {
    let data =
        Array::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    let variance: f64 =
        result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
    let std_dev = variance.sqrt();

    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
}

#[test]
fn test_standardize_preserve_shape() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let original_shape = data.shape().to_vec();
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    assert_eq!(result.shape(), &original_shape[..]);
}

#[test]
fn test_standardize_with_negative_values() {
    let data = array![[-1.0, -2.0], [3.0, 4.0]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    let variance: f64 =
        result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
    let std_dev = variance.sqrt();

    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
}

// Input validation tests
#[test]
fn test_standardize_empty_array() {
    let data: Array2<f64> = Array2::zeros((0, 0));
    let result = standardize(&data, StandardizationAxis::Global, EPSILON);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Cannot standardize empty array"));
    }
}

#[test]
fn test_standardize_nan_values() {
    let data = array![[1.0, f64::NAN], [3.0, 4.0]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Input contains NaN or infinite values"));
    }
}

#[test]
fn test_standardize_infinite_values() {
    let data = array![[1.0, 2.0], [f64::INFINITY, 4.0]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Input contains NaN or infinite values"));
    }
}

#[test]
fn test_standardize_negative_epsilon() {
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let result = standardize(&data, StandardizationAxis::Global, -1e-8);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Epsilon must be positive and finite"));
    }
}

#[test]
fn test_standardize_zero_epsilon() {
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let result = standardize(&data, StandardizationAxis::Global, 0.0);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Epsilon must be positive and finite"));
    }
}

#[test]
fn test_standardize_infinite_epsilon() {
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let result = standardize(&data, StandardizationAxis::Global, f64::INFINITY);

    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Epsilon must be positive and finite"));
    }
}

#[test]
fn test_standardize_row_1d_array() {
    let data = Array1::from(vec![1.0, 2.0, 3.0]);
    let result = standardize(&data, StandardizationAxis::Row, EPSILON);

    // 1D array cannot be standardized by rows
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

#[test]
fn test_standardize_column_1d_array() {
    let data = Array1::from(vec![1.0, 2.0, 3.0]);
    let result = standardize(&data, StandardizationAxis::Column, EPSILON);

    // 1D array cannot be standardized by columns
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

#[test]
fn test_standardize_large_values() {
    let data = array![[1e6, 2e6], [3e6, 4e6]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    let variance: f64 =
        result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
    let std_dev = variance.sqrt();

    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-10);
}

#[test]
fn test_standardize_small_values() {
    let data = array![[1e-6, 2e-6], [3e-6, 4e-6]];
    let result = standardize(&data, StandardizationAxis::Global, EPSILON).unwrap();

    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    let variance: f64 =
        result.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64;
    let std_dev = variance.sqrt();

    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-2);
    assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-2);
}

#[test]
fn test_standardization_axis_equality() {
    // Test StandardizationAxis enum equality
    assert_eq!(StandardizationAxis::Row, StandardizationAxis::Row);
    assert_eq!(StandardizationAxis::Column, StandardizationAxis::Column);
    assert_eq!(StandardizationAxis::Global, StandardizationAxis::Global);

    assert_ne!(StandardizationAxis::Row, StandardizationAxis::Column);
    assert_ne!(StandardizationAxis::Row, StandardizationAxis::Global);
    assert_ne!(StandardizationAxis::Column, StandardizationAxis::Global);
}

#[test]
fn test_standardization_axis_debug() {
    // Test StandardizationAxis Debug implementation
    let row_axis = StandardizationAxis::Row;
    let debug_str = format!("{:?}", row_axis);
    assert_eq!(debug_str, "Row");
}

#[test]
fn test_standardization_axis_clone() {
    // Test StandardizationAxis Clone implementation
    let original = StandardizationAxis::Column;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_standardization_axis_copy() {
    // Test StandardizationAxis Copy implementation
    let original = StandardizationAxis::Global;
    let copied = original;
    assert_eq!(original, copied);
}

#[test]
fn test_standardize_column_axis_large_array() {
    // Test column standardization on large array
    let data = Array2::from_shape_fn((100, 50), |(i, j)| (i as f64 + j as f64) * 0.1);
    let data2 = data.clone();
    let result = standardize(&data, StandardizationAxis::Column, EPSILON).unwrap();

    // Check mean and standard deviation for each column
    for col_idx in 0..data2.ncols() {
        let col = result.column(col_idx);
        let (mean, std_dev) = calculate_mean_and_std(col.iter().cloned());

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_standardize_row_axis_large_array() {
    // Test row standardization on large array
    let data = Array2::from_shape_fn((50, 100), |(i, j)| (i as f64 + j as f64) * 0.1);
    let data2 = data.clone();
    let result = standardize(&data, StandardizationAxis::Row, EPSILON).unwrap();

    // Check mean and standard deviation for each row
    for row_idx in 0..data2.nrows() {
        let row = result.row(row_idx);
        let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
        let variance: f64 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / row.len() as f64;
        let std_dev = variance.sqrt();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(std_dev, 1.0, epsilon = 1e-4);
    }
}
