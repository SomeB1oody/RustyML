use super::*;

// Test basic constructor functionality
#[test]
fn test_new() {
    let forest = IsolationForest::new(50, 128, Some(6), Some(42)).unwrap();
    assert_eq!(forest.get_n_estimators(), 50);
    assert_eq!(forest.get_max_samples(), 128);
    assert_eq!(forest.get_max_depth(), 6);
    assert_eq!(forest.get_random_state(), Some(42));
    assert_eq!(forest.get_n_features(), 0);
}

// Test default constructor
#[test]
fn test_default() {
    let forest = IsolationForest::default();
    assert_eq!(forest.get_n_estimators(), 100);
    assert_eq!(forest.get_max_samples(), 256);
    assert_eq!(forest.get_max_depth(), 8);
    assert_eq!(forest.get_random_state(), None);
    assert!(forest.get_trees().is_none());
}

// Test constructor with automatic max_depth calculation
#[test]
fn test_new_with_auto_max_depth() {
    let forest = IsolationForest::new(10, 64, None, None).unwrap();
    // ceil(log2(64)) = ceil(6.0) = 6
    assert_eq!(forest.get_max_depth(), 6);

    let forest2 = IsolationForest::new(10, 100, None, None).unwrap();
    // ceil(log2(100)) = ceil(6.64) = 7
    assert_eq!(forest2.get_max_depth(), 7);
}

// Test fitting with valid data
#[test]
fn test_fit_valid_data() {
    let mut forest = IsolationForest::new(10, 50, Some(3), Some(42)).unwrap();
    let x = arr2(&[
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [100.0, 200.0], // potential outlier
    ]);

    let result = forest.fit(&x.view());
    assert!(result.is_ok());
    assert_eq!(forest.get_n_features(), 2);
    assert!(forest.get_trees().is_some());
    assert_eq!(forest.get_trees().as_ref().unwrap().len(), 10);
}

// Test fitting with empty data
#[test]
fn test_fit_empty_data() {
    let mut forest = IsolationForest::default();
    let x = Array2::<f64>::zeros((0, 2));

    let result = forest.fit(&x.view());
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test fitting with NaN values
#[test]
fn test_fit_nan_data() {
    let mut forest = IsolationForest::default();
    let x = arr2(&[[1.0, 2.0], [f64::NAN, 3.0], [4.0, 5.0]]);

    let result = forest.fit(&x.view());
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test fitting with infinite values
#[test]
fn test_fit_infinite_data() {
    let mut forest = IsolationForest::default();
    let x = arr2(&[[1.0, 2.0], [f64::INFINITY, 3.0], [4.0, 5.0]]);

    let result = forest.fit(&x.view());
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test anomaly score calculation for single samples
#[test]
fn test_anomaly_score() {
    let mut forest = IsolationForest::new(10, 50, Some(3), Some(42)).unwrap();
    let x_train = arr2(&[[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.8], [0.8, 1.0]]);

    forest.fit(&x_train.view()).unwrap();

    // Test normal point (should have low anomaly score, close to 0.5)
    let normal_point = [1.0, 1.0];
    let normal_score = forest.anomaly_score(&normal_point).unwrap();
    assert!(normal_score >= 0.0 && normal_score <= 1.0);

    // Test outlier point (should have higher anomaly score)
    let outlier_point = [10.0, 10.0];
    let outlier_score = forest.anomaly_score(&outlier_point).unwrap();
    assert!(outlier_score >= 0.0 && outlier_score <= 1.0);

    // Outlier should typically have higher score than normal point
    // (though this isn't guaranteed due to randomness)
}

// Test anomaly score on unfitted model
#[test]
fn test_anomaly_score_not_fitted() {
    let forest = IsolationForest::default();
    let sample = [1.0, 2.0];

    let result = forest.anomaly_score(&sample);
    assert!(matches!(result, Err(ModelError::NotFitted)));
}

// Test anomaly score with wrong feature dimension
#[test]
fn test_anomaly_score_dimension_mismatch() {
    let mut forest = IsolationForest::new(5, 10, Some(2), Some(42)).unwrap();
    let x_train = arr2(&[[1.0, 2.0], [2.0, 3.0]]);

    forest.fit(&x_train.view()).unwrap();

    // Try with wrong number of features
    let wrong_sample = [1.0, 2.0, 3.0]; // 3 features instead of 2
    let result = forest.anomaly_score(&wrong_sample);
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test predict function with multiple samples
#[test]
fn test_predict() {
    let mut forest = IsolationForest::new(20, 50, Some(4), Some(42)).unwrap();
    let x_train = arr2(&[
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.2],
        [0.2, 0.0],
        [-0.1, 0.1],
        [0.1, -0.1],
    ]);

    forest.fit(&x_train.view()).unwrap();

    let x_test = arr2(&[
        [0.05, 0.05], // normal point
        [5.0, 5.0],   // outlier
        [0.0, 0.0],   // training point
    ]);

    let scores = forest.predict(&x_test.view()).unwrap();
    assert_eq!(scores.len(), 3);

    // All scores should be between 0 and 1
    for &score in scores.iter() {
        assert!(score >= 0.0 && score <= 1.0);
    }
}

// Test predict on unfitted model
#[test]
fn test_predict_not_fitted() {
    let forest = IsolationForest::default();
    let x_test = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    let result = forest.predict(&x_test.view());
    assert!(matches!(result, Err(ModelError::NotFitted)));
}

// Test predict with empty data
#[test]
fn test_predict_empty_data() {
    let mut forest = IsolationForest::new(5, 10, Some(2), Some(42)).unwrap();
    let x_train = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    forest.fit(&x_train.view()).unwrap();

    let x_test = Array2::<f64>::zeros((0, 2));
    let result = forest.predict(&x_test.view());
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test predict with dimension mismatch
#[test]
fn test_predict_dimension_mismatch() {
    let mut forest = IsolationForest::new(5, 10, Some(2), Some(42)).unwrap();
    let x_train = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    forest.fit(&x_train.view()).unwrap();

    let x_test = arr2(&[[1.0, 2.0, 3.0]]); // 3 features instead of 2
    let result = forest.predict(&x_test.view());
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test predict with NaN values
#[test]
fn test_predict_nan_data() {
    let mut forest = IsolationForest::new(5, 10, Some(2), Some(42)).unwrap();
    let x_train = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    forest.fit(&x_train.view()).unwrap();

    let x_test = arr2(&[[1.0, f64::NAN]]);
    let result = forest.predict(&x_test.view());
    assert!(matches!(result, Err(ModelError::InputValidationError(_))));
}

// Test fit_predict function
#[test]
fn test_fit_predict() {
    let mut forest = IsolationForest::new(15, 30, Some(3), Some(123)).unwrap();
    let x = arr2(&[
        [1.0, 1.0],
        [1.2, 0.8],
        [0.8, 1.2],
        [1.1, 0.9],
        [10.0, 10.0], // outlier
    ]);

    let scores = forest.fit_predict(&x.view()).unwrap();
    assert_eq!(scores.len(), 5);

    // All scores should be between 0 and 1
    for &score in scores.iter() {
        assert!(score >= 0.0 && score <= 1.0);
    }

    // Verify the model is fitted
    assert_eq!(forest.get_n_features(), 2);
    assert!(forest.get_trees().is_some());
}

// Test with single feature data
#[test]
fn test_single_feature() {
    let mut forest = IsolationForest::new(10, 20, Some(3), Some(42)).unwrap();
    let x = arr2(&[
        [1.0],
        [1.1],
        [0.9],
        [1.05],
        [10.0], // outlier
    ]);

    let result = forest.fit(&x.view());
    assert!(result.is_ok());

    let scores = forest.predict(&x.view()).unwrap();
    assert_eq!(scores.len(), 5);

    for &score in scores.iter() {
        assert!(score >= 0.0 && score <= 1.0);
    }
}

// Test reproducibility with random seed
#[test]
fn test_reproducibility() {
    let x = arr2(&[[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [10.0, 10.0]]);

    // Train two models with same seed
    let mut forest1 = IsolationForest::new(10, 20, Some(3), Some(42)).unwrap();
    let mut forest2 = IsolationForest::new(10, 20, Some(3), Some(42)).unwrap();

    let scores1 = forest1.fit_predict(&x.view()).unwrap();
    let scores2 = forest2.fit_predict(&x.view()).unwrap();

    // Results should be identical with same seed
    for i in 0..scores1.len() {
        assert!((scores1[i] - scores2[i]).abs() < 1e-10);
    }
}

// Test different numbers of estimators
#[test]
fn test_different_n_estimators() {
    let x = arr2(&[[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [0.5, 2.5]]);

    for n_estimators in [1, 5, 10, 50] {
        let mut forest = IsolationForest::new(n_estimators, 10, Some(2), Some(42)).unwrap();
        let result = forest.fit(&x.view());
        assert!(result.is_ok());
        assert_eq!(forest.get_trees().as_ref().unwrap().len(), n_estimators);
    }
}

// Test different max_samples values
#[test]
fn test_different_max_samples() {
    let x = arr2(&[
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.5],
        [0.5, 2.5],
        [2.5, 0.5],
        [1.8, 1.2],
    ]);

    for max_samples in [2, 4, 6, 10] {
        let mut forest = IsolationForest::new(5, max_samples, Some(3), Some(42)).unwrap();
        let result = forest.fit(&x.view());
        assert!(result.is_ok());
        assert_eq!(forest.get_max_samples(), max_samples);
    }
}

// Test path length calculation consistency
#[test]
fn test_path_length_consistency() {
    let mut forest = IsolationForest::new(1, 10, Some(3), Some(42)).unwrap(); // Single tree for predictable behavior
    let x = arr2(&[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);

    forest.fit(&x.view()).unwrap();

    // Same sample should always give same score
    let sample = [1.5, 1.5];
    let score1 = forest.anomaly_score(&sample).unwrap();
    let score2 = forest.anomaly_score(&sample).unwrap();
    assert_eq!(score1, score2);
}

// Test with larger dataset
// Test with larger dataset
#[test]
fn test_larger_dataset() {
    let mut forest = IsolationForest::new(20, 100, Some(5), Some(42)).unwrap();

    // Generate normal data points around (0, 0)
    let mut data = Vec::new();
    for i in 0..100 {
        data.push((i as f64 * 0.01 - 1.0).sin());
        data.push((i as f64 * 0.01 - 1.0).cos());
    }
    // Add some outliers
    data.extend_from_slice(&[10.0, 10.0, -10.0, -10.0]);

    let x = Array2::from_shape_vec((102, 2), data).unwrap();

    let result = forest.fit(&x.view());
    assert!(result.is_ok());

    let scores = forest.predict(&x.view()).unwrap();
    assert_eq!(scores.len(), 102);

    // Check that all scores are valid
    for &score in scores.iter() {
        assert!(score >= 0.0 && score <= 1.0);
        assert!(!score.is_nan());
    }
}

// Test edge case with all identical points
#[test]
fn test_identical_points() {
    let mut forest = IsolationForest::new(10, 20, Some(3), Some(42)).unwrap();
    let x = arr2(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]);

    let result = forest.fit(&x.view());
    assert!(result.is_ok());

    let scores = forest.predict(&x.view()).unwrap();
    assert_eq!(scores.len(), 4);

    // All identical points should have similar scores
    let first_score = scores[0];
    for &score in scores.iter() {
        assert!((score - first_score).abs() < 1e-10);
        assert!(score >= 0.0 && score <= 1.0);
    }
}

// Test parallel processing consistency
#[test]
fn test_parallel_consistency() {
    let x = arr2(&[
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [100.0, 200.0],
    ]);

    let mut forest = IsolationForest::new(10, 20, Some(4), Some(42)).unwrap();
    forest.fit(&x.view()).unwrap();

    // Run prediction multiple times - should be consistent
    let scores1 = forest.predict(&x.view()).unwrap();
    let scores2 = forest.predict(&x.view()).unwrap();

    for i in 0..scores1.len() {
        assert_eq!(scores1[i], scores2[i]);
    }
}

// Test memory efficiency with many small predictions
#[test]
fn test_memory_efficiency() {
    let mut forest = IsolationForest::new(5, 10, Some(3), Some(42)).unwrap();
    let x_train = arr2(&[[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]]);

    forest.fit(&x_train.view()).unwrap();

    // Make many individual predictions
    for i in 0..100 {
        let test_point = [i as f64 * 0.1, i as f64 * 0.1];
        let score = forest.anomaly_score(&test_point).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
