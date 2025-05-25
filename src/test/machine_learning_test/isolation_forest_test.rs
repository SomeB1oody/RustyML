use crate::machine_learning::isolation_forest::IsolationForest;
use ndarray::{Array, Array2};
use rand::Rng;

#[test]
fn test_isolation_forest_new() {
    // Test creation with default parameters
    let forest = IsolationForest::default();
    assert_eq!(forest.get_n_estimators(), 100); // Assuming default value is 100
    assert_eq!(forest.get_max_samples(), 256); // Assuming default value is 256
    assert_eq!(forest.get_max_depth(), 8); // Assuming default value is 8
    assert_eq!(forest.get_random_state(), None);

    // Test creation with custom parameters
    let forest = IsolationForest::new(50, 100, Some(10), Some(42));
    assert_eq!(forest.get_n_estimators(), 50);
    assert_eq!(forest.get_max_samples(), 100);
    assert_eq!(forest.get_max_depth(), 10);
    assert_eq!(forest.get_random_state(), Some(42));
}

#[test]
fn test_isolation_forest_getters() {
    let forest = IsolationForest::new(50, 100, Some(10), Some(42));

    assert_eq!(forest.get_n_estimators(), 50);
    assert_eq!(forest.get_max_samples(), 100);
    assert_eq!(forest.get_max_depth(), 10);
    assert_eq!(forest.get_random_state(), Some(42));

    // Test that get_trees returns an error before model is fitted
    match forest.get_trees() {
        Err(_) => assert!(true), // Expected failure
        Ok(_) => panic!("Should return an error before fitting"),
    }
}

#[test]
fn test_isolation_forest_fit() {
    // Create a 2D sample data
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8,
            0.8, 10.0, 10.0, // Outlier
        ],
    )
    .unwrap();

    let mut forest = IsolationForest::new(10, 8, Some(5), Some(42));
    forest.fit(data.view()).unwrap();

    // After fitting, we should be able to get the trees
    assert!(forest.get_trees().is_ok());
    if let Ok(trees) = forest.get_trees() {
        assert_eq!(trees.len(), 10); // Should have n_estimators trees
    }
}

#[test]
fn test_isolation_forest_anomaly_score() {
    // Create a 2D sample data with one clear outlier
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8,
            0.8, 10.0, 10.0, // Outlier
        ],
    )
    .unwrap();

    let mut forest = IsolationForest::new(100, 8, Some(8), Some(42));
    forest.fit(data.view()).unwrap();

    // Test anomaly score for a normal point
    let normal_point = vec![0.5, 0.5];
    let normal_score = forest.anomaly_score(&normal_point).unwrap();

    // Test anomaly score for an outlier point
    let anomaly_point = vec![10.0, 10.0];
    let anomaly_score = forest.anomaly_score(&anomaly_point).unwrap();

    // The outlier's score should be higher than the normal point's
    assert!(anomaly_score > normal_score);
}

#[test]
fn test_isolation_forest_predict() {
    // Create a 2D sample training data
    let train_data = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8,
            0.8, 0.9, 0.9,
        ],
    )
    .unwrap();

    let mut forest = IsolationForest::new(100, 8, Some(8), Some(42));
    forest.fit(train_data.view()).unwrap();

    // Create test data with both normal and anomaly points
    let test_data = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.5, 0.5, // Normal point
            0.7, 0.7, // Normal point
            10.0, 10.0, // Outlier
            -5.0, -5.0, // Outlier
        ],
    )
    .unwrap();

    // Predict
    let predictions = forest.predict(test_data.view()).unwrap();
    assert_eq!(predictions.len(), 4);

    // The outliers' scores in the predictions array should be higher
    assert!(predictions[2] > predictions[0]); // 10.0, 10.0 > 0.5, 0.5
    assert!(predictions[3] > predictions[1]); // -5.0, -5.0 > 0.7, 0.7
}

#[test]
fn test_error_handling() {
    let forest = IsolationForest::default();

    // Test calling anomaly detection functions before fitting
    let sample = vec![0.5, 0.5];
    match forest.anomaly_score(&sample) {
        Err(_) => assert!(true), // Expected failure
        Ok(_) => panic!("Should return error when model is not fitted"),
    }

    // Test calling predict function before fitting
    let test_data = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.7, 0.7]).unwrap();
    match forest.predict(test_data.view()) {
        Err(_) => assert!(true), // Expected failure
        Ok(_) => panic!("Should return error when model is not fitted"),
    }
}

#[test]
fn test_dimension_validation() {
    // Train on 2D data
    let train_data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
    )
    .unwrap();

    let mut forest = IsolationForest::new(10, 4, Some(5), Some(42));
    forest.fit(train_data.view()).unwrap();

    // Test with 2D point (should work)
    let correct_sample = vec![0.2, 0.2];
    assert!(forest.anomaly_score(&correct_sample).is_ok());

    // Test with 1D point (should fail)
    let too_small_sample = vec![0.2];
    assert!(forest.anomaly_score(&too_small_sample).is_err());

    // Test with 3D point (should fail)
    let too_large_sample = vec![0.2, 0.2, 0.2];
    assert!(forest.anomaly_score(&too_large_sample).is_err());
}

#[test]
fn test_fit_predict() {
    // Create a simple dataset with some outliers
    // Normal points clustered around origin
    let mut data = Vec::new();

    // Generate 50 normal points around origin
    let mut rng = rand::rng();
    for _ in 0..50 {
        let x = rng.random_range(-1.0..1.0);
        let y = rng.random_range(-1.0..1.0);
        data.push(vec![x, y]);
    }

    // Add 5 outlier points far from origin
    data.push(vec![10.0, 10.0]);
    data.push(vec![-10.0, 10.0]);
    data.push(vec![10.0, -10.0]);
    data.push(vec![-10.0, -10.0]);
    data.push(vec![15.0, 0.0]);

    // Convert to ndarray
    let n_samples = data.len();
    let n_features = data[0].len();
    let x_flat: Vec<f64> = data.into_iter().flatten().collect();
    let x = Array::from_shape_vec((n_samples, n_features), x_flat).unwrap();

    // Create and fit IsolationForest
    let mut forest = IsolationForest::new(100, 25, Some(8), Some(42));

    // Using fit_predict to get anomaly scores
    let scores = forest.fit_predict(x.view()).unwrap();

    // We expect 5 outliers to have higher anomaly scores than normal points
    // Let's sort the scores and check the indices
    let mut score_indices: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    score_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by score in descending order

    // The top 5 scores should correspond to our outliers (indices 50-54)
    let top_indices: Vec<usize> = score_indices.iter().take(5).map(|(i, _)| *i).collect();

    assert!(top_indices.contains(&50));
    assert!(top_indices.contains(&51));
    assert!(top_indices.contains(&52));
    assert!(top_indices.contains(&53));
    assert!(top_indices.contains(&54));

    // Also verify that fit_predict gives the same results as calling fit and predict separately
    let mut forest2 = IsolationForest::new(100, 25, Some(8), Some(42));
    forest2.fit(x.view()).unwrap();
    let scores2 = forest2.predict(x.view()).unwrap();

    // Compare results (should be identical since we used the same random seed)
    for (s1, s2) in scores.iter().zip(scores2.iter()) {
        assert_eq!(*s1, *s2);
    }
}
