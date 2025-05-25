use crate::ModelError;
use crate::machine_learning::logistic_regression::*;
use ndarray::{Array1, Array2, arr1, arr2};

#[test]
fn test_default_constructor() {
    let model = LogisticRegression::default();

    assert_eq!(model.get_fit_intercept(), true);
    assert_eq!(model.get_learning_rate(), 0.01);
    assert_eq!(model.get_max_iterations(), 100);
    assert_eq!(model.get_tolerance(), 1e-4);
    assert!(matches!(
        model.get_actual_iterations(),
        Err(ModelError::NotFitted)
    ));
    assert!(matches!(model.get_weights(), Err(ModelError::NotFitted)));
}

#[test]
fn test_new_constructor() {
    let model = LogisticRegression::new(false, 0.05, 200, 1e-5, None);

    assert_eq!(model.get_fit_intercept(), false);
    assert_eq!(model.get_learning_rate(), 0.05);
    assert_eq!(model.get_max_iterations(), 200);
    assert_eq!(model.get_tolerance(), 1e-5);
    assert!(matches!(
        model.get_actual_iterations(),
        Err(ModelError::NotFitted)
    ));
    assert!(matches!(model.get_weights(), Err(ModelError::NotFitted)));
}

#[test]
fn test_fit_and_predict_simple_case() {
    // Simple linearly separable data
    let x = arr2(&[
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
    ]);
    let y = arr1(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let mut model = LogisticRegression::default();
    model.fit(x.view(), y.view()).unwrap();

    // Check if weights exist
    assert!(matches!(model.get_weights(), Ok(_)));

    // Predict training data
    let predictions = model.predict(x.view()).unwrap();

    let correct_predictions = predictions
        .iter()
        .zip(y.iter())
        .filter(|&(ref pred, &actual)| (**pred as f64) == actual)
        .count();

    let accuracy = correct_predictions as f64 / predictions.len() as f64;
    assert!(accuracy >= 0.875); // 1 mistake is acceptable
}

#[test]
fn test_predict_proba() {
    // Create a simple model and set weights
    let mut model = LogisticRegression::new(false, 0.01, 100, 1e-4, None);

    // Since predict_proba is private, we'll test it indirectly
    // First, fit the model with simple data
    let x = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    let y = arr1(&[0.0, 0.0, 0.0, 1.0]);

    model.fit(x.view(), y.view()).unwrap();

    // Check predictions
    let predictions = model.predict(x.view()).unwrap();
    assert_eq!(predictions.len(), 4);
}

#[test]
fn test_generate_polynomial_features() {
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    // Test degree = 1 (should return intercept + original features)
    let poly_features_1 = generate_polynomial_features(x.view(), 1);
    let expected_degree_1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(poly_features_1, expected_degree_1);

    // Test degree = 2
    let poly_features_2 = generate_polynomial_features(x.view(), 2);
    let expected_degree_2 = arr2(&[[1.0, 2.0, 1.0, 2.0, 4.0], [3.0, 4.0, 9.0, 12.0, 16.0]]);

    assert_eq!(poly_features_2.shape(), expected_degree_2.shape());

    // Check floating-point values with approximate equality
    for i in 0..poly_features_2.nrows() {
        for j in 0..poly_features_2.ncols() {
            assert!(
                (poly_features_2[[i, j]] - expected_degree_2[[i, j]]).abs() < f64::EPSILON * 100.0
            );
        }
    }
}

#[test]
fn test_fit_with_intercept() {
    // Training with intercept
    let x = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
    let y = arr1(&[0.0, 0.0, 1.0, 1.0]);

    let mut model_with_intercept = LogisticRegression::new(true, 0.1, 1000, 1e-6, None);
    model_with_intercept.fit(x.view(), y.view()).unwrap();

    // Training without intercept
    let mut model_without_intercept = LogisticRegression::new(false, 0.1, 1000, 1e-6, None);
    model_without_intercept.fit(x.view(), y.view()).unwrap();

    // Check predictions from both models
    let predictions_with_intercept = model_with_intercept.predict(x.view()).unwrap();

    // The model with intercept should fit this data better
    assert_eq!(predictions_with_intercept, arr1(&[0, 0, 1, 1]));
}

#[test]
fn test_fit_predict() {
    // Create a simple logistic regression instance
    let mut model = LogisticRegression::new(true, 0.01, 10000, 1e-5, None);

    // Create training data: simple 2D features
    // For example: two features forming two linearly separable classes
    let train_x = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0, // Class 0
            0.0, 1.0, // Class 1
            1.0, 0.0, // Class 1
            1.0, 1.0, // Class 1
        ],
    )
    .unwrap();

    // Corresponding target values: 0 for first class, 1 for second class
    let train_y = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0]);

    // Create test data
    let test_x = Array2::from_shape_vec(
        (2, 2),
        vec![
            0.0, 0.2, // Should be close to class 0
            0.9, 0.8, // Should predict class 1
        ],
    )
    .unwrap();

    // Use fit_predict method for prediction
    let predictions = model
        .fit_predict(train_x.view(), train_y.view(), test_x.view())
        .unwrap();

    // Assert: Check that the predictions are a 1D array of length 2 (number of test samples)
    assert_eq!(predictions.len(), 2);

    // Check if predictions match expectations
    // Since the training data is simple, we expect first prediction as 0, second as 1
    assert_eq!(predictions[0], 0);
    assert_eq!(predictions[1], 1);
}
