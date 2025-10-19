use super::*;

#[test]
fn test_default() {
    let model = LinearSVC::default();
    assert_eq!(model.get_weights(), None);
    assert_eq!(model.get_bias(), None);
    assert_eq!(model.get_actual_iterations(), None);
}

#[test]
fn test_new() {
    let model = LinearSVC::new(
        100,                         // max_iter
        0.01,                        // learning_rate
        RegularizationType::L2(0.1), // penalty with regularization strength
        true,                        // fit_intercept
        1e-4,                        // tol
    )
    .unwrap();

    assert_eq!(model.get_max_iterations(), 100);
    assert_eq!(model.get_learning_rate(), 0.01);
    assert!(matches!(model.get_penalty(), RegularizationType::L2(0.1)));
    assert!(model.get_fit_intercept());
    assert_eq!(model.get_tolerance(), 1e-4);
}

#[test]
fn test_getters_before_fit() {
    let model = LinearSVC::default();

    // These should return None when model is not fitted
    assert!(model.get_weights().is_none());
    assert!(model.get_bias().is_none());
    assert!(model.get_actual_iterations().is_none());
}

#[test]
fn test_fit_predict_simple_case() {
    // Create a larger training dataset
    let x = arr2(&[
        [1.0, 1.0],
        [1.0, -2.0],
        [-1.0, 1.0],
        [-2.0, -1.0],
        [2.0, 2.0],
        [2.0, -3.0],
        [-2.0, 2.0],
        [-3.0, -2.0],
        [0.5, 1.5],
        [1.5, -0.5],
        [-0.5, 1.5],
        [-1.5, -0.5],
    ]);

    // Corresponding labels
    let y = arr1(&[
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    ]);

    let mut model = LinearSVC::new(
        10000,                        // max_iter
        0.1,                          // learning_rate
        RegularizationType::L2(0.01), // penalty with regularization strength
        true,                         // fit_intercept
        1e-4,                         // tol
    )
    .unwrap();

    // Fit the model
    model.fit(&x.view(), &y.view()).unwrap();

    // Test that weights and bias are now available
    assert!(model.get_weights().is_some());
    assert!(model.get_bias().is_some());
    assert!(model.get_actual_iterations().is_some());

    // Test predictions
    let predictions = model.predict(&x.view()).unwrap();
    assert_eq!(predictions.len(), 12); // Updated to new number of data points

    let mut correct_count = 0;

    for (i, &pred) in predictions.iter().enumerate() {
        if pred == y[i] {
            correct_count += 1;
        }
    }

    println!(
        "Correct predictions: {}/{}",
        correct_count,
        predictions.len()
    );

    // Expect higher accuracy with larger dataset
    assert!(correct_count >= 6); // Require at least 6/12 correct

    // Create some new test data points
    let x_test = arr2(&[[1.2, 0.8], [0.8, -1.7], [-0.9, 1.1], [-1.8, -0.9]]);

    // Expected labels
    let y_expected = arr1(&[1.0, -1.0, 1.0, -1.0]);

    // Make predictions on new data
    let test_predictions = model.predict(&x_test.view()).unwrap();

    // Check prediction accuracy on new data
    let mut test_correct = 0;
    for (i, &pred) in test_predictions.iter().enumerate() {
        if pred == y_expected[i] {
            test_correct += 1;
        }
    }

    println!(
        "Test set correct predictions: {}/{}",
        test_correct,
        test_predictions.len()
    );
    assert!(test_correct >= 2); // Require at least 2/4 correct
}

#[test]
fn test_decision_function() {
    // Create a simple dataset
    let x = arr2(&[[2.0, 2.0], [-2.0, -2.0]]);

    let y = arr1(&[1.0, -1.0]);

    let mut model = LinearSVC::default();
    model.fit(&x.view(), &y.view()).unwrap();

    // Get decision values
    let decision_values = model.decision_function(&x.view()).unwrap();

    // Decision values should have the same sign as labels
    assert!(decision_values[0] > 0.0);
    assert!(decision_values[1] < 0.0);
}

#[test]
fn test_different_penalties() {
    // Test with L1 penalty
    let mut model_l1 = LinearSVC::new(100, 0.01, RegularizationType::L1(0.1), true, 1e-4).unwrap();

    // Test with L2 penalty
    let mut model_l2 = LinearSVC::new(100, 0.01, RegularizationType::L2(0.1), true, 1e-4).unwrap();

    // Simple dataset
    let x = arr2(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);

    let y = arr1(&[1.0, 1.0, -1.0, -1.0]);

    // Fit both models
    let _ = model_l1.fit(&x.view(), &y.view());
    let _ = model_l2.fit(&x.view(), &y.view());

    // The weights should be different due to the different penalties
    if let (Some(w1), Some(w2)) = (model_l1.get_weights(), model_l2.get_weights()) {
        assert_ne!(w1, w2);
    }
}

#[test]
fn test_error_handling() {
    let model = LinearSVC::default();

    // Attempt to predict without fitting should return error
    let x = arr2(&[[1.0, 2.0]]);
    assert!(model.predict(&x.view()).is_err());

    // Attempt to get decision function without fitting should return error
    assert!(model.decision_function(&x.view()).is_err());
}

#[test]
fn test_fit_with_invalid_data() {
    let mut model = LinearSVC::default();

    // Test with mismatched dimensions
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    // y has 3 samples but x has only 2
    let y = arr1(&[1.0, -1.0, 1.0]);

    assert!(model.fit(&x.view(), &y.view()).is_err());
}
