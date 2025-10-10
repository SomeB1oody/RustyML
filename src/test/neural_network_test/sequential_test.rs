use super::*;

#[test]
fn fit_with_batches_test() {
    // create training data
    let x = Array::ones((1000, 784)).into_dyn(); // 1000 samples, 784 features
    let y = Array::ones((1000, 10)).into_dyn(); // 1000 samples, 10 categories

    // build a neural network
    let mut model = Sequential::new();
    model
        .add(Dense::new(784, 128, Activation::ReLU))
        .add(Dense::new(128, 64, Activation::ReLU))
        .add(Dense::new(64, 10, Activation::Softmax))
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8),
            CategoricalCrossEntropy::new(),
        );

    // use batch processing to train the model
    model.fit_with_batches(&x, &y, 1, 32).unwrap();
}

#[test]
fn test_fit_linear_regression_convergence() {
    // Create a simple linearly separable dataset for testing
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    // Generate simple linear relationship data: y = 2*x + 1
    for i in 0..100 {
        let x_val = i as f32 / 50.0; // x ranges from 0 to 2
        let y_val = 2.0 * x_val + 1.0; // linear relationship
        x_data.push(x_val);
        y_data.push(y_val);
    }

    // Convert to ndarray format
    let x = Array::from_shape_vec((100, 1), x_data).unwrap().into_dyn();
    let y = Array::from_shape_vec((100, 1), y_data).unwrap().into_dyn();

    // Build a simple network to learn the linear relationship
    let mut model = Sequential::new();
    model
        .add(Dense::new(1, 1, Activation::Linear)) // Single linear layer
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Record initial predictions
    let initial_predictions = model.predict(&x);
    let initial_loss = calculate_mse(&y, &initial_predictions);

    // Train the model
    model.fit(&x, &y, 100).unwrap();

    // Get predictions after training
    let final_predictions = model.predict(&x);
    let final_loss = calculate_mse(&y, &final_predictions);

    // Validation 1: Loss should decrease significantly
    assert!(
        final_loss < initial_loss,
        "Final loss ({:.6}) should be less than initial loss ({:.6})",
        final_loss,
        initial_loss
    );

    // Validation 2: Final loss should be relatively small
    assert!(
        final_loss < 0.2,
        "For simple linear relationship, final loss ({:.6}) should be less than 0.2",
        final_loss
    );

    // Validation 3: Predictions for known inputs should be close to expected values
    let test_x = Array::from_shape_vec((1, 1), vec![1.0]).unwrap().into_dyn();
    let prediction = model.predict(&test_x);
    let expected = 3.0; // 2*1 + 1 = 3

    assert!(
        (prediction[[0, 0]] - expected).abs() <= 0.3,
        "Prediction ({:.3}) for input 1.0 should be close to expected value ({:.3})",
        prediction[[0, 0]],
        expected
    );
}

#[test]
fn test_fit_classification_convergence() {
    // Create a more challenging binary classification dataset
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    // Class 0: scattered points in one region with some overlap
    for i in 0..50 {
        let x1 = -2.0 + (i as f32 / 25.0) + (rand::random::<f32>() - 0.5) * 0.5; // Add noise
        let x2 = -2.0 + (i as f32 / 25.0) + (rand::random::<f32>() - 0.5) * 0.5; // Add noise
        x_data.extend_from_slice(&[x1, x2]);
        y_data.extend_from_slice(&[1.0, 0.0]); // one-hot encoding [1,0]
    }

    // Class 1: scattered points in another region with some overlap
    for i in 0..50 {
        let x1 = 0.5 + (i as f32 / 25.0) + (rand::random::<f32>() - 0.5) * 0.5; // Add noise
        let x2 = 0.5 + (i as f32 / 25.0) + (rand::random::<f32>() - 0.5) * 0.5; // Add noise
        x_data.extend_from_slice(&[x1, x2]);
        y_data.extend_from_slice(&[0.0, 1.0]); // one-hot encoding [0,1]
    }

    let x = Array::from_shape_vec((100, 2), x_data).unwrap().into_dyn();
    let y = Array::from_shape_vec((100, 2), y_data).unwrap().into_dyn();

    // Build classification network with smaller learning rate
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 4, Activation::ReLU))
        .add(Dense::new(4, 2, Activation::Softmax))
        .compile(
            Adam::new(0.01, 0.9, 0.999, 1e-8),
            CategoricalCrossEntropy::new(),
        ); // Smaller learning rate

    // Record initial prediction accuracy
    let initial_predictions = model.predict(&x);
    let initial_accuracy = calculate_accuracy(&y, &initial_predictions);

    println!("Initial accuracy: {:.3}", initial_accuracy);

    // Train the model for more epochs
    model.fit(&x, &y, 150).unwrap();

    // Get predictions after training
    let final_predictions = model.predict(&x);
    let final_accuracy = calculate_accuracy(&y, &final_predictions);

    println!("Final accuracy: {:.3}", final_accuracy);

    // Validation 1: Accuracy should improve (with tolerance for cases where initial is already high)
    if initial_accuracy < 0.9 {
        assert!(
            final_accuracy > initial_accuracy,
            "Final accuracy ({:.3}) should be higher than initial accuracy ({:.3})",
            final_accuracy,
            initial_accuracy
        );
    } else {
        // If initial accuracy is already very high, just ensure final accuracy is maintained
        assert!(
            final_accuracy >= initial_accuracy - 0.05,
            "Final accuracy ({:.3}) should not be significantly worse than initial accuracy ({:.3})",
            final_accuracy,
            initial_accuracy
        );
    }

    // Validation 2: Final accuracy should be reasonable
    assert!(
        final_accuracy > 0.7,
        "For classification task, final accuracy ({:.3}) should be greater than 0.7",
        final_accuracy
    );

    // Validation 3: Test predictions for specific inputs
    let test_x = Array::from_shape_vec((2, 2), vec![-1.5, -1.5, 1.5, 1.5])
        .unwrap()
        .into_dyn();
    let predictions = model.predict(&test_x);

    // First sample [-1.5, -1.5] should predict class 0 (first element larger)
    assert!(
        predictions[[0, 0]] > predictions[[0, 1]],
        "Sample [-1.5, -1.5] should be classified as class 0"
    );

    // Second sample [1.5, 1.5] should predict class 1 (second element larger)
    assert!(
        predictions[[1, 1]] > predictions[[1, 0]],
        "Sample [1.5, 1.5] should be classified as class 1"
    );
}

#[test]
fn test_fit_parameter_updates() {
    // Create simple training data
    let x = Array::from_shape_vec(
        (10, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5,
            5.5, 5.5, 6.5,
        ],
    )
    .unwrap()
    .into_dyn();
    let y = Array::from_shape_vec(
        (10, 1),
        vec![3.0, 5.0, 7.0, 9.0, 11.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    )
    .unwrap()
    .into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::Linear))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Get weights before training
    let initial_weights = model.get_weights();
    let mut initial_weight_sums = Vec::new();
    for layer_weight in &initial_weights {
        match layer_weight {
            LayerWeight::Dense(DenseLayerWeight { weight, bias: _ }) => {
                initial_weight_sums.push(weight.sum());
            }
            _ => {}
        }
    }

    // Train the model
    model.fit(&x, &y, 10).unwrap();

    // Get weights after training
    let final_weights = model.get_weights();
    let mut final_weight_sums = Vec::new();
    for layer_weight in &final_weights {
        match layer_weight {
            LayerWeight::Dense(DenseLayerWeight { weight, bias: _ }) => {
                final_weight_sums.push(weight.sum());
            }
            _ => {}
        }
    }

    // Verify that weights have actually changed
    for (i, (&initial_sum, &final_sum)) in initial_weight_sums
        .iter()
        .zip(final_weight_sums.iter())
        .enumerate()
    {
        assert!(
            (final_sum - initial_sum).abs() > 1e-6,
            "Layer {} weights should change during training (initial: {:.6}, final: {:.6})",
            i,
            initial_sum,
            final_sum
        );
    }
}

#[test]
fn test_fit_error_handling() {
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, Activation::Linear));

    // Test 1: Uncompiled model should return error
    let x = Array::ones((5, 2)).into_dyn();
    let y = Array::ones((5, 1)).into_dyn();

    let result = model.fit(&x, &y, 10);
    assert!(result.is_err(), "Uncompiled model should return error");

    // Compile the model
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Test 2: Empty data should return error
    let empty_x = Array::zeros((0, 2)).into_dyn();
    let empty_y = Array::zeros((0, 1)).into_dyn();

    let result = model.fit(&empty_x, &empty_y, 10);
    assert!(result.is_err(), "Empty data should return error");

    // Test 3: Dimension mismatch should return error
    let x_mismatch = Array::ones((5, 2)).into_dyn();
    let y_mismatch = Array::ones((3, 1)).into_dyn(); // Sample count mismatch

    let result = model.fit(&x_mismatch, &y_mismatch, 10);
    assert!(result.is_err(), "Sample count mismatch should return error");
}

// Helper function: calculate mean squared error
fn calculate_mse(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let diff = y_pred - y_true;
    let squared_diff = &diff * &diff;
    squared_diff.sum() / (y_true.len() as f32)
}

// Helper function: calculate classification accuracy
fn calculate_accuracy(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let mut correct = 0;
    let n_samples = y_true.shape()[0];

    for i in 0..n_samples {
        // Find the maximum value indices for true labels and predicted labels
        let true_class = if y_true[[i, 0]] > y_true[[i, 1]] {
            0
        } else {
            1
        };
        let pred_class = if y_pred[[i, 0]] > y_pred[[i, 1]] {
            0
        } else {
            1
        };

        if true_class == pred_class {
            correct += 1;
        }
    }

    correct as f32 / n_samples as f32
}
