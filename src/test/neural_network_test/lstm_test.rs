use super::*;

#[test]
fn test_lstm_layer() {
    // Create input data with batch_size=2, timesteps=5, input_dim=4,
    // and target data with batch_size=2, units=3 (same dimension as the final hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: an LSTM layer with Tanh activation function
    let mut model = Sequential::new();
    model
        .add(LSTM::new(4, 3, Activation::Tanh))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);
    println!("LSTM prediction:\n{:#?}\n", pred);
}

#[test]
fn test_lstm_layer_basic() {
    // Create input data with batch_size=2, timesteps=5, input_dim=4
    // Create target data with batch_size=2, units=3 (same dimension as the final hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: an LSTM layer with Tanh activation function
    let mut model = Sequential::new();
    model
        .add(LSTM::new(4, 3, Activation::Tanh))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);

    // Check output shape
    assert_eq!(pred.shape(), &[2, 3]);
}

#[test]
fn test_lstm_different_activations() {
    // Test different activation functions
    let x = Array::ones((3, 4, 2)).into_dyn();
    let y = Array::ones((3, 6)).into_dyn();

    // Test ReLU activation function
    let mut model_relu = Sequential::new();
    model_relu
        .add(LSTM::new(2, 6, Activation::ReLU))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    model_relu.fit(&x, &y, 3).unwrap();
    let pred_relu = model_relu.predict(&x);

    // Test Sigmoid activation function
    let mut model_sigmoid = Sequential::new();
    model_sigmoid
        .add(LSTM::new(2, 6, Activation::Sigmoid))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    model_sigmoid.fit(&x, &y, 3).unwrap();
    let pred_sigmoid = model_sigmoid.predict(&x);

    // Check output shapes
    assert_eq!(pred_relu.shape(), &[3, 6]);
    assert_eq!(pred_sigmoid.shape(), &[3, 6]);

    // When using ReLU, outputs should be greater than or equal to 0
    for v in pred_relu.iter() {
        assert!(*v >= 0.0);
    }

    // When using Sigmoid, outputs should be between 0 and 1
    for v in pred_sigmoid.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_lstm_sequential_composition() {
    // Test LSTM combined with other layers
    let x = Array::ones((2, 5, 3)).into_dyn();
    let y = Array::ones((2, 4)).into_dyn();

    // Build a model containing LSTM and Dense layers
    let mut model = Sequential::new();
    model
        .add(LSTM::new(3, 6, Activation::Tanh))
        .add(Dense::new(6, 4, Activation::Sigmoid))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 5).unwrap();

    // Predict
    let pred = model.predict(&x);

    // Check output shape
    assert_eq!(pred.shape(), &[2, 4]);

    // Since the last layer uses Sigmoid activation, all outputs should be between 0 and 1
    for v in pred.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_lstm_overfitting() {
    // Test if the model can overfit a simple dataset
    let x = Array::ones((2, 4, 3)).into_dyn();
    let y = Array::ones((2, 7)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(LSTM::new(3, 7, Activation::Tanh))
        .compile(RMSprop::new(0.01, 0.9, 1e-8), MeanSquaredError::new());

    // Train long enough to overfit
    model.fit(&x, &y, 50).unwrap();

    // Predictions should be very close to target values
    let pred = model.predict(&x);
    for (pred_val, target_val) in pred.iter().zip(y.iter()) {
        assert_abs_diff_eq!(*pred_val, *target_val, epsilon = 0.1);
    }
}

#[test]
fn test_lstm_sequence_learning() {
    // Test LSTM's ability to learn sequence patterns
    // Create sequences where output depends on the sum of inputs
    let batch_size = 8;
    let seq_len = 6;
    let input_dim = 1;

    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    let mut y = Array2::<f32>::zeros((batch_size, 1));

    // Generate sequences with different patterns
    for b in 0..batch_size {
        let mut sequence_sum = 0.0;
        for t in 0..seq_len {
            // Create a simple pattern: alternating values
            let val = if t % 2 == 0 { 0.5 } else { -0.3 };
            x[[b, t, 0]] = val + 0.1 * (b as f32 / batch_size as f32); // Add slight variation per batch
            sequence_sum += x[[b, t, 0]];
        }
        // Target is normalized sum of sequence
        y[[b, 0]] = (sequence_sum / seq_len as f32 + 1.0) / 2.0; // Normalize to [0,1]
    }

    let x = x.into_dyn();
    let y = y.into_dyn();

    let mut model = Sequential::new();
    model
        .add(LSTM::new(input_dim, 12, Activation::Tanh))
        .add(Dense::new(12, 1, Activation::Sigmoid))
        .compile(Adam::new(0.005, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Train for sequence learning
    model.fit(&x, &y, 80).unwrap();

    let pred = model.predict(&x);

    // Calculate average error
    let mut total_error = 0.0;
    for b in 0..batch_size {
        let error = (pred[[b, 0]] - y[[b, 0]]).abs();
        total_error += error;
    }
    let avg_error = total_error / batch_size as f32;

    // LSTM should learn the sequence pattern reasonably well
    assert!(avg_error < 0.2, "Average error too high: {}", avg_error);

    println!("LSTM Sequence Learning - Average Error: {:.4}", avg_error);
    println!("Sample predictions vs targets:");
    for b in 0..4.min(batch_size) {
        println!(
            "  Batch {}: pred={:.3}, target={:.3}",
            b,
            pred[[b, 0]],
            y[[b, 0]]
        );
    }
}

#[test]
fn test_lstm_state_evolution() {
    // Test that LSTM internal states evolve meaningfully across timesteps
    let batch_size = 2;
    let seq_len = 5;
    let input_dim = 3;
    let units = 4;

    // Create input with clear temporal structure
    let mut x = Array3::<f32>::zeros((batch_size, seq_len, input_dim));
    for b in 0..batch_size {
        for t in 0..seq_len {
            x[[b, t, 0]] = (t as f32 / seq_len as f32).sin(); // Sinusoidal pattern
            x[[b, t, 1]] = t as f32 / seq_len as f32; // Linear increase
            x[[b, t, 2]] = if t % 2 == 0 { 1.0 } else { -1.0 }; // Alternating pattern
        }
    }

    let x = x.into_dyn();
    let y = Array::ones((batch_size, units)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(LSTM::new(input_dim, units, Activation::Tanh))
        .compile(Adam::new(0.01, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Test multiple training steps to see evolution
    let mut predictions = Vec::new();

    for _ in 0..5 {
        model.fit(&x, &y, 1).unwrap();
        let pred = model.predict(&x);
        predictions.push(pred);
    }

    // Check that predictions evolve over training epochs
    let initial_pred = &predictions[0];
    let final_pred = &predictions[predictions.len() - 1];

    let change = (final_pred - initial_pred).mapv(|x| x.abs()).sum();
    assert!(change > 1e-4, "Predictions should evolve during training");

    // Check that final predictions are reasonable (not all zeros or ones)
    let pred_mean = final_pred.mean().unwrap();
    assert!(
        pred_mean > -0.9 && pred_mean < 0.9,
        "Final predictions should be in reasonable range, got mean: {}",
        pred_mean
    );

    println!(
        "LSTM State Evolution Test - Prediction change: {:.6}",
        change
    );
    println!("Final prediction mean: {:.4}", pred_mean);
}
