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
