use crate::neural_network::*;
use ndarray::prelude::*;

#[test]
fn mse_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::ReLU));
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn mae_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::ReLU));
    model.compile(SGD::new(0.01), MeanAbsoluteError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn binary_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::ReLU));
    model.compile(SGD::new(0.01), BinaryCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn categorical_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::ReLU));
    model.compile(SGD::new(0.01), CategoricalCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn sparse_categorical_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    // Assume there are 3 classes, labels should be one of 0, 1, 2
    let y: ArrayD<f32> = Array::from_shape_vec((2, 1), vec![0.0, 1.0])
        .unwrap()
        .into_dyn();

    // Build the model, note that the second Dense layer must use Dense::new(3, 3) because it's a multi-class task
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 3, Activation::ReLU));
    model.compile(SGD::new(0.01), SparseCategoricalCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn adam_test() {
    // Create an input tensor with shape (batch_size=2, input_dim=4) and corresponding target tensor
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model: add two Dense layers, use Adam optimizer (learning rate, beta1, beta2, epsilon) with MSE loss function
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::ReLU));
    model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}

#[test]
fn rmsprop_test() {
    // Create input (batch_size=2, input_dim=4) and target tensors (batch_size=2, output_dim=3)
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model, add two Dense layers; choose RMSprop optimizer when compiling
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(Dense::new(3, 1, Activation::ReLU));
    model.compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}

#[test]
fn with_activation_test() {
    // Create input tensor with shape (batch_size=2, input_dim=4) and target tensor (batch_size=2, output_dim=3)
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build model: using Dense layers with specified activation functions (such as ReLU or Softmax)
    // Here we use Sigmoid activation for the first layer and Softmax for the second layer (you can modify as needed)
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::Sigmoid))
        .add(Dense::new(3, 1, Activation::Softmax));

    // Choose an optimizer, e.g., RMSprop, Adam or SGD - using RMSprop as an example here
    model.compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model (simple iteration example)
    model.fit(&x, &y, 3).unwrap();

    // Get output using predict
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}

#[test]
fn test_simple_rnn_layer() {
    // Create input with batch_size=2, timesteps=5, input_dim=4,
    // and target with batch_size=2, units=3 (same dimension as the last hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: one SimpleRnn layer with tanh activation
    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(4, 3, Activation::Tanh))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);
    println!("SimpleRnn prediction:\n{:#?}\n", pred);
}

#[test]
fn test_lstm_layer() {
    // Create input with batch_size=2, timesteps=5, input_dim=4,
    // and target with batch_size=2, units=3 (same dimension as the last hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: one SimpleRnn layer with tanh activation
    let mut model = Sequential::new();
    model
        .add(LSTM::new(4, 3, Activation::Tanh))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print structure
    model.summary();

    // Train for 1 epoch
    model.fit(&x, &y, 1).unwrap();

    // Predict
    let pred = model.predict(&x);
    println!("LSTM prediction:\n{:#?}\n", pred);
}
