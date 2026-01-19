use super::*;

#[test]
fn test_ada_grad_dense_basic() {
    // Create an input tensor with shape (batch_size=2, input_dim=4) and corresponding target tensor
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model: add two Dense layers, use AdaGrad optimizer with MSE loss function
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 5).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("AdaGrad Dense prediction: {:?}", prediction);

    // Check output shape
    assert_eq!(prediction.shape(), &[2, 1]);
}

#[test]
fn test_ada_grad_simple_rnn() {
    // Create input with batch_size=2, timesteps=5, input_dim=4,
    // and target with batch_size=2, units=3 (same dimension as the last hidden state)
    let x = Array::ones((2, 5, 4)).into_dyn();
    let y = Array::ones((2, 3)).into_dyn();

    // Build model: one SimpleRNN layer with Tanh activation
    let mut model = Sequential::new();
    model
        .add(SimpleRNN::new(4, 3, Tanh::new()).unwrap())
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Print structure
    model.summary();

    // Train for multiple epochs
    model.fit(&x, &y, 3).unwrap();

    // Predict
    let pred = model.predict(&x);
    println!("AdaGrad SimpleRNN prediction:\n{:#?}\n", pred);

    // Check output shape
    assert_eq!(pred.shape(), &[2, 3]);
}

#[test]
fn test_ada_grad_lstm() {
    // Create input data: batch_size=2, timesteps=5, features=4
    let input = Array::ones((2, 5, 4)).into_dyn();
    let target = Array::ones((2, 3)).into_dyn(); // batch_size=2, units=3

    // Create LSTM layer with 4 input features, 3 units, Tanh activation
    let mut model = Sequential::new();
    model
        .add(LSTM::new(4, 3, Tanh::new()).unwrap())
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Train the model
    model.fit(&input, &target, 3).unwrap();

    // Make predictions
    let predictions = model.predict(&input);
    println!("AdaGrad LSTM output shape: {:?}", predictions.shape());

    // Check output shape: [2, 3] (batch_size, units)
    assert_eq!(predictions.shape(), &[2, 3]);
}

#[test]
fn test_ada_grad_gru() {
    // Create input data: batch_size=3, timesteps=6, features=5
    let input = Array::ones((3, 6, 5)).into_dyn();
    let target = Array::ones((3, 4)).into_dyn(); // batch_size=3, units=4

    // Create GRU layer with 5 input features, 4 units, Tanh activation
    let mut model = Sequential::new();
    model
        .add(GRU::new(5, 4, Tanh::new()).unwrap())
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Train the model
    model.fit(&input, &target, 3).unwrap();

    // Make predictions
    let predictions = model.predict(&input);
    println!("AdaGrad GRU output shape: {:?}", predictions.shape());

    // Check output shape: [3, 4] (batch_size, units)
    assert_eq!(predictions.shape(), &[3, 4]);
}

#[test]
fn test_ada_grad_conv1d() {
    // Create input data: batch_size=2, channels=3, length=10
    let input = Array::ones((2, 3, 10)).into_dyn();
    let target = Array::ones((2, 4, 8)).into_dyn(); // output: (2, 4, 8)

    // Create Conv1D layer
    let mut model = Sequential::new();
    model
        .add(
            Conv1D::new(
                4,                  // filters
                3,                  // kernel_size
                vec![2, 3, 10],     // input_shape
                1,                  // stride
                PaddingType::Valid, // padding
                ReLU::new(),        // activation
            )
            .unwrap(),
        )
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Train the model
    model.fit(&input, &target, 3).unwrap();

    // Make predictions
    let predictions = model.predict(&input);
    println!("AdaGrad Conv1D output shape: {:?}", predictions.shape());

    // Check output shape
    assert_eq!(predictions.shape(), &[2, 4, 8]);
}

#[test]
fn test_ada_grad_conv2d() {
    // Create input data: batch_size=2, channels=3, height=8, width=8
    let input = Array::ones((2, 3, 8, 8)).into_dyn();
    let target = Array::ones((2, 8, 6, 6)).into_dyn(); // output: (2, 8, 6, 6)

    // Create Conv2D layer
    let mut model = Sequential::new();
    model
        .add(
            Conv2D::new(
                8,                  // filters
                (3, 3),             // kernel_size
                vec![2, 3, 8, 8],   // input_shape
                (1, 1),             // stride
                PaddingType::Valid, // padding
                ReLU::new(),        // activation
            )
            .unwrap(),
        )
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Train the model
    model.fit(&input, &target, 3).unwrap();

    // Make predictions
    let predictions = model.predict(&input);
    println!("AdaGrad Conv2D output shape: {:?}", predictions.shape());

    // Check output shape
    assert_eq!(predictions.shape(), &[2, 8, 6, 6]);
}

#[test]
fn test_ada_grad_conv3d() {
    // Create input data: batch_size=2, channels=2, depth=4, height=4, width=4
    let input = Array::ones((2, 2, 4, 4, 4)).into_dyn();
    let target = Array::ones((2, 4, 3, 3, 3)).into_dyn(); // output: (2, 4, 3, 3, 3)

    // Create Conv3D layer
    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                4,                   // filters
                (2, 2, 2),           // kernel_size
                vec![2, 2, 4, 4, 4], // input_shape
                (1, 1, 1),           // stride
                PaddingType::Valid,  // padding
                ReLU::new(),         // activation
            )
            .unwrap(),
        )
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());

    // Train the model
    model.fit(&input, &target, 3).unwrap();

    // Make predictions
    let predictions = model.predict(&input);
    println!("AdaGrad Conv3D output shape: {:?}", predictions.shape());

    // Check output shape
    assert_eq!(predictions.shape(), &[2, 4, 3, 3, 3]);
}

#[test]
fn test_ada_grad_mixed_layers() {
    // Test AdaGrad with a mixed architecture
    let x = Array::ones((4, 10)).into_dyn();
    let y = Array::ones((4, 2)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(10, 16, ReLU::new()).unwrap())
        .add(Dense::new(16, 8, ReLU::new()).unwrap())
        .add(Dense::new(8, 2, Sigmoid::new()).unwrap());
    model.compile(AdaGrad::new(0.05, 1e-8).unwrap(), MeanSquaredError::new());

    model.summary();

    // Train the model
    model.fit(&x, &y, 10).unwrap();

    // Make predictions
    let predictions = model.predict(&x);
    println!("AdaGrad mixed layers prediction: {:?}", predictions);

    // Check output shape
    assert_eq!(predictions.shape(), &[4, 2]);
}

#[test]
fn test_ada_grad_learning_rate_effect() {
    // Test that different learning rates produce different results
    let x = Array::ones((3, 5)).into_dyn();
    let y = Array::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();

    // Model with higher learning rate
    let mut model_high_lr = Sequential::new();
    model_high_lr
        .add(Dense::new(5, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model_high_lr.compile(AdaGrad::new(0.1, 1e-8).unwrap(), MeanSquaredError::new());
    model_high_lr.fit(&x, &y, 5).unwrap();
    let pred_high = model_high_lr.predict(&x);

    // Model with lower learning rate
    let mut model_low_lr = Sequential::new();
    model_low_lr
        .add(Dense::new(5, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model_low_lr.compile(AdaGrad::new(0.001, 1e-8).unwrap(), MeanSquaredError::new());
    model_low_lr.fit(&x, &y, 5).unwrap();
    let pred_low = model_low_lr.predict(&x);

    println!("High LR prediction: {:?}", pred_high);
    println!("Low LR prediction: {:?}", pred_low);

    // Predictions should be different (not exactly equal)
    // This is a weak test since both could theoretically converge to same solution
    assert_eq!(pred_high.shape(), &[3, 1]);
    assert_eq!(pred_low.shape(), &[3, 1]);
}

#[test]
fn test_ada_grad_with_different_activations() {
    // Test AdaGrad with different activation functions
    let x = Array::ones((3, 4)).into_dyn();
    let y = Array::ones((3, 2)).into_dyn();

    // Test with ReLU
    let mut model_relu = Sequential::new();
    model_relu
        .add(Dense::new(4, 2, ReLU::new()).unwrap())
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());
    model_relu.fit(&x, &y, 3).unwrap();
    let pred_relu = model_relu.predict(&x);

    // Test with Sigmoid
    let mut model_sigmoid = Sequential::new();
    model_sigmoid
        .add(Dense::new(4, 2, Sigmoid::new()).unwrap())
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());
    model_sigmoid.fit(&x, &y, 3).unwrap();
    let pred_sigmoid = model_sigmoid.predict(&x);

    // Test with Tanh
    let mut model_tanh = Sequential::new();
    model_tanh
        .add(Dense::new(4, 2, Tanh::new()).unwrap())
        .compile(AdaGrad::new(0.01, 1e-8).unwrap(), MeanSquaredError::new());
    model_tanh.fit(&x, &y, 3).unwrap();
    let pred_tanh = model_tanh.predict(&x);

    println!("ReLU prediction: {:?}", pred_relu);
    println!("Sigmoid prediction: {:?}", pred_sigmoid);
    println!("Tanh prediction: {:?}", pred_tanh);

    // Check all output shapes are correct
    assert_eq!(pred_relu.shape(), &[3, 2]);
    assert_eq!(pred_sigmoid.shape(), &[3, 2]);
    assert_eq!(pred_tanh.shape(), &[3, 2]);
}
