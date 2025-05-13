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

#[test]
fn test_get_weights() {
    // Create and compile the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Activation::ReLU))
        .add(SimpleRNN::new(3, 2, Activation::Tanh));
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Get all layer weights
    let weights = model.get_weights();

    // Examine the weights of the first layer (Dense layer)
    if let LayerWeight::Dense(dense_weights) = &weights[0] {
        println!("Dense layer weights: {:?}", dense_weights.weight);
        println!("Dense layer bias: {:?}", dense_weights.bias);
    }

    // Examine the weights of the second layer (SimpleRNN layer)
    if let LayerWeight::SimpleRNN(rnn_weights) = &weights[1] {
        println!("SimpleRNN layer input weights: {:?}", rnn_weights.kernel);
        println!(
            "SimpleRNN layer recurrent weights: {:?}",
            rnn_weights.recurrent_kernel
        );
        println!("SimpleRNN layer bias: {:?}", rnn_weights.bias);
    }
}

#[test]
fn conv2d_test() {
    // Create a simple 4D input tensor: [batch_size, channels, height, width]
    // Batch size=2, 1 input channel, 5x5 pixels
    let x = Array4::ones((2, 1, 5, 5)).into_dyn();

    // Create target tensor - assuming we'll have 3 filters with output size 3x3
    let y = Array4::ones((2, 3, 3, 3)).into_dyn();

    // Build model: add a Conv2D layer with 3 filters and 3x3 kernel
    let mut model = Sequential::new();
    model
        .add(Conv2D::new(
            3,                      // Number of filters
            (3, 3),                 // Kernel size
            vec![2, 1, 5, 5],       // Input shape
            (1, 1),                 // Stride
            PaddingType::Valid,     // No padding
            Some(Activation::ReLU), // ReLU activation function
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model (run a few epochs)
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Convolution layer prediction results: {:?}", prediction);

    // Check if output shape is correct - should be [2, 3, 3, 3]
    assert_eq!(prediction.shape(), &[2, 3, 3, 3]);

    // Create another convolution layer with different padding strategy and stride
    let mut model2 = Sequential::new();
    model2
        .add(Conv2D::new(
            2,                         // Number of filters
            (3, 3),                    // Kernel size
            vec![2, 1, 5, 5],          // Input shape
            (2, 2),                    // Larger stride
            PaddingType::Same,         // Same padding
            Some(Activation::Sigmoid), // Sigmoid activation function
        ))
        .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Create appropriate target tensor - for Same padding and stride (2,2), output size should be 3x3
    let y2 = Array4::ones((2, 2, 3, 3)).into_dyn();

    // Print model structure
    model2.summary();

    // Train the model
    model2.fit(&x, &y2, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction2 = model2.predict(&x);
    println!(
        "Convolution layer prediction results (Same padding, stride 2): {:?}",
        prediction2
    );

    // Check if output shape is correct
    assert_eq!(prediction2.shape(), &[2, 2, 3, 3]);
}

#[test]
fn max_pooling_2d_test() {
    // Create a simple 4D input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 input channels, 6x6 pixels
    let mut input_data = Array4::zeros((2, 3, 6, 6));

    // Set some specific values so we can predict the max pooling result
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..6 {
                for j in 0..6 {
                    // Create input data with an easily observable pattern
                    input_data[[b, c, i, j]] = (i * j) as f32 + b as f32 * 0.1 + c as f32 * 0.01;
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Create MaxPooling2D layer with pool size (2,2) and stride (2,2)
    let mut pool_layer = MaxPooling2D::new(
        (2, 2),           // Pool window size
        vec![2, 3, 6, 6], // Input shape
        Some((2, 2)),     // Stride
    );

    // Perform forward propagation
    let output = pool_layer.forward(&x);

    // Check if output shape is correct - should be [2, 3, 3, 3]
    assert_eq!(output.shape(), &[2, 3, 3, 3]);

    // Manually check some pooling results
    // For input region [[0,0], [0,1], [1,0], [1,1]], the max value should be at [1,1]
    assert_eq!(output[[0, 0, 0, 0]], input_data[[0, 0, 1, 1]]);

    // Test backward propagation
    let mut grad_output = ArrayD::zeros(output.dim());
    grad_output.fill(1.0); // Set uniform gradient

    let grad_input = pool_layer.backward(&grad_output).unwrap();

    // Check if gradient shape is correct
    assert_eq!(grad_input.shape(), x.shape());

    // Gradients should only have non-zero values at maximum value positions
    let nonzero_count = grad_input.iter().filter(|&&x| x > 0.0).count();

    // Output size is 2x3x3x3=54 elements, so there should be 54 non-zero gradients
    assert_eq!(nonzero_count, 54);

    // Test using MaxPooling2D in a model
    let mut model = Sequential::new();
    model
        .add(MaxPooling2D::new(
            (2, 2),           // Pool window size
            vec![2, 3, 6, 6], // Input shape
            None,             // Use default stride (2,2)
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Create target tensor - corresponding to the pooled shape
    let y = Array4::ones((2, 3, 3, 3)).into_dyn();

    // Print model structure
    model.summary();

    // Train the model (run a few epochs)
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("MaxPooling2D prediction results: {:?}", prediction);

    // Check if output shape is correct
    assert_eq!(prediction.shape(), &[2, 3, 3, 3]);

    // Test different stride cases
    let mut model2 = Sequential::new();
    model2
        .add(MaxPooling2D::new(
            (3, 3),           // Larger pool window size
            vec![2, 3, 6, 6], // Input shape
            Some((1, 1)),     // Smaller stride
        ))
        .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Print second model structure
    model2.summary();

    // Calculate expected output shape
    let expected_shape = [2, 3, 4, 4]; // Using 3x3 window and 1x1 stride

    // Use predict to confirm output shape
    let prediction2 = model2.predict(&x);
    println!(
        "MaxPooling2D prediction results (pool_size=(3,3), stride=(1,1)): {:?}",
        prediction2
    );

    // Check if output shape matches expectation
    assert_eq!(prediction2.shape(), &expected_shape);
}

#[test]
fn max_pooling_2d_edge_cases() {
    // Test edge cases: input size equals pool window size
    let input_data = Array4::zeros((1, 1, 2, 2)).into_dyn();

    let mut pool_layer = MaxPooling2D::new(
        (2, 2),           // Pool window equals input size
        vec![1, 1, 2, 2], // Input shape
        None,             // Default stride
    );

    let output = pool_layer.forward(&input_data);
    // Output should be [1, 1, 1, 1]
    assert_eq!(output.shape(), &[1, 1, 1, 1]);

    // Test asymmetric pool window and stride
    let input_data = Array4::zeros((1, 2, 5, 4)).into_dyn();

    let mut pool_layer = MaxPooling2D::new(
        (3, 2),           // Asymmetric pool window
        vec![1, 2, 5, 4], // Input shape
        Some((2, 1)),     // Asymmetric stride
    );

    let output = pool_layer.forward(&input_data);
    // Output should be [1, 2, 2, 3]
    assert_eq!(output.shape(), &[1, 2, 2, 3]);

    // Test parameter count - pooling layer has no trainable parameters
    assert_eq!(pool_layer.param_count(), 0);
}

#[test]
fn test_flatten_in_sequential() {
    // Create a simple 4D input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 channels, each 4x4 pixels
    let x = Array4::ones((2, 3, 4, 4)).into_dyn();

    // Create target tensor - assuming final output is a single value
    let y = Array2::ones((2, 1)).into_dyn();

    // Build model: convolution layer -> max pooling layer -> flatten layer -> fully connected layer
    let mut model = Sequential::new();
    model
        .add(Conv2D::new(
            6,                      // Number of filters
            (3, 3),                 // Kernel size
            vec![2, 3, 4, 4],       // Input shape
            (1, 1),                 // Stride
            PaddingType::Valid,     // No padding
            Some(Activation::ReLU), // ReLU activation function
        ))
        .add(MaxPooling2D::new(
            (2, 2),           // Pooling window size
            vec![2, 6, 2, 2], // Input shape (after convolution)
            None,             // Use default stride
        ))
        .add(Flatten::new(vec![2, 6, 1, 1])) // Flatten layer
        .add(Dense::new(6, 1, Activation::Sigmoid)) // Fully connected layer
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("CNN+Flatten prediction results: {:?}", prediction);

    // Check if output shape is correct
    assert_eq!(prediction.shape(), &[2, 1]);
}

#[test]
fn test_flatten_only_model() {
    // Create 4D input tensor
    let x = Array4::from_shape_fn((2, 3, 4, 4), |(b, c, h, w)| {
        (b * 100 + c * 10 + h + w) as f32 / 10.0
    })
    .into_dyn();

    // Target tensor - already flattened format
    let flattened_size = 3 * 4 * 4; // channels * height * width

    // Create model with only Flatten layer
    let mut model = Sequential::new();
    model
        .add(Flatten::new(vec![2, 3, 4, 4]))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Perform forward propagation
    let output = model.predict(&x);

    // Check output shape
    assert_eq!(output.shape(), &[2, flattened_size]);
}

#[test]
fn test_multiple_layers_with_flatten() {
    // Create 4D input tensor
    let x = Array4::ones((2, 1, 8, 8)).into_dyn();

    // Create target tensor
    let y = Array2::ones((2, 10)).into_dyn();

    // Build model with multiple layers including two convolution layers, one pooling layer,
    // one flatten layer and two dense layers
    let mut model = Sequential::new();
    model
        .add(Conv2D::new(
            8,                      // 8 filters
            (3, 3),                 // 3x3 kernel
            vec![2, 1, 8, 8],       // Input shape
            (1, 1),                 // Stride
            PaddingType::Same,      // Same padding
            Some(Activation::ReLU), // ReLU activation
        ))
        .add(Conv2D::new(
            16,                     // 16 filters
            (3, 3),                 // 3x3 kernel
            vec![2, 8, 8, 8],       // Input shape (after first convolution)
            (1, 1),                 // Stride
            PaddingType::Valid,     // No padding
            Some(Activation::ReLU), // ReLU activation
        ))
        .add(MaxPooling2D::new(
            (2, 2),            // 2x2 pooling window
            vec![2, 16, 6, 6], // Input shape (after second convolution)
            Some((2, 2)),      // 2x2 stride
        ))
        .add(Flatten::new(vec![2, 16, 3, 3])) // Flatten layer
        .add(Dense::new(16 * 3 * 3, 20, Activation::ReLU)) // Fully connected layer
        .add(Dense::new(20, 10, Activation::Softmax)) // Output layer
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8),
            CategoricalCrossEntropy::new(),
        );

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Forward propagation prediction
    let prediction = model.predict(&x);
    println!(
        "Complex CNN with Flatten prediction results: {:?}",
        prediction
    );

    // Check output shape
    assert_eq!(prediction.shape(), &[2, 10]);
}
