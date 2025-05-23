use crate::neural_network::*;
use approx::assert_relative_eq;
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

#[test]
fn average_pooling_basic_test() {
    // Create a simple input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 input channels, each channel is 4x4 pixels
    let mut input_data = Array4::zeros((2, 3, 4, 4));

    // Set test data to make average pooling results predictable
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..4 {
                for j in 0..4 {
                    input_data[[b, c, i, j]] = (i + j) as f32;
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Test AveragePooling with Sequential model
    let mut model = Sequential::new();
    model
        .add(AveragePooling2D::new(
            (2, 2),           // Pooling window size
            (2, 2),           // Stride
            vec![2, 3, 4, 4], // Input shape
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Output shape should be [2, 3, 2, 2]
    let output = model.predict(&x);
    assert_eq!(output.shape(), &[2, 3, 2, 2]);

    // Verify correctness of pooling results
    // For a 2x2 window with stride 2, we expect the result to be the average of the elements in the window
    for b in 0..2 {
        for c in 0..3 {
            // First window (0,0), (0,1), (1,0), (1,1) -> average should be (0+1+1+2)/4 = 1.0
            assert_relative_eq!(output[[b, c, 0, 0]], 1.0);
            // Second window (0,2), (0,3), (1,2), (1,3) -> average should be (2+3+3+4)/4 = 3.0
            assert_relative_eq!(output[[b, c, 0, 1]], 3.0);
            // Third window (2,0), (2,1), (3,0), (3,1) -> average should be (2+3+3+4)/4 = 3.0
            assert_relative_eq!(output[[b, c, 1, 0]], 3.0);
            // Fourth window (2,2), (2,3), (3,2), (3,3) -> average should be (4+5+5+6)/4 = 5.0
            assert_relative_eq!(output[[b, c, 1, 1]], 5.0);
        }
    }
}

#[test]
fn average_pooling_non_even_input_test() {
    // Test input with non-even dimensions
    let mut input_data = Array4::zeros((2, 3, 5, 5));

    // Set test data
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..5 {
                for j in 0..5 {
                    input_data[[b, c, i, j]] = (i + j) as f32;
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Pooling window size (3,3), stride (2,2)
    let mut model = Sequential::new();
    model
        .add(AveragePooling2D::new(
            (3, 3),           // Pooling window size
            (2, 2),           // Stride
            vec![2, 3, 5, 5], // Input shape
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Output shape should be [2, 3, 2, 2]
    // (5-3)/2+1 = 2
    let output = model.predict(&x);
    assert_eq!(output.shape(), &[2, 3, 2, 2]);

    // Verify the average pooling result for the first window
    // Window (0,0) to (2,2) contains 9 elements: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
    // Corresponding values: 0+1+2+1+2+3+2+3+4 = 18/9 = 2.0
    assert_relative_eq!(output[[0, 0, 0, 0]], 2.0);
}

#[test]
fn average_pooling_different_strides_test() {
    // Test different stride cases
    let mut input_data = Array4::zeros((1, 1, 6, 6));

    // Set test data - using increasing values
    for i in 0..6 {
        for j in 0..6 {
            input_data[[0, 0, i, j]] = (i * 6 + j) as f32;
        }
    }

    let x = input_data.clone().into_dyn();

    // Pooling with stride (1,1)
    let mut model = Sequential::new();
    model
        .add(AveragePooling2D::new(
            (2, 2),           // Pooling window size
            (1, 1),           // Stride
            vec![1, 1, 6, 6], // Input shape
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Output shape should be [1, 1, 5, 5]
    // (6-2)/1+1 = 5
    let output = model.predict(&x);
    assert_eq!(output.shape(), &[1, 1, 5, 5]);

    // Check the average of the first window
    // Window (0,0),(0,1),(1,0),(1,1) corresponding values: 0+1+6+7=14/4=3.5
    assert_relative_eq!(output[[0, 0, 0, 0]], 3.5);
}

#[test]
fn average_pooling_backprop_test() {
    // Test backpropagation
    let input_data = Array4::ones((2, 2, 4, 4)).into_dyn();
    let target_data = Array4::ones((2, 2, 2, 2)).into_dyn();

    // Create model and train
    let mut model = Sequential::new();
    model
        .add(AveragePooling2D::new(
            (2, 2),           // Pooling window size
            (2, 2),           // Stride
            vec![2, 2, 4, 4], // Input shape
        ))
        .compile(RMSprop::new(0.01, 0.9, 1e-8), MeanSquaredError::new());

    // Train the model
    let result = model.fit(&input_data, &target_data, 5);
    assert!(result.is_ok(), "Model training failed");

    // Verify that predictions after training are close to target values
    let prediction = model.predict(&input_data);
    for b in 0..2 {
        for c in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(
                        prediction[[b, c, i, j]],
                        target_data[[b, c, i, j]],
                        epsilon = 0.1
                    );
                }
            }
        }
    }
}

#[test]
fn test_global_max_pooling_forward() {
    // Create a test input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 channels, each channel 4x4 pixels
    let mut input_data = Array::zeros(IxDyn(&[2, 3, 4, 4]));

    // Set some values for the first sample's first channel
    input_data[[0, 0, 1, 2]] = 5.0; // Max value
    input_data[[0, 0, 0, 0]] = 1.0;

    // Set some values for the first sample's second channel
    input_data[[0, 1, 3, 3]] = 7.0; // Max value
    input_data[[0, 1, 2, 2]] = 3.0;

    // Set some values for the first sample's third channel
    input_data[[0, 2, 0, 3]] = 9.0; // Max value
    input_data[[0, 2, 1, 1]] = 4.0;

    // Set some values for the second sample
    input_data[[1, 0, 2, 1]] = 6.0; // Max value
    input_data[[1, 1, 1, 0]] = 8.0; // Max value
    input_data[[1, 2, 3, 2]] = 10.0; // Max value

    // Create a Sequential model containing a GlobalMaxPooling2D layer
    let mut model = Sequential::new();
    model.add(GlobalMaxPooling2D::new());

    // Forward propagation
    let output = model.predict(&input_data);

    // Check output shape - should be [2, 3]
    assert_eq!(output.shape(), &[2, 3]);

    // Verify that the maximum value of each channel is correctly extracted
    assert_relative_eq!(output[[0, 0]], 5.0);
    assert_relative_eq!(output[[0, 1]], 7.0);
    assert_relative_eq!(output[[0, 2]], 9.0);
    assert_relative_eq!(output[[1, 0]], 6.0);
    assert_relative_eq!(output[[1, 1]], 8.0);
    assert_relative_eq!(output[[1, 2]], 10.0);
}

#[test]
fn test_global_max_pooling_backward() {
    // Create a test input tensor: [batch_size, channels, height, width]
    let mut input_data = Array::zeros(IxDyn(&[2, 2, 3, 3]));

    // Set some values
    input_data[[0, 0, 1, 1]] = 5.0; // Max value
    input_data[[0, 1, 0, 2]] = 7.0; // Max value
    input_data[[1, 0, 2, 0]] = 6.0; // Max value
    input_data[[1, 1, 2, 2]] = 8.0; // Max value

    // Create GlobalMaxPooling2D layer
    let mut pool_layer = GlobalMaxPooling2D::new();

    // Forward propagation
    let _forward_output = pool_layer.forward(&input_data);

    // Create output gradient
    let grad_output = Array::from_elem(IxDyn(&[2, 2]), 1.0);

    // Backward propagation
    let grad_input = pool_layer.backward(&grad_output).unwrap();

    // Check backward propagation gradient shape
    assert_eq!(grad_input.shape(), input_data.shape());

    // Verify that gradient is 1.0 only at max value positions, 0.0 elsewhere
    assert_relative_eq!(grad_input[[0, 0, 1, 1]], 1.0);
    assert_relative_eq!(grad_input[[0, 1, 0, 2]], 1.0);
    assert_relative_eq!(grad_input[[1, 0, 2, 0]], 1.0);
    assert_relative_eq!(grad_input[[1, 1, 2, 2]], 1.0);

    // Check that gradient is 0.0 at some non-maximum positions
    assert_relative_eq!(grad_input[[0, 0, 0, 0]], 0.0);
    assert_relative_eq!(grad_input[[0, 1, 1, 1]], 0.0);
    assert_relative_eq!(grad_input[[1, 0, 0, 0]], 0.0);
    assert_relative_eq!(grad_input[[1, 1, 0, 0]], 0.0);

    // Calculate total gradient sum, should equal the number of elements in input gradient tensor
    let total_grad = grad_input.iter().filter(|&&x| x > 0.0).count();
    assert_eq!(total_grad, 4); // Only one maximum value for each channel of each sample
}

#[test]
fn test_global_max_pooling_in_sequential() {
    // Create a Sequential model containing multiple layers
    let mut model = Sequential::new();

    // Add a GlobalMaxPooling2D layer
    model.add(GlobalMaxPooling2D::new());

    // Create a test input tensor: [batch_size, channels, height, width]
    let input_data = Array::from_elem(IxDyn(&[3, 4, 5, 5]), 1.0);

    // Forward propagation
    let output = model.predict(&input_data);

    // Check output shape - should be [3, 4]
    assert_eq!(output.shape(), &[3, 4]);

    // Since all input values are 1.0, all output values should also be 1.0
    for b in 0..3 {
        for c in 0..4 {
            assert_relative_eq!(output[[b, c]], 1.0);
        }
    }
}

#[test]
fn test_global_average_pooling_2d() {
    // Create test input data: shape is [batch_size, channels, height, width]
    let input = Array::from_elem(IxDyn(&[2, 3, 4, 4]), 1.0);

    // Create Sequential model
    let mut model = Sequential::new();

    // Add GlobalAveragePooling2D layer
    model
        .add(GlobalAveragePooling2D::new())
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Forward propagation
    let output = model.predict(&input);

    // Check output shape - should be [2, 3]
    assert_eq!(output.shape(), &[2, 3]);

    // Since all input values are 1.0, all output values should also be 1.0
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c]], 1.0);
        }
    }

    // Test different input values
    let mut varied_input = Array::from_elem(IxDyn(&[1, 2, 2, 2]), 0.0);

    // First channel: [[1, 2], [3, 4]], average should be 2.5
    varied_input[[0, 0, 0, 0]] = 1.0;
    varied_input[[0, 0, 0, 1]] = 2.0;
    varied_input[[0, 0, 1, 0]] = 3.0;
    varied_input[[0, 0, 1, 1]] = 4.0;

    // Second channel: [[5, 6], [7, 8]], average should be 6.5
    varied_input[[0, 1, 0, 0]] = 5.0;
    varied_input[[0, 1, 0, 1]] = 6.0;
    varied_input[[0, 1, 1, 0]] = 7.0;
    varied_input[[0, 1, 1, 1]] = 8.0;

    // Reset model and predict
    let mut model = Sequential::new();
    model.add(GlobalAveragePooling2D::new());

    let varied_output = model.predict(&varied_input);

    // Check output shape - should be [1, 2]
    assert_eq!(varied_output.shape(), &[1, 2]);

    // Check average values
    assert_relative_eq!(varied_output[[0, 0]], 2.5);
    assert_relative_eq!(varied_output[[0, 1]], 6.5);
}

#[test]
fn test_average_pooling_1d_shape() {
    // Create a simple input tensor: [batch_size, channels, length]
    // Batch size=2, 3 input channels, each channel has 10 elements
    let input_data = Array3::<f32>::zeros((2, 3, 10)).into_dyn();

    // Test different pooling window sizes and strides
    let test_cases = vec![
        // (pool_size, stride, expected_output_length)
        (2, 2, 5), // Pool size=2, stride=2, output length=(10-2)/2+1=5
        (3, 2, 4), // Pool size=3, stride=2, output length=(10-3)/2+1=4
        (4, 3, 3), // Pool size=4, stride=3, output length=(10-4)/3+1=3
        (5, 5, 2), // Pool size=5, stride=5, output length=(10-5)/5+1=2
    ];

    for (pool_size, stride, expected_length) in test_cases {
        let mut layer = AveragePooling1D::new(pool_size, stride, vec![2, 3, 10]);

        let output = layer.forward(&input_data);

        // Verify output shape
        assert_eq!(output.shape(), &[2, 3, expected_length]);
    }
}

#[test]
fn test_average_pooling_1d_forward() {
    // Create an input tensor with predefined values
    let mut input_data = Array3::<f32>::zeros((2, 3, 8));

    // First batch: use increasing values
    for c in 0..3 {
        for i in 0..8 {
            input_data[[0, c, i]] = (i as f32) + (c as f32 * 0.1);
        }
    }

    // Second batch: use more complex pattern
    for c in 0..3 {
        for i in 0..8 {
            // Generate data using different patterns, such as sine waves or alternating values
            input_data[[1, c, i]] = if i % 2 == 0 {
                (i as f32) * 1.5
            } else {
                (i as f32) * 0.5
            };
        }
    }

    let input = input_data.clone().into_dyn();

    // Create pooling layer, pool size=2, stride=2
    let mut layer = AveragePooling1D::new(2, 2, vec![2, 3, 8]);

    // Perform forward propagation
    let output = layer.forward(&input);

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Verify pooling results for batch 1
    for c in 0..3 {
        let c_offset = c as f32 * 0.1;
        // First window (0+c_offset, 1+c_offset) -> average should be (0+1)/2 + c_offset = 0.5 + c_offset
        assert_relative_eq!(output[[0, c, 0]], 0.5 + c_offset);
        // Second window (2+c_offset, 3+c_offset) -> average should be (2+3)/2 + c_offset = 2.5 + c_offset
        assert_relative_eq!(output[[0, c, 1]], 2.5 + c_offset);
        // Third window (4+c_offset, 5+c_offset) -> average should be (4+5)/2 + c_offset = 4.5 + c_offset
        assert_relative_eq!(output[[0, c, 2]], 4.5 + c_offset);
        // Fourth window (6+c_offset, 7+c_offset) -> average should be (6+7)/2 + c_offset = 6.5 + c_offset
        assert_relative_eq!(output[[0, c, 3]], 6.5 + c_offset);
    }

    // Verify pooling results for batch 2
    for c in 0..3 {
        // First window (0*1.5, 1*0.5) -> average should be (0 + 0.5)/2 = 0.25
        assert_relative_eq!(output[[1, c, 0]], 0.25);
        // Second window (2*1.5, 3*0.5) -> average should be (3 + 1.5)/2 = 2.25
        assert_relative_eq!(output[[1, c, 1]], 2.25);
        // Third window (4*1.5, 5*0.5) -> average should be (6 + 2.5)/2 = 4.25
        assert_relative_eq!(output[[1, c, 2]], 4.25);
        // Fourth window (6*1.5, 7*0.5) -> average should be (9 + 3.5)/2 = 6.25
        assert_relative_eq!(output[[1, c, 3]], 6.25);
    }
}

#[test]
fn test_average_pooling_1d_backward() {
    // Create input tensor
    let mut input_data = Array3::<f32>::zeros((1, 1, 4));
    for i in 0..4 {
        input_data[[0, 0, i]] = i as f32;
    }
    let input = input_data.clone().into_dyn();

    // Create pooling layer, pool size=2, stride=1
    let mut layer = AveragePooling1D::new(2, 1, vec![1, 1, 4]);

    // Perform forward propagation
    let output = layer.forward(&input);

    // Verify output shape and values
    assert_eq!(output.shape(), &[1, 1, 3]);
    assert_relative_eq!(output[[0, 0, 0]], 0.5); // (0+1)/2
    assert_relative_eq!(output[[0, 0, 1]], 1.5); // (1+2)/2
    assert_relative_eq!(output[[0, 0, 2]], 2.5); // (2+3)/2

    // Create upstream gradient - all values are 1.0
    let grad_output = Array3::<f32>::ones((1, 1, 3)).into_dyn();

    // Perform backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Verify gradient shape
    assert_eq!(grad_input.shape(), &[1, 1, 4]);

    // Verify gradient values
    // Each input element is affected by the gradients of all windows it belongs to
    // For pool size=2, stride=1:
    // Element 0 is in 1 window (window 0)
    // Element 1 is in 2 windows (windows 0,1)
    // Element 2 is in 2 windows (windows 1,2)
    // Element 3 is in 1 window (window 2)
    // Gradient contribution is 1.0/2.0 = 0.5 (average gradient per window)
    assert_relative_eq!(grad_input[[0, 0, 0]], 0.5); // In 1 window
    assert_relative_eq!(grad_input[[0, 0, 1]], 1.0); // In 2 windows
    assert_relative_eq!(grad_input[[0, 0, 2]], 1.0); // In 2 windows
    assert_relative_eq!(grad_input[[0, 0, 3]], 0.5); // In 1 window
}

#[test]
fn test_average_pooling_1d_with_sequential() {
    // Create input tensor
    let mut input_data = Array3::<f32>::zeros((2, 3, 8));

    // Set test data
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..8 {
                input_data[[b, c, i]] = i as f32;
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Create Sequential model
    let mut model = Sequential::new();
    model
        .add(AveragePooling1D::new(
            2,             // Pool size
            2,             // Stride
            vec![2, 3, 8], // Input shape
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Perform prediction
    let output = model.predict(&x);

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Verify output values
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c, 0]], 0.5);
            assert_relative_eq!(output[[b, c, 1]], 2.5);
            assert_relative_eq!(output[[b, c, 2]], 4.5);
            assert_relative_eq!(output[[b, c, 3]], 6.5);
        }
    }
}

#[test]
fn test_average_pooling_1d_odd_window_size() {
    // Create input tensor
    let mut input_data = Array3::<f32>::zeros((1, 1, 5));
    for i in 0..5 {
        input_data[[0, 0, i]] = i as f32;
    }
    let input = input_data.clone().into_dyn();

    // Create pooling layer, pool size=3, stride=1
    let mut layer = AveragePooling1D::new(3, 1, vec![1, 1, 5]);

    // Perform forward propagation
    let output = layer.forward(&input);

    // Verify output shape
    assert_eq!(output.shape(), &[1, 1, 3]);

    // Verify output values
    assert_relative_eq!(output[[0, 0, 0]], 1.0); // (0+1+2)/3
    assert_relative_eq!(output[[0, 0, 1]], 2.0); // (1+2+3)/3
    assert_relative_eq!(output[[0, 0, 2]], 3.0); // (2+3+4)/3
}

#[test]
fn test_layer_type_and_output_shape() {
    let layer = AveragePooling1D::new(2, 2, vec![1, 3, 10]);

    // Test layer type
    assert_eq!(layer.layer_type(), "AveragePooling1D");

    // Test output shape
    let expected_shape = "[1, 3, 5]"; // (10-2)/2+1 = 5
    assert_eq!(layer.output_shape(), expected_shape);
}
