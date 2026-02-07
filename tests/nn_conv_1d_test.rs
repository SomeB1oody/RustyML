#![cfg(feature = "neural_network")]

use ndarray::Array3;
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::layer::activation_layer::relu::ReLU;
use rustyml::neural_network::layer::activation_layer::sigmoid::Sigmoid;
use rustyml::neural_network::layer::activation_layer::tanh::Tanh;
use rustyml::neural_network::layer::convolution_layer::PaddingType;
use rustyml::neural_network::layer::convolution_layer::conv_1d::Conv1D;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::rms_prop::RMSprop;
use rustyml::neural_network::optimizer::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_conv1d_sequential_with_sgd() {
    // Create a 3D input tensor: [batch_size, channels, length]
    let x = Array3::ones((2, 1, 10)).into_dyn();
    // Create a target tensor - assume output length is 8 (Valid padding)
    let y = Array3::ones((2, 3, 8)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(
            Conv1D::new(
                3,                  // filters
                3,                  // kernel_size
                vec![2, 1, 10],     // input_shape
                1,                  // stride
                PaddingType::Valid, // padding
                ReLU::new(),        // activation
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Print the model structure
    model.summary();

    // Train the model
    let result = model.fit(&x, &y, 3);
    assert!(result.is_ok());

    // Make predictions
    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[2, 3, 8]);

    // Verify that the predictions are non-negative (ReLU activation function)
    for value in prediction.iter() {
        assert!(*value >= 0.0);
    }
}

#[test]
fn test_conv1d_sequential_with_rmsprop() {
    // Create more complex training data
    let x = Array3::from_shape_fn((3, 2, 8), |(b, c, l)| {
        ((b * 2 + c * 3 + l) as f32).sin() * 0.5
    })
    .into_dyn();

    let y = Array3::zeros((3, 2, 6)).into_dyn(); // Valid padding output

    // Build the model
    let mut model = Sequential::new();
    model
        .add(
            Conv1D::new(
                2,                  // filters
                3,                  // kernel_size
                vec![3, 2, 8],      // input_shape
                1,                  // stride
                PaddingType::Valid, // padding
                Tanh::new(),        // activation
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    model.summary();

    // Train the model
    let result = model.fit(&x, &y, 4);
    assert!(result.is_ok());

    // Make predictions
    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[3, 2, 6]);

    // Verify that Tanh output is within [-1, 1]
    for value in prediction.iter() {
        assert!(*value >= -1.0 && *value <= 1.0);
    }
}

#[test]
fn test_conv1d_different_strides() {
    let x = Array3::ones((1, 1, 20)).into_dyn();

    // Test with different stride values
    let stride_2_conv = Conv1D::new(
        1,
        3,
        vec![1, 1, 20],
        2, // stride = 2
        PaddingType::Valid,
        ReLU::new(),
    )
    .unwrap();

    let mut model = Sequential::new();
    model
        .add(stride_2_conv)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[1, 1, 9]);
}

#[test]
fn test_conv1d_multiple_channels() {
    // Test multi-channel input
    let x = Array3::from_shape_fn((2, 3, 15), |(b, c, l)| (b + c + l) as f32 * 0.1).into_dyn();

    let y = Array3::ones((2, 5, 13)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            Conv1D::new(
                5,                  // filters
                3,                  // kernel_size
                vec![2, 3, 15],     // input_shape (3 channels)
                1,                  // stride
                PaddingType::Valid, // padding
                ReLU::new(),        // activation
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    model.summary();

    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());

    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[2, 5, 13]);
}

#[test]
fn test_conv1d_activation_functions() {
    let x = Array3::from_shape_fn((1, 1, 5), |(_, _, l)| {
        l as f32 - 2.0 // Generate negative values to test activation functions
    })
    .into_dyn();

    // Test ReLU activation function
    let mut relu_model = Sequential::new();
    relu_model
        .add(Conv1D::new(1, 3, vec![1, 1, 5], 1, PaddingType::Valid, ReLU::new()).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let relu_output = relu_model.predict(&x).unwrap();
    // ReLU output should be non-negative
    for value in relu_output.iter() {
        assert!(*value >= 0.0);
    }

    // Test Sigmoid activation function
    let mut sigmoid_model = Sequential::new();
    sigmoid_model
        .add(Conv1D::new(1, 3, vec![1, 1, 5], 1, PaddingType::Valid, Sigmoid::new()).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let sigmoid_output = sigmoid_model.predict(&x).unwrap();
    // Sigmoid output should be within [0, 1]
    for value in sigmoid_output.iter() {
        assert!(*value >= 0.0 && *value <= 1.0);
    }
}

#[test]
fn test_conv1d_parameter_count() {
    let conv1d = Conv1D::new(
        4,              // filters
        3,              // kernel_size
        vec![2, 2, 10], // input_shape (2 channels)
        1,
        PaddingType::Valid,
        ReLU::new(),
    )
    .unwrap();

    // Parameter count = weights + bias = (4 * 2 * 3) + (1 * 4) = 24 + 4 = 28
    assert_eq!(conv1d.param_count(), TrainingParameters::Trainable(28));
}
