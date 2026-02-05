use ndarray::Array5;
use rustyml::neural_network::layer::ReLU;
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::layer::activation_layer::sigmoid::Sigmoid;
use rustyml::neural_network::layer::activation_layer::tanh::Tanh;
use rustyml::neural_network::layer::convolution_layer::PaddingType;
use rustyml::neural_network::layer::convolution_layer::conv_3d::Conv3D;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::adam::Adam;
use rustyml::neural_network::optimizer::rms_prop::RMSprop;
use rustyml::neural_network::optimizer::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_conv3d_sequential_with_sgd() {
    // Create a 5D input tensor: [batch_size, channels, depth, height, width]
    let x = Array5::ones((2, 1, 8, 8, 8)).into_dyn();
    // Create a target tensor - assume output size is 6x6x6 (Valid padding)
    let y = Array5::ones((2, 3, 6, 6, 6)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                3,                   // filters
                (3, 3, 3),           // kernel_size
                vec![2, 1, 8, 8, 8], // input_shape
                (1, 1, 1),           // stride
                PaddingType::Valid,  // padding
                ReLU::new(),         // activation
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
    assert_eq!(prediction.shape(), &[2, 3, 6, 6, 6]);

    // Verify that the predictions are non-negative (ReLU activation function)
    for value in prediction.iter() {
        assert!(*value >= 0.0);
    }
}

#[test]
fn test_conv3d_sequential_with_rmsprop() {
    // Create more complex training data
    let x = Array5::from_shape_fn((2, 2, 6, 6, 6), |(b, c, d, h, w)| {
        ((b * 2 + c * 3 + d + h + w) as f32).sin() * 0.5
    })
    .into_dyn();

    let y = Array5::zeros((2, 2, 4, 4, 4)).into_dyn(); // Valid padding output

    // Build the model
    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                2,                   // filters
                (3, 3, 3),           // kernel_size
                vec![2, 2, 6, 6, 6], // input_shape
                (1, 1, 1),           // stride
                PaddingType::Valid,  // padding
                Tanh::new(),         // activation
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
    assert_eq!(prediction.shape(), &[2, 2, 4, 4, 4]);

    // Verify that Tanh output is within [-1, 1]
    for value in prediction.iter() {
        assert!(*value >= -1.0 && *value <= 1.0);
    }
}

#[test]
fn test_conv3d_different_strides() {
    let x = Array5::ones((1, 1, 10, 10, 10)).into_dyn();

    // Test with different stride values
    let stride_2_conv = Conv3D::new(
        1,
        (3, 3, 3),
        vec![1, 1, 10, 10, 10],
        (2, 2, 2), // stride = 2
        PaddingType::Valid,
        ReLU::new(),
    )
    .unwrap();

    let mut model = Sequential::new();
    model
        .add(stride_2_conv)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[1, 1, 4, 4, 4]);
}

#[test]
fn test_conv3d_multiple_channels() {
    // Test multi-channel input
    let x = Array5::from_shape_fn((2, 3, 6, 6, 6), |(b, c, d, h, w)| {
        (b + c + d + h + w) as f32 * 0.1
    })
    .into_dyn();

    let y = Array5::ones((2, 5, 4, 4, 4)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                5,                   // filters
                (3, 3, 3),           // kernel_size
                vec![2, 3, 6, 6, 6], // input_shape (3 channels)
                (1, 1, 1),           // stride
                PaddingType::Valid,  // padding
                ReLU::new(),         // activation
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    model.summary();

    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());

    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[2, 5, 4, 4, 4]);
}

#[test]
fn test_conv3d_activation_functions() {
    let x = Array5::from_shape_fn((1, 1, 4, 4, 4), |(_, _, d, _, _)| {
        d as f32 - 2.0 // Generate negative values to test activation functions
    })
    .into_dyn();

    // Test ReLU activation function
    let mut relu_model = Sequential::new();
    relu_model
        .add(
            Conv3D::new(
                1,
                (2, 2, 2),
                vec![1, 1, 4, 4, 4],
                (1, 1, 1),
                PaddingType::Valid,
                ReLU::new(),
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let relu_output = relu_model.predict(&x).unwrap();
    // ReLU output should be non-negative
    for value in relu_output.iter() {
        assert!(*value >= 0.0);
    }

    // Test Sigmoid activation function
    let mut sigmoid_model = Sequential::new();
    sigmoid_model
        .add(
            Conv3D::new(
                1,
                (2, 2, 2),
                vec![1, 1, 4, 4, 4],
                (1, 1, 1),
                PaddingType::Valid,
                Sigmoid::new(),
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let sigmoid_output = sigmoid_model.predict(&x).unwrap();
    // Sigmoid output should be within [0, 1]
    for value in sigmoid_output.iter() {
        assert!(*value >= 0.0 && *value <= 1.0);
    }
}

#[test]
fn test_conv3d_parameter_count() {
    let conv3d = Conv3D::new(
        4,                   // filters
        (3, 3, 3),           // kernel_size
        vec![2, 2, 5, 5, 5], // input_shape (2 channels)
        (1, 1, 1),
        PaddingType::Valid,
        ReLU::new(),
    )
    .unwrap();

    // Parameter count = weights + bias = (4 * 2 * 3 * 3 * 3) + (1 * 4) = 216 + 4 = 220
    assert_eq!(conv3d.param_count(), TrainingParameters::Trainable(220));
}

#[test]
fn test_conv3d_same_padding() {
    let x = Array5::ones((1, 1, 8, 8, 8)).into_dyn();

    // Test with Same padding
    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                2,
                (3, 3, 3),
                vec![1, 1, 8, 8, 8],
                (1, 1, 1),
                PaddingType::Same,
                ReLU::new(),
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let prediction = model.predict(&x).unwrap();
    // With Same padding and stride 1, output should have the same spatial dimensions
    assert_eq!(prediction.shape(), &[1, 2, 8, 8, 8]);
}

#[test]
fn test_conv3d_with_adam() {
    // Test Conv3D with Adam optimizer
    let x = Array5::from_shape_fn((2, 1, 6, 6, 6), |(b, _, d, h, w)| {
        ((b + d + h + w) as f32) * 0.1
    })
    .into_dyn();

    let y = Array5::ones((2, 2, 4, 4, 4)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                2,
                (3, 3, 3),
                vec![2, 1, 6, 6, 6],
                (1, 1, 1),
                PaddingType::Valid,
                ReLU::new(),
            )
            .unwrap(),
        )
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    model.summary();

    let result = model.fit(&x, &y, 3);
    assert!(result.is_ok());

    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[2, 2, 4, 4, 4]);
}

#[test]
fn test_conv3d_asymmetric_stride() {
    // Test with asymmetric strides
    let x = Array5::ones((1, 1, 12, 8, 10)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            Conv3D::new(
                1,
                (3, 3, 3),
                vec![1, 1, 12, 8, 10],
                (2, 1, 2), // Different strides for each dimension
                PaddingType::Valid,
                ReLU::new(),
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let prediction = model.predict(&x).unwrap();
    // Output: depth = (12-3)/2+1 = 5, height = (8-3)/1+1 = 6, width = (10-3)/2+1 = 4
    assert_eq!(prediction.shape(), &[1, 1, 5, 6, 4]);
}
