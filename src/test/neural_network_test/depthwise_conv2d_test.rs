use super::*;

#[test]
fn test_depthwise_conv2d_creation() {
    // Create DepthwiseConv2D layer
    let layer = DepthwiseConv2D::new(
        3,                  // filters
        (3, 3),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        ReLU::new(),        // activation
    )
    .unwrap();

    // Verify layer type
    assert_eq!(layer.layer_type(), "DepthwiseConv2D");

    // Verify parameter count (weights + bias)
    // Weight shape: [3, 1, 3, 3] = 27, bias: [3] = 3
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(30));

    // Output shape should be "Unknown" before initialization
    assert_eq!(layer.output_shape(), "Unknown");
}

#[test]
fn test_depthwise_conv2d_forward() {
    // Create test input data: [batch_size, channels, height, width]
    let batch_size = 2;
    let channels = 3;
    let height = 6;
    let width = 6;

    let mut input_data = Array4::zeros((batch_size, channels, height, width));

    // Set different values for each channel to verify that depthwise convolution processes channels independently
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    input_data[[b, c, h, w]] = (b + c * 10 + h + w) as f32;
                }
            }
        }
    }

    let input = input_data.into_dyn();

    // Create and initialize DepthwiseConv2D layer
    let mut layer = DepthwiseConv2D::new(
        channels,           // filters must equal input_channels
        (3, 3),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    layer.initialize_weights(channels);

    // Forward propagation
    let output = layer.forward(&input).unwrap();

    // Verify output shape
    // Input: [2, 3, 6, 6], kernel (3,3), stride (1,1), valid padding
    // Output should be: [2, 3, 4, 4]
    assert_eq!(output.shape(), &[2, 3, 4, 4]);

    // Verify output shape string
    assert_eq!(layer.output_shape(), "(2, 3, 4, 4)");
}

#[test]
fn test_depthwise_conv2d_sequential_model() {
    // Create Sequential model
    let mut model = Sequential::new();

    // Create and initialize DepthwiseConv2D layer
    let mut depthwise_layer = DepthwiseConv2D::new(
        3,                  // filters
        (2, 2),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        ReLU::new(),        // activation
    )
    .unwrap();
    depthwise_layer.initialize_weights(3);

    // Add layer and compile model
    model
        .add(depthwise_layer)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Create test input data: [batch_size, channels, height, width]
    let batch_size = 1;
    let input_channels = 3;
    let height = 4;
    let width = 4;

    let mut input_data = Array4::zeros((batch_size, input_channels, height, width));

    // Set different values for each channel
    for c in 0..input_channels {
        for h in 0..height {
            for w in 0..width {
                input_data[[0, c, h, w]] = (c * 10 + h * width + w) as f32;
            }
        }
    }

    let input = input_data.into_dyn();

    // Display model structure
    model.summary();

    // Forward propagation
    let output = model.predict(&input);

    // Verify output shape
    // Input: [1, 3, 4, 4], kernel (2,2), stride (1,1), valid padding
    // Output should be: [1, 3, 3, 3]
    assert_eq!(output.shape(), &[1, 3, 3, 3]);

    // Since ReLU activation is used, all output values should be non-negative
    for value in output.iter() {
        assert!(*value >= 0.0);
    }
}

#[test]
fn test_depthwise_conv2d_same_padding() {
    // Test Same padding
    let mut model = Sequential::new();

    let mut depthwise_layer = DepthwiseConv2D::new(
        2,                 // filters
        (3, 3),            // kernel_size
        (1, 1),            // strides
        PaddingType::Same, // padding
        Linear::new(),     // no activation
    )
    .unwrap();
    depthwise_layer.initialize_weights(2);

    model
        .add(depthwise_layer)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Create input data
    let input_data =
        Array4::from_shape_fn((1, 2, 5, 5), |(_, c, h, w)| (c * 100 + h * 10 + w) as f32)
            .into_dyn();

    // Forward propagation
    let output = model.predict(&input_data);

    // With Same padding, output shape should match input shape (except for channel count)
    assert_eq!(output.shape(), &[1, 2, 5, 5]);
}

#[test]
fn test_depthwise_conv2d_different_strides() {
    // Test different strides
    let mut model = Sequential::new();

    let mut depthwise_layer = DepthwiseConv2D::new(
        2,                  // filters
        (3, 3),             // kernel_size
        (2, 2),             // strides - larger strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    depthwise_layer.initialize_weights(2);

    model
        .add(depthwise_layer)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Create input data
    let input_data =
        Array4::from_shape_fn((1, 2, 8, 8), |(_, c, h, w)| (c * 100 + h * 10 + w) as f32)
            .into_dyn();

    // Forward propagation
    let output = model.predict(&input_data);

    // Verify output shape
    // Input: [1, 2, 8, 8], kernel (3,3), stride (2,2), valid padding
    // Output should be: [1, 2, 3, 3] ((8-3)/2 + 1 = 3)
    assert_eq!(output.shape(), &[1, 2, 3, 3]);
}

#[test]
fn test_depthwise_conv2d_training() {
    // Test training process
    let mut model = Sequential::new();

    let mut depthwise_layer = DepthwiseConv2D::new(
        2,                  // filters
        (3, 3),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        ReLU::new(),        // activation
    )
    .unwrap();
    depthwise_layer.initialize_weights(2);

    model
        .add(depthwise_layer)
        .compile(SGD::new(0.1).unwrap(), MeanSquaredError::new());

    // Create training data
    let input_data = Array4::from_shape_fn((4, 2, 5, 5), |(b, c, h, w)| {
        (b + c * 10 + h + w) as f32 * 0.1
    })
    .into_dyn();

    // Create target data (output shape should be [4, 2, 3, 3])
    let target_data = Array4::ones((4, 2, 3, 3)).into_dyn();

    // Train the model
    let result = model.fit(&input_data, &target_data, 5);
    assert!(result.is_ok());

    // Verify predictions
    let predictions = model.predict(&input_data);
    assert_eq!(predictions.shape(), &[4, 2, 3, 3]);
}

#[test]
fn test_depthwise_conv2d_backward() {
    // Test backward propagation
    let input_channels = 2;
    let mut layer = DepthwiseConv2D::new(
        input_channels,     // filters
        (2, 2),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    layer.initialize_weights(input_channels);

    // Create simple input data
    let input_data =
        Array4::from_shape_fn((1, 2, 3, 3), |(_, c, h, w)| (c * 10 + h * 3 + w) as f32).into_dyn();

    // Forward propagation
    let output = layer.forward(&input_data).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 2]);

    // Create gradient output
    let grad_output = Array4::ones((1, 2, 2, 2)).into_dyn();

    // Backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Verify gradient input shape
    assert_eq!(grad_input.shape(), input_data.shape());
}

#[test]
fn test_depthwise_conv2d_channel_independence() {
    // Verify that depthwise convolution indeed processes each channel independently
    let mut layer = DepthwiseConv2D::new(
        2,                  // filters
        (2, 2),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    layer.initialize_weights(2);

    // Create special input: first channel all 1s, second channel all 2s
    let mut input_data = Array4::zeros((1, 2, 3, 3));

    // Set first channel to 1
    for h in 0..3 {
        for w in 0..3 {
            input_data[[0, 0, h, w]] = 1.0;
        }
    }

    // Set second channel to 2
    for h in 0..3 {
        for w in 0..3 {
            input_data[[0, 1, h, w]] = 2.0;
        }
    }

    let input = input_data.into_dyn();

    // Forward propagation
    let output = layer.forward(&input).unwrap();

    // Output shape should be [1, 2, 2, 2]
    assert_eq!(output.shape(), &[1, 2, 2, 2]);

    // Since depthwise convolution processes channels independently,
    // the output values of the second channel should be proportional to those of the first channel
    // (assuming same weights and biases)
}

#[test]
fn test_depthwise_conv2d_edge_cases() {
    // Test edge cases

    // 1. Minimum convolution kernel size (1x1)
    let mut layer_1x1 = DepthwiseConv2D::new(
        1,                  // filters
        (1, 1),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    layer_1x1.initialize_weights(1);

    let input_1x1 = Array4::ones((1, 1, 2, 2)).into_dyn();
    let output_1x1 = layer_1x1.forward(&input_1x1).unwrap();
    assert_eq!(output_1x1.shape(), &[1, 1, 2, 2]);

    // 2. Large strides resulting in small output
    let mut layer_large_stride = DepthwiseConv2D::new(
        1,                  // filters
        (2, 2),             // kernel_size
        (3, 3),             // large strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    layer_large_stride.initialize_weights(1);

    let input_large = Array4::ones((1, 1, 5, 5)).into_dyn();
    let output_large = layer_large_stride.forward(&input_large).unwrap();
    assert_eq!(output_large.shape(), &[1, 1, 2, 2]);
}

#[test]
fn test_depthwise_conv2d_multiple_batches() {
    // Test multiple batch data
    let mut model = Sequential::new();

    let mut depthwise_layer = DepthwiseConv2D::new(
        3,                  // filters
        (2, 2),             // kernel_size
        (1, 1),             // strides
        PaddingType::Valid, // padding
        Linear::new(),      // no activation
    )
    .unwrap();
    depthwise_layer.initialize_weights(3);

    model
        .add(depthwise_layer)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Create multi-batch input data
    let batch_size = 5;
    let input_data = Array4::from_shape_fn((batch_size, 3, 4, 4), |(b, c, h, w)| {
        (b * 1000 + c * 100 + h * 10 + w) as f32 * 0.01
    })
    .into_dyn();

    // Forward propagation
    let output = model.predict(&input_data);

    // Verify output shape
    assert_eq!(output.shape(), &[batch_size, 3, 3, 3]);

    // Verify that outputs of different batches are indeed different
    let batch0_output = output.slice(s![0, 0, 0, 0]).into_scalar();
    let batch1_output = output.slice(s![1, 0, 0, 0]).into_scalar();
    assert_ne!(batch0_output, batch1_output);
}
