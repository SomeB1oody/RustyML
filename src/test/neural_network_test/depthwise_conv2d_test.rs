use super::*;

#[test]
fn test_depthwise_conv2d_basic() {
    // Create test input data: [batch_size, channels, height, width]
    let batch_size = 1;
    let input_channels = 3;
    let height = 4;
    let width = 4;

    let mut input_data = Array4::zeros((batch_size, input_channels, height, width));

    // Set different values for each channel to verify that depthwise convolution processes channels separately
    for c in 0..input_channels {
        for h in 0..height {
            for w in 0..width {
                input_data[[0, c, h, w]] = (c * 10 + h * width + w) as f32;
            }
        }
    }

    let input = input_data.into_dyn();

    // Test DepthwiseConv2D using Sequential model
    let mut model = Sequential::new();

    // Create layer and initialize weights
    let mut depthwise_layer = DepthwiseConv2D::new(
        input_channels, // filters must equal input_channels
        (2, 2),         // kernel_size
        (1, 1),         // strides
        PaddingType::Valid,
    );
    depthwise_layer.initialize_weights(input_channels);

    model
        .add(depthwise_layer)
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Forward propagation
    let output = model.predict(&input);

    // Check output shape
    // Input: [1, 3, 4, 4], kernel (2,2), stride (1,1), valid padding
    // Output should be: [1, 3, 3, 3]
    assert_eq!(output.shape(), &[1, 3, 3, 3]);

    // Print model structure
    model.summary();
}

#[test]
fn test_depthwise_conv2d_with_same_padding() {
    let batch_size = 2;
    let channels = 2;
    let height = 3;
    let width = 3;

    let input_data = Array4::ones((batch_size, channels, height, width));
    let input = input_data.into_dyn();

    // Create layer and initialize weights
    let mut depthwise_layer = DepthwiseConv2D::new(
        channels,
        (3, 3), // kernel_size
        (1, 1), // strides
        PaddingType::Same,
    );
    depthwise_layer.initialize_weights(channels);

    // Test DepthwiseConv2D with Same padding using Sequential model
    let mut model = Sequential::new();
    model.add(depthwise_layer);

    let output = model.predict(&input);

    // With Same padding, output size should match input size
    assert_eq!(output.shape(), &[2, 2, 3, 3]);
}

#[test]
fn test_depthwise_conv2d_with_strides() {
    let input_data = Array4::from_elem((1, 2, 6, 6), 1.0);

    // Create layer and initialize weights
    let mut depthwise_layer = DepthwiseConv2D::new(
        2,      // filters = input_channels
        (3, 3), // kernel_size
        (2, 2), // strides
        PaddingType::Valid,
    );
    depthwise_layer.initialize_weights(2);

    // Test DepthwiseConv2D with strides using Sequential model
    let mut model = Sequential::new();
    model.add(depthwise_layer);

    let output = model.predict(&input_data.into_dyn());

    // Input: [1, 2, 6, 6], kernel (3,3), stride (2,2), valid padding
    // Output should be: [1, 2, 2, 2]
    assert_eq!(output.shape(), &[1, 2, 2, 2]);
}

#[test]
fn test_depthwise_conv2d_backward() {
    let input_data = Array4::from_elem((1, 2, 4, 4), 1.0);

    // Create DepthwiseConv2D layer
    let mut layer = DepthwiseConv2D::new(
        2,      // filters
        (2, 2), // kernel_size
        (1, 1), // strides
        PaddingType::Valid,
    );

    // Initialize weights
    layer.initialize_weights(2);

    // Forward propagation
    let output = layer.forward(&input_data.into_dyn());
    assert_eq!(output.shape(), &[1, 2, 3, 3]);

    // Create gradients
    let grad_output = Array4::ones((1, 2, 3, 3)).into_dyn();

    // Backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();
    assert_eq!(grad_input.shape(), &[1, 2, 4, 4]);

    // Check if weight and bias gradients are computed
    assert!(layer.weight_gradients.is_some());
    assert!(layer.bias_gradients.is_some());

    let weight_grads = layer.weight_gradients.as_ref().unwrap();
    let bias_grads = layer.bias_gradients.as_ref().unwrap();

    assert_eq!(weight_grads.shape(), &[2, 1, 2, 2]);
    assert_eq!(bias_grads.shape(), &[2]);
}

#[test]
fn test_depthwise_conv2d_sequential_with_sgd() {
    let input_data = Array4::from_shape_fn((2, 3, 4, 4), |(b, c, h, w)| {
        (b as f32 + c as f32 + h as f32 + w as f32) / 10.0
    })
    .into_dyn();

    let target_data = Array4::ones((2, 3, 3, 3)).into_dyn();

    // Create layer and initialize weights
    let mut depthwise_layer = DepthwiseConv2D::new(
        3,      // filters
        (2, 2), // kernel_size
        (1, 1), // strides
        PaddingType::Valid,
    );
    depthwise_layer.initialize_weights(3);

    // Use Sequential model with SGD optimizer
    let mut model = Sequential::new();
    model
        .add(depthwise_layer)
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Test training - compute initial loss
    let initial_output = model.predict(&input_data);
    let loss_fn = MeanSquaredError::new();
    let initial_loss = loss_fn.compute_loss(&target_data, &initial_output);

    // Train model
    let result = model.fit(&input_data, &target_data, 10);
    assert!(result.is_ok());

    // Compute loss after training
    let final_output = model.predict(&input_data);
    let final_loss = loss_fn.compute_loss(&target_data, &final_output);

    // Loss should decrease
    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_depthwise_conv2d_sequential_with_adam() {
    let input_data = Array4::from_shape_fn((1, 2, 5, 5), |(_, c, h, w)| {
        c as f32 * 0.1 + h as f32 * 0.2 + w as f32 * 0.3
    })
    .into_dyn();

    let target_data = Array4::from_elem((1, 2, 4, 4), 0.5);

    // Create layer and initialize weights
    let mut depthwise_layer = DepthwiseConv2D::new(
        2,      // filters
        (2, 2), // kernel_size
        (1, 1), // strides
        PaddingType::Valid,
    );
    depthwise_layer.initialize_weights(2);

    // Use Sequential model with Adam optimizer
    let mut model = Sequential::new();
    model
        .add(depthwise_layer)
        .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Test training - compute initial loss
    let initial_output = model.predict(&input_data);
    let loss_fn = MeanSquaredError::new();
    let initial_loss = loss_fn.compute_loss(&target_data.clone().into_dyn(), &initial_output);

    // Train model
    let result = model.fit(&input_data, &target_data.clone().into_dyn(), 20);
    assert!(result.is_ok());

    // Compute loss after training
    let final_output = model.predict(&input_data);
    let final_loss = loss_fn.compute_loss(&target_data.into_dyn(), &final_output);

    // Loss should decrease
    assert!(
        final_loss < initial_loss,
        "Loss should decrease with Adam optimizer: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_depthwise_conv2d_sequential_with_rmsprop() {
    let input_data = Array4::from_elem((1, 1, 6, 6), 2.0);
    let target_data = Array4::from_elem((1, 1, 5, 5), 1.0);

    // Create layer and initialize weights
    let mut depthwise_layer = DepthwiseConv2D::new(
        1,      // filters
        (2, 2), // kernel_size
        (1, 1), // strides
        PaddingType::Valid,
    );
    depthwise_layer.initialize_weights(1);

    // Use Sequential model with RMSprop optimizer
    let mut model = Sequential::new();
    model
        .add(depthwise_layer)
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Test training - compute initial loss
    let initial_output = model.predict(&input_data.clone().into_dyn());
    let loss_fn = MeanSquaredError::new();
    let initial_loss = loss_fn.compute_loss(&target_data.clone().into_dyn(), &initial_output);

    // Train model
    let result = model.fit(
        &input_data.clone().into_dyn(),
        &target_data.clone().into_dyn(),
        15,
    );
    assert!(result.is_ok());

    // Compute loss after training
    let final_output = model.predict(&input_data.into_dyn());
    let final_loss = loss_fn.compute_loss(&target_data.into_dyn(), &final_output);

    // Loss should decrease
    assert!(
        final_loss < initial_loss,
        "Loss should decrease with RMSprop optimizer: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_depthwise_conv2d_parameter_count() {
    let layer = DepthwiseConv2D::new(
        4,      // filters
        (3, 3), // kernel_size
        (1, 1), // strides
        PaddingType::Same,
    );

    // Parameter count = weight parameters + bias parameters
    // Weights: [filters, 1, kernel_height, kernel_width] = 4 * 1 * 3 * 3 = 36
    // Biases: [filters] = 4
    // Total: 36 + 4 = 40
    assert_eq!(layer.param_count(), 40);
}

#[test]
fn test_depthwise_conv2d_output_shape() {
    let mut layer = DepthwiseConv2D::new(
        3,      // filters
        (2, 2), // kernel_size
        (1, 1), // strides
        PaddingType::Valid,
    );

    // When there's no input shape information
    assert_eq!(layer.output_shape(), "Unknown");

    // Set input shape and test
    layer.input_shape = vec![2, 3, 5, 5]; // [batch, channels, height, width]
    let expected_output_shape = "(2, 3, 4, 4)"; // valid padding, kernel 2x2
    assert_eq!(layer.output_shape(), expected_output_shape);
}

#[test]
fn test_depthwise_conv2d_layer_type() {
    let layer = DepthwiseConv2D::new(2, (3, 3), (1, 1), PaddingType::Valid);

    assert_eq!(layer.layer_type(), "DepthwiseConv2D");
}

#[test]
#[should_panic(expected = "For depthwise convolution, number of filters must equal input channels")]
fn test_depthwise_conv2d_filter_channel_mismatch() {
    let mut layer = DepthwiseConv2D::new(
        3, // filters
        (2, 2),
        (1, 1),
        PaddingType::Valid,
    );

    // Try to initialize weights with mismatched channel count (should panic)
    layer.initialize_weights(2); // input_channels != filters
}

#[test]
#[should_panic(expected = "Input channels must equal number of filters for depthwise convolution")]
fn test_depthwise_conv2d_forward_channel_mismatch() {
    let mut layer = DepthwiseConv2D::new(
        2, // filters
        (2, 2),
        (1, 1),
        PaddingType::Valid,
    );

    layer.initialize_weights(2);

    // Create input data with mismatched channel count
    let input_data = Array4::ones((1, 3, 4, 4)).into_dyn(); // 3 channels, but layer expects 2

    // This should panic
    layer.forward(&input_data);
}

#[test]
fn test_depthwise_conv2d_sequential_summary() {
    // Create layers and initialize weights
    let mut depthwise_layer1 = DepthwiseConv2D::new(2, (3, 3), (1, 1), PaddingType::Same);
    depthwise_layer1.initialize_weights(2);

    let mut depthwise_layer2 = DepthwiseConv2D::new(2, (2, 2), (2, 2), PaddingType::Valid);
    depthwise_layer2.initialize_weights(2);

    let mut model = Sequential::new();
    model.add(depthwise_layer1).add(depthwise_layer2);

    // Test that summary doesn't panic
    model.summary();
}

#[test]
fn test_depthwise_conv2d_weights() {
    let mut layer = DepthwiseConv2D::new(2, (2, 2), (1, 1), PaddingType::Valid);

    layer.initialize_weights(2);

    let weights = layer.get_weights();

    match weights {
        LayerWeight::DepthwiseConv2DLayer(depthwise_weights) => {
            assert_eq!(depthwise_weights.weight.shape(), &[2, 1, 2, 2]);
            assert_eq!(depthwise_weights.bias.shape(), &[2]);
        }
        _ => panic!("Expected DepthwiseConv2DLayer weights"),
    }
}

#[test]
fn test_depthwise_conv2d_multiple_layers() {
    let input_data =
        Array4::from_shape_fn((1, 3, 8, 8), |(_, c, h, w)| (c + h + w) as f32 * 0.1).into_dyn();

    // Create multiple depthwise convolution layers
    let mut layer1 = DepthwiseConv2D::new(3, (3, 3), (1, 1), PaddingType::Same);
    layer1.initialize_weights(3);

    let mut layer2 = DepthwiseConv2D::new(3, (3, 3), (2, 2), PaddingType::Valid);
    layer2.initialize_weights(3);

    let mut model = Sequential::new();
    model
        .add(layer1)
        .add(layer2)
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let output = model.predict(&input_data);

    // First layer: [1, 3, 8, 8] -> [1, 3, 8, 8] (Same padding)
    // Second layer: [1, 3, 8, 8] -> [1, 3, 3, 3] (Valid padding, kernel 3x3, stride 2x2)
    assert_eq!(output.shape(), &[1, 3, 3, 3]);

    model.summary();
}
