use super::*;

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
