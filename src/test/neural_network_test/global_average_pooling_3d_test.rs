use super::*;

#[test]
fn test_global_average_pooling_3d_basic_functionality() {
    // Create a Sequential model
    let mut model = Sequential::new();

    // Add GlobalAveragePooling3D layer
    model.add(GlobalAveragePooling3D::new());

    // Compile the model
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Create test input tensor: [batch_size, channels, depth, height, width]
    let input_data = Array::from_elem(IxDyn(&[2, 3, 4, 4, 4]), 2.0);

    // Forward propagation
    let output = model.predict(&input_data);

    // Check output shape - should be [2, 3]
    assert_eq!(output.shape(), &[2, 3]);

    // Since all input values are 2.0, all output values should also be 2.0
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c]], 2.0, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_global_average_pooling_3d_with_random_values() {
    // Create a Sequential model
    let mut model = Sequential::new();
    model.add(GlobalAveragePooling3D::new());
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Create test tensor with known values
    let mut input_data = Array::zeros(IxDyn(&[1, 2, 2, 2, 2]));

    // Set first channel values to 1.0
    for d in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                input_data[[0, 0, d, h, w]] = 1.0;
            }
        }
    }

    // Set second channel values to 2.0
    for d in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                input_data[[0, 1, d, h, w]] = 2.0;
            }
        }
    }

    let output = model.predict(&input_data);

    // Check output shape
    assert_eq!(output.shape(), &[1, 2]);

    // Verify that average calculation is correct
    assert_relative_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_relative_eq!(output[[0, 1]], 2.0, epsilon = 1e-6);
}

#[test]
fn test_global_average_pooling_3d_gradient_computation() {
    // Create a GlobalAveragePooling3D layer instance (not using Sequential)
    let mut layer = GlobalAveragePooling3D::new();

    // Create test input
    let input = Array::from_elem(IxDyn(&[1, 1, 2, 2, 2]), 3.0);

    // Forward propagation
    let output = layer.forward(&input).unwrap();

    // Check output
    assert_eq!(output.shape(), &[1, 1]);
    assert_relative_eq!(output[[0, 0]], 3.0, epsilon = 1e-6);

    // Create gradient output
    let grad_output = Array::from_elem(IxDyn(&[1, 1]), 1.0);

    // Backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Check gradient shape
    assert_eq!(grad_input.shape(), input.shape());

    // Gradients should be uniformly distributed across all spatial positions
    let expected_grad = 1.0 / 8.0; // 1.0 / (2*2*2)
    for d in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                assert_relative_eq!(grad_input[[0, 0, d, h, w]], expected_grad, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn test_global_average_pooling_3d_multiple_batches_and_channels() {
    let mut model = Sequential::new();
    model.add(GlobalAveragePooling3D::new());
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Create multi-batch, multi-channel test data
    let mut input_data = Array::zeros(IxDyn(&[3, 4, 3, 3, 3]));

    // Set different values for each batch and channel
    for b in 0..3 {
        for c in 0..4 {
            let value = (b + 1) as f32 * (c + 1) as f32;
            for d in 0..3 {
                for h in 0..3 {
                    for w in 0..3 {
                        input_data[[b, c, d, h, w]] = value;
                    }
                }
            }
        }
    }

    let output = model.predict(&input_data);

    // Check output shape
    assert_eq!(output.shape(), &[3, 4]);

    // Verify average values for each position
    for b in 0..3 {
        for c in 0..4 {
            let expected_value = (b + 1) as f32 * (c + 1) as f32;
            assert_relative_eq!(output[[b, c]], expected_value, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_global_average_pooling_3d_different_spatial_dimensions() {
    let mut model = Sequential::new();
    model.add(GlobalAveragePooling3D::new());
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Test with different spatial dimensions
    let input_data = Array::from_elem(IxDyn(&[1, 2, 5, 6, 7]), 4.0);

    let output = model.predict(&input_data);

    // Check output shape
    assert_eq!(output.shape(), &[1, 2]);

    // All output values should equal input values (since input is constant)
    for c in 0..2 {
        assert_relative_eq!(output[[0, c]], 4.0, epsilon = 1e-6);
    }
}

#[test]
fn test_global_average_pooling_3d_backward_without_forward() {
    let mut layer = GlobalAveragePooling3D::new();

    // Try to perform backward propagation without forward propagation
    let grad_output = Array::zeros(IxDyn(&[1, 1]));

    let result = layer.backward(&grad_output);

    // Should return an error
    assert!(result.is_err());
    if let Err(ModelError::ProcessingError(msg)) = result {
        assert_eq!(msg, "Forward pass has not been run yet");
    } else {
        panic!("Expected ProcessingError");
    }
}

#[test]
fn test_global_average_pooling_3d_layer_properties() {
    let layer = GlobalAveragePooling3D::new();

    // Test layer type
    assert_eq!(layer.layer_type(), "GlobalAveragePooling3D");

    // Test parameter count (should be TrainingParameters::NoTrainable, since this is a parameter-free layer)
    assert_eq!(layer.param_count(), TrainingParameters::NoTrainable);
}

#[test]
fn test_global_average_pooling_3d_with_sequential_training() {
    // Create a more complex model for training tests
    let mut model = Sequential::new();
    model.add(GlobalAveragePooling3D::new());
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Create training data
    let x = Array::ones(IxDyn(&[2, 2, 3, 3, 3]));
    let y = Array::ones(IxDyn(&[2, 2]));

    // Display model structure
    model.summary();

    // Perform training (since there are no trainable parameters, this mainly tests the pipeline)
    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());

    // Test prediction
    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[2, 2]);
}
