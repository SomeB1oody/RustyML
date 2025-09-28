use super::*;

#[test]
fn test_global_average_pooling_1d_creation() {
    // Create GlobalAveragePooling1D layer
    let layer = GlobalAveragePooling1D::new();

    // Verify layer type
    assert_eq!(layer.layer_type(), "GlobalAveragePooling1D");

    // Verify parameter count should be TrainingParameters::NoTrainable
    assert_eq!(layer.param_count(), TrainingParameters::NoTrainable);

    // Verify output shape before initialization
    assert_eq!(layer.output_shape(), "Unknown");
}

#[test]
fn test_global_average_pooling_1d_forward() {
    // Create input tensor: [batch_size, channels, length]
    let batch_size = 2;
    let channels = 3;
    let length = 4;

    // Generate sequentially increasing data
    let input_data = generate_data(batch_size, channels, length);

    // Create layer and perform forward propagation
    let mut layer = GlobalAveragePooling1D::new();
    let output = layer.forward(&input_data);

    // Verify output shape - should be [batch_size, channels]
    assert_eq!(output.shape(), &[batch_size, channels]);

    // Verify output values - should be the average of each channel
    for b in 0..batch_size {
        for c in 0..channels {
            // Calculate expected average for each channel
            let sum: f32 = (0..length).map(|l| (b * 100 + c * 10 + l) as f32).sum();
            let expected_avg = sum / (length as f32);
            assert_relative_eq!(output[[b, c]], expected_avg);
        }
    }

    // Verify layer's output shape string
    assert_eq!(
        layer.output_shape(),
        format!("({}, {})", batch_size, channels)
    );
}

#[test]
fn test_global_average_pooling_1d_sequential() {
    // Create Sequential model
    let mut model = Sequential::new();

    // Add GlobalAveragePooling1D layer
    model.add(GlobalAveragePooling1D::new());

    // Create test input data: [batch_size, channels, length]
    let batch_size = 3;
    let channels = 4;
    let length = 8;
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // Fill data - make all values in each channel the same for easy verification
    for b in 0..batch_size {
        for c in 0..channels {
            let value = (b * 10 + c) as f32;
            for l in 0..length {
                input_data[[b, c, l]] = value;
            }
        }
    }

    // Forward propagation
    let output = model.predict(&input_data);

    // Verify output shape - should be [batch_size, channels]
    assert_eq!(output.shape(), &[batch_size, channels]);

    // Verify output values - should be the same as input values since all values in each channel are the same
    for b in 0..batch_size {
        for c in 0..channels {
            let expected_value = (b * 10 + c) as f32;
            assert_relative_eq!(output[[b, c]], expected_value);
        }
    }
}

#[test]
fn test_global_average_pooling_1d_backward() {
    // Create input tensor: [batch_size, channels, length]
    let batch_size = 2;
    let channels = 3;
    let length = 4;

    // Create input data
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // Fill data - use random values
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                input_data[[b, c, l]] = (b + c + l) as f32;
            }
        }
    }

    // Create layer and perform forward propagation
    let mut layer = GlobalAveragePooling1D::new();
    let _output = layer.forward(&input_data);

    // Create gradient output - shape should match output [batch_size, channels]
    let grad_output = Tensor::ones(IxDyn(&[batch_size, channels]));

    // Perform backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Verify gradient input shape - should match original input
    assert_eq!(grad_input.shape(), input_data.shape());

    // Verify gradients - should be uniformly distributed at each position
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                // Gradient should be uniformly distributed, equal to 1.0 / length
                assert_relative_eq!(grad_input[[b, c, l]], 1.0 / (length as f32));
            }
        }
    }
}

#[test]
fn test_global_average_pooling_1d_zero_input() {
    // Create all-zero input tensor
    let input_data = Tensor::zeros(IxDyn(&[2, 3, 4]));

    // Create layer and perform forward propagation
    let mut layer = GlobalAveragePooling1D::new();
    let output = layer.forward(&input_data);

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3]);

    // Verify output values - all values should be 0
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c]], 0.0);
        }
    }
}
