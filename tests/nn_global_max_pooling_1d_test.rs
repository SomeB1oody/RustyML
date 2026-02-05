use approx::assert_relative_eq;
use ndarray::IxDyn;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::layer::pooling_layer::global_max_pooling_1d::GlobalMaxPooling1D;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_global_max_pooling_1d_creation() {
    // Create a GlobalMaxPooling1D layer
    let layer = GlobalMaxPooling1D::new();

    // Verify layer type
    assert_eq!(layer.layer_type(), "GlobalMaxPooling1D");

    // Verify parameter count should be TrainingParameters::NoTrainable
    assert_eq!(layer.param_count(), TrainingParameters::NoTrainable);

    // Verify output shape before initialization
    assert_eq!(layer.output_shape(), "Unknown");
}

#[test]
fn test_global_max_pooling_1d_forward() {
    // Create an input tensor: [batch_size, channels, length]
    let batch_size = 2;
    let channels = 3;
    let length = 4;

    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // Initialize input data, set specific values for testing average calculation
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                input_data[[b, c, l]] = (b * 100 + c * 10 + l) as f32;
            }
        }
    }

    // Create layer and perform forward propagation
    let mut layer = GlobalMaxPooling1D::new();
    let output = layer.forward(&input_data).unwrap();

    // Verify output shape - should be [batch_size, channels]
    assert_eq!(output.shape(), &[batch_size, channels]);

    // Verify output values - should be the maximum value of each channel
    for b in 0..batch_size {
        for c in 0..channels {
            // Calculate the expected maximum value for each channel
            let expected_max = (b * 100 + c * 10 + (length - 1)) as f32;
            assert_relative_eq!(output[[b, c]], expected_max);
        }
    }

    // Verify the layer's output shape string
    assert_eq!(
        layer.output_shape(),
        format!("({}, {})", batch_size, channels)
    );
}

#[test]
fn test_global_max_pooling_1d_sequential() {
    // Create Sequential model
    let mut model = Sequential::new();

    // Add GlobalMaxPooling1D layer
    model.add(GlobalMaxPooling1D::new());

    // Create test input data: [batch_size, channels, length]
    let batch_size = 3;
    let channels = 4;
    let length = 8;
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // Fill data - use specific pattern so that maximum values are in predictable positions
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                // Ensure each channel has a clear maximum value
                input_data[[b, c, l]] = if l == b % length {
                    10.0 // This will be the maximum value
                } else {
                    l as f32
                };
            }
        }
    }

    // Forward propagation
    let output = model.predict(&input_data).unwrap();

    // Verify output shape - should be [batch_size, channels]
    assert_eq!(output.shape(), &[batch_size, channels]);

    // Verify output values - should be the maximum value of each channel
    for b in 0..batch_size {
        for c in 0..channels {
            // Our expected maximum value is 10.0
            assert_relative_eq!(output[[b, c]], 10.0);
        }
    }
}

#[test]
fn test_global_max_pooling_1d_backward() {
    // Create input tensor: [batch_size, channels, length]
    let batch_size = 2;
    let channels = 3;
    let length = 4;

    // Create input data - ensure a clear maximum value position
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));
    let mut expected_max_positions = vec![0; batch_size * channels];

    // Fill data - each channel has a maximum value at different position
    for b in 0..batch_size {
        for c in 0..channels {
            let max_pos = (b + c) % length;
            expected_max_positions[b * channels + c] = max_pos;

            for l in 0..length {
                input_data[[b, c, l]] = if l == max_pos {
                    10.0 // Maximum value
                } else {
                    1.0 // Other values
                };
            }
        }
    }

    // Create layer and perform forward propagation
    let mut layer = GlobalMaxPooling1D::new();
    let _output = layer.forward(&input_data);

    // Create gradient output - shape should match the output [batch_size, channels]
    let grad_output = Tensor::ones(IxDyn(&[batch_size, channels]));

    // Perform backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Verify gradient input shape - should match the original input
    assert_eq!(grad_input.shape(), input_data.shape());

    // Verify gradients - only the maximum value position should have non-zero gradient
    for b in 0..batch_size {
        for c in 0..channels {
            let max_pos = expected_max_positions[b * channels + c];

            for l in 0..length {
                if l == max_pos {
                    assert_relative_eq!(grad_input[[b, c, l]], grad_output[[b, c]]);
                } else {
                    assert_relative_eq!(grad_input[[b, c, l]], 0.0);
                }
            }
        }
    }
}

#[test]
fn test_global_max_pooling_1d_zero_input() {
    // Create an all-zero input tensor
    let input_data = Tensor::zeros(IxDyn(&[2, 3, 4]));

    // Create layer and perform forward propagation
    let mut layer = GlobalMaxPooling1D::new();
    let output = layer.forward(&input_data).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3]);

    // Verify output values - all values should be 0
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c]], 0.0);
        }
    }
}

#[test]
fn test_global_max_pooling_1d_negative_input() {
    // Create an input tensor with negative values
    let mut input_data = Tensor::zeros(IxDyn(&[2, 3, 4]));

    // Fill with negative data
    for b in 0..2 {
        for c in 0..3 {
            for l in 0..4 {
                input_data[[b, c, l]] = -((b + c + l) as f32);
            }
        }
    }

    // Create layer and perform forward propagation
    let mut layer = GlobalMaxPooling1D::new();
    let output = layer.forward(&input_data).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3]);

    // Verify output values - each channel should have the maximum negative value (i.e., the least negative value)
    for b in 0..2 {
        for c in 0..3 {
            // The maximum negative value should be -(b+c+0)
            let expected_max = -((b + c) as f32);
            assert_relative_eq!(output[[b, c]], expected_max);
        }
    }
}
