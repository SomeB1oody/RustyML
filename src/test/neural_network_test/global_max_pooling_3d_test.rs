use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::pooling_layer::global_max_pooling_3d::GlobalMaxPooling3D;
use crate::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use crate::neural_network::neural_network_trait::Layer;
use crate::neural_network::optimizer::sgd::SGD;
use crate::neural_network::sequential::Sequential;
use approx::assert_relative_eq;
use ndarray::{Array, IxDyn};

#[test]
fn test_global_max_pooling_3d_forward_basic() {
    // Create a simple GlobalMaxPooling3D layer
    let mut layer = GlobalMaxPooling3D::new();

    // Create test input: [batch_size, channels, depth, height, width] = [1, 2, 3, 4, 5]
    let mut input_data = Array::zeros(IxDyn(&[1, 2, 3, 4, 5]));

    // Set maximum value for first channel at position (1, 2, 3)
    input_data[[0, 0, 1, 2, 3]] = 10.0;

    // Set maximum value for second channel at position (2, 1, 4)
    input_data[[0, 1, 2, 1, 4]] = 15.0;

    // Perform forward propagation
    let output = layer.forward(&input_data).unwrap();

    // Check output shape should be [1, 2]
    assert_eq!(output.shape(), &[1, 2]);

    // Check output values
    assert_relative_eq!(output[[0, 0]], 10.0);
    assert_relative_eq!(output[[0, 1]], 15.0);
}

#[test]
fn test_global_max_pooling_3d_forward_batch() {
    // Test batch processing
    let mut layer = GlobalMaxPooling3D::new();

    // Create test input: [batch_size, channels, depth, height, width] = [2, 3, 2, 2, 2]
    let mut input_data = Array::from_elem(IxDyn(&[2, 3, 2, 2, 2]), 1.0);

    // Set different maximum values for each batch and channel
    input_data[[0, 0, 0, 1, 1]] = 5.0; // batch 0, channel 0
    input_data[[0, 1, 1, 0, 0]] = 7.0; // batch 0, channel 1
    input_data[[0, 2, 1, 1, 0]] = 9.0; // batch 0, channel 2

    input_data[[1, 0, 0, 0, 1]] = 3.0; // batch 1, channel 0
    input_data[[1, 1, 1, 1, 1]] = 8.0; // batch 1, channel 1
    input_data[[1, 2, 0, 1, 0]] = 6.0; // batch 1, channel 2

    // Perform forward propagation
    let output = layer.forward(&input_data).unwrap();

    // Check output shape
    assert_eq!(output.shape(), &[2, 3]);

    // Check output values
    assert_relative_eq!(output[[0, 0]], 5.0);
    assert_relative_eq!(output[[0, 1]], 7.0);
    assert_relative_eq!(output[[0, 2]], 9.0);
    assert_relative_eq!(output[[1, 0]], 3.0);
    assert_relative_eq!(output[[1, 1]], 8.0);
    assert_relative_eq!(output[[1, 2]], 6.0);
}

#[test]
fn test_global_max_pooling_3d_backward() {
    let mut layer = GlobalMaxPooling3D::new();

    // Create input data
    let mut input_data = Array::zeros(IxDyn(&[1, 1, 2, 2, 2]));
    input_data[[0, 0, 1, 0, 1]] = 10.0; // Maximum value position

    // Forward propagation
    let _output = layer.forward(&input_data);

    // Create gradient output
    let grad_output = Array::from_elem(IxDyn(&[1, 1]), 2.0);

    // Perform backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Check gradient shape
    assert_eq!(grad_input.shape(), input_data.shape());

    // Check that only the maximum value position has gradient
    assert_relative_eq!(grad_input[[0, 0, 1, 0, 1]], 2.0);

    // Check that other positions have zero gradient
    for d in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                if (d, h, w) != (1, 0, 1) {
                    assert_relative_eq!(grad_input[[0, 0, d, h, w]], 0.0);
                }
            }
        }
    }
}

#[test]
fn test_global_max_pooling_3d_with_sequential() {
    // Create Sequential model
    let mut model = Sequential::new();

    // Add GlobalMaxPooling3D layer
    model.add(GlobalMaxPooling3D::new());

    // Compile model
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Create test input: [batch_size, channels, depth, height, width] = [2, 3, 4, 5, 5]
    let input_data = Array::from_elem(IxDyn(&[2, 3, 4, 5, 5]), 1.0);

    // Perform prediction
    let output = model.predict(&input_data).unwrap();

    // Check output shape
    assert_eq!(output.shape(), &[2, 3]);

    // Since all input values are 1.0, all output values should also be 1.0
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c]], 1.0);
        }
    }
}

#[test]
fn test_global_max_pooling_3d_with_negative_values() {
    let mut layer = GlobalMaxPooling3D::new();

    // Create test input containing negative values
    let mut input_data = Array::from_elem(IxDyn(&[1, 2, 2, 2, 2]), -5.0);

    // Set some larger negative values as "maximum values"
    input_data[[0, 0, 0, 1, 1]] = -1.0; // Maximum value for channel 0
    input_data[[0, 1, 1, 0, 0]] = -2.0; // Maximum value for channel 1

    // Perform forward propagation
    let output = layer.forward(&input_data).unwrap();

    // Check output
    assert_eq!(output.shape(), &[1, 2]);
    assert_relative_eq!(output[[0, 0]], -1.0);
    assert_relative_eq!(output[[0, 1]], -2.0);
}

#[test]
fn test_global_max_pooling_3d_layer_properties() {
    let layer = GlobalMaxPooling3D::new();

    // Check layer type
    assert_eq!(layer.layer_type(), "GlobalMaxPooling3D");

    // Check parameter count (should be TrainingParameters::NoTrainable since it's a pooling layer)
    assert_eq!(layer.param_count(), TrainingParameters::NoTrainable);
}

#[test]
fn test_global_max_pooling_3d_backward_without_forward() {
    let mut layer = GlobalMaxPooling3D::new();

    // Create gradient output
    let grad_output = Array::from_elem(IxDyn(&[1, 1]), 1.0);

    // Try backward propagation without performing forward propagation
    let result = layer.backward(&grad_output);

    // Should return an error
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Forward pass has not been run yet")
    );
}

#[test]
fn test_global_max_pooling_3d_sequential_training() {
    // Create Sequential model for training test
    let mut model = Sequential::new();

    // Add layer
    model.add(GlobalMaxPooling3D::new());

    // Compile model
    model.compile(SGD::new(0.1).unwrap(), MeanSquaredError::new());

    // Create training data
    let x = Array::from_elem(IxDyn(&[4, 2, 3, 3, 3]), 1.0);
    let y = Array::from_elem(IxDyn(&[4, 2]), 1.0);

    // Train model (should run successfully)
    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());

    // Verify prediction functionality
    let prediction = model.predict(&x).unwrap();
    assert_eq!(prediction.shape(), &[4, 2]);
}
