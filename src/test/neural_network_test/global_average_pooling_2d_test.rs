use super::*;

#[test]
fn test_global_average_pooling_2d() {
    // Create test input data: shape is [batch_size, channels, height, width]
    let input = Array::from_elem(IxDyn(&[2, 3, 4, 4]), 1.0);

    // Create Sequential model
    let mut model = Sequential::new();

    // Add GlobalAveragePooling2D layer
    model
        .add(GlobalAveragePooling2D::new())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

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
