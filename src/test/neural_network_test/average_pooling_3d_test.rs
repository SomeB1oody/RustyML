use super::*;

#[test]
fn test_average_pooling_3d_basic() {
    // Create a 3D average pooling layer
    let pool_size = (2, 2, 2);
    let input_shape = vec![1, 2, 4, 4, 4];
    let layer = AveragePooling3D::new(pool_size, input_shape.clone(), None);

    // Check the basic properties of the layer
    assert_eq!(layer.layer_type(), "AveragePooling3D");
    assert_eq!(layer.param_count(), TrainingParameters::NoTrainable);
    assert_eq!(layer.output_shape(), "(1, 2, 2, 2, 2)");
}

#[test]
fn test_average_pooling_3d_forward() {
    // Create test data
    let input_shape = vec![1, 1, 4, 4, 4];
    let mut layer = AveragePooling3D::new((2, 2, 2), input_shape, None);

    // Create the input tensor, all values are 8.0, so the average value of the 2x2x2 pooling window should also be 8.0
    let input = Array5::from_elem((1, 1, 4, 4, 4), 8.0).into_dyn();

    // Perform forward propagation
    let output = layer.forward(&input).unwrap();

    // Check the output shape
    assert_eq!(output.shape(), &[1, 1, 2, 2, 2]);

    // Check the output values
    for &value in output.iter() {
        assert_abs_diff_eq!(value, 8.0, epsilon = 1e-6);
    }
}

#[test]
fn test_average_pooling_3d_with_strides() {
    let input_shape = vec![1, 1, 6, 6, 6];
    let mut layer = AveragePooling3D::new((2, 2, 2), input_shape, Some((3, 3, 3)));

    // Create the input tensor
    let input =
        Array5::from_shape_fn((1, 1, 6, 6, 6), |(_, _, d, h, w)| (d + h + w) as f32).into_dyn();

    // Perform forward propagation
    let output = layer.forward(&input).unwrap();

    // Check the output shape (dimensions of 6 reduced to 2 with stride=3)
    assert_eq!(output.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn test_average_pooling_3d_sequential_model() {
    // Create a Sequential model
    let mut model = Sequential::new();

    // Add an AveragePooling3D layer
    model.add(AveragePooling3D::new(
        (2, 2, 2),           // Pooling window size
        vec![1, 2, 8, 8, 8], // Input shape
        Some((2, 2, 2)),     // Strides
    ));

    // Compile the model
    model.compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Create input data
    let input_data = Array5::from_shape_fn((1, 2, 8, 8, 8), |(_, c, d, h, w)| {
        ((d + h + w) as f32 * 0.1) + (c as f32 * 0.01)
    })
    .into_dyn();

    // Create target data (pooled shape should be [1, 2, 4, 4, 4])
    let target_data = Array5::ones((1, 2, 4, 4, 4)).into_dyn();

    // Display the model structure
    model.summary();

    // Train the model
    model.fit(&input_data, &target_data, 2).unwrap();

    // Perform predictions
    let predictions = model.predict(&input_data);

    // Check the shape of the prediction results
    assert_eq!(predictions.shape(), &[1, 2, 4, 4, 4]);
    println!("Shape of output after pooling: {:?}", predictions.shape());
}

#[test]
fn test_average_pooling_3d_multiple_channels() {
    let input_shape = vec![2, 3, 4, 4, 4]; // Batch size=2, channels=3
    let mut layer = AveragePooling3D::new((2, 2, 2), input_shape, None);

    // Create multi-channel input
    let input = Array5::from_shape_fn((2, 3, 4, 4, 4), |(b, c, d, h, w)| {
        (b * 100 + c * 10 + d + h + w) as f32
    })
    .into_dyn();

    // Perform forward propagation
    let output = layer.forward(&input).unwrap();

    // Check the output shape
    assert_eq!(output.shape(), &[2, 3, 2, 2, 2]);

    // Verify that values differ across batches and channels
    let val_batch0_ch0 = output[[0, 0, 0, 0, 0]];
    let val_batch1_ch0 = output[[1, 0, 0, 0, 0]];
    let val_batch0_ch1 = output[[0, 1, 0, 0, 0]];

    assert_ne!(val_batch0_ch0, val_batch1_ch0);
    assert_ne!(val_batch0_ch0, val_batch0_ch1);
}

#[test]
fn test_average_pooling_3d_edge_cases() {
    // Test edge case: pooling window exceeds input boundaries
    let input_shape = vec![1, 1, 3, 3, 3];
    let mut layer = AveragePooling3D::new((2, 2, 2), input_shape, Some((2, 2, 2)));

    let input = Array5::ones((1, 1, 3, 3, 3)).into_dyn();
    let output = layer.forward(&input).unwrap();

    // Output shape should be [1, 1, 1, 1, 1] as there is only one full pooling window
    assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);

    // Since all input values are 1, the average should also be 1
    assert_abs_diff_eq!(output[[0, 0, 0, 0, 0]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_average_pooling_3d_gradient_flow() {
    // Test if gradient flow is correct
    let input_shape = vec![1, 1, 4, 4, 4];
    let mut layer = AveragePooling3D::new((2, 2, 2), input_shape, None);

    // Create input tensor with varying values
    let input = Array5::from_shape_fn((1, 1, 4, 4, 4), |(_, _, d, h, w)| {
        (d * 16 + h * 4 + w) as f32
    })
    .into_dyn();

    // Forward propagation
    let output = layer.forward(&input).unwrap();

    // Create a specific gradient tensor
    let mut grad_output = ArrayD::zeros(output.raw_dim());
    grad_output[[0, 0, 0, 0, 0]] = 1.0; // Gradient set only at one position

    // Backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Check if the gradient is correctly distributed to the corresponding pooling window
    let expected_grad = 1.0 / 8.0; // Average allocation for 2x2x2 window

    // Gradient should be distributed to input positions of the first pooling window
    for d in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                assert_abs_diff_eq!(grad_input[[0, 0, d, h, w]], expected_grad, epsilon = 1e-6);
            }
        }
    }

    // Other positions should not have gradients
    assert_abs_diff_eq!(grad_input[[0, 0, 2, 2, 2]], 0.0, epsilon = 1e-6);
}
