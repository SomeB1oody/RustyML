use super::*;

#[test]
fn max_pooling_2d_test() {
    // Create a simple 4D input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 input channels, 6x6 pixels
    let mut input_data = Array4::zeros((2, 3, 6, 6));

    // Set some specific values so we can predict the max pooling result
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..6 {
                for j in 0..6 {
                    // Create input data with an easily observable pattern
                    input_data[[b, c, i, j]] = (i * j) as f32 + b as f32 * 0.1 + c as f32 * 0.01;
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Create MaxPooling2D layer with pool size (2,2) and stride (2,2)
    let mut pool_layer = MaxPooling2D::new(
        (2, 2),           // Pool window size
        vec![2, 3, 6, 6], // Input shape
        Some((2, 2)),     // Stride
    );

    // Perform forward propagation
    let output = pool_layer.forward(&x);

    // Check if output shape is correct - should be [2, 3, 3, 3]
    assert_eq!(output.shape(), &[2, 3, 3, 3]);

    // Manually check some pooling results
    // For input region [[0,0], [0,1], [1,0], [1,1]], the max value should be at [1,1]
    assert_eq!(output[[0, 0, 0, 0]], input_data[[0, 0, 1, 1]]);

    // Test backward propagation
    let mut grad_output = ArrayD::zeros(output.dim());
    grad_output.fill(1.0); // Set uniform gradient

    let grad_input = pool_layer.backward(&grad_output).unwrap();

    // Check if gradient shape is correct
    assert_eq!(grad_input.shape(), x.shape());

    // Gradients should only have non-zero values at maximum value positions
    let nonzero_count = grad_input.iter().filter(|&&x| x > 0.0).count();

    // Output size is 2x3x3x3=54 elements, so there should be 54 non-zero gradients
    assert_eq!(nonzero_count, 54);

    // Test using MaxPooling2D in a model
    let mut model = Sequential::new();
    model
        .add(MaxPooling2D::new(
            (2, 2),           // Pool window size
            vec![2, 3, 6, 6], // Input shape
            None,             // Use default stride (2,2)
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Create target tensor - corresponding to the pooled shape
    let y = Array4::ones((2, 3, 3, 3)).into_dyn();

    // Print model structure
    model.summary();

    // Train the model (run a few epochs)
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("MaxPooling2D prediction results: {:?}", prediction);

    // Check if output shape is correct
    assert_eq!(prediction.shape(), &[2, 3, 3, 3]);

    // Test different stride cases
    let mut model2 = Sequential::new();
    model2
        .add(MaxPooling2D::new(
            (3, 3),           // Larger pool window size
            vec![2, 3, 6, 6], // Input shape
            Some((1, 1)),     // Smaller stride
        ))
        .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Print second model structure
    model2.summary();

    // Calculate expected output shape
    let expected_shape = [2, 3, 4, 4]; // Using 3x3 window and 1x1 stride

    // Use predict to confirm output shape
    let prediction2 = model2.predict(&x);
    println!(
        "MaxPooling2D prediction results (pool_size=(3,3), stride=(1,1)): {:?}",
        prediction2
    );

    // Check if output shape matches expectation
    assert_eq!(prediction2.shape(), &expected_shape);
}

#[test]
fn max_pooling_2d_edge_cases() {
    // Test edge cases: input size equals pool window size
    let input_data = Array4::zeros((1, 1, 2, 2)).into_dyn();

    let mut pool_layer = MaxPooling2D::new(
        (2, 2),           // Pool window equals input size
        vec![1, 1, 2, 2], // Input shape
        None,             // Default stride
    );

    let output = pool_layer.forward(&input_data);
    // Output should be [1, 1, 1, 1]
    assert_eq!(output.shape(), &[1, 1, 1, 1]);

    // Test asymmetric pool window and stride
    let input_data = Array4::zeros((1, 2, 5, 4)).into_dyn();

    let mut pool_layer = MaxPooling2D::new(
        (3, 2),           // Asymmetric pool window
        vec![1, 2, 5, 4], // Input shape
        Some((2, 1)),     // Asymmetric stride
    );

    let output = pool_layer.forward(&input_data);
    // Output should be [1, 2, 2, 3]
    assert_eq!(output.shape(), &[1, 2, 2, 3]);

    // Test parameter count - pooling layer has no trainable parameters
    assert_eq!(pool_layer.param_count(), 0);
}
