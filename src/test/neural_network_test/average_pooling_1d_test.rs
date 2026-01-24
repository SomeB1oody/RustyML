use super::*;

#[test]
fn test_average_pooling_1d_shape() {
    // Create a simple input tensor: [batch_size, channels, length]
    // Batch size=2, 3 input channels, each channel has 10 elements
    let input_data = Array3::<f32>::zeros((2, 3, 10)).into_dyn();

    // Test different pooling window sizes and strides
    let test_cases = vec![
        // (pool_size, stride, expected_output_length)
        (2, 2, 5), // Pool size=2, stride=2, output length=(10-2)/2+1=5
        (3, 2, 4), // Pool size=3, stride=2, output length=(10-3)/2+1=4
        (4, 3, 3), // Pool size=4, stride=3, output length=(10-4)/3+1=3
        (5, 5, 2), // Pool size=5, stride=5, output length=(10-5)/5+1=2
    ];

    for (pool_size, stride, expected_length) in test_cases {
        let mut layer = AveragePooling1D::new(pool_size, vec![2, 3, 10], Some(stride)).unwrap();

        let output = layer.forward(&input_data).unwrap();

        // Verify output shape
        assert_eq!(output.shape(), &[2, 3, expected_length]);
    }
}

#[test]
fn test_average_pooling_1d_forward() {
    // Create an input tensor with predefined values
    let mut input_data = Array3::<f32>::zeros((2, 3, 8));

    // First batch: use increasing values
    for c in 0..3 {
        for i in 0..8 {
            input_data[[0, c, i]] = (i as f32) + (c as f32 * 0.1);
        }
    }

    // Second batch: use more complex pattern
    for c in 0..3 {
        for i in 0..8 {
            // Generate data using different patterns, such as sine waves or alternating values
            input_data[[1, c, i]] = if i % 2 == 0 {
                (i as f32) * 1.5
            } else {
                (i as f32) * 0.5
            };
        }
    }

    let input = input_data.clone().into_dyn();

    // Create pooling layer, pool size=2, stride=2
    let mut layer = AveragePooling1D::new(2, vec![2, 3, 8], Some(2)).unwrap();

    // Perform forward propagation
    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Verify pooling results for batch 1
    for c in 0..3 {
        let c_offset = c as f32 * 0.1;
        // First window (0+c_offset, 1+c_offset) -> average should be (0+1)/2 + c_offset = 0.5 + c_offset
        assert_relative_eq!(output[[0, c, 0]], 0.5 + c_offset);
        // Second window (2+c_offset, 3+c_offset) -> average should be (2+3)/2 + c_offset = 2.5 + c_offset
        assert_relative_eq!(output[[0, c, 1]], 2.5 + c_offset);
        // Third window (4+c_offset, 5+c_offset) -> average should be (4+5)/2 + c_offset = 4.5 + c_offset
        assert_relative_eq!(output[[0, c, 2]], 4.5 + c_offset);
        // Fourth window (6+c_offset, 7+c_offset) -> average should be (6+7)/2 + c_offset = 6.5 + c_offset
        assert_relative_eq!(output[[0, c, 3]], 6.5 + c_offset);
    }

    // Verify pooling results for batch 2
    for c in 0..3 {
        // First window (0*1.5, 1*0.5) -> average should be (0 + 0.5)/2 = 0.25
        assert_relative_eq!(output[[1, c, 0]], 0.25);
        // Second window (2*1.5, 3*0.5) -> average should be (3 + 1.5)/2 = 2.25
        assert_relative_eq!(output[[1, c, 1]], 2.25);
        // Third window (4*1.5, 5*0.5) -> average should be (6 + 2.5)/2 = 4.25
        assert_relative_eq!(output[[1, c, 2]], 4.25);
        // Fourth window (6*1.5, 7*0.5) -> average should be (9 + 3.5)/2 = 6.25
        assert_relative_eq!(output[[1, c, 3]], 6.25);
    }
}

#[test]
fn test_average_pooling_1d_backward() {
    // Create input tensor
    let mut input_data = Array3::<f32>::zeros((1, 1, 4));
    for i in 0..4 {
        input_data[[0, 0, i]] = i as f32;
    }
    let input = input_data.clone().into_dyn();

    // Create pooling layer, pool size=2, stride=1
    let mut layer = AveragePooling1D::new(2, vec![1, 1, 4], Some(1)).unwrap();

    // Perform forward propagation
    let output = layer.forward(&input).unwrap();

    // Verify output shape and values
    assert_eq!(output.shape(), &[1, 1, 3]);
    assert_relative_eq!(output[[0, 0, 0]], 0.5); // (0+1)/2
    assert_relative_eq!(output[[0, 0, 1]], 1.5); // (1+2)/2
    assert_relative_eq!(output[[0, 0, 2]], 2.5); // (2+3)/2

    // Create upstream gradient - all values are 1.0
    let grad_output = Array3::<f32>::ones((1, 1, 3)).into_dyn();

    // Perform backward propagation
    let grad_input = layer.backward(&grad_output).unwrap();

    // Verify gradient shape
    assert_eq!(grad_input.shape(), &[1, 1, 4]);

    // Verify gradient values
    // Each input element is affected by the gradients of all windows it belongs to
    // For pool size=2, stride=1:
    // Element 0 is in 1 window (window 0)
    // Element 1 is in 2 windows (windows 0,1)
    // Element 2 is in 2 windows (windows 1,2)
    // Element 3 is in 1 window (window 2)
    // Gradient contribution is 1.0/2.0 = 0.5 (average gradient per window)
    assert_relative_eq!(grad_input[[0, 0, 0]], 0.5); // In 1 window
    assert_relative_eq!(grad_input[[0, 0, 1]], 1.0); // In 2 windows
    assert_relative_eq!(grad_input[[0, 0, 2]], 1.0); // In 2 windows
    assert_relative_eq!(grad_input[[0, 0, 3]], 0.5); // In 1 window
}

#[test]
fn test_average_pooling_1d_with_sequential() {
    // Create input tensor
    let mut input_data = Array3::<f32>::zeros((2, 3, 8));

    // Set test data
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..8 {
                input_data[[b, c, i]] = i as f32;
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Create Sequential model
    let mut model = Sequential::new();
    model
        .add(
            AveragePooling1D::new(
                2,             // Pool size
                vec![2, 3, 8], // Input shape
                Some(2),       // Stride (optional)
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Perform prediction
    let output = model.predict(&x).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Verify output values
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c, 0]], 0.5);
            assert_relative_eq!(output[[b, c, 1]], 2.5);
            assert_relative_eq!(output[[b, c, 2]], 4.5);
            assert_relative_eq!(output[[b, c, 3]], 6.5);
        }
    }
}

#[test]
fn test_average_pooling_1d_odd_window_size() {
    // Create input tensor
    let mut input_data = Array3::<f32>::zeros((1, 1, 5));
    for i in 0..5 {
        input_data[[0, 0, i]] = i as f32;
    }
    let input = input_data.clone().into_dyn();

    // Create pooling layer, pool size=3, stride=1
    let mut layer = AveragePooling1D::new(3, vec![1, 1, 5], Some(1)).unwrap();

    // Perform forward propagation
    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[1, 1, 3]);

    // Verify output values
    assert_relative_eq!(output[[0, 0, 0]], 1.0); // (0+1+2)/3
    assert_relative_eq!(output[[0, 0, 1]], 2.0); // (1+2+3)/3
    assert_relative_eq!(output[[0, 0, 2]], 3.0); // (2+3+4)/3
}

#[test]
fn test_layer_type_and_output_shape() {
    let layer = AveragePooling1D::new(2, vec![1, 3, 10], Some(2)).unwrap();

    // Test layer type
    assert_eq!(layer.layer_type(), "AveragePooling1D");

    // Test output shape
    let expected_shape = "(1, 3, 5)"; // (10-2)/2+1 = 5
    assert_eq!(layer.output_shape(), expected_shape);
}
