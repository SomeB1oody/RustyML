use super::*;

#[test]
fn test_separable_conv2d_basic() {
    // Create simple 4D input tensor: [batch_size, channels, height, width]
    let x = Array4::ones((1, 3, 16, 16)).into_dyn();

    // Create Sequential model
    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                32,                 // filters
                (3, 3),             // kernel_size
                vec![1, 3, 16, 16], // input_shape
                (1, 1),             // strides
                PaddingType::Same,  // padding
                1,                  // depth_multiplier
                ReLU::new(),        // activation
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Forward pass
    let output = model.predict(&x);

    // Verify output shape: [1, 32, 16, 16]
    assert_eq!(output.shape(), &[1, 32, 16, 16]);

    // Verify all output values are non-negative (due to ReLU activation)
    for &val in output.iter() {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_separable_conv2d_different_depth_multiplier() {
    let x = Array4::ones((2, 4, 8, 8)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                16,                // filters
                (3, 3),            // kernel_size
                vec![2, 4, 8, 8],  // input_shape
                (1, 1),            // strides
                PaddingType::Same, // padding
                2,                 // depth_multiplier = 2
                Linear::new(),     // no activation
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    let output = model.predict(&x);

    // Verify output shape
    assert_eq!(output.shape(), &[2, 16, 8, 8]);
}

#[test]
fn test_separable_conv2d_valid_padding() {
    let x = Array4::ones((1, 2, 10, 10)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                8,                  // filters
                (3, 3),             // kernel_size
                vec![1, 2, 10, 10], // input_shape
                (1, 1),             // strides
                PaddingType::Valid, // valid padding
                1,                  // depth_multiplier
                Sigmoid::new(),     // sigmoid activation
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let output = model.predict(&x);

    // Valid padding: output size = (10-3)/1 + 1 = 8
    assert_eq!(output.shape(), &[1, 8, 8, 8]);

    // Verify sigmoid activation output values are in [0,1] range
    for &val in output.iter() {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_separable_conv2d_with_strides() {
    let x = Array4::ones((1, 3, 32, 32)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                64,                 // filters
                (3, 3),             // kernel_size
                vec![1, 3, 32, 32], // input_shape
                (2, 2),             // strides = 2
                PaddingType::Same,  // same padding
                1,                  // depth_multiplier
                Tanh::new(),        // tanh activation
            )
            .unwrap(),
        )
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    let output = model.predict(&x);

    // Same padding with stride 2: output size = ceil(32/2) = 16
    assert_eq!(output.shape(), &[1, 64, 16, 16]);

    // Verify tanh activation output values are in [-1,1] range
    for &val in output.iter() {
        assert!(val >= -1.0 && val <= 1.0);
    }
}

#[test]
fn test_separable_conv2d_training() {
    // Create simple training data
    let x = Array4::from_elem((2, 3, 8, 8), 0.5).into_dyn();
    let y = Array4::from_elem((2, 16, 8, 8), 1.0).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                16,                // filters
                (3, 3),            // kernel_size
                vec![2, 3, 8, 8],  // input_shape
                (1, 1),            // strides
                PaddingType::Same, // same padding
                1,                 // depth_multiplier
                ReLU::new(),       // activation
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Train the model
    let result = model.fit(&x, &y, 3);
    assert!(result.is_ok());

    // Verify model can make predictions
    let output = model.predict(&x);
    assert_eq!(output.shape(), &[2, 16, 8, 8]);
}

#[test]
fn test_separable_conv2d_multiple_training_calls() {
    let x = Array4::ones((1, 2, 4, 4)).into_dyn();
    let y = Array4::ones((1, 8, 4, 4)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                8,                 // filters
                (3, 3),            // kernel_size
                vec![1, 2, 4, 4],  // input_shape
                (1, 1),            // strides
                PaddingType::Same, // padding
                1,                 // depth_multiplier
                Linear::new(),     // no activation
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // First training
    let result1 = model.fit(&x, &y, 1);
    assert!(result1.is_ok());

    // Second training should also work normally
    let result2 = model.fit(&x, &y, 1);
    assert!(result2.is_ok());
}

#[test]
fn test_separable_conv2d_different_optimizers() {
    let x = Array4::ones((1, 3, 16, 16)).into_dyn();
    let y = Array4::ones((1, 8, 16, 16)).into_dyn();

    // Test SGD optimizer
    let mut model_sgd = Sequential::new();
    model_sgd
        .add(
            SeparableConv2D::new(
                8,
                (3, 3),
                vec![1, 3, 16, 16],
                (1, 1),
                PaddingType::Same,
                1,
                Linear::new(),
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let result_sgd = model_sgd.fit(&x, &y, 1);
    assert!(result_sgd.is_ok());

    // Test Adam optimizer
    let mut model_adam = Sequential::new();
    model_adam
        .add(
            SeparableConv2D::new(
                8,
                (3, 3),
                vec![1, 3, 16, 16],
                (1, 1),
                PaddingType::Same,
                1,
                Linear::new(),
            )
            .unwrap(),
        )
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    let result_adam = model_adam.fit(&x, &y, 1);
    assert!(result_adam.is_ok());

    // Test RMSprop optimizer
    let mut model_rmsprop = Sequential::new();
    model_rmsprop
        .add(
            SeparableConv2D::new(
                8,
                (3, 3),
                vec![1, 3, 16, 16],
                (1, 1),
                PaddingType::Same,
                1,
                Linear::new(),
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    let result_rmsprop = model_rmsprop.fit(&x, &y, 1);
    assert!(result_rmsprop.is_ok());
}

#[test]
fn test_separable_conv2d_batch_processing() {
    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8];

    for &batch_size in &batch_sizes {
        let x = Array4::ones((batch_size, 3, 8, 8)).into_dyn();

        let mut model = Sequential::new();
        model
            .add(
                SeparableConv2D::new(
                    16,
                    (3, 3),
                    vec![batch_size, 3, 8, 8],
                    (1, 1),
                    PaddingType::Same,
                    1,
                    Linear::new(),
                )
                .unwrap(),
            )
            .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

        let output = model.predict(&x);
        assert_eq!(output.shape(), &[batch_size, 16, 8, 8]);
    }
}

#[test]
fn test_separable_conv2d_large_kernel() {
    let x = Array4::ones((1, 4, 32, 32)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            SeparableConv2D::new(
                64,                 // filters
                (5, 5),             // larger kernel
                vec![1, 4, 32, 32], // input_shape
                (1, 1),             // strides
                PaddingType::Same,  // padding
                1,                  // depth_multiplier
                ReLU::new(),        // activation
            )
            .unwrap(),
        )
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    let output = model.predict(&x);
    assert_eq!(output.shape(), &[1, 64, 32, 32]);
}

#[test]
fn test_separable_conv2d_get_weights() {
    let mut model = Sequential::new();
    model.add(
        SeparableConv2D::new(
            8,
            (3, 3),
            vec![1, 3, 16, 16],
            (1, 1),
            PaddingType::Same,
            1,
            Linear::new(),
        )
        .unwrap(),
    );

    let weights = model.get_weights();
    assert_eq!(weights.len(), 1);

    // Verify weight type
    match &weights[0] {
        LayerWeight::SeparableConv2DLayer(_) => {
            // Correct weight type
        }
        _ => panic!("Expected Conv2D weight type"),
    }
}

#[test]
fn test_separable_conv2d_output_shape_calculation() {
    let layer = SeparableConv2D::new(
        32,
        (3, 3),
        vec![2, 16, 64, 64],
        (2, 2),
        PaddingType::Valid,
        1,
        Linear::new(),
    )
    .unwrap();

    // Valid padding with stride 2: (64-3)/2 + 1 = 31
    let expected_shape = "(2, 32, 31, 31)";
    assert_eq!(layer.output_shape(), expected_shape);
}
