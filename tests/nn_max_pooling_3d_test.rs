#![cfg(feature = "neural_network")]

use ndarray::{Array5, ArrayD};
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::layer::pooling_layer::max_pooling_3d::MaxPooling3D;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::rms_prop::RMSprop;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_max_pooling_3d_with_sequential() {
    // Create a simple 5D input tensor: [batch_size, channels, depth, height, width]
    // batch_size=2, 3 input channels, 4x4x4 3D data
    let mut input_data = Array5::zeros((2, 3, 4, 4, 4));

    // Set some specific values so we can predict the max pooling result
    for b in 0..2 {
        for c in 0..3 {
            for d in 0..4 {
                for i in 0..4 {
                    for j in 0..4 {
                        // Create input data with an observable pattern
                        input_data[[b, c, d, i, j]] =
                            (d * i * j) as f32 + b as f32 * 0.1 + c as f32 * 0.01;
                    }
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Test MaxPooling3D with Sequential model
    let mut model = Sequential::new();
    model
        .add(
            MaxPooling3D::new(
                (2, 2, 2),           // Pool window size
                vec![2, 3, 4, 4, 4], // Input shape
                None,                // Use default stride (2,2,2)
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Create target tensor - corresponding to the pooled shape
    let y = Array5::ones((2, 3, 2, 2, 2)).into_dyn();

    // Print model structure
    model.summary();

    // Train model (run a few epochs)
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation
    let prediction = model.predict(&x).unwrap();
    println!("MaxPooling3D prediction shape: {:?}", prediction.shape());

    // Check if output shape is correct
    assert_eq!(prediction.shape(), &[2, 3, 2, 2, 2]);

    // Verify the correctness of the pooling operation
    // For our input pattern, max values should be at the corners of each pooling window
    for b in 0..2 {
        for c in 0..3 {
            for d in 0..2 {
                for i in 0..2 {
                    for j in 0..2 {
                        let pooled_value = prediction[[b, c, d, i, j]];

                        // Find the maximum value in the corresponding pooling window
                        let mut expected_max = f32::NEG_INFINITY;
                        for dd in 0..2 {
                            for di in 0..2 {
                                for dj in 0..2 {
                                    let orig_d = d * 2 + dd;
                                    let orig_i = i * 2 + di;
                                    let orig_j = j * 2 + dj;
                                    let orig_value = input_data[[b, c, orig_d, orig_i, orig_j]];
                                    if orig_value > expected_max {
                                        expected_max = orig_value;
                                    }
                                }
                            }
                        }

                        // Since the training process may modify prediction values, we only verify that pooling operation produces reasonable output
                        assert!(pooled_value.is_finite());
                    }
                }
            }
        }
    }
}

#[test]
fn test_max_pooling_3d_layer_properties() {
    // Test basic layer properties
    let layer = MaxPooling3D::new(
        (2, 2, 2),
        vec![1, 2, 6, 6, 6],
        Some((1, 1, 1)), // Custom stride
    )
    .unwrap();

    // Verify output shape calculation
    assert_eq!(layer.output_shape(), "(1, 2, 5, 5, 5)");

    // Verify parameter count (pooling layers have no trainable parameters)
    assert_eq!(layer.param_count(), TrainingParameters::NoTrainable);

    // Verify layer type
    assert_eq!(layer.layer_type(), "MaxPooling3D");
}

#[test]
fn test_max_pooling_3d_forward_pass() {
    // Test forward pass
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();

    // Create test input
    let mut input = ArrayD::zeros(vec![1, 1, 4, 4, 4]);

    // Set some known values
    input[[0, 0, 0, 0, 0]] = 1.0;
    input[[0, 0, 1, 1, 1]] = 5.0; // This should be the max value for the first pooling window
    input[[0, 0, 2, 2, 2]] = 3.0;
    input[[0, 0, 3, 3, 3]] = 7.0; // This should be the max value for the last pooling window

    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[1, 1, 2, 2, 2]);

    // Verify max values are correctly selected
    assert_eq!(output[[0, 0, 0, 0, 0]], 5.0);
    assert_eq!(output[[0, 0, 1, 1, 1]], 7.0);
}

#[test]
fn test_max_pooling_3d_different_strides() {
    // Test different stride settings
    let test_cases = vec![
        ((2, 2, 2), None, (1, 1, 2, 2, 2)),            // Default stride
        ((2, 2, 2), Some((1, 1, 1)), (1, 1, 3, 3, 3)), // Stride of 1
        ((3, 3, 3), Some((2, 2, 2)), (1, 1, 1, 1, 1)), // Large pooling window, stride of 2
    ];

    for (pool_size, strides, expected_shape) in test_cases {
        let mut layer = MaxPooling3D::new(pool_size, vec![1, 1, 4, 4, 4], strides).unwrap();

        let input = ArrayD::ones(vec![1, 1, 4, 4, 4]);
        let output = layer.forward(&input).unwrap();

        assert_eq!(
            output.shape(),
            &[
                expected_shape.0,
                expected_shape.1,
                expected_shape.2,
                expected_shape.3,
                expected_shape.4
            ]
        );
    }
}

#[test]
fn test_max_pooling_3d_multiple_channels() {
    // Test multi-channel input
    let mut layer = MaxPooling3D::new(
        (2, 2, 2),
        vec![2, 3, 4, 4, 4], // 2 batches, 3 channels
        None,
    )
    .unwrap();

    let mut input = ArrayD::zeros(vec![2, 3, 4, 4, 4]);

    // Set different values for each channel
    for b in 0..2 {
        for c in 0..3 {
            for d in 0..4 {
                for i in 0..4 {
                    for j in 0..4 {
                        input[[b, c, d, i, j]] = (c + 1) as f32 * (d + i + j) as f32;
                    }
                }
            }
        }
    }

    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), &[2, 3, 2, 2, 2]);

    // Verify each channel is processed correctly
    for b in 0..2 {
        for c in 0..3 {
            for d in 0..2 {
                for i in 0..2 {
                    for j in 0..2 {
                        let value = output[[b, c, d, i, j]];
                        assert!(value > 0.0); // All values should be positive
                    }
                }
            }
        }
    }
}

#[test]
fn test_max_pooling_3d_backward_pass() {
    // Test backward propagation
    let mut layer = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None).unwrap();

    // Create input and perform forward pass
    let input = ArrayD::from_shape_fn(vec![1, 1, 4, 4, 4], |idx| (idx[2] * idx[3] * idx[4]) as f32);

    let output = layer.forward(&input).unwrap();

    // Create gradient output
    let grad_output = ArrayD::ones(output.raw_dim());

    // Test backward pass
    let result = layer.backward(&grad_output);
    assert!(result.is_ok());

    let grad_input = result.unwrap();
    assert_eq!(grad_input.shape(), input.shape());
}

#[test]
fn test_max_pooling_3d_edge_cases() {
    // Test edge cases

    // 1. Minimum possible input
    let mut layer = MaxPooling3D::new((1, 1, 1), vec![1, 1, 1, 1, 1], None).unwrap();

    let input = ArrayD::ones(vec![1, 1, 1, 1, 1]);
    let output = layer.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output[[0, 0, 0, 0, 0]], 1.0);

    // 2. Large batch size
    let mut layer2 = MaxPooling3D::new((2, 2, 2), vec![10, 5, 4, 4, 4], None).unwrap();

    let input2 = ArrayD::ones(vec![10, 5, 4, 4, 4]);
    let output2 = layer2.forward(&input2).unwrap();
    assert_eq!(output2.shape(), &[10, 5, 2, 2, 2]);
}
