use approx::assert_abs_diff_eq;
use ndarray::{Array, Array3, IxDyn};
use rustyml::neural_network::layer::regularization_layer::dropout_layer::spatial_dropout_1d::SpatialDropout1D;
use rustyml::neural_network::neural_network_trait::Layer;

#[test]
fn test_spatial_dropout_1d_forward_pass_dimensions() {
    // Test that spatial dropout preserves input dimensions
    let mut dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();
    let input = Array3::ones((2, 8, 10)).into_dyn();

    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 8, 10]);
    println!(
        "SpatialDropout1D dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_spatial_dropout_1d_training_mode() {
    // Test that spatial dropout drops entire channels during training
    let mut dropout = SpatialDropout1D::new(0.5, vec![10, 20, 50]).unwrap();
    dropout.set_training(true);

    let input = Array3::ones((10, 20, 50)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    let shape = output.shape();
    let batch_size = shape[0];
    let channels = shape[1];
    let length = shape[2];

    // Check that entire channels are dropped (not individual elements)
    let mut dropped_channel_count = 0;
    for b in 0..batch_size {
        for c in 0..channels {
            let mut channel_sum = 0.0;
            for l in 0..length {
                channel_sum += output[[b, c, l]];
            }

            // Either entire channel is dropped (sum=0) or kept (sum != 0)
            if channel_sum == 0.0 {
                dropped_channel_count += 1;
            } else {
                // Verify all values in this channel are the same (either all 0 or all scaled)
                let first_val = output[[b, c, 0]];
                for l in 1..length {
                    assert_abs_diff_eq!(output[[b, c, l]], first_val, epsilon = 0.001);
                }
            }
        }
    }

    // With rate=0.5, approximately 50% of channels should be dropped
    let total_channels = batch_size * channels;
    let drop_ratio = dropped_channel_count as f32 / total_channels as f32;

    // Allow statistical variance (30%-70%)
    assert!(
        drop_ratio > 0.3 && drop_ratio < 0.7,
        "Expected ~50% channels dropped, got {:.1}%",
        drop_ratio * 100.0
    );

    println!(
        "SpatialDropout1D training mode test passed: {}/{} channels dropped ({:.1}%)",
        dropped_channel_count,
        total_channels,
        drop_ratio * 100.0
    );
}

#[test]
fn test_spatial_dropout_1d_channel_consistency() {
    // Test that when a channel is kept, all spatial positions have the same mask value
    let mut dropout = SpatialDropout1D::new(0.3, vec![5, 10, 20]).unwrap();
    dropout.set_training(true);

    let input = Array3::from_shape_fn((5, 10, 20), |(b, c, l)| (b * 200 + c * 20 + l + 1) as f32)
        .into_dyn();

    let output = dropout.forward(&input).unwrap();

    // Check channel consistency
    for b in 0..5 {
        for c in 0..10 {
            let first_val = output[[b, c, 0]];
            let input_first = input[[b, c, 0]];

            if first_val == 0.0 {
                // Channel dropped - all positions should be 0
                for l in 0..20 {
                    assert_eq!(output[[b, c, l]], 0.0);
                }
            } else {
                // Channel kept - all positions should be scaled by same factor
                let scale_factor = first_val / input_first;
                for l in 1..20 {
                    let expected = input[[b, c, l]] * scale_factor;
                    assert_abs_diff_eq!(output[[b, c, l]], expected, epsilon = 0.001);
                }
            }
        }
    }

    println!("SpatialDropout1D channel consistency test passed");
}

#[test]
fn test_spatial_dropout_1d_inference_mode() {
    // Test that dropout passes through input unchanged during inference
    let mut dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();
    dropout.set_training(false);

    let input =
        Array3::from_shape_fn((2, 8, 10), |(i, j, k)| (i * 80 + j * 10 + k) as f32).into_dyn();

    let output = dropout.forward(&input).unwrap();

    // During inference, output should be identical to input
    assert_eq!(output, input);
    println!("SpatialDropout1D inference mode test passed: output equals input");
}

#[test]
fn test_spatial_dropout_1d_rate_zero() {
    // Test that dropout with rate=0 keeps all values
    let mut dropout = SpatialDropout1D::new(0.0, vec![2, 8, 10]).unwrap();
    dropout.set_training(true);

    let input = Array3::ones((2, 8, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // All values should be retained
    assert_eq!(output, input);
    println!("SpatialDropout1D rate=0 test passed: all values retained");
}

#[test]
fn test_spatial_dropout_1d_rate_one() {
    // Test that dropout with rate=1 drops all values
    let mut dropout = SpatialDropout1D::new(1.0, vec![2, 8, 10]).unwrap();
    dropout.set_training(true);

    let input = Array3::ones((2, 8, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // All values should be zero
    for val in output.iter() {
        assert_eq!(*val, 0.0);
    }
    println!("SpatialDropout1D rate=1 test passed: all values dropped");
}

#[test]
fn test_spatial_dropout_1d_invalid_rate() {
    // Test that invalid rates are caught
    let dropout_negative = SpatialDropout1D::new(-0.1, vec![2, 8, 10]);
    let dropout_over_one = SpatialDropout1D::new(1.5, vec![2, 8, 10]);

    assert!(dropout_negative.is_err());
    assert!(dropout_over_one.is_err());
    println!("SpatialDropout1D invalid rate test passed");
}

#[test]
fn test_spatial_dropout_1d_shape_validation() {
    // Test that shape mismatch is detected
    let mut dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();
    let wrong_input = Array3::ones((3, 8, 10)).into_dyn();

    let result = dropout.forward(&wrong_input);
    assert!(result.is_err());
    println!("SpatialDropout1D shape validation test passed");
}

#[test]
fn test_spatial_dropout_1d_dimension_validation() {
    // Test that non-3D input is rejected
    let mut dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();

    // Test 2D input
    let input_2d = Array::ones((2, 8)).into_dyn();
    let result = dropout.forward(&input_2d);
    assert!(result.is_err());

    // Test 4D input
    let input_4d = Array::ones(IxDyn(&[2, 8, 10, 5]));
    let result = dropout.forward(&input_4d);
    assert!(result.is_err());

    println!("SpatialDropout1D dimension validation test passed");
}

#[test]
fn test_spatial_dropout_1d_backward_pass() {
    // Test backward pass preserves gradient dimensions and applies mask
    let mut dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();
    dropout.set_training(true);

    let input = Array3::ones((2, 8, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 8, 10)).into_dyn();
    let grad_input = dropout.backward(&grad_output).unwrap();

    // Gradient should have same shape as input
    assert_eq!(grad_input.shape(), input.shape());

    // Gradient should be zero where output was zero (dropped channels)
    for b in 0..2 {
        for c in 0..8 {
            for l in 0..10 {
                let out_val = output[[b, c, l]];
                let grad_val = grad_input[[b, c, l]];

                if out_val == 0.0 {
                    assert_eq!(
                        grad_val, 0.0,
                        "Gradient should be zero where dropout occurred"
                    );
                }
            }
        }
    }

    println!("SpatialDropout1D backward pass test passed");
}

#[test]
fn test_spatial_dropout_1d_different_rates() {
    // Test spatial dropout with different rates
    let rates = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    for rate in rates {
        let mut dropout = SpatialDropout1D::new(rate, vec![10, 50, 20]).unwrap();
        dropout.set_training(true);

        let input = Array3::ones((10, 50, 20)).into_dyn();
        let output = dropout.forward(&input).unwrap();

        // Count dropped channels
        let mut dropped_channels = 0;
        let total_channels = 10 * 50; // batch_size * channels

        for b in 0..10 {
            for c in 0..50 {
                if output[[b, c, 0]] == 0.0 {
                    dropped_channels += 1;
                }
            }
        }

        let dropped_ratio = dropped_channels as f32 / total_channels as f32;
        let expected_ratio = rate;

        // Allow 15% variance for statistical fluctuation
        assert!(
            (dropped_ratio - expected_ratio).abs() < 0.15,
            "Rate {:.1}: expected ~{:.1}% dropped, got {:.1}%",
            rate,
            expected_ratio * 100.0,
            dropped_ratio * 100.0
        );

        println!(
            "SpatialDropout1D rate={:.1} test passed: {:.1}% channels dropped (expected {:.1}%)",
            rate,
            dropped_ratio * 100.0,
            expected_ratio * 100.0
        );
    }
}

#[test]
fn test_spatial_dropout_1d_maintains_expected_value() {
    // Test that inverted dropout maintains expected value
    let mut dropout = SpatialDropout1D::new(0.5, vec![20, 30, 40]).unwrap();
    dropout.set_training(true);

    let input = Array3::from_elem((20, 30, 40), 2.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Calculate mean of output
    let sum: f32 = output.iter().sum();
    let total_elements = 20 * 30 * 40;
    let mean = sum / total_elements as f32;

    // Mean should be close to input value (2.0) due to scaling
    assert_abs_diff_eq!(mean, 2.0, epsilon = 0.3);
    println!(
        "SpatialDropout1D expected value test passed: mean = {:.2} (expected 2.0)",
        mean
    );
}

#[test]
fn test_spatial_dropout_1d_layer_type() {
    // Test that layer type is correctly reported
    let dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();
    assert_eq!(dropout.layer_type(), "SpatialDropout1D");
    println!("SpatialDropout1D layer type test passed");
}

#[test]
fn test_spatial_dropout_1d_output_shape() {
    // Test that output shape is correctly reported
    let dropout = SpatialDropout1D::new(0.5, vec![2, 8, 10]).unwrap();
    let shape_str = dropout.output_shape();
    assert!(shape_str.contains("2") && shape_str.contains("8") && shape_str.contains("10"));
    println!("SpatialDropout1D output shape test passed: {}", shape_str);
}

#[test]
fn test_spatial_dropout_1d_consistency_across_calls() {
    // Test that each forward pass generates different masks
    let mut dropout = SpatialDropout1D::new(0.5, vec![5, 10, 20]).unwrap();
    dropout.set_training(true);

    let input = Array3::ones((5, 10, 20)).into_dyn();

    let output1 = dropout.forward(&input).unwrap();
    let output2 = dropout.forward(&input).unwrap();

    // The two outputs should be different (different random masks)
    assert_ne!(output1, output2);
    println!("SpatialDropout1D consistency test passed: different masks on each forward pass");
}

#[test]
fn test_spatial_dropout_1d_various_shapes() {
    // Test spatial dropout with various input shapes
    let shapes = vec![(1, 4, 8), (4, 16, 32), (8, 32, 64), (16, 8, 128)];

    for (batch_size, channels, length) in shapes.iter() {
        let mut dropout =
            SpatialDropout1D::new(0.5, vec![*batch_size, *channels, *length]).unwrap();
        dropout.set_training(true);

        let input = Array3::ones((*batch_size, *channels, *length)).into_dyn();
        let output = dropout.forward(&input).unwrap();

        assert_eq!(output.shape(), &[*batch_size, *channels, *length]);
        println!(
            "SpatialDropout1D shape test passed for ({}, {}, {})",
            batch_size, channels, length
        );
    }
}

#[test]
fn test_spatial_dropout_1d_scaling() {
    // Test that kept channels are properly scaled by 1/(1-rate)
    let rate = 0.5;
    let expected_scale = 1.0 / (1.0 - rate);

    let mut dropout = SpatialDropout1D::new(rate, vec![2, 10, 5]).unwrap();
    dropout.set_training(true);

    let input = Array3::from_elem((2, 10, 5), 1.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Check that non-zero values are scaled correctly
    for &val in output.iter() {
        if val != 0.0 {
            assert_abs_diff_eq!(val, expected_scale, epsilon = 0.001);
        }
    }

    println!(
        "SpatialDropout1D scaling test passed: kept values scaled by {:.2}",
        expected_scale
    );
}
