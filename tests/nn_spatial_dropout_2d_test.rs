use approx::assert_abs_diff_eq;
use ndarray::{Array, Array4, IxDyn};
use rustyml::neural_network::layer::regularization_layer::dropout_layer::spatial_dropout_2d::SpatialDropout2D;
use rustyml::neural_network::neural_network_trait::Layer;

#[test]
fn test_spatial_dropout_2d_forward_pass_dimensions() {
    // Test that spatial dropout preserves input dimensions
    let mut dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();
    let input = Array4::ones((2, 8, 10, 10)).into_dyn();

    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 8, 10, 10]);
    println!(
        "SpatialDropout2D dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_spatial_dropout_2d_training_mode() {
    // Test that spatial dropout drops entire feature maps during training
    let mut dropout = SpatialDropout2D::new(0.5, vec![10, 20, 16, 16]).unwrap();
    dropout.set_training(true);

    let input = Array4::ones((10, 20, 16, 16)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    let shape = output.shape();
    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    // Check that entire feature maps (channels) are dropped (not individual elements)
    let mut dropped_channel_count = 0;
    for b in 0..batch_size {
        for c in 0..channels {
            let mut channel_sum = 0.0;
            for h in 0..height {
                for w in 0..width {
                    channel_sum += output[[b, c, h, w]];
                }
            }

            // Either entire channel is dropped (sum=0) or kept (sum != 0)
            if channel_sum == 0.0 {
                dropped_channel_count += 1;
            } else {
                // Verify all values in this channel are the same (either all 0 or all scaled)
                let first_val = output[[b, c, 0, 0]];
                for h in 0..height {
                    for w in 0..width {
                        assert_abs_diff_eq!(output[[b, c, h, w]], first_val, epsilon = 0.001);
                    }
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
        "SpatialDropout2D training mode test passed: {}/{} channels dropped ({:.1}%)",
        dropped_channel_count,
        total_channels,
        drop_ratio * 100.0
    );
}

#[test]
fn test_spatial_dropout_2d_feature_map_consistency() {
    // Test that when a feature map is kept, all spatial positions have the same mask value
    let mut dropout = SpatialDropout2D::new(0.3, vec![5, 10, 8, 8]).unwrap();
    dropout.set_training(true);

    let input = Array4::from_shape_fn((5, 10, 8, 8), |(b, c, h, w)| {
        (b * 640 + c * 64 + h * 8 + w + 1) as f32 // +1 to avoid zero values
    })
    .into_dyn();

    let output = dropout.forward(&input).unwrap();

    // Check feature map consistency
    for b in 0..5 {
        for c in 0..10 {
            // Calculate the sum of the entire channel to determine if it's dropped
            let mut channel_sum = 0.0;
            for h in 0..8 {
                for w in 0..8 {
                    channel_sum += output[[b, c, h, w]];
                }
            }

            if channel_sum == 0.0 {
                // Channel dropped - all positions should be 0
                for h in 0..8 {
                    for w in 0..8 {
                        assert_eq!(output[[b, c, h, w]], 0.0);
                    }
                }
            } else {
                // Channel kept - all positions should be scaled by same factor
                let scale_factor = output[[b, c, 0, 0]] / input[[b, c, 0, 0]];
                for h in 0..8 {
                    for w in 0..8 {
                        let expected = input[[b, c, h, w]] * scale_factor;
                        assert_abs_diff_eq!(output[[b, c, h, w]], expected, epsilon = 0.001);
                    }
                }
            }
        }
    }

    println!("SpatialDropout2D feature map consistency test passed");
}

#[test]
fn test_spatial_dropout_2d_inference_mode() {
    // Test that dropout passes through input unchanged during inference
    let mut dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();
    dropout.set_training(false);

    let input = Array4::from_shape_fn((2, 8, 10, 10), |(i, j, k, l)| {
        (i * 800 + j * 100 + k * 10 + l) as f32
    })
    .into_dyn();

    let output = dropout.forward(&input).unwrap();

    // During inference, output should be identical to input
    assert_eq!(output, input);
    println!("SpatialDropout2D inference mode test passed: output equals input");
}

#[test]
fn test_spatial_dropout_2d_rate_zero() {
    // Test that dropout with rate=0 keeps all values
    let mut dropout = SpatialDropout2D::new(0.0, vec![2, 8, 10, 10]).unwrap();
    dropout.set_training(true);

    let input = Array4::ones((2, 8, 10, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // All values should be retained
    assert_eq!(output, input);
    println!("SpatialDropout2D rate=0 test passed: all values retained");
}

#[test]
fn test_spatial_dropout_2d_rate_one() {
    // Test that dropout with rate=1 drops all values
    let mut dropout = SpatialDropout2D::new(1.0, vec![2, 8, 10, 10]).unwrap();
    dropout.set_training(true);

    let input = Array4::ones((2, 8, 10, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // All values should be zero
    for val in output.iter() {
        assert_eq!(*val, 0.0);
    }
    println!("SpatialDropout2D rate=1 test passed: all values dropped");
}

#[test]
fn test_spatial_dropout_2d_invalid_rate() {
    // Test that invalid rates are caught
    let dropout_negative = SpatialDropout2D::new(-0.1, vec![2, 8, 10, 10]);
    let dropout_over_one = SpatialDropout2D::new(1.5, vec![2, 8, 10, 10]);

    assert!(dropout_negative.is_err());
    assert!(dropout_over_one.is_err());
    println!("SpatialDropout2D invalid rate test passed");
}

#[test]
fn test_spatial_dropout_2d_shape_validation() {
    // Test that shape mismatch is detected
    let mut dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();
    let wrong_input = Array4::ones((3, 8, 10, 10)).into_dyn();

    let result = dropout.forward(&wrong_input);
    assert!(result.is_err());
    println!("SpatialDropout2D shape validation test passed");
}

#[test]
fn test_spatial_dropout_2d_dimension_validation() {
    // Test that non-4D input is rejected
    let mut dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();

    // Test 3D input
    let input_3d = Array::ones(IxDyn(&[2, 8, 10]));
    let result = dropout.forward(&input_3d);
    assert!(result.is_err());

    // Test 5D input
    let input_5d = Array::ones(IxDyn(&[2, 8, 10, 10, 5]));
    let result = dropout.forward(&input_5d);
    assert!(result.is_err());

    println!("SpatialDropout2D dimension validation test passed");
}

#[test]
fn test_spatial_dropout_2d_backward_pass() {
    // Test backward pass preserves gradient dimensions and applies mask
    let mut dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();
    dropout.set_training(true);

    let input = Array4::ones((2, 8, 10, 10)).into_dyn();
    let output = dropout.forward(&input).unwrap();

    let grad_output = Array4::ones((2, 8, 10, 10)).into_dyn();
    let grad_input = dropout.backward(&grad_output).unwrap();

    // Gradient should have same shape as input
    assert_eq!(grad_input.shape(), input.shape());

    // Gradient should be zero where output was zero (dropped channels)
    for b in 0..2 {
        for c in 0..8 {
            for h in 0..10 {
                for w in 0..10 {
                    let out_val = output[[b, c, h, w]];
                    let grad_val = grad_input[[b, c, h, w]];

                    if out_val == 0.0 {
                        assert_eq!(
                            grad_val, 0.0,
                            "Gradient should be zero where dropout occurred"
                        );
                    }
                }
            }
        }
    }

    println!("SpatialDropout2D backward pass test passed");
}

#[test]
fn test_spatial_dropout_2d_different_rates() {
    // Test spatial dropout with different rates
    let rates = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    for rate in rates {
        let mut dropout = SpatialDropout2D::new(rate, vec![10, 50, 8, 8]).unwrap();
        dropout.set_training(true);

        let input = Array4::ones((10, 50, 8, 8)).into_dyn();
        let output = dropout.forward(&input).unwrap();

        // Count dropped channels
        let mut dropped_channels = 0;
        let total_channels = 10 * 50; // batch_size * channels

        for b in 0..10 {
            for c in 0..50 {
                if output[[b, c, 0, 0]] == 0.0 {
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
            "SpatialDropout2D rate={:.1} test passed: {:.1}% channels dropped (expected {:.1}%)",
            rate,
            dropped_ratio * 100.0,
            expected_ratio * 100.0
        );
    }
}

#[test]
fn test_spatial_dropout_2d_maintains_expected_value() {
    // Test that inverted dropout maintains expected value
    let mut dropout = SpatialDropout2D::new(0.5, vec![20, 30, 12, 12]).unwrap();
    dropout.set_training(true);

    let input = Array4::from_elem((20, 30, 12, 12), 2.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Calculate mean of output
    let sum: f32 = output.iter().sum();
    let total_elements = 20 * 30 * 12 * 12;
    let mean = sum / total_elements as f32;

    // Mean should be close to input value (2.0) due to scaling
    assert_abs_diff_eq!(mean, 2.0, epsilon = 0.3);
    println!(
        "SpatialDropout2D expected value test passed: mean = {:.2} (expected 2.0)",
        mean
    );
}

#[test]
fn test_spatial_dropout_2d_layer_type() {
    // Test that layer type is correctly reported
    let dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();
    assert_eq!(dropout.layer_type(), "SpatialDropout2D");
    println!("SpatialDropout2D layer type test passed");
}

#[test]
fn test_spatial_dropout_2d_output_shape() {
    // Test that output shape is correctly reported
    let dropout = SpatialDropout2D::new(0.5, vec![2, 8, 10, 10]).unwrap();
    let shape_str = dropout.output_shape();
    assert!(shape_str.contains("2") && shape_str.contains("8") && shape_str.contains("10"));
    println!("SpatialDropout2D output shape test passed: {}", shape_str);
}

#[test]
fn test_spatial_dropout_2d_consistency_across_calls() {
    // Test that each forward pass generates different masks
    let mut dropout = SpatialDropout2D::new(0.5, vec![5, 10, 8, 8]).unwrap();
    dropout.set_training(true);

    let input = Array4::ones((5, 10, 8, 8)).into_dyn();

    let output1 = dropout.forward(&input).unwrap();
    let output2 = dropout.forward(&input).unwrap();

    // The two outputs should be different (different random masks)
    assert_ne!(output1, output2);
    println!("SpatialDropout2D consistency test passed: different masks on each forward pass");
}

#[test]
fn test_spatial_dropout_2d_various_shapes() {
    // Test spatial dropout with various input shapes
    let shapes = vec![
        (1, 4, 8, 8),
        (4, 16, 16, 16),
        (8, 32, 32, 32),
        (16, 8, 64, 64),
    ];

    for (batch_size, channels, height, width) in shapes.iter() {
        let mut dropout =
            SpatialDropout2D::new(0.5, vec![*batch_size, *channels, *height, *width]).unwrap();
        dropout.set_training(true);

        let input = Array4::ones((*batch_size, *channels, *height, *width)).into_dyn();
        let output = dropout.forward(&input).unwrap();

        assert_eq!(output.shape(), &[*batch_size, *channels, *height, *width]);
        println!(
            "SpatialDropout2D shape test passed for ({}, {}, {}, {})",
            batch_size, channels, height, width
        );
    }
}

#[test]
fn test_spatial_dropout_2d_scaling() {
    // Test that kept channels are properly scaled by 1/(1-rate)
    let rate = 0.5;
    let expected_scale = 1.0 / (1.0 - rate);

    let mut dropout = SpatialDropout2D::new(rate, vec![2, 10, 8, 8]).unwrap();
    dropout.set_training(true);

    let input = Array4::from_elem((2, 10, 8, 8), 1.0).into_dyn();
    let output = dropout.forward(&input).unwrap();

    // Check that non-zero values are scaled correctly
    for &val in output.iter() {
        if val != 0.0 {
            assert_abs_diff_eq!(val, expected_scale, epsilon = 0.001);
        }
    }

    println!(
        "SpatialDropout2D scaling test passed: kept values scaled by {:.2}",
        expected_scale
    );
}

#[test]
fn test_spatial_dropout_2d_spatial_structure() {
    // Test that entire spatial dimensions are treated uniformly per channel
    let mut dropout = SpatialDropout2D::new(0.5, vec![3, 4, 16, 16]).unwrap();
    dropout.set_training(true);

    let input = Array4::from_shape_fn((3, 4, 16, 16), |(b, c, h, w)| {
        (b as f32 * 100.0) + (c as f32 * 10.0) + (h as f32) + (w as f32 * 0.01) + 1.0
    })
    .into_dyn();

    let output = dropout.forward(&input).unwrap();

    // For each batch and channel, verify spatial consistency
    for b in 0..3 {
        for c in 0..4 {
            // Get the mask value from first spatial position
            let mask_indicator = if output[[b, c, 0, 0]] == 0.0 {
                0.0
            } else {
                1.0
            };

            // All spatial positions in this channel should have the same mask
            for h in 0..16 {
                for w in 0..16 {
                    let current_indicator = if output[[b, c, h, w]] == 0.0 {
                        0.0
                    } else {
                        1.0
                    };
                    assert_eq!(
                        current_indicator, mask_indicator,
                        "Spatial position ({},{}) in batch {} channel {} has inconsistent mask",
                        h, w, b, c
                    );
                }
            }
        }
    }

    println!(
        "SpatialDropout2D spatial structure test passed: all positions in each feature map share the same mask"
    );
}

#[test]
fn test_spatial_dropout_2d_non_square_feature_maps() {
    // Test spatial dropout with non-square feature maps
    let shapes = vec![
        (2, 8, 16, 32), // width > height
        (2, 8, 32, 16), // height > width
        (4, 16, 7, 13), // odd dimensions
    ];

    for (batch_size, channels, height, width) in shapes.iter() {
        let mut dropout =
            SpatialDropout2D::new(0.5, vec![*batch_size, *channels, *height, *width]).unwrap();
        dropout.set_training(true);

        let input = Array4::ones((*batch_size, *channels, *height, *width)).into_dyn();
        let output = dropout.forward(&input).unwrap();

        assert_eq!(output.shape(), &[*batch_size, *channels, *height, *width]);

        // Verify spatial consistency in non-square maps
        for b in 0..*batch_size {
            for c in 0..*channels {
                let first_val = output[[b, c, 0, 0]];
                for h in 0..*height {
                    for w in 0..*width {
                        assert_abs_diff_eq!(output[[b, c, h, w]], first_val, epsilon = 0.001);
                    }
                }
            }
        }

        println!(
            "SpatialDropout2D non-square test passed for ({}, {}, {}, {})",
            batch_size, channels, height, width
        );
    }
}
