use ndarray::{s, Array, Array3};
use rustyml::neural_network::layer::layer_weight::LayerWeight;
use rustyml::neural_network::layer::regularization_layer::normalization_layer::group_normalization::GroupNormalization;
use rustyml::neural_network::layer::regularization_layer::normalization_layer::instance_normalization::InstanceNormalization;
use rustyml::neural_network::layer::serialize_weight::SerializableLayerWeight;
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::neural_network_trait::Layer;

#[test]
fn test_group_normalization_forward_pass_dimensions() {
    // Test forward propagation dimension correctness
    let mut gn_layer = GroupNormalization::new(vec![4, 8, 16], 4, 1, 1e-5).unwrap();
    let input = Array3::ones((4, 8, 16)).into_dyn(); // batch=4, channels=8, spatial=16

    let output = gn_layer.forward(&input).unwrap();
    assert_eq!(output.shape(), &[4, 8, 16]);
    println!(
        "Forward pass dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_group_normalization_3d_input() {
    // Test group normalization with 3D input (batch, channels, spatial)
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    // Create input with varying values
    let input = Array::from_shape_vec(
        (2, 4, 8),
        vec![
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 0, Channel 1
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // Batch 0, Channel 2
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, // Batch 0, Channel 3
            25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, // Batch 1, Channel 0
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Batch 1, Channel 1
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, // Batch 1, Channel 2
            18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, // Batch 1, Channel 3
            26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
        ],
    )
    .unwrap()
    .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    // Check that the output has the same shape
    assert_eq!(output.shape(), &[2, 4, 8]);

    // For group normalization, each group (2 channels per group) should be normalized
    // We can verify the output is finite and within reasonable bounds
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should be finite"
    );

    println!("3D input normalization test passed");
}

#[test]
fn test_group_normalization_4d_input() {
    // Test group normalization with 4D input (batch, channels, height, width)
    let batch_size = 2;
    let channels = 8;
    let height = 4;
    let width = 4;
    let num_groups = 4;

    let mut gn_layer = GroupNormalization::new(
        vec![batch_size, channels, height, width],
        num_groups,
        1,
        1e-5,
    )
    .unwrap();

    let input = Array::from_shape_fn((batch_size, channels, height, width), |(b, c, h, w)| {
        (b * 100 + c * 10 + h + w) as f32
    })
    .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, channels, height, width]);

    // Verify output is finite
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should be finite"
    );

    println!("4D input normalization test passed");
}

#[test]
fn test_group_normalization_group_statistics() {
    // Test that group normalization computes statistics correctly within each group
    let batch_size = 1;
    let channels = 4;
    let spatial = 4;
    let num_groups = 2; // 2 channels per group

    let mut gn_layer =
        GroupNormalization::new(vec![batch_size, channels, spatial], num_groups, 1, 1e-5).unwrap();

    // Create input where each group has different statistics
    let mut input_vec = vec![0.0; batch_size * channels * spatial];

    // Group 0 (channels 0-1): values around 10
    for i in 0..(2 * spatial) {
        input_vec[i] = 10.0 + i as f32;
    }

    // Group 1 (channels 2-3): values around 100
    for i in (2 * spatial)..(4 * spatial) {
        input_vec[i] = 100.0 + i as f32;
    }

    let input = Array::from_shape_vec((batch_size, channels, spatial), input_vec)
        .unwrap()
        .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, channels, spatial]);

    // Each group should be normalized independently
    // Verify output values are within normalized range (roughly [-3, 3] for normalized data)
    assert!(
        output.iter().all(|&x| x.abs() < 10.0),
        "Normalized values should be within reasonable bounds"
    );

    println!("Group statistics test passed");
}

#[test]
fn test_group_normalization_invalid_num_groups() {
    // Test that num_groups must divide num_channels evenly
    let mut gn_layer = GroupNormalization::new(vec![2, 7, 4], 3, 1, 1e-5).unwrap(); // 7 channels, 3 groups (not divisible)

    let input = Array::from_shape_fn((2, 7, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let result = gn_layer.forward(&input);

    assert!(
        result.is_err(),
        "Should return error when channels not divisible by num_groups"
    );
    println!("Invalid num_groups test passed");
}

#[test]
fn test_group_normalization_single_group() {
    // Test with num_groups = 1 (equivalent to Layer Normalization)
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 1, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b * 10 + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4, 8]);
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should be finite"
    );

    println!("Single group (layer norm equivalent) test passed");
}

#[test]
fn test_group_normalization_all_groups() {
    // Test with num_groups = num_channels (equivalent to Instance Normalization)
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 4, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b * 10 + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4, 8]);
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should be finite"
    );

    println!("All groups (instance norm equivalent) test passed");
}

#[test]
fn test_group_normalization_invalid_channel_axis() {
    // Test that channel axis cannot be 0 (batch axis)
    let gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 0, 1e-5);

    assert!(gn_layer.is_err());
    println!("Invalid channel axis test passed");
}

#[test]
fn test_group_normalization_training_mode() {
    // Test that group normalization works in training mode
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let input =
        Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b * 100 + c * 10 + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4, 8]);
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should be finite"
    );

    println!("Training mode test passed");
}

#[test]
fn test_group_normalization_inference_mode() {
    // Test that group normalization works in inference mode
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let train_input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b + c + s) as f32).into_dyn();

    // First do a forward pass in training mode
    gn_layer.set_training(true);
    gn_layer.forward(&train_input).unwrap();

    // Switch to inference mode
    gn_layer.set_training(false);

    let test_input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b * 2 + c + s) as f32).into_dyn();

    let output = gn_layer.forward(&test_input).unwrap();
    assert_eq!(output.shape(), &[2, 4, 8]);

    println!("Inference mode test passed");
}

#[test]
fn test_group_normalization_backward_pass() {
    // Test backward pass gradient computation
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b * 10 + c * 5 + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let _output = gn_layer.forward(&input).unwrap();

    // Create gradient
    let grad_output = Array3::ones((2, 4, 8)).into_dyn();

    let grad_input = gn_layer.backward(&grad_output).unwrap();

    // Verify gradient dimensions
    assert_eq!(grad_input.shape(), input.shape());

    // Verify gradients are finite
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "Gradients should be finite"
    );

    println!("Backward pass test passed");
}

#[test]
fn test_group_normalization_parameter_update_sgd() {
    // Test parameter update with SGD optimizer
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let _output = gn_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 4, 8)).into_dyn();
    let _grad_input = gn_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::GroupNormalizationLayer(weights) = gn_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters
        gn_layer.update_parameters_sgd(0.01);

        // Get updated weights
        if let LayerWeight::GroupNormalizationLayer(updated_weights) = gn_layer.get_weights() {
            // Verify parameters have changed
            let gamma_changed = updated_weights
                .gamma
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_gamma.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            let beta_changed = updated_weights
                .beta
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_beta.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            assert!(
                gamma_changed || beta_changed,
                "Parameters should be updated"
            );
        }
    }

    println!("SGD parameter update test passed");
}

#[test]
fn test_group_normalization_parameter_update_adam() {
    // Test parameter update with Adam optimizer
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let _output = gn_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 4, 8)).into_dyn();
    let _grad_input = gn_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::GroupNormalizationLayer(weights) = gn_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with Adam
        gn_layer.update_parameters_adam(0.001, 0.9, 0.999, 1e-8, 1);

        // Get updated weights
        if let LayerWeight::GroupNormalizationLayer(updated_weights) = gn_layer.get_weights() {
            // Verify parameters have changed
            let gamma_changed = updated_weights
                .gamma
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_gamma.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            let beta_changed = updated_weights
                .beta
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_beta.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            assert!(
                gamma_changed || beta_changed,
                "Parameters should be updated"
            );
        }
    }

    println!("Adam parameter update test passed");
}

#[test]
fn test_group_normalization_parameter_update_rmsprop() {
    // Test parameter update with RMSprop optimizer
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let _output = gn_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 4, 8)).into_dyn();
    let _grad_input = gn_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::GroupNormalizationLayer(weights) = gn_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with RMSprop
        gn_layer.update_parameters_rmsprop(0.001, 0.9, 1e-8);

        // Get updated weights
        if let LayerWeight::GroupNormalizationLayer(updated_weights) = gn_layer.get_weights() {
            // Verify parameters have changed
            let gamma_changed = updated_weights
                .gamma
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_gamma.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            let beta_changed = updated_weights
                .beta
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_beta.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            assert!(
                gamma_changed || beta_changed,
                "Parameters should be updated"
            );
        }
    }

    println!("RMSprop parameter update test passed");
}

#[test]
fn test_group_normalization_parameter_update_adagrad() {
    // Test parameter update with AdaGrad optimizer
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (b + c + s) as f32).into_dyn();

    gn_layer.set_training(true);
    let _output = gn_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 4, 8)).into_dyn();
    let _grad_input = gn_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::GroupNormalizationLayer(weights) = gn_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with AdaGrad
        gn_layer.update_parameters_ada_grad(0.01, 1e-8);

        // Get updated weights
        if let LayerWeight::GroupNormalizationLayer(updated_weights) = gn_layer.get_weights() {
            // Verify parameters have changed
            let gamma_changed = updated_weights
                .gamma
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_gamma.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            let beta_changed = updated_weights
                .beta
                .as_slice()
                .unwrap()
                .iter()
                .zip(initial_beta.as_slice().unwrap().iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

            assert!(
                gamma_changed || beta_changed,
                "Parameters should be updated"
            );
        }
    }

    println!("AdaGrad parameter update test passed");
}

#[test]
fn test_group_normalization_parameter_count() {
    // Test parameter count correctness
    let gn_layer = GroupNormalization::new(vec![4, 8, 16], 4, 1, 1e-5).unwrap();
    let expected_params = 8 + 8; // gamma + beta (one per channel)
    assert_eq!(
        gn_layer.param_count(),
        TrainingParameters::Trainable(expected_params)
    );
    println!(
        "Parameter count test passed: {} parameters",
        expected_params
    );
}

#[test]
fn test_group_normalization_layer_type() {
    // Test layer type identification
    let gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();
    assert_eq!(gn_layer.layer_type(), "GroupNormalization");
    println!("Layer type test passed");
}

#[test]
fn test_group_normalization_output_shape() {
    // Test output shape string formatting
    let gn_layer = GroupNormalization::new(vec![4, 8, 16], 4, 1, 1e-5).unwrap();
    let output_shape = gn_layer.output_shape();
    assert_eq!(output_shape, "(4, 8, 16)");
    println!("Output shape test passed: {}", output_shape);
}

#[test]
fn test_group_normalization_set_weights() {
    // Test manual weight setting
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    let new_gamma = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0]).into_dyn();
    let new_beta = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]).into_dyn();

    gn_layer.set_weights(new_gamma.clone(), new_beta.clone());

    // Verify weights were set correctly
    if let LayerWeight::GroupNormalizationLayer(weights) = gn_layer.get_weights() {
        assert_eq!(
            weights.gamma.as_slice().unwrap(),
            new_gamma.as_slice().unwrap()
        );
        assert_eq!(
            weights.beta.as_slice().unwrap(),
            new_beta.as_slice().unwrap()
        );
    }

    println!("Set weights test passed");
}

#[test]
fn test_group_normalization_epsilon_effect() {
    // Test that epsilon prevents division by zero
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    // Create input where each group has constant values (zero variance)
    let mut input_vec = vec![0.0; 2 * 4 * 8];
    // Set constant values for each group
    for batch in 0..2 {
        for group in 0..2 {
            let channels_per_group = 2;
            for ch in 0..channels_per_group {
                let channel_idx = group * channels_per_group + ch;
                let base_idx = (batch * 4 + channel_idx) * 8;
                let constant_val = (batch * 10 + group) as f32;
                for spatial in 0..8 {
                    input_vec[base_idx + spatial] = constant_val;
                }
            }
        }
    }

    let input = Array::from_shape_vec((2, 4, 8), input_vec)
        .unwrap()
        .into_dyn();

    gn_layer.set_training(true);
    let result = gn_layer.forward(&input);

    // Should not panic or produce NaN/Inf
    assert!(
        result.is_ok(),
        "Forward pass should succeed with zero variance"
    );

    let output = result.unwrap();
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should not contain NaN or Inf"
    );

    println!("Epsilon effect test passed");
}

#[test]
fn test_group_normalization_vs_instance_normalization() {
    // Demonstrate the difference between group norm and instance norm
    let input =
        Array::from_shape_fn((2, 8, 4), |(b, c, s)| (b * 100 + c * 10 + s) as f32).into_dyn();

    // Group normalization with 4 groups (2 channels per group)
    let mut gn_layer = GroupNormalization::new(vec![2, 8, 4], 4, 1, 1e-5).unwrap();
    gn_layer.set_training(true);
    let gn_output = gn_layer.forward(&input).unwrap();

    // Instance normalization (equivalent to num_groups = num_channels)
    let mut in_layer = InstanceNormalization::new(vec![2, 8, 4], 1, 1e-5).unwrap();
    in_layer.set_training(true);
    let in_output = in_layer.forward(&input).unwrap();

    // The outputs should be different (unless all inputs are identical)
    let outputs_different = gn_output
        .iter()
        .zip(in_output.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);

    assert!(
        outputs_different,
        "Group norm (4 groups) and instance norm should produce different outputs"
    );

    println!("Group norm vs instance norm difference test passed");
}

#[test]
fn test_group_normalization_large_batch_parallel() {
    // Test parallel computation with large tensor (should trigger parallel path)
    let batch_size = 8;
    let channels = 16;
    let spatial = 32;
    let num_groups = 8;

    let mut gn_layer =
        GroupNormalization::new(vec![batch_size, channels, spatial], num_groups, 1, 1e-5).unwrap();
    let input = Array::from_shape_fn((batch_size, channels, spatial), |(b, c, s)| {
        (b * 1000 + c * 100 + s) as f32
    })
    .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, channels, spatial]);
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Output should be finite"
    );

    println!("Large batch parallel computation test passed");
}

#[test]
fn test_group_normalization_gradient_flow() {
    // Test that gradients flow properly through the layer
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    // Use varied input to ensure non-uniform gradients
    let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| {
        ((b * 100 + c * 10 + s) as f32) * 0.5 + 1.0
    })
    .into_dyn();

    gn_layer.set_training(true);
    let _output = gn_layer.forward(&input).unwrap();

    // Use non-uniform gradient output
    let grad_output = Array::from_shape_fn((2, 4, 8), |(b, c, s)| {
        ((b * 3 + c * 2 + s) as f32) * 0.1 + 0.5
    })
    .into_dyn();

    let grad_input = gn_layer.backward(&grad_output).unwrap();

    // Verify gradients are not all zero (gradients are flowing)
    let has_nonzero_grad = grad_input.iter().any(|&x| x.abs() > 1e-6);
    assert!(has_nonzero_grad, "Gradients should flow through the layer");

    // Verify no NaN or Inf in gradients
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "Gradients should be finite"
    );

    println!("Gradient flow test passed");
}

#[test]
fn test_group_normalization_multiple_forward_backward() {
    // Test multiple forward and backward passes
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    for i in 0..5 {
        let input = Array::from_shape_fn((2, 4, 8), |(b, c, s)| (i + b + c + s) as f32).into_dyn();

        gn_layer.set_training(true);
        let output = gn_layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 4, 8]);

        let grad_output = Array3::ones((2, 4, 8)).into_dyn();
        let grad_input = gn_layer.backward(&grad_output).unwrap();

        assert_eq!(grad_input.shape(), &[2, 4, 8]);

        gn_layer.update_parameters_sgd(0.01);
    }

    println!("Multiple forward-backward passes test passed");
}

#[test]
fn test_group_normalization_serialization() {
    // Test weight serialization and deserialization
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    // Do a forward pass to initialize
    let input = Array3::ones((2, 4, 8)).into_dyn();
    gn_layer.set_training(true);
    gn_layer.forward(&input).unwrap();

    // Get weights and convert to serializable format
    let weights = gn_layer.get_weights();
    let serializable_weights = SerializableLayerWeight::from_layer_weight(&weights);

    // Verify we can convert back (this tests the serialization structure)
    match serializable_weights {
        SerializableLayerWeight::GroupNormalization(w) => {
            assert_eq!(w.gamma.len(), 4); // 4 channels
            assert_eq!(w.beta.len(), 4);
            assert_eq!(w.shape, vec![4]);
        }
        _ => panic!("Expected GroupNormalization weights"),
    }

    println!("Serialization test passed");
}

#[test]
fn test_group_normalization_different_group_sizes() {
    // Test with different numbers of groups
    let channels = 16;
    let group_configs = vec![1, 2, 4, 8, 16]; // Different num_groups

    for num_groups in group_configs {
        let mut gn_layer =
            GroupNormalization::new(vec![2, channels, 8], num_groups, 1, 1e-5).unwrap();
        let input =
            Array::from_shape_fn((2, channels, 8), |(b, c, s)| (b * 100 + c * 10 + s) as f32)
                .into_dyn();

        gn_layer.set_training(true);
        let output = gn_layer.forward(&input).unwrap();

        assert_eq!(
            output.shape(),
            &[2, channels, 8],
            "Output shape should match input shape"
        );

        println!(
            "Group config {} groups (channels_per_group={}) test passed: {:?} -> {:?}",
            num_groups,
            channels / num_groups,
            input.shape(),
            output.shape()
        );
    }
}

#[test]
fn test_group_normalization_custom_channel_axis() {
    // Test group normalization with custom channel axis
    // Format: [batch, height, width, channels]
    let batch_size = 2;
    let height = 4;
    let width = 4;
    let channels = 8;
    let num_groups = 4;

    let mut gn_layer = GroupNormalization::new(
        vec![batch_size, height, width, channels],
        num_groups,
        3,
        1e-5,
    )
    .unwrap();

    let input = Array::from_shape_fn((batch_size, height, width, channels), |(b, h, w, c)| {
        (b * 100 + c * 10 + h + w) as f32
    })
    .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, height, width, channels]);

    println!("Custom channel axis test passed");
}

#[test]
fn test_group_normalization_5d_input() {
    // Test with 5D input (batch, channels, depth, height, width)
    let batch_size = 2;
    let channels = 8;
    let depth = 2;
    let height = 3;
    let width = 3;
    let num_groups = 4;

    let mut gn_layer = GroupNormalization::new(
        vec![batch_size, channels, depth, height, width],
        num_groups,
        1,
        1e-5,
    )
    .unwrap();

    let input = Array::from_shape_fn(
        (batch_size, channels, depth, height, width),
        |(b, c, d, h, w)| (b * 1000 + c * 100 + d * 10 + h + w) as f32,
    )
    .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(
        output.shape(),
        &[batch_size, channels, depth, height, width]
    );

    println!("5D input test passed");
}

#[test]
fn test_group_normalization_batch_independence() {
    // Test that different batches are normalized independently
    let mut gn_layer = GroupNormalization::new(vec![2, 4, 8], 2, 1, 1e-5).unwrap();

    // Create input where batch 0 has much larger values than batch 1
    let mut input_vec = vec![0.0; 2 * 4 * 8];

    // Batch 0: large values
    for i in 0..(4 * 8) {
        input_vec[i] = 100.0 + i as f32;
    }

    // Batch 1: small values
    for i in (4 * 8)..(2 * 4 * 8) {
        input_vec[i] = 1.0 + (i as f32 * 0.1);
    }

    let input = Array::from_shape_vec((2, 4, 8), input_vec)
        .unwrap()
        .into_dyn();

    gn_layer.set_training(true);
    let output = gn_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4, 8]);

    // Each batch should be normalized independently, so output ranges should be similar
    let output_3d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();

    let batch0_max = output_3d
        .slice(s![0, .., ..])
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let batch1_max = output_3d
        .slice(s![1, .., ..])
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Both batches should have similar normalized ranges
    assert!(
        (batch0_max - batch1_max).abs() < 5.0,
        "Batches should be normalized independently"
    );

    println!("Batch independence test passed");
}
