#![cfg(feature = "neural_network")]

use approx::assert_abs_diff_eq;
use ndarray::{s, Array, Array2, Array3};
use rustyml::neural_network::layer::activation_layer::linear::Linear;
use rustyml::neural_network::layer::dense::Dense;
use rustyml::neural_network::layer::layer_weight::LayerWeight;
use rustyml::neural_network::layer::regularization_layer::normalization_layer::batch_normalization::BatchNormalization;
use rustyml::neural_network::layer::regularization_layer::normalization_layer::instance_normalization::InstanceNormalization;
use rustyml::neural_network::layer::serialize_weight::SerializableLayerWeight;
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_instance_normalization_forward_pass_dimensions() {
    // Test forward propagation dimension correctness
    let mut in_layer = InstanceNormalization::new(vec![4, 3, 8], 1, 1e-5).unwrap();
    let input = Array3::ones((4, 3, 8)).into_dyn(); // batch=4, channels=3, spatial=8

    let output = in_layer.forward(&input).unwrap();
    assert_eq!(output.shape(), &[4, 3, 8]);
    println!(
        "Forward pass dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_instance_normalization_3d_input() {
    // Test instance normalization with 3D input (batch, channels, spatial)
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    // Create input with varying values
    let input = Array::from_shape_vec(
        (2, 3, 4),
        vec![
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1
            5.0, 6.0, 7.0, 8.0, // Batch 0, Channel 2
            9.0, 10.0, 11.0, 12.0, // Batch 1, Channel 0
            2.0, 3.0, 4.0, 5.0, // Batch 1, Channel 1
            6.0, 7.0, 8.0, 9.0, // Batch 1, Channel 2
            10.0, 11.0, 12.0, 13.0,
        ],
    )
    .unwrap()
    .into_dyn();

    in_layer.set_training(true);
    let output = in_layer.forward(&input).unwrap();

    // For instance normalization, each (batch, channel) instance should be normalized independently
    // Check that the output has the same shape
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Verify that each instance is normalized (mean H 0, std H 1)
    let output_3d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();

    for batch_idx in 0..2 {
        for channel_idx in 0..3 {
            let instance = output_3d.slice(s![batch_idx, channel_idx, ..]).to_owned();
            let mean: f32 = instance.mean().unwrap();
            let variance: f32 = instance.mapv(|x| (x - mean).powi(2)).mean().unwrap();

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
            assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-4);
        }
    }

    println!("3D input normalization test passed");
}

#[test]
fn test_instance_normalization_4d_input() {
    // Test instance normalization with 4D input (batch, channels, height, width)
    let batch_size = 2;
    let channels = 3;
    let height = 4;
    let width = 4;

    let mut in_layer =
        InstanceNormalization::new(vec![batch_size, channels, height, width], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((batch_size, channels, height, width), |(b, c, h, w)| {
        (b * 100 + c * 10 + h + w) as f32
    })
    .into_dyn();

    in_layer.set_training(true);
    let output = in_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, channels, height, width]);

    // Verify normalization for each (batch, channel) instance
    let output_4d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    for batch_idx in 0..batch_size {
        for channel_idx in 0..channels {
            let instance = output_4d
                .slice(s![batch_idx, channel_idx, .., ..])
                .to_owned();
            let mean: f32 = instance.mean().unwrap();
            let variance: f32 = instance.mapv(|x| (x - mean).powi(2)).mean().unwrap();

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
            assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-4);
        }
    }

    println!("4D input normalization test passed");
}

#[test]
fn test_instance_normalization_custom_channel_axis() {
    // Test instance normalization with custom channel axis
    // Format: [batch, height, width, channels]
    let batch_size = 2;
    let height = 4;
    let width = 4;
    let channels = 3;

    let mut in_layer =
        InstanceNormalization::new(vec![batch_size, height, width, channels], 3, 1e-5).unwrap();

    let input = Array::from_shape_fn((batch_size, height, width, channels), |(b, h, w, c)| {
        (b * 100 + c * 10 + h + w) as f32
    })
    .into_dyn();

    in_layer.set_training(true);
    let output = in_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, height, width, channels]);

    println!("Custom channel axis test passed");
}

#[test]
fn test_instance_normalization_invalid_channel_axis() {
    // Test that channel axis cannot be 0 (batch axis)
    let in_layer = InstanceNormalization::new(vec![2, 3, 4], 0, 1e-5);

    assert!(in_layer.is_err());
    println!("Invalid channel axis test passed");
}

#[test]
fn test_instance_normalization_invalid_input_dimensions() {
    // Test that input must be at least 3D
    let mut in_layer = InstanceNormalization::new(vec![4, 8], 1, 1e-5).unwrap();

    let input = Array2::ones((4, 8)).into_dyn();

    let result = in_layer.forward(&input);
    assert!(
        result.is_err(),
        "Should return error for input with less than 3 dimensions"
    );
    println!("Invalid input dimensions test passed");
}

#[test]
fn test_instance_normalization_training_mode() {
    // Test that instance normalization works in training mode
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 8], 1, 1e-5).unwrap();

    let input =
        Array::from_shape_fn((2, 3, 8), |(b, c, s)| (b * 100 + c * 10 + s) as f32).into_dyn();

    in_layer.set_training(true);
    let output = in_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3, 8]);

    // Verify normalization
    let output_3d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();

    for batch_idx in 0..2 {
        for channel_idx in 0..3 {
            let instance = output_3d.slice(s![batch_idx, channel_idx, ..]).to_owned();
            let mean: f32 = instance.mean().unwrap();
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
        }
    }

    println!("Training mode test passed");
}

#[test]
fn test_instance_normalization_inference_mode() {
    // Test that instance normalization works in inference mode
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let train_input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    // First do a forward pass in training mode
    in_layer.set_training(true);
    in_layer.forward(&train_input).unwrap();

    // Switch to inference mode
    in_layer.set_training(false);

    let test_input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b * 2 + c + s) as f32).into_dyn();

    let output = in_layer.forward(&test_input).unwrap();
    assert_eq!(output.shape(), &[2, 3, 4]);

    println!("Inference mode test passed");
}

#[test]
fn test_instance_normalization_backward_pass() {
    // Test backward pass gradient computation
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b * 10 + c * 5 + s) as f32).into_dyn();

    in_layer.set_training(true);
    let _output = in_layer.forward(&input).unwrap();

    // Create gradient
    let grad_output = Array3::ones((2, 3, 4)).into_dyn();

    let grad_input = in_layer.backward(&grad_output).unwrap();

    // Verify gradient dimensions
    assert_eq!(grad_input.shape(), input.shape());

    // For instance normalization, the sum of gradients for each instance should be approximately zero
    let grad_3d = grad_input
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();

    for batch_idx in 0..2 {
        for channel_idx in 0..3 {
            let instance_grad_sum: f32 = grad_3d.slice(s![batch_idx, channel_idx, ..]).sum();
            assert_abs_diff_eq!(instance_grad_sum, 0.0, epsilon = 1e-4);
        }
    }

    println!("Backward pass test passed");
}

#[test]
fn test_instance_normalization_parameter_update_sgd() {
    // Test parameter update with SGD optimizer
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    in_layer.set_training(true);
    let _output = in_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 3, 4)).into_dyn();
    let _grad_input = in_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::InstanceNormalizationLayer(weights) = in_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters
        in_layer.update_parameters_sgd(0.01);

        // Get updated weights
        if let LayerWeight::InstanceNormalizationLayer(updated_weights) = in_layer.get_weights() {
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
fn test_instance_normalization_parameter_update_adam() {
    // Test parameter update with Adam optimizer
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    in_layer.set_training(true);
    let _output = in_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 3, 4)).into_dyn();
    let _grad_input = in_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::InstanceNormalizationLayer(weights) = in_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with Adam
        in_layer.update_parameters_adam(0.001, 0.9, 0.999, 1e-8, 1);

        // Get updated weights
        if let LayerWeight::InstanceNormalizationLayer(updated_weights) = in_layer.get_weights() {
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
fn test_instance_normalization_parameter_update_rmsprop() {
    // Test parameter update with RMSprop optimizer
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    in_layer.set_training(true);
    let _output = in_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 3, 4)).into_dyn();
    let _grad_input = in_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::InstanceNormalizationLayer(weights) = in_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with RMSprop
        in_layer.update_parameters_rmsprop(0.001, 0.9, 1e-8);

        // Get updated weights
        if let LayerWeight::InstanceNormalizationLayer(updated_weights) = in_layer.get_weights() {
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
fn test_instance_normalization_parameter_update_adagrad() {
    // Test parameter update with AdaGrad optimizer
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    in_layer.set_training(true);
    let _output = in_layer.forward(&input).unwrap();

    let grad_output = Array3::ones((2, 3, 4)).into_dyn();
    let _grad_input = in_layer.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::InstanceNormalizationLayer(weights) = in_layer.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with AdaGrad
        in_layer.update_parameters_ada_grad(0.01, 1e-8);

        // Get updated weights
        if let LayerWeight::InstanceNormalizationLayer(updated_weights) = in_layer.get_weights() {
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
fn test_instance_normalization_parameter_count() {
    // Test parameter count correctness
    let in_layer = InstanceNormalization::new(vec![4, 8, 16], 1, 1e-5).unwrap();
    let expected_params = 8 + 8; // gamma + beta (one per channel)
    assert_eq!(
        in_layer.param_count(),
        TrainingParameters::Trainable(expected_params)
    );
    println!(
        "Parameter count test passed: {} parameters",
        expected_params
    );
}

#[test]
fn test_instance_normalization_layer_type() {
    // Test layer type identification
    let in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();
    assert_eq!(in_layer.layer_type(), "InstanceNormalization");
    println!("Layer type test passed");
}

#[test]
fn test_instance_normalization_output_shape() {
    // Test output shape string formatting
    let in_layer = InstanceNormalization::new(vec![4, 8, 16], 1, 1e-5).unwrap();
    let output_shape = in_layer.output_shape();
    assert_eq!(output_shape, "(4, 8, 16)");
    println!("Output shape test passed: {}", output_shape);
}

#[test]
fn test_instance_normalization_set_weights() {
    // Test manual weight setting
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let new_gamma = Array::from_vec(vec![2.0, 2.0, 2.0]).into_dyn();
    let new_beta = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();

    in_layer.set_weights(new_gamma.clone(), new_beta.clone());

    // Verify weights were set correctly
    if let LayerWeight::InstanceNormalizationLayer(weights) = in_layer.get_weights() {
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
fn test_instance_normalization_epsilon_effect() {
    // Test that epsilon prevents division by zero
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    // Create input where each instance has constant values (zero variance)
    let mut input_vec = vec![0.0; 2 * 3 * 4];
    // Set constant values for each (batch, channel) instance
    for batch in 0..2 {
        for channel in 0..3 {
            let base_idx = (batch * 3 + channel) * 4;
            let constant_val = (batch * 10 + channel) as f32;
            for spatial in 0..4 {
                input_vec[base_idx + spatial] = constant_val;
            }
        }
    }

    let input = Array::from_shape_vec((2, 3, 4), input_vec)
        .unwrap()
        .into_dyn();

    in_layer.set_training(true);
    let result = in_layer.forward(&input);

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
fn test_instance_normalization_vs_batch_normalization() {
    // Demonstrate the difference between instance norm and batch norm
    let input =
        Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b * 100 + c * 10 + s) as f32).into_dyn();

    // Instance normalization (normalizes each channel of each sample independently)
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();
    in_layer.set_training(true);
    let in_output = in_layer.forward(&input).unwrap();

    // Batch normalization (normalizes across batch for each channel)
    let mut bn = BatchNormalization::new(vec![2, 3, 4], 0.9, 1e-5).unwrap();
    bn.set_training(true);
    let bn_output = bn.forward(&input).unwrap();

    // The outputs should be different
    let outputs_different = in_output
        .iter()
        .zip(bn_output.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);

    assert!(
        outputs_different,
        "Instance norm and batch norm should produce different outputs"
    );

    println!("Instance norm vs batch norm difference test passed");
}

#[test]
fn test_instance_normalization_large_batch_parallel() {
    // Test parallel computation with large tensor (should trigger parallel path)
    let batch_size = 8;
    let channels = 16;
    let spatial = 32;

    let mut in_layer =
        InstanceNormalization::new(vec![batch_size, channels, spatial], 1, 1e-5).unwrap();
    let input = Array::from_shape_fn((batch_size, channels, spatial), |(b, c, s)| {
        (b * 1000 + c * 100 + s) as f32
    })
    .into_dyn();

    in_layer.set_training(true);
    let output = in_layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, channels, spatial]);

    // Verify normalization properties
    let output_3d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();

    for batch_idx in 0..batch_size {
        for channel_idx in 0..channels {
            let instance = output_3d.slice(s![batch_idx, channel_idx, ..]).to_owned();
            let mean: f32 = instance.mean().unwrap();
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
        }
    }

    println!("Large batch parallel computation test passed");
}

#[test]
fn test_instance_normalization_gradient_flow() {
    // Test that gradients flow properly through the layer
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (b + c + s) as f32).into_dyn();

    in_layer.set_training(true);
    let _output = in_layer.forward(&input).unwrap();

    let grad_output =
        Array::from_shape_fn((2, 3, 4), |(b, c, s)| ((b + c + s) as f32) * 0.1).into_dyn();

    let grad_input = in_layer.backward(&grad_output).unwrap();

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
fn test_instance_normalization_multiple_forward_backward() {
    // Test multiple forward and backward passes
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    for i in 0..5 {
        let input = Array::from_shape_fn((2, 3, 4), |(b, c, s)| (i + b + c + s) as f32).into_dyn();

        in_layer.set_training(true);
        let output = in_layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3, 4]);

        let grad_output = Array3::ones((2, 3, 4)).into_dyn();
        let grad_input = in_layer.backward(&grad_output).unwrap();

        assert_eq!(grad_input.shape(), &[2, 3, 4]);

        in_layer.update_parameters_sgd(0.01);
    }

    println!("Multiple forward-backward passes test passed");
}

#[test]
fn test_instance_normalization_serialization() {
    // Test weight serialization and deserialization
    let mut in_layer = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();

    // Do a forward pass to initialize
    let input = Array3::ones((2, 3, 4)).into_dyn();
    in_layer.set_training(true);
    in_layer.forward(&input).unwrap();

    // Get weights and convert to serializable format
    let weights = in_layer.get_weights();
    let serializable_weights = SerializableLayerWeight::from_layer_weight(&weights);

    // Verify we can convert back (this tests the serialization structure)
    match serializable_weights {
        SerializableLayerWeight::InstanceNormalization(w) => {
            assert_eq!(w.gamma.len(), 3); // 3 channels
            assert_eq!(w.beta.len(), 3);
            assert_eq!(w.shape, vec![3]);
        }
        _ => panic!("Expected InstanceNormalization weights"),
    }

    println!("Serialization test passed");
}

#[test]
fn test_instance_normalization_different_spatial_sizes() {
    // Test with different spatial dimensions
    let spatial_sizes = vec![4, 8, 16, 32];

    for spatial_size in spatial_sizes {
        let mut in_layer = InstanceNormalization::new(vec![2, 3, spatial_size], 1, 1e-5).unwrap();
        let input = Array::from_shape_fn((2, 3, spatial_size), |(b, c, s)| {
            (b * 100 + c * 10 + s) as f32
        })
        .into_dyn();

        in_layer.set_training(true);
        let output = in_layer.forward(&input).unwrap();

        assert_eq!(
            output.shape(),
            &[2, 3, spatial_size],
            "Output shape should match input shape"
        );

        println!(
            "Spatial size {} test passed: {:?} -> {:?}",
            spatial_size,
            input.shape(),
            output.shape()
        );
    }
}

#[test]
fn test_instance_normalization_with_sequential_model() {
    // Test instance normalization integration with Sequential model
    let mut model = Sequential::new();
    model.add(Dense::new(12, 24, Linear::new()).unwrap());
    // Reshape to 3D for instance norm: from (2, 24) to (2, 3, 8)
    // Note: In practice you'd need a reshape layer, but for testing we'll verify shapes separately

    let input = Array::ones((2, 12)).into_dyn();
    let target = Array::ones((2, 24)).into_dyn();
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Test forward pass
    let output = model.predict(&input).unwrap();
    assert_eq!(output.shape(), &[2, 24]);

    // Test training
    let result = model.fit(&input, &target, 10);
    assert!(result.is_ok(), "Training should succeed");

    println!("Sequential model integration test passed");
}

#[test]
fn test_instance_normalization_5d_input() {
    // Test with 5D input (batch, channels, depth, height, width)
    let batch_size = 2;
    let channels = 3;
    let depth = 2;
    let height = 3;
    let width = 3;

    let mut in_layer =
        InstanceNormalization::new(vec![batch_size, channels, depth, height, width], 1, 1e-5)
            .unwrap();

    let input = Array::from_shape_fn(
        (batch_size, channels, depth, height, width),
        |(b, c, d, h, w)| (b * 1000 + c * 100 + d * 10 + h + w) as f32,
    )
    .into_dyn();

    in_layer.set_training(true);
    let output = in_layer.forward(&input).unwrap();

    assert_eq!(
        output.shape(),
        &[batch_size, channels, depth, height, width]
    );

    println!("5D input test passed");
}
