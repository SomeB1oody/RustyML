#![cfg(feature = "neural_network")]

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::layer::activation_layer::linear::Linear;
use rustyml::neural_network::layer::dense::Dense;
use rustyml::neural_network::layer::layer_weight::LayerWeight;
use rustyml::neural_network::layer::regularization_layer::normalization_layer::batch_normalization::BatchNormalization;
use rustyml::neural_network::layer::regularization_layer::normalization_layer::layer_normalization::{LayerNormalization, LayerNormalizationAxis};
use rustyml::neural_network::layer::serialize_weight::SerializableLayerWeight;
use rustyml::neural_network::layer::TrainingParameters;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_layer_normalization_forward_pass_dimensions() {
    // Test forward propagation dimension correctness with default axis
    let mut ln =
        LayerNormalization::new(vec![4, 8], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn(); // batch_size=4, features=8

    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[4, 8]);
    println!(
        "Forward pass dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_layer_normalization_default_axis() {
    // Test that layer normalization properly normalizes along the last axis (default)
    let mut ln =
        LayerNormalization::new(vec![3, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    // Create input with non-zero mean and variance
    let input = Array::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap()
    .into_dyn();

    ln.set_training(true);
    let output = ln.forward(&input).unwrap();

    // For layer normalization, each sample should be normalized independently
    let output_2d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    // Check that each sample (row) has approximately zero mean and unit variance
    for sample_idx in 0..3 {
        let sample_row = output_2d.row(sample_idx);
        let mean: f32 = sample_row.mean().unwrap();
        let variance: f32 = sample_row.mapv(|x| (x - mean).powi(2)).mean().unwrap();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-4);
    }

    println!("Default axis normalization test passed");
}

#[test]
fn test_layer_normalization_custom_axis() {
    // Test layer normalization with custom axis (axis 0)
    let mut ln =
        LayerNormalization::new(vec![3, 4], LayerNormalizationAxis::Custom(0), 1e-5).unwrap();

    let input = Array::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap()
    .into_dyn();

    ln.set_training(true);
    let output = ln.forward(&input).unwrap();

    // When normalizing along axis 0, each column should be normalized
    let output_2d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for col_idx in 0..4 {
        let col = output_2d.column(col_idx);
        let mean: f32 = col.mean().unwrap();
        let variance: f32 = col.mapv(|x| (x - mean).powi(2)).mean().unwrap();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-4);
    }

    println!("Custom axis normalization test passed");
}

#[test]
fn test_layer_normalization_invalid_axis() {
    // Test that invalid axis returns an error
    let mut ln =
        LayerNormalization::new(vec![3, 4], LayerNormalizationAxis::Custom(5), 1e-5).unwrap();

    let input = Array::ones((3, 4)).into_dyn();

    let result = ln.forward(&input);
    assert!(result.is_err(), "Should return error for invalid axis");
    println!("Invalid axis error handling test passed");
}

#[test]
fn test_layer_normalization_training_mode() {
    // Test that layer normalization works in training mode
    let mut ln =
        LayerNormalization::new(vec![2, 6], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec(
        (2, 6),
        vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0,
        ],
    )
    .unwrap()
    .into_dyn();

    ln.set_training(true);
    let output = ln.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 6]);

    // Verify normalization properties
    let output_2d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for sample_idx in 0..2 {
        let sample = output_2d.row(sample_idx);
        let mean: f32 = sample.mean().unwrap();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
    }

    println!("Training mode test passed");
}

#[test]
fn test_layer_normalization_inference_mode() {
    // Test that layer normalization works in inference mode
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    // First do a forward pass in training mode
    ln.set_training(true);
    ln.forward(&input).unwrap();

    // Switch to inference mode
    ln.set_training(false);

    let test_input = Array::from_shape_vec((2, 4), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .unwrap()
        .into_dyn();

    let output = ln.forward(&test_input).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    println!("Inference mode test passed");
}

#[test]
fn test_layer_normalization_backward_pass() {
    // Test backward pass gradient computation
    let mut ln =
        LayerNormalization::new(vec![3, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec(
        (3, 4),
        vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap()
    .into_dyn();

    ln.set_training(true);
    let _output = ln.forward(&input).unwrap();

    // Create gradient
    let grad_output = Array::ones((3, 4)).into_dyn();

    let grad_input = ln.backward(&grad_output).unwrap();

    // Verify gradient dimensions
    assert_eq!(grad_input.shape(), input.shape());

    // For layer normalization, the sum of gradients along the normalized axis should be approximately zero
    let grad_2d = grad_input
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for sample_idx in 0..3 {
        let sample_grad_sum: f32 = grad_2d.row(sample_idx).sum();
        assert_abs_diff_eq!(sample_grad_sum, 0.0, epsilon = 1e-4);
    }

    println!("Backward pass test passed");
}

#[test]
fn test_layer_normalization_parameter_update_sgd() {
    // Test parameter update with SGD optimizer
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    ln.set_training(true);
    let _output = ln.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = ln.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::LayerNormalizationLayer(weights) = ln.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters
        ln.update_parameters_sgd(0.01);

        // Get updated weights
        if let LayerWeight::LayerNormalizationLayer(updated_weights) = ln.get_weights() {
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
fn test_layer_normalization_parameter_update_adam() {
    // Test parameter update with Adam optimizer
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    ln.set_training(true);
    let _output = ln.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = ln.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::LayerNormalizationLayer(weights) = ln.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with Adam
        ln.update_parameters_adam(0.001, 0.9, 0.999, 1e-8, 1);

        // Get updated weights
        if let LayerWeight::LayerNormalizationLayer(updated_weights) = ln.get_weights() {
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
fn test_layer_normalization_parameter_update_rmsprop() {
    // Test parameter update with RMSprop optimizer
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    ln.set_training(true);
    let _output = ln.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = ln.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::LayerNormalizationLayer(weights) = ln.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with RMSprop
        ln.update_parameters_rmsprop(0.001, 0.9, 1e-8);

        // Get updated weights
        if let LayerWeight::LayerNormalizationLayer(updated_weights) = ln.get_weights() {
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
fn test_layer_normalization_parameter_update_adagrad() {
    // Test parameter update with AdaGrad optimizer
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    ln.set_training(true);
    let _output = ln.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = ln.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::LayerNormalizationLayer(weights) = ln.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with AdaGrad
        ln.update_parameters_ada_grad(0.01, 1e-8);

        // Get updated weights
        if let LayerWeight::LayerNormalizationLayer(updated_weights) = ln.get_weights() {
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
fn test_layer_normalization_different_batch_sizes() {
    // Test layer normalization with different batch sizes
    let batch_sizes = vec![2, 4, 8, 16];
    let features = 5;

    for batch_size in batch_sizes {
        let mut ln = LayerNormalization::new(
            vec![batch_size, features],
            LayerNormalizationAxis::Default,
            1e-5,
        )
        .unwrap();
        let input =
            Array::from_shape_fn((batch_size, features), |(i, j)| (i * features + j) as f32)
                .into_dyn();

        ln.set_training(true);
        let output = ln.forward(&input).unwrap();

        assert_eq!(
            output.shape(),
            &[batch_size, features],
            "Output shape should match input shape"
        );

        println!(
            "Batch size {} test passed: {:?} -> {:?}",
            batch_size,
            input.shape(),
            output.shape()
        );
    }
}

#[test]
fn test_layer_normalization_parameter_count() {
    // Test parameter count correctness
    let ln = LayerNormalization::new(vec![4, 10], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let expected_params = 10 + 10; // gamma + beta
    assert_eq!(
        ln.param_count(),
        TrainingParameters::Trainable(expected_params)
    );
    println!(
        "Parameter count test passed: {} parameters",
        expected_params
    );
}

#[test]
fn test_layer_normalization_layer_type() {
    // Test layer type identification
    let ln = LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();
    assert_eq!(ln.layer_type(), "LayerNormalization");
    println!("Layer type test passed");
}

#[test]
fn test_layer_normalization_output_shape() {
    // Test output shape string formatting
    let ln = LayerNormalization::new(vec![4, 8], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let output_shape = ln.output_shape();
    assert_eq!(output_shape, "(4, 8)");
    println!("Output shape test passed: {}", output_shape);
}

#[test]
fn test_layer_normalization_with_sequential_model() {
    // Test layer normalization integration with Sequential model
    let mut model = Sequential::new();
    model.add(Dense::new(4, 8, Linear::new()).unwrap());
    model.add(LayerNormalization::new(vec![2, 8], LayerNormalizationAxis::Default, 1e-5).unwrap());
    model.add(Dense::new(8, 1, Linear::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let input = Array::ones((2, 4)).into_dyn();
    let target = Array::ones((2, 1)).into_dyn();

    // Test forward pass
    let output = model.predict(&input).unwrap();
    assert_eq!(output.shape(), &[2, 1]);

    // Test training
    let result = model.fit(&input, &target, 10);
    assert!(result.is_ok(), "Training should succeed");

    println!("Sequential model integration test passed");
}

#[test]
fn test_layer_normalization_set_weights() {
    // Test manual weight setting
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let new_gamma = Array::from_vec(vec![2.0, 2.0, 2.0]).into_dyn();
    let new_beta = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();

    ln.set_weights(new_gamma.clone(), new_beta.clone());

    // Verify weights were set correctly
    if let LayerWeight::LayerNormalizationLayer(weights) = ln.get_weights() {
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
fn test_layer_normalization_3d_input() {
    // Test layer normalization with 3D input (batch, sequence, features)
    let batch_size = 2;
    let sequence_len = 4;
    let features = 3;

    let mut ln = LayerNormalization::new(
        vec![batch_size, sequence_len, features],
        LayerNormalizationAxis::Default,
        1e-5,
    )
    .unwrap();

    let input = Array::from_shape_fn((batch_size, sequence_len, features), |(i, j, k)| {
        (i * 100 + j * 10 + k) as f32
    })
    .into_dyn();

    ln.set_training(true);
    let output = ln.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, sequence_len, features]);

    println!("3D input test passed");
}

#[test]
fn test_layer_normalization_epsilon_effect() {
    // Test that epsilon prevents division by zero
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    // Create input with constant values (zero variance)
    let input = Array::from_shape_vec((2, 3), vec![5.0, 5.0, 5.0, 3.0, 3.0, 3.0])
        .unwrap()
        .into_dyn();

    ln.set_training(true);
    let result = ln.forward(&input);

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
fn test_layer_normalization_vs_batch_normalization_difference() {
    // Demonstrate the difference between layer norm and batch norm
    let input = Array::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap()
    .into_dyn();

    // Layer normalization (normalizes across features for each sample)
    let mut ln =
        LayerNormalization::new(vec![3, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    ln.set_training(true);
    let ln_output = ln.forward(&input).unwrap();

    // Batch normalization (normalizes across batch for each feature)
    let mut bn = BatchNormalization::new(vec![3, 4], 0.9, 1e-5).unwrap();
    bn.set_training(true);
    let bn_output = bn.forward(&input).unwrap();

    // The outputs should be different
    let ln_2d = ln_output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let bn_2d = bn_output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    let outputs_different = ln_2d
        .iter()
        .zip(bn_2d.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);

    assert!(
        outputs_different,
        "Layer norm and batch norm should produce different outputs"
    );

    println!("Layer norm vs batch norm difference test passed");
}

#[test]
fn test_layer_normalization_gradient_flow() {
    // Test that gradients flow properly through the layer
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    ln.set_training(true);
    let _output = ln.forward(&input).unwrap();

    let grad_output = Array::from_shape_vec((2, 4), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        .unwrap()
        .into_dyn();

    let grad_input = ln.backward(&grad_output).unwrap();

    // Verify gradients are not zero (gradients are flowing)
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
fn test_layer_normalization_multiple_forward_backward() {
    // Test multiple forward and backward passes
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    for i in 0..5 {
        let input = Array::from_shape_fn((2, 3), |(b, f)| (i + b + f) as f32).into_dyn();

        ln.set_training(true);
        let output = ln.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3]);

        let grad_output = Array::ones((2, 3)).into_dyn();
        let grad_input = ln.backward(&grad_output).unwrap();

        assert_eq!(grad_input.shape(), &[2, 3]);

        ln.update_parameters_sgd(0.01);
    }

    println!("Multiple forward-backward passes test passed");
}

#[test]
fn test_layer_normalization_serialization() {
    // Test weight serialization and deserialization
    let mut ln =
        LayerNormalization::new(vec![2, 3], LayerNormalizationAxis::Default, 1e-5).unwrap();

    // Do a forward pass to initialize
    let input = Array::ones((2, 3)).into_dyn();
    ln.set_training(true);
    ln.forward(&input).unwrap();

    // Get weights and convert to serializable format
    let weights = ln.get_weights();
    let serializable_weights = SerializableLayerWeight::from_layer_weight(&weights);

    // Verify we can convert back (this tests the serialization structure)
    match serializable_weights {
        SerializableLayerWeight::LayerNormalization(w) => {
            assert_eq!(w.gamma.len(), 3);
            assert_eq!(w.beta.len(), 3);
            assert_eq!(w.shape, vec![3]);
        }
        _ => panic!("Expected LayerNormalization weights"),
    }

    println!("Serialization test passed");
}
