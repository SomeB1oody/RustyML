use super::*;

#[test]
fn test_batch_normalization_forward_pass_dimensions() {
    // Test forward propagation dimension correctness
    let mut bn = BatchNormalization::new(vec![4, 8], 0.9, 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn(); // batch_size=4, features=8

    let output = bn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[4, 8]);
    println!(
        "Forward pass dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_batch_normalization_training_mode() {
    // Test that batch normalization properly normalizes in training mode
    let mut bn = BatchNormalization::new(vec![3, 4], 0.9, 1e-5).unwrap();

    // Create input with non-zero mean and variance
    let input = Array::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap()
    .into_dyn();

    bn.set_training(true);
    let output = bn.forward(&input).unwrap();

    // Compute mean and variance of output along batch dimension
    let output_2d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    // Check that each feature has approximately zero mean and unit variance
    for feature_idx in 0..4 {
        let feature_col = output_2d.column(feature_idx);
        let mean: f32 = feature_col.mean().unwrap();
        let variance: f32 = feature_col.mapv(|x| (x - mean).powi(2)).mean().unwrap();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-4);
    }

    println!("Training mode normalization test passed");
}

#[test]
fn test_batch_normalization_inference_mode() {
    // Test that batch normalization uses running statistics in inference mode
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    // Train on some data to update running statistics
    let train_input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    bn.set_training(true);
    bn.forward(&train_input).unwrap();

    // Switch to inference mode
    bn.set_training(false);

    // Use different input
    let test_input = Array::from_shape_vec((2, 3), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .unwrap()
        .into_dyn();

    let output = bn.forward(&test_input).unwrap();

    // Output should be computed using running statistics, not batch statistics
    assert_eq!(output.shape(), &[2, 3]);
    println!("Inference mode test passed");
}

#[test]
fn test_batch_normalization_backward_pass() {
    // Test backward pass gradient computation
    let mut bn = BatchNormalization::new(vec![3, 4], 0.9, 1e-5).unwrap();

    let input = Array::from_shape_vec(
        (3, 4),
        vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap()
    .into_dyn();

    bn.set_training(true);
    let _output = bn.forward(&input).unwrap();

    // Create gradient
    let grad_output = Array::ones((3, 4)).into_dyn();

    let grad_input = bn.backward(&grad_output).unwrap();

    // Verify gradient dimensions
    assert_eq!(grad_input.shape(), input.shape());

    // Verify gradient sum along batch dimension is approximately zero
    // (property of batch normalization gradient)
    let grad_2d = grad_input
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for feature_idx in 0..4 {
        let feature_grad_sum: f32 = grad_2d.column(feature_idx).sum();
        assert_abs_diff_eq!(feature_grad_sum, 0.0, epsilon = 1e-4);
    }

    println!("Backward pass test passed");
}

#[test]
fn test_batch_normalization_parameter_update_sgd() {
    // Test parameter update with SGD optimizer
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    bn.set_training(true);
    let _output = bn.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = bn.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters
        bn.update_parameters_sgd(0.01);

        // Get updated weights
        if let LayerWeight::BatchNormalization(updated_weights) = bn.get_weights() {
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
fn test_batch_normalization_parameter_update_adam() {
    // Test parameter update with Adam optimizer
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    bn.set_training(true);
    let _output = bn.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = bn.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with Adam
        bn.update_parameters_adam(0.001, 0.9, 0.999, 1e-8, 1);

        // Get updated weights
        if let LayerWeight::BatchNormalization(updated_weights) = bn.get_weights() {
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
fn test_batch_normalization_parameter_update_rmsprop() {
    // Test parameter update with RMSprop optimizer
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    bn.set_training(true);
    let _output = bn.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = bn.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with RMSprop
        bn.update_parameters_rmsprop(0.001, 0.9, 1e-8);

        // Get updated weights
        if let LayerWeight::BatchNormalization(updated_weights) = bn.get_weights() {
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
fn test_batch_normalization_parameter_update_adagrad() {
    // Test parameter update with AdaGrad optimizer
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    bn.set_training(true);
    let _output = bn.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let _grad_input = bn.backward(&grad_output).unwrap();

    // Get initial weights
    if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        let initial_gamma = weights.gamma.clone();
        let initial_beta = weights.beta.clone();

        // Update parameters with AdaGrad
        bn.update_parameters_ada_grad(0.01, 1e-8);

        // Get updated weights
        if let LayerWeight::BatchNormalization(updated_weights) = bn.get_weights() {
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
fn test_batch_normalization_running_statistics() {
    // Test that running statistics are properly updated during training
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let input = Array::from_shape_vec((2, 3), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        .unwrap()
        .into_dyn();

    // Get initial running statistics
    let initial_running_mean = if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        weights.running_mean.clone()
    } else {
        panic!("Failed to get weights");
    };

    bn.set_training(true);
    bn.forward(&input).unwrap();

    // Get updated running statistics
    let updated_running_mean = if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        weights.running_mean.clone()
    } else {
        panic!("Failed to get weights");
    };

    // Verify running statistics have changed
    let stats_changed = updated_running_mean
        .as_slice()
        .unwrap()
        .iter()
        .zip(initial_running_mean.as_slice().unwrap().iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);

    assert!(
        stats_changed,
        "Running statistics should be updated during training"
    );
    println!("Running statistics update test passed");
}

#[test]
fn test_batch_normalization_different_batch_sizes() {
    // Test batch normalization with different batch sizes
    let batch_sizes = vec![2, 4, 8, 16];
    let features = 5;

    for batch_size in batch_sizes {
        let mut bn = BatchNormalization::new(vec![batch_size, features], 0.9, 1e-5).unwrap();
        let input = Array::ones((batch_size, features)).into_dyn();

        bn.set_training(true);
        let output = bn.forward(&input).unwrap();

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
fn test_batch_normalization_parameter_count() {
    // Test parameter count correctness
    let bn = BatchNormalization::new(vec![4, 10], 0.9, 1e-5).unwrap();
    let expected_params = 10 + 10; // gamma + beta
    assert_eq!(
        bn.param_count(),
        TrainingParameters::Trainable(expected_params)
    );
    println!(
        "Parameter count test passed: {} parameters",
        expected_params
    );
}

#[test]
fn test_batch_normalization_layer_type() {
    // Test layer type identification
    let bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();
    assert_eq!(bn.layer_type(), "BatchNormalization");
    println!("Layer type test passed");
}

#[test]
fn test_batch_normalization_output_shape() {
    // Test output shape string formatting
    let bn = BatchNormalization::new(vec![4, 8], 0.9, 1e-5).unwrap();
    let output_shape = bn.output_shape();
    assert_eq!(output_shape, "(4, 8)");
    println!("Output shape test passed: {}", output_shape);
}

#[test]
fn test_batch_normalization_with_sequential_model() {
    // Test batch normalization integration with Sequential model
    let mut model = Sequential::new();
    model.add(Dense::new(4, 8, Linear::new()).unwrap());
    model.add(BatchNormalization::new(vec![2, 8], 0.9, 1e-5).unwrap());
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
fn test_batch_normalization_large_batch_parallel() {
    // Test parallel computation with large batch
    let batch_size = 64; // Should trigger parallel computation
    let features = 32;

    let mut bn = BatchNormalization::new(vec![batch_size, features], 0.9, 1e-5).unwrap();
    let input =
        Array::from_shape_fn((batch_size, features), |(i, j)| (i * features + j) as f32).into_dyn();

    bn.set_training(true);
    let output = bn.forward(&input).unwrap();

    assert_eq!(output.shape(), &[batch_size, features]);

    // Verify normalization properties
    let output_2d = output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for feature_idx in 0..features {
        let feature_col = output_2d.column(feature_idx);
        let mean: f32 = feature_col.mean().unwrap();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    }

    println!("Large batch parallel computation test passed");
}

#[test]
fn test_batch_normalization_set_weights() {
    // Test manual weight setting
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();

    let new_gamma = Array::from_vec(vec![2.0, 2.0, 2.0]).into_dyn();
    let new_beta = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();
    let new_running_mean = Array::from_vec(vec![5.0, 5.0, 5.0]).into_dyn();
    let new_running_var = Array::from_vec(vec![2.0, 2.0, 2.0]).into_dyn();

    bn.set_weights(
        new_gamma.clone(),
        new_beta.clone(),
        new_running_mean.clone(),
        new_running_var.clone(),
    );

    // Verify weights were set correctly
    if let LayerWeight::BatchNormalization(weights) = bn.get_weights() {
        assert_eq!(
            weights.gamma.as_slice().unwrap(),
            new_gamma.as_slice().unwrap()
        );
        assert_eq!(
            weights.beta.as_slice().unwrap(),
            new_beta.as_slice().unwrap()
        );
        assert_eq!(
            weights.running_mean.as_slice().unwrap(),
            new_running_mean.as_slice().unwrap()
        );
        assert_eq!(
            weights.running_var.as_slice().unwrap(),
            new_running_var.as_slice().unwrap()
        );
    }

    println!("Set weights test passed");
}
