use super::*;

#[test]
fn test_dense_forward_pass_dimensions() {
    // Test forward propagation dimension correctness
    let mut dense = Dense::new(4, 3, ReLU::new()).unwrap();
    let input = Array::ones((2, 4)).into_dyn(); // batch_size=2, input_dim=4

    let output = dense.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 3]); // batch_size=2, output_dim=3
    println!(
        "Forward pass dimension test passed: {:?} -> {:?}",
        input.shape(),
        output.shape()
    );
}

#[test]
fn test_dense_activation_functions() {
    // Test effects of different activation functions
    let input = Array::from_shape_vec((1, 2), vec![1.0, -1.0])
        .unwrap()
        .into_dyn();

    // Test ReLU activation function
    let mut dense_relu = Dense::new(2, 2, ReLU::new()).unwrap();
    let output_relu = dense_relu.forward(&input);

    // Test Sigmoid activation function
    let mut dense_sigmoid = Dense::new(2, 2, Sigmoid::new()).unwrap();
    let output_sigmoid = dense_sigmoid.forward(&input);

    // Test Tanh activation function
    let mut dense_tanh = Dense::new(2, 2, Tanh::new()).unwrap();
    let output_tanh = dense_tanh.forward(&input).unwrap();

    println!("ReLU output: {:?}", output_relu);
    println!("Sigmoid output: {:?}", output_sigmoid);
    println!("Tanh output: {:?}", output_tanh);

    // Verify Sigmoid output is in (0,1) range
    let sigmoid_output = output_sigmoid.unwrap();
    let sigmoid_2d = sigmoid_output
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for value in sigmoid_2d.iter() {
        assert!(
            *value > 0.0 && *value < 1.0,
            "Sigmoid output should be in (0,1) range: {}",
            value
        );
    }

    // Verify Tanh output is in (-1,1) range
    let tanh_2d = output_tanh
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for value in tanh_2d.iter() {
        assert!(
            *value > -1.0 && *value < 1.0,
            "Tanh output should be in (-1,1) range: {}",
            value
        );
    }
}

#[test]
fn test_dense_learning_capability() {
    // Test Dense layer learning capability - simple linear regression problem
    // y = 2*x1 + 3*x2 + 1

    // Create training data
    let x_train = Array::from_shape_vec(
        (4, 2),
        vec![
            1.0, 2.0, // y = 2*1 + 3*2 + 1 = 9
            2.0, 1.0, // y = 2*2 + 3*1 + 1 = 8
            3.0, 3.0, // y = 2*3 + 3*3 + 1 = 16
            0.0, 1.0, // y = 2*0 + 3*1 + 1 = 4
        ],
    )
    .unwrap()
    .into_dyn();

    let y_train = Array::from_shape_vec((4, 1), vec![9.0, 8.0, 16.0, 4.0])
        .unwrap()
        .into_dyn();

    // Build model
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, Linear::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Record initial prediction
    let initial_prediction = model.predict(&x_train);
    let initial_loss = calculate_mse(&initial_prediction, &y_train);
    println!("Initial loss: {:.4}", initial_loss);

    // Train model
    model.fit(&x_train, &y_train, 1000).unwrap();

    // Test final prediction
    let final_prediction = model.predict(&x_train);
    let final_loss = calculate_mse(&final_prediction, &y_train);
    println!("Final loss: {:.4}", final_loss);
    println!("Predictions: {:?}", final_prediction);
    println!("Ground truth: {:?}", y_train);

    // Verify model actually learned something (loss should be significantly reduced)
    assert!(
        final_loss < initial_loss * 0.1,
        "Model should learn patterns, loss reduced from {:.4} to {:.4}",
        initial_loss,
        final_loss
    );

    // Verify prediction accuracy
    let pred_2d = final_prediction
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let target_2d = y_train
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for i in 0..pred_2d.nrows() {
        let pred_val = pred_2d[[i, 0]];
        let true_val = target_2d[[i, 0]];
        assert_abs_diff_eq!(pred_val, true_val, epsilon = 1.0);
    }
}

#[test]
fn test_dense_batch_processing() {
    // Test batch processing capability
    let batch_sizes = vec![1, 5, 10, 32];

    for batch_size in batch_sizes {
        let input = Array::ones((batch_size, 4)).into_dyn();
        let mut dense = Dense::new(4, 3, ReLU::new()).unwrap();

        let output = dense.forward(&input).unwrap();
        assert_eq!(
            output.shape()[0],
            batch_size,
            "Batch size should remain unchanged"
        );
        assert_eq!(
            output.shape()[1],
            3,
            "Output feature dimension should be correct"
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
fn test_dense_parameter_count() {
    // Test parameter count correctness
    let dense = Dense::new(10, 5, ReLU::new()).unwrap();
    let expected_params = 10 * 5 + 5; // weights + bias
    assert_eq!(
        dense.param_count(),
        TrainingParameters::Trainable(expected_params)
    );
    println!(
        "Parameter count test passed: {} parameters",
        expected_params
    );

    // Test different layer sizes
    let sizes = vec![(1, 1), (100, 50), (784, 128), (512, 10)];
    for (input_dim, output_dim) in sizes {
        let layer = Dense::new(input_dim, output_dim, Linear::new()).unwrap();
        let expected = input_dim * output_dim + output_dim;
        assert_eq!(
            layer.param_count(),
            TrainingParameters::Trainable(expected),
            "({}, {}) layer parameter count error",
            input_dim,
            output_dim
        );
    }
}

// Helper functions
fn calculate_mse(pred: &Tensor, target: &Tensor) -> f32 {
    let pred_2d = pred
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let target_2d = target
        .as_standard_layout()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    let diff = &pred_2d - &target_2d;
    let squared_diff = diff.mapv(|x| x * x);
    squared_diff.mean().unwrap()
}
