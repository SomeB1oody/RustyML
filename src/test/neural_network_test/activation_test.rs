use super::*;

#[test]
fn with_activation_test() {
    // Create input tensor with shape (batch_size=2, input_dim=4) and target tensor (batch_size=2, output_dim=3)
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build model: using Dense layers with specified activation functions (such as ReLU or Softmax)
    // Here we use Sigmoid activation for the first layer and Softmax for the second layer (you can modify as needed)
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, Sigmoid::new()))
        .add(Dense::new(3, 1, Softmax::new()));

    // Choose an optimizer, e.g., RMSprop, Adam or SGD - using RMSprop as an example here
    model.compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model (simple iteration example)
    model.fit(&x, &y, 3).unwrap();

    // Get output using predict
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}

#[test]
fn relu_forward_test() {
    // Test ReLU forward pass with positive and negative values
    let input = Array::from_shape_vec((2, 3), vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();

    let mut relu = ReLU::new();
    let output = relu.forward(&input).unwrap();

    // Expected: all negative values become 0, positive values unchanged
    assert_eq!(output[[0, 0]], 0.0);
    assert_eq!(output[[0, 1]], 0.0);
    assert_eq!(output[[0, 2]], 0.0);
    assert_eq!(output[[1, 0]], 1.0);
    assert_eq!(output[[1, 1]], 2.0);
    assert_eq!(output[[1, 2]], 3.0);
}

#[test]
fn relu_backward_test() {
    // Test ReLU backward pass (gradient computation)
    let input = Array::from_shape_vec((2, 3), vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();

    let mut relu = ReLU::new();
    relu.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let grad_input = relu.backward(&grad_output).unwrap();

    // Expected: gradient is 0 for negative inputs, 1 for positive inputs
    assert_eq!(grad_input[[0, 0]], 0.0);
    assert_eq!(grad_input[[0, 1]], 0.0);
    assert_eq!(grad_input[[0, 2]], 0.0);
    assert_eq!(grad_input[[1, 0]], 1.0);
    assert_eq!(grad_input[[1, 1]], 1.0);
    assert_eq!(grad_input[[1, 2]], 1.0);
}

#[test]
fn relu_empty_input_test() {
    // Test ReLU with empty input tensor - should return error
    let input = Array::from_shape_vec((0, 0), vec![]).unwrap().into_dyn();

    let mut relu = ReLU::new();
    let result = relu.forward(&input);

    assert!(result.is_err());
}

#[test]
fn sigmoid_forward_test() {
    // Test Sigmoid forward pass with various input values
    let input = Array::from_shape_vec((2, 2), vec![-10.0, 0.0, 1.0, 10.0])
        .unwrap()
        .into_dyn();

    let mut sigmoid = Sigmoid::new();
    let output = sigmoid.forward(&input).unwrap();

    // Sigmoid should map values to (0, 1) range
    // Sigmoid(0) = 0.5, Sigmoid(-inf) → 0, Sigmoid(+inf) → 1
    assert!(output[[0, 0]] < 0.01); // Very close to 0
    assert!((output[[0, 1]] - 0.5).abs() < 1e-5); // Equal to 0.5
    assert!((output[[1, 0]] - 0.731).abs() < 0.01); // Approximately 0.731
    assert!(output[[1, 1]] > 0.99); // Very close to 1

    // All values should be in range (0, 1)
    for val in output.iter() {
        assert!(*val > 0.0 && *val < 1.0);
    }
}

#[test]
fn sigmoid_backward_test() {
    // Test Sigmoid backward pass (gradient computation)
    let input = Array::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 2.0])
        .unwrap()
        .into_dyn();

    let mut sigmoid = Sigmoid::new();
    sigmoid.forward(&input).unwrap();

    let grad_output = Array::ones((2, 2)).into_dyn();
    let grad_input = sigmoid.backward(&grad_output).unwrap();

    // Sigmoid derivative is sigmoid(x) * (1 - sigmoid(x))
    // Maximum derivative is 0.25 at x=0
    assert!((grad_input[[0, 0]] - 0.25).abs() < 1e-5);

    // All gradients should be positive and less than 0.25
    for val in grad_input.iter() {
        assert!(*val > 0.0 && *val <= 0.25);
    }
}

#[test]
fn sigmoid_empty_input_test() {
    // Test Sigmoid with empty input tensor - should return error
    let input = Array::from_shape_vec((0, 0), vec![]).unwrap().into_dyn();

    let mut sigmoid = Sigmoid::new();
    let result = sigmoid.forward(&input);

    assert!(result.is_err());
}

#[test]
fn tanh_forward_test() {
    // Test Tanh forward pass with various input values
    let input = Array::from_shape_vec((2, 3), vec![-10.0, -1.0, 0.0, 1.0, 2.0, 10.0])
        .unwrap()
        .into_dyn();

    let mut tanh = Tanh::new();
    let output = tanh.forward(&input).unwrap();

    // Tanh should map values to (-1, 1) range
    // Tanh(0) = 0, Tanh(-inf) → -1, Tanh(+inf) → 1
    assert!(output[[0, 0]] < -0.99); // Very close to -1
    assert!((output[[0, 1]] - (-0.761)).abs() < 0.01); // Approximately -0.761
    assert!(output[[0, 2]].abs() < 1e-5); // Equal to 0
    assert!((output[[1, 0]] - 0.761).abs() < 0.01); // Approximately 0.761
    assert!((output[[1, 1]] - 0.964).abs() < 0.01); // Approximately 0.964
    assert!(output[[1, 2]] > 0.99); // Very close to 1

    // All values should be in range [-1, 1]
    for val in output.iter() {
        assert!(*val >= -1.0 && *val <= 1.0);
    }
}

#[test]
fn tanh_backward_test() {
    // Test Tanh backward pass (gradient computation)
    let input = Array::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 2.0])
        .unwrap()
        .into_dyn();

    let mut tanh = Tanh::new();
    tanh.forward(&input).unwrap();

    let grad_output = Array::ones((2, 2)).into_dyn();
    let grad_input = tanh.backward(&grad_output).unwrap();

    // Tanh derivative is 1 - tanh^2(x)
    // Maximum derivative is 1 at x=0
    assert!((grad_input[[0, 0]] - 1.0).abs() < 1e-5);

    // All gradients should be positive and less than or equal to 1
    for val in grad_input.iter() {
        assert!(*val > 0.0 && *val <= 1.0);
    }
}

#[test]
fn tanh_empty_input_test() {
    // Test Tanh with empty input tensor - should return error
    let input = Array::from_shape_vec((0, 0), vec![]).unwrap().into_dyn();

    let mut tanh = Tanh::new();
    let result = tanh.forward(&input);

    assert!(result.is_err());
}

#[test]
fn softmax_forward_test() {
    // Test Softmax forward pass with 2D input
    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
        .unwrap()
        .into_dyn();

    let mut softmax = Softmax::new();
    let output = softmax.forward(&input).unwrap();

    // Each row should sum to 1.0 (probability distribution)
    let row0_sum: f32 = output.slice(s![0, ..]).iter().sum();
    let row1_sum: f32 = output.slice(s![1, ..]).iter().sum();

    assert!((row0_sum - 1.0).abs() < 1e-5);
    assert!((row1_sum - 1.0).abs() < 1e-5);

    // All values should be in range (0, 1)
    for val in output.iter() {
        assert!(*val > 0.0 && *val < 1.0);
    }

    // For uniform input [1, 1, 1], output should be approximately [1/3, 1/3, 1/3]
    assert!((output[[1, 0]] - 1.0 / 3.0).abs() < 1e-5);
    assert!((output[[1, 1]] - 1.0 / 3.0).abs() < 1e-5);
    assert!((output[[1, 2]] - 1.0 / 3.0).abs() < 1e-5);
}

#[test]
fn softmax_backward_test() {
    // Test Softmax backward pass (gradient computation)
    let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
        .unwrap()
        .into_dyn();

    let mut softmax = Softmax::new();
    softmax.forward(&input).unwrap();

    let grad_output = Array::ones((2, 3)).into_dyn();
    let grad_input = softmax.backward(&grad_output).unwrap();

    // For uniform gradient input, sum of gradients along each row should be close to 0
    // This is a property of softmax derivative
    let row0_grad_sum: f32 = grad_input.slice(s![0, ..]).iter().sum();
    let row1_grad_sum: f32 = grad_input.slice(s![1, ..]).iter().sum();

    assert!(row0_grad_sum.abs() < 1e-5);
    assert!(row1_grad_sum.abs() < 1e-5);
}

#[test]
fn softmax_1d_input_error_test() {
    // Test Softmax with 1D input - should return error (requires at least 2D)
    let input = Array::from_shape_vec((5,), vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut softmax = Softmax::new();
    let result = softmax.forward(&input);

    assert!(result.is_err());
}

#[test]
fn softmax_empty_input_test() {
    // Test Softmax with empty input tensor - should return error
    let input = Array::from_shape_vec((0, 0), vec![]).unwrap().into_dyn();

    let mut softmax = Softmax::new();
    let result = softmax.forward(&input);

    assert!(result.is_err());
}

#[test]
fn activation_numerical_stability_test() {
    // Test all activation functions with extreme values to ensure numerical stability

    // Test with very large positive values
    let large_pos = Array::from_shape_vec((1, 3), vec![100.0, 500.0, 1000.0])
        .unwrap()
        .into_dyn();

    let mut relu = ReLU::new();
    let relu_out = relu.forward(&large_pos).unwrap();
    assert!(!relu_out.iter().any(|x| x.is_nan() || x.is_infinite()));

    let mut sigmoid = Sigmoid::new();
    let sigmoid_out = sigmoid.forward(&large_pos).unwrap();
    assert!(!sigmoid_out.iter().any(|x| x.is_nan() || x.is_infinite()));

    let mut tanh = Tanh::new();
    let tanh_out = tanh.forward(&large_pos).unwrap();
    assert!(!tanh_out.iter().any(|x| x.is_nan() || x.is_infinite()));

    // Test with very large negative values
    let large_neg = Array::from_shape_vec((1, 3), vec![-100.0, -500.0, -1000.0])
        .unwrap()
        .into_dyn();

    let mut relu2 = ReLU::new();
    let relu_out2 = relu2.forward(&large_neg).unwrap();
    assert!(!relu_out2.iter().any(|x| x.is_nan() || x.is_infinite()));

    let mut sigmoid2 = Sigmoid::new();
    let sigmoid_out2 = sigmoid2.forward(&large_neg).unwrap();
    assert!(!sigmoid_out2.iter().any(|x| x.is_nan() || x.is_infinite()));

    let mut tanh2 = Tanh::new();
    let tanh_out2 = tanh2.forward(&large_neg).unwrap();
    assert!(!tanh_out2.iter().any(|x| x.is_nan() || x.is_infinite()));
}

#[test]
fn activation_layer_properties_test() {
    // Test that activation layers correctly implement Layer trait methods

    let mut relu = ReLU::new();
    assert_eq!(relu.layer_type(), "ReLU");
    assert_eq!(relu.output_shape(), "Unknown"); // Before forward pass

    let input = Array::ones((2, 3)).into_dyn();
    relu.forward(&input).unwrap();
    assert_eq!(relu.output_shape(), "(2, 3)"); // After forward pass

    let sigmoid = Sigmoid::new();
    assert_eq!(sigmoid.layer_type(), "Sigmoid");

    let tanh = Tanh::new();
    assert_eq!(tanh.layer_type(), "Tanh");

    let softmax = Softmax::new();
    assert_eq!(softmax.layer_type(), "Softmax");
}

#[test]
fn activation_backward_without_forward_test() {
    // Test that backward pass fails when forward pass hasn't been called

    let grad_output = Array::ones((2, 3)).into_dyn();

    let mut relu = ReLU::new();
    assert!(relu.backward(&grad_output).is_err());

    let mut sigmoid = Sigmoid::new();
    assert!(sigmoid.backward(&grad_output).is_err());

    let mut tanh = Tanh::new();
    assert!(tanh.backward(&grad_output).is_err());

    let mut softmax = Softmax::new();
    assert!(softmax.backward(&grad_output).is_err());
}

#[test]
fn activation_integration_test() {
    // Integration test: Build a simple neural network with activation layers
    let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((4, 2), vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        .unwrap()
        .into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 4, ReLU::new()))
        .add(Dense::new(4, 4, Tanh::new()))
        .add(Dense::new(4, 2, Sigmoid::new()));

    model.compile(Adam::new(0.01, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Train for a few epochs
    model.fit(&x, &y, 5).unwrap();

    // Make predictions
    let prediction = model.predict(&x);

    // Check output shape is correct
    assert_eq!(prediction.shape(), &[4, 2]);

    // Check all predictions are in valid range [0, 1] due to Sigmoid
    for val in prediction.iter() {
        assert!(*val >= 0.0 && *val <= 1.0);
    }
}
