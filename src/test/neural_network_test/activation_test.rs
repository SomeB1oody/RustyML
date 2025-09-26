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
        .add(Dense::new(4, 3, Activation::Sigmoid))
        .add(Dense::new(3, 1, Activation::Softmax));

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
fn test_apply_activation_inplace() {
    // ReLU
    let mut input_relu = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
    let expected_relu = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]);

    Activation::apply_activation_inplace(&Activation::ReLU, &mut input_relu);
    for (actual, expected) in input_relu.iter().zip(expected_relu.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }

    // Sigmoid
    let mut input_sigmoid = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
    let expected_sigmoid = vec![0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708];

    Activation::apply_activation_inplace(&Activation::Sigmoid, &mut input_sigmoid);
    for (actual, expected) in input_sigmoid.iter().zip(expected_sigmoid.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }

    // Tanh
    let mut input_tanh = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
    let expected_tanh = vec![-0.76159415, 0.0, 0.76159415];

    Activation::apply_activation_inplace(&Activation::Tanh, &mut input_tanh);
    for (actual, expected) in input_tanh.iter().zip(expected_tanh.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }

    // Linear
    let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut input_linear = Array::from_vec(original_data.clone()).into_dyn();

    Activation::apply_activation_inplace(&Activation::Linear, &mut input_linear);
    for (actual, expected) in input_linear.iter().zip(original_data.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }
}

#[test]
#[should_panic(expected = "Cannot use Softmax for convolution layers")]
fn test_apply_activation_inplace_softmax_panic() {
    let mut input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
    Activation::apply_activation_inplace(&Activation::Softmax, &mut input);
}

#[test]
fn test_activation_derivative_inplace() {
    use approx::assert_relative_eq;

    // ReLU
    let mut relu_output = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]).into_dyn();
    let expected_relu_deriv = vec![0.0, 0.0, 0.0, 1.0, 1.0];

    Activation::activation_derivative_inplace(&Activation::ReLU, &mut relu_output);
    for (actual, expected) in relu_output.iter().zip(expected_relu_deriv.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }

    // Sigmoid
    let sigmoid_values = vec![0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708];
    let mut sigmoid_output = Array::from_vec(sigmoid_values.clone()).into_dyn();

    Activation::activation_derivative_inplace(&Activation::Sigmoid, &mut sigmoid_output);

    // Sigmoid: f'(x) = f(x) * (1 - f(x))
    for (i, actual) in sigmoid_output.iter().enumerate() {
        let expected = sigmoid_values[i] * (1.0 - sigmoid_values[i]);
        assert_relative_eq!(*actual, expected, epsilon = 1e-6);
    }

    // Tanh
    let tanh_values = vec![-0.76159415, 0.0, 0.76159415];
    let mut tanh_output = Array::from_vec(tanh_values.clone()).into_dyn();

    Activation::activation_derivative_inplace(&Activation::Tanh, &mut tanh_output);

    // Tanh: f'(x) = 1 - f(x)^2
    for (i, actual) in tanh_output.iter().enumerate() {
        let expected = 1.0 - tanh_values[i] * tanh_values[i];
        assert_relative_eq!(*actual, expected, epsilon = 1e-6);
    }

    // Linear
    let mut linear_output = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn();

    Activation::activation_derivative_inplace(&Activation::Linear, &mut linear_output);
    for actual in linear_output.iter() {
        assert_relative_eq!(*actual, 1.0, epsilon = 1e-6);
    }
}

#[test]
#[should_panic(expected = "Cannot use Softmax for convolution layers")]
fn test_activation_derivative_inplace_softmax_panic() {
    let mut output = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
    Activation::activation_derivative_inplace(&Activation::Softmax, &mut output);
}
