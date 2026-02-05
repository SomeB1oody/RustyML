use approx::assert_relative_eq;
use ndarray::Array3;
use rustyml::neural_network::layer::pooling_layer::max_pooling_1d::MaxPooling1D;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::rms_prop::RMSprop;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_max_pooling_1d() {
    // Create a simple input tensor: [batch_size, channels, length]
    // batch_size=2, 3 input channels, 8 elements per channel
    let mut input_data = Array3::zeros((2, 3, 8));

    // Set test data to make max pooling results predictable
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..8 {
                input_data[[b, c, i]] = i as f32;
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Test MaxPooling1D using Sequential model
    let mut model = Sequential::new();
    model
        .add(
            MaxPooling1D::new(
                2,             // pool window size
                2,             // stride
                vec![2, 3, 8], // input shape
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Output shape should be [2, 3, 4]
    let output = model.predict(&x).unwrap();
    assert_eq!(output.shape(), &[2, 3, 4]);

    // Verify correctness of pooling results
    // For window size 2 and stride 2, we expect the maximum element in each window
    for b in 0..2 {
        for c in 0..3 {
            // First window (0,1) -> max value should be 1.0
            assert_relative_eq!(output[[b, c, 0]], 1.0);
            // Second window (2,3) -> max value should be 3.0
            assert_relative_eq!(output[[b, c, 1]], 3.0);
            // Third window (4,5) -> max value should be 5.0
            assert_relative_eq!(output[[b, c, 2]], 5.0);
            // Fourth window (6,7) -> max value should be 7.0
            assert_relative_eq!(output[[b, c, 3]], 7.0);
        }
    }
}

#[test]
fn test_max_pooling_1d_backward() {
    // Create a simple input tensor
    let mut input_data = Array3::zeros((1, 1, 4));
    for i in 0..4 {
        input_data[[0, 0, i]] = i as f32;
    }
    let x = input_data.clone().into_dyn();

    // Create MaxPooling1D layer
    let mut pool_layer = MaxPooling1D::new(2, 2, vec![1, 1, 4]).unwrap();

    // Forward propagation
    let output = pool_layer.forward(&x).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2]);
    assert_relative_eq!(output[[0, 0, 0]], 1.0); // Max value of first window
    assert_relative_eq!(output[[0, 0, 1]], 3.0); // Max value of second window

    // Create gradient
    let grad_output = Array3::ones((1, 1, 2)).into_dyn();

    // Backward propagation
    let grad_input = pool_layer.backward(&grad_output).unwrap();
    assert_eq!(grad_input.shape(), &[1, 1, 4]);

    // For max pooling, gradient should only flow to the input elements that produced the max values
    assert_relative_eq!(grad_input[[0, 0, 0]], 0.0); // Not the max value
    assert_relative_eq!(grad_input[[0, 0, 1]], 1.0); // Is the max value of first window
    assert_relative_eq!(grad_input[[0, 0, 2]], 0.0); // Not the max value
    assert_relative_eq!(grad_input[[0, 0, 3]], 1.0); // Is the max value of second window
}
