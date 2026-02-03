use crate::neural_network::layer::pooling_layer::average_pooling_2d::AveragePooling2D;
use crate::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use crate::neural_network::optimizer::rms_prop::RMSprop;
use crate::neural_network::sequential::Sequential;
use approx::assert_relative_eq;
use ndarray::Array4;

#[test]
fn average_pooling_basic_test() {
    // Create a simple input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 input channels, each channel is 4x4 pixels
    let mut input_data = Array4::zeros((2, 3, 4, 4));

    // Set test data to make average pooling results predictable
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..4 {
                for j in 0..4 {
                    input_data[[b, c, i, j]] = (i + j) as f32;
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Test AveragePooling with Sequential model
    let mut model = Sequential::new();
    model
        .add(
            AveragePooling2D::new(
                (2, 2),           // Pooling window size
                vec![2, 3, 4, 4], // Input shape
                Some((2, 2)),     // Strides (optional)
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Output shape should be [2, 3, 2, 2]
    let output = model.predict(&x).unwrap();
    assert_eq!(output.shape(), &[2, 3, 2, 2]);

    // Verify correctness of pooling results
    // For a 2x2 window with stride 2, we expect the result to be the average of the elements in the window
    for b in 0..2 {
        for c in 0..3 {
            // First window (0,0), (0,1), (1,0), (1,1) -> average should be (0+1+1+2)/4 = 1.0
            assert_relative_eq!(output[[b, c, 0, 0]], 1.0);
            // Second window (0,2), (0,3), (1,2), (1,3) -> average should be (2+3+3+4)/4 = 3.0
            assert_relative_eq!(output[[b, c, 0, 1]], 3.0);
            // Third window (2,0), (2,1), (3,0), (3,1) -> average should be (2+3+3+4)/4 = 3.0
            assert_relative_eq!(output[[b, c, 1, 0]], 3.0);
            // Fourth window (2,2), (2,3), (3,2), (3,3) -> average should be (4+5+5+6)/4 = 5.0
            assert_relative_eq!(output[[b, c, 1, 1]], 5.0);
        }
    }
}

#[test]
fn average_pooling_non_even_input_test() {
    // Test input with non-even dimensions
    let mut input_data = Array4::zeros((2, 3, 5, 5));

    // Set test data
    for b in 0..2 {
        for c in 0..3 {
            for i in 0..5 {
                for j in 0..5 {
                    input_data[[b, c, i, j]] = (i + j) as f32;
                }
            }
        }
    }

    let x = input_data.clone().into_dyn();

    // Pooling window size (3,3), stride (2,2)
    let mut model = Sequential::new();
    model
        .add(
            AveragePooling2D::new(
                (3, 3),           // Pooling window size
                vec![2, 3, 5, 5], // Input shape
                Some((2, 2)),     // Strides (optional)
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Output shape should be [2, 3, 2, 2]
    // (5-3)/2+1 = 2
    let output = model.predict(&x).unwrap();
    assert_eq!(output.shape(), &[2, 3, 2, 2]);

    // Verify the average pooling result for the first window
    // Window (0,0) to (2,2) contains 9 elements: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
    // Corresponding values: 0+1+2+1+2+3+2+3+4 = 18/9 = 2.0
    assert_relative_eq!(output[[0, 0, 0, 0]], 2.0);
}

#[test]
fn average_pooling_different_strides_test() {
    // Test different stride cases
    let mut input_data = Array4::zeros((1, 1, 6, 6));

    // Set test data - using increasing values
    for i in 0..6 {
        for j in 0..6 {
            input_data[[0, 0, i, j]] = (i * 6 + j) as f32;
        }
    }

    let x = input_data.clone().into_dyn();

    // Pooling with stride (1,1)
    let mut model = Sequential::new();
    model
        .add(
            AveragePooling2D::new(
                (2, 2),           // Pooling window size
                vec![1, 1, 6, 6], // Input shape
                Some((1, 1)),     // Strides (optional)
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Output shape should be [1, 1, 5, 5]
    // (6-2)/1+1 = 5
    let output = model.predict(&x).unwrap();
    assert_eq!(output.shape(), &[1, 1, 5, 5]);

    // Check the average of the first window
    // Window (0,0),(0,1),(1,0),(1,1) corresponding values: 0+1+6+7=14/4=3.5
    assert_relative_eq!(output[[0, 0, 0, 0]], 3.5);
}

#[test]
fn average_pooling_backprop_test() {
    // Test backpropagation
    let input_data = Array4::ones((2, 2, 4, 4)).into_dyn();
    let target_data = Array4::ones((2, 2, 2, 2)).into_dyn();

    // Create model and train
    let mut model = Sequential::new();
    model
        .add(
            AveragePooling2D::new(
                (2, 2),           // Pooling window size
                vec![2, 2, 4, 4], // Input shape
                Some((2, 2)),     // Strides (optional)
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.01, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Train the model
    let result = model.fit(&input_data, &target_data, 5);
    assert!(result.is_ok(), "Model training failed");

    // Verify that predictions after training are close to target values
    let prediction = model.predict(&input_data).unwrap();
    for b in 0..2 {
        for c in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(
                        prediction[[b, c, i, j]],
                        target_data[[b, c, i, j]],
                        epsilon = 0.1
                    );
                }
            }
        }
    }
}
