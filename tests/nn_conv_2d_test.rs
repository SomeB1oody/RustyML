#![cfg(feature = "neural_network")]

use ndarray::Array4;
use rustyml::neural_network::layer::activation_layer::relu::ReLU;
use rustyml::neural_network::layer::activation_layer::sigmoid::Sigmoid;
use rustyml::neural_network::layer::convolution_layer::PaddingType;
use rustyml::neural_network::layer::convolution_layer::conv_2d::Conv2D;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizer::adam::Adam;
use rustyml::neural_network::optimizer::rms_prop::RMSprop;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn conv2d_test() {
    // Create a simple 4D input tensor: [batch_size, channels, height, width]
    // Batch size=2, 1 input channel, 5x5 pixels
    let x = Array4::ones((2, 1, 5, 5)).into_dyn();

    // Create target tensor - assuming we'll have 3 filters with output size 3x3
    let y = Array4::ones((2, 3, 3, 3)).into_dyn();

    // Build model: add a Conv2D layer with 3 filters and 3x3 kernel
    let mut model = Sequential::new();
    model
        .add(
            Conv2D::new(
                3,                  // Number of filters
                (3, 3),             // Kernel size
                vec![2, 1, 5, 5],   // Input shape
                (1, 1),             // Stride
                PaddingType::Valid, // No padding
                ReLU::new(),        // ReLU activation function
            )
            .unwrap(),
        )
        .compile(
            RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Print model structure
    model.summary();

    // Train the model (run a few epochs)
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x).unwrap();
    println!("Convolution layer prediction results: {:?}", prediction);

    // Check if output shape is correct - should be [2, 3, 3, 3]
    assert_eq!(prediction.shape(), &[2, 3, 3, 3]);

    // Create another convolution layer with different padding strategy and stride
    let mut model2 = Sequential::new();
    model2
        .add(
            Conv2D::new(
                2,                 // Number of filters
                (3, 3),            // Kernel size
                vec![2, 1, 5, 5],  // Input shape
                (2, 2),            // Larger stride
                PaddingType::Same, // Same padding
                Sigmoid::new(),    // Sigmoid activation function
            )
            .unwrap(),
        )
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );

    // Create appropriate target tensor - for Same padding and stride (2,2), output size should be 3x3
    let y2 = Array4::ones((2, 2, 3, 3)).into_dyn();

    // Print model structure
    model2.summary();

    // Train the model
    model2.fit(&x, &y2, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction2 = model2.predict(&x).unwrap();
    println!(
        "Convolution layer prediction results (Same padding, stride 2): {:?}",
        prediction2
    );

    // Check if output shape is correct
    assert_eq!(prediction2.shape(), &[2, 2, 3, 3]);
}
