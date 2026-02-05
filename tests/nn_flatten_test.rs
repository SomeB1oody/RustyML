use ndarray::{Array2, Array3, Array4, Array5};
use rustyml::neural_network::layer::activation_layer::relu::ReLU;
use rustyml::neural_network::layer::activation_layer::sigmoid::Sigmoid;
use rustyml::neural_network::layer::activation_layer::softmax::Softmax;
use rustyml::neural_network::layer::convolution_layer::PaddingType;
use rustyml::neural_network::layer::convolution_layer::conv_2d::Conv2D;
use rustyml::neural_network::layer::dense::Dense;
use rustyml::neural_network::layer::flatten::Flatten;
use rustyml::neural_network::layer::pooling_layer::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::loss_function::categorical_cross_entropy::CategoricalCrossEntropy;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Layer;
use rustyml::neural_network::optimizer::adam::Adam;
use rustyml::neural_network::optimizer::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_flatten_in_sequential() {
    // Create a simple 4D input tensor: [batch_size, channels, height, width]
    // Batch size=2, 3 channels, each 4x4 pixels
    let x = Array4::ones((2, 3, 4, 4)).into_dyn();

    // Create target tensor - assuming final output is a single value
    let y = Array2::ones((2, 1)).into_dyn();

    // Build model: convolution layer -> max pooling layer -> flatten layer -> fully connected layer
    let mut model = Sequential::new();
    model
        .add(
            Conv2D::new(
                6,                  // Number of filters
                (3, 3),             // Kernel size
                vec![2, 3, 4, 4],   // Input shape
                (1, 1),             // Stride
                PaddingType::Valid, // No padding
                ReLU::new(),        // ReLU activation function
            )
            .unwrap(),
        )
        .add(
            MaxPooling2D::new(
                (2, 2),           // Pooling window size
                vec![2, 6, 2, 2], // Input shape (after convolution)
                None,             // Use default stride
            )
            .unwrap(),
        )
        .add(Flatten::new(vec![2, 6, 1, 1]).unwrap()) // Flatten layer
        .add(Dense::new(6, 1, Sigmoid::new()).unwrap()) // Fully connected layer
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x).unwrap();
    println!("CNN+Flatten prediction results: {:?}", prediction);

    // Check if output shape is correct
    assert_eq!(prediction.shape(), &[2, 1]);
}

#[test]
fn test_flatten_only_model() {
    // Create 4D input tensor
    let x = Array4::from_shape_fn((2, 3, 4, 4), |(b, c, h, w)| {
        (b * 100 + c * 10 + h + w) as f32 / 10.0
    })
    .into_dyn();

    // Target tensor - already flattened format
    let flattened_size = 3 * 4 * 4; // channels * height * width

    // Create model with only Flatten layer
    let mut model = Sequential::new();
    model
        .add(Flatten::new(vec![2, 3, 4, 4]).unwrap())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Perform forward propagation
    let output = model.predict(&x).unwrap();

    // Check output shape
    assert_eq!(output.shape(), &[2, flattened_size]);
}

#[test]
fn test_multiple_layers_with_flatten() {
    // Create 4D input tensor
    let x = Array4::ones((2, 1, 8, 8)).into_dyn();

    // Create target tensor
    let y = Array2::ones((2, 10)).into_dyn();

    // Build model with multiple layers including two convolution layers, one pooling layer,
    // one flatten layer and two dense layers
    let mut model = Sequential::new();
    model
        .add(
            Conv2D::new(
                8,                 // 8 filters
                (3, 3),            // 3x3 kernel
                vec![2, 1, 8, 8],  // Input shape
                (1, 1),            // Stride
                PaddingType::Same, // Same padding
                ReLU::new(),       // ReLU activation
            )
            .unwrap(),
        )
        .add(
            Conv2D::new(
                16,                 // 16 filters
                (3, 3),             // 3x3 kernel
                vec![2, 8, 8, 8],   // Input shape (after first convolution)
                (1, 1),             // Stride
                PaddingType::Valid, // No padding
                ReLU::new(),        // ReLU activation
            )
            .unwrap(),
        )
        .add(
            MaxPooling2D::new(
                (2, 2),            // 2x2 pooling window
                vec![2, 16, 6, 6], // Input shape (after second convolution)
                Some((2, 2)),      // 2x2 stride
            )
            .unwrap(),
        )
        .add(Flatten::new(vec![2, 16, 3, 3]).unwrap()) // Flatten layer
        .add(Dense::new(16 * 3 * 3, 20, ReLU::new()).unwrap()) // Fully connected layer
        .add(Dense::new(20, 10, Softmax::new()).unwrap()) // Output layer
        .compile(
            Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
            CategoricalCrossEntropy::new(),
        );

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Forward propagation prediction
    let prediction = model.predict(&x).unwrap();
    println!(
        "Complex CNN with Flatten prediction results: {:?}",
        prediction
    );

    // Check output shape
    assert_eq!(prediction.shape(), &[2, 10]);
}

#[test]
fn test_flatten_3d() {
    let input = Array3::ones((2, 10, 5)).into_dyn(); // batch=2, features=10, length=5
    let mut flatten = Flatten::new(vec![2, 10, 5]).unwrap();

    let output = flatten.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 50]); // 10 * 5 = 50

    // Test backward pass
    let grad_output = Array2::ones((2, 50)).into_dyn();
    let grad_input = flatten.backward(&grad_output).unwrap();
    assert_eq!(grad_input.shape(), input.shape());
}

#[test]
fn test_flatten_4d() {
    let input = Array4::ones((2, 3, 4, 4)).into_dyn(); // batch=2, channels=3, height=4, width=4
    let mut flatten = Flatten::new(vec![2, 3, 4, 4]).unwrap();

    let output = flatten.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 48]); // 3 * 4 * 4 = 48

    // Test backward pass
    let grad_output = Array2::ones((2, 48)).into_dyn();
    let grad_input = flatten.backward(&grad_output).unwrap();
    assert_eq!(grad_input.shape(), input.shape());
}

#[test]
fn test_flatten_5d() {
    let input = Array5::ones((2, 3, 4, 8, 8)).into_dyn(); // batch=2, channels=3, depth=4, height=8, width=8
    let mut flatten = Flatten::new(vec![2, 3, 4, 8, 8]).unwrap();

    let output = flatten.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 768]); // 3 * 4 * 8 * 8 = 768

    // Test backward pass
    let grad_output = Array2::ones((2, 768)).into_dyn();
    let grad_input = flatten.backward(&grad_output).unwrap();
    assert_eq!(grad_input.shape(), input.shape());
}
