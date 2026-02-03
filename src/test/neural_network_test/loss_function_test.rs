use crate::neural_network::layer::activation_layer::relu::ReLU;
use crate::neural_network::layer::dense::Dense;
use crate::neural_network::loss_function::binary_cross_entropy::BinaryCrossEntropy;
use crate::neural_network::loss_function::categorical_cross_entropy::CategoricalCrossEntropy;
use crate::neural_network::loss_function::mean_absolute_error::MeanAbsoluteError;
use crate::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use crate::neural_network::loss_function::sparse_categorical_cross_entropy::SparseCategoricalCrossEntropy;
use crate::neural_network::optimizer::sgd::SGD;
use crate::neural_network::sequential::Sequential;
use ndarray::{Array, ArrayD};

#[test]
fn mse_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn mae_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), MeanAbsoluteError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn binary_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), BinaryCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn categorical_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), CategoricalCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn sparse_categorical_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    // Assume there are 3 classes, labels should be one of 0, 1, 2
    let y: ArrayD<f32> = Array::from_shape_vec((2, 1), vec![0.0, 1.0])
        .unwrap()
        .into_dyn();

    // Build the model, note that the second Dense layer must use Dense::new(3, 3) because it's a multi-class task
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 3, ReLU::new()).unwrap());
    model.compile(
        SGD::new(0.01).unwrap(),
        SparseCategoricalCrossEntropy::new(),
    );

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}
