use super::*;

#[test]
fn adam_test() {
    // Create an input tensor with shape (batch_size=2, input_dim=4) and corresponding target tensor
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model: add two Dense layers, use Adam optimizer (learning rate, beta1, beta2, epsilon) with MSE loss function
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(
        Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}

#[test]
fn rmsprop_test() {
    // Create input (batch_size=2, input_dim=4) and target tensors (batch_size=2, output_dim=3)
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model, add two Dense layers; choose RMSprop optimizer when compiling
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(Dense::new(3, 1, ReLU::new()).unwrap());
    model.compile(
        RMSprop::new(0.001, 0.9, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3).unwrap();

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}
