use super::*;

#[test]
fn test_sequential_new() {
    let model = Sequential::new();
    // Test that we can create a new model without errors
    // We can't access private fields directly, but we can test behavior

    // Test that get_weights returns empty vector for new model
    let weights = model.get_weights();
    assert_eq!(weights.len(), 0);
}

#[test]
fn test_sequential_add_layer() {
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3, ReLU::new()));

    // Test that one layer was added
    let weights = model.get_weights();
    assert_eq!(weights.len(), 1);

    model.add(Dense::new(3, 1, Sigmoid::new()));

    // Test that second layer was added
    let weights = model.get_weights();
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_sequential_method_chaining() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()))
        .add(Dense::new(3, 1, Sigmoid::new()))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // Test that both layers were added
    let weights = model.get_weights();
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_sequential_fit_success() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, ReLU::new()))
        .add(Dense::new(3, 1, Sigmoid::new()))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let x = Array::ones((3, 2)).into_dyn();
    let y = Array::ones((3, 1)).into_dyn();

    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());
}

#[test]
fn test_sequential_fit_no_optimizer() {
    let mut model = Sequential::new();
    model.add(Dense::new(2, 1, ReLU::new()));

    let x = Array::ones((3, 2)).into_dyn();
    let y = Array::ones((3, 1)).into_dyn();

    let result = model.fit(&x, &y, 1);
    assert!(result.is_err());
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Optimizer not specified"));
    }
}

#[test]
fn test_sequential_fit_no_layers() {
    let mut model = Sequential::new();
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    let x = Array::ones((3, 2)).into_dyn();
    let y = Array::ones((3, 1)).into_dyn();

    let result = model.fit(&x, &y, 1);
    assert!(result.is_err());
    if let Err(ModelError::InputValidationError(msg)) = result {
        assert!(msg.contains("Layers not specified"));
    }
}

#[test]
fn test_sequential_predict() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, ReLU::new()))
        .add(Dense::new(3, 1, Sigmoid::new()));

    let x = Array::ones((3, 2)).into_dyn();
    let prediction = model.predict(&x);

    assert_eq!(prediction.shape(), &[3, 1]);
}

#[test]
fn test_sequential_summary() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()))
        .add(Dense::new(3, 2, Sigmoid::new()))
        .add(Dense::new(2, 1, Tanh::new()));

    // Test that summary doesn't panic
    model.summary();
}

#[test]
fn test_sequential_get_weights() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, ReLU::new()))
        .add(Dense::new(3, 1, Sigmoid::new()));

    let weights = model.get_weights();
    assert_eq!(weights.len(), 2);

    // Check that the first layer returns Dense weights
    if let LayerWeight::Dense(dense_weights) = &weights[0] {
        assert_eq!(dense_weights.weight.shape(), &[2, 3]);
        assert_eq!(dense_weights.bias.shape(), &[1, 3]);
    } else {
        panic!("Expected Dense layer weights");
    }
}

#[test]
fn test_sequential_different_optimizers() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(3, 2, ReLU::new()))
        .add(Dense::new(2, 1, Sigmoid::new()))
        .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanAbsoluteError::new());

    let x = Array::ones((2, 3)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    let result = model.fit(&x, &y, 1);
    assert!(result.is_ok());
}

#[test]
fn test_sequential_complex_model() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 8, ReLU::new()))
        .add(Dense::new(8, 4, ReLU::new()))
        .add(Dense::new(4, 2, Tanh::new()))
        .add(Dense::new(2, 1, Sigmoid::new()))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let x = Array::ones((5, 4)).into_dyn();
    let y = Array::zeros((5, 1)).into_dyn();

    let result = model.fit(&x, &y, 3);
    assert!(result.is_ok());

    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[5, 1]);
}

#[test]
fn test_sequential_different_activations() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(3, 4, ReLU::new()))
        .add(Dense::new(4, 3, Tanh::new()))
        .add(Dense::new(3, 2, Sigmoid::new()))
        .add(Dense::new(2, 1, ReLU::new()))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let x = Array::ones((2, 3)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());

    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[2, 1]);
}

#[test]
fn test_sequential_empty_model_summary() {
    let model = Sequential::new();
    // Test that summary works even for empty model
    model.summary();
}

#[test]
fn test_sequential_predict_before_compile() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, ReLU::new()))
        .add(Dense::new(3, 1, Sigmoid::new()));

    let x = Array::ones((2, 2)).into_dyn();

    // Should be able to predict even without compiling
    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[2, 1]);
}

#[test]
fn test_sequential_multiple_fit_calls() {
    let mut model = Sequential::new();
    model
        .add(Dense::new(2, 3, ReLU::new()))
        .add(Dense::new(3, 1, Sigmoid::new()))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let x = Array::ones((3, 2)).into_dyn();
    let y = Array::ones((3, 1)).into_dyn();

    // First fit
    let result1 = model.fit(&x, &y, 1);
    assert!(result1.is_ok());

    // Second fit should also work
    let result2 = model.fit(&x, &y, 1);
    assert!(result2.is_ok());
}
