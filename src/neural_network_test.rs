use ndarray::prelude::*;
use crate::neural_network::*;

#[test]
fn mse_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

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
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), MeanAbsoluteError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

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
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), BinaryCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

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
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), CategoricalCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn sparse_categorical_cross_entropy_test() {
    // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    // Assume there are 3 classes, labels should be one of 0, 1, 2
    let y: ArrayD<f32> = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap().into_dyn();

    // Build the model, note that the second Dense layer must use Dense::new(3, 3) because it's a multi-class task
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 3));
    model.compile(SGD::new(0.01), SparseCategoricalCrossEntropy::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction results: {:?}", prediction);
}

#[test]
fn adam_test() {
    // Create an input tensor with shape (batch_size=2, input_dim=4) and corresponding target tensor
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // Build the model: add two Dense layers, use Adam optimizer (learning rate, beta1, beta2, epsilon) with MSE loss function
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // Print model structure
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

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
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // Print model structure (summary)
    model.summary();

    // Train the model
    model.fit(&x, &y, 3);

    // Use predict for forward propagation prediction
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}

#[test]
fn with_activation_test() {
    // 构造形状为 (batch_size=2, input_dim=4) 的输入张量，以及目标张量 (batch_size=2, output_dim=3)
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型：示例中使用 Dense 层并指定激活函数（例如使用 ReLU 或 Softmax）
    // 这里第一层采用 Sigmoid 激活，第二层采用 Softmax 激活（你可以根据需要修改）
    let mut model = Sequential::new();
    model.add(Dense::new_with_activation(4, 3, Some(Activation::Sigmoid)))
        .add(Dense::new_with_activation(3, 1, Some(Activation::Softmax)));

    // 选择优化器，例如 RMSprop、Adam 或 SGD，这里选择 RMSprop 作为示例
    model.compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    // 打印模型结构
    model.summary();

    // 训练模型（这里只做简单迭代示例）
    model.fit(&x, &y, 3);

    // 使用 predict 得到输出
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}