use ndarray::prelude::*;
use crate::neural_network::*;

#[test]
fn mse_test() {
    // 构造输入和目标张量，假设输入维度为 4，输出维度为 3，batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // 打印模型结构（summary）
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("预测结果: {:?}", prediction);
}

#[test]
fn mae_test() {
    // 构造输入和目标张量，假设输入维度为 4，输出维度为 3，batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), MeanAbsoluteError::new());

    // 打印模型结构（summary）
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("预测结果: {:?}", prediction);
}

#[test]
fn binary_cross_entropy_test() {
    // 构造输入和目标张量，假设输入维度为 4，输出维度为 3，batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), BinaryCrossEntropy::new());

    // 打印模型结构（summary）
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("预测结果: {:?}", prediction);
}

#[test]
fn categorical_cross_entropy_test() {
    // 构造输入和目标张量，假设输入维度为 4，输出维度为 3，batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), CategoricalCrossEntropy::new());

    // 打印模型结构（summary）
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("预测结果: {:?}", prediction);
}

#[test]
fn sparse_categorical_cross_entropy_test() {
    // 构造输入和目标张量，假设输入维度为 4，输出维度为 3，batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    // 假设有3个类别，标签应该是0,1,2中的一个
    let y: ArrayD<f32> = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap().into_dyn();

    // 构建模型，注意第二个 Dense 层必须用 Dense::new(3, 3),因为是多分类任务
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 3));
    model.compile(SGD::new(0.01), SparseCategoricalCrossEntropy::new());

    // 打印模型结构（summary）
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("预测结果: {:?}", prediction);
}

#[test]
fn adam_test() {
    // 创建形状为 (batch_size=2, input_dim=4) 的输入张量，以及对应目标张量
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型：添加两个 Dense 层，使用 Adam 优化器（学习率、beta1、beta2、epsilon）与 MSE 损失函数
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8), MeanSquaredError::new());

    // 打印模型结构
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("Prediction: {:?}", prediction);
}
