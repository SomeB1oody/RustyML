use super::*;

#[test]
fn test_conv1d_sequential_with_sgd() {
    // 创建3D输入张量: [batch_size, channels, length]
    let x = Array3::ones((2, 1, 10)).into_dyn();
    // 创建目标张量 - 假设输出长度为8 (Valid padding)
    let y = Array3::ones((2, 3, 8)).into_dyn();

    // 构建模型
    let mut model = Sequential::new();
    model
        .add(Conv1D::new(
            3,                  // filters
            3,                  // kernel_size
            vec![2, 1, 10],     // input_shape
            1,                  // stride
            PaddingType::Valid, // padding
            Activation::ReLU,   // activation
        ))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    // 打印模型结构
    model.summary();

    // 训练模型
    let result = model.fit(&x, &y, 3);
    assert!(result.is_ok());

    // 预测
    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[2, 3, 8]);

    // 验证预测结果是非负的（ReLU激活函数）
    for value in prediction.iter() {
        assert!(*value >= 0.0);
    }
}

#[test]
fn test_conv1d_sequential_with_rmsprop() {
    // 创建更复杂的训练数据
    let x = Array3::from_shape_fn((3, 2, 8), |(b, c, l)| {
        ((b * 2 + c * 3 + l) as f32).sin() * 0.5
    })
    .into_dyn();

    let y = Array3::zeros((3, 2, 6)).into_dyn(); // Valid padding输出

    // 构建模型
    let mut model = Sequential::new();
    model
        .add(Conv1D::new(
            2,                  // filters
            3,                  // kernel_size
            vec![3, 2, 8],      // input_shape
            1,                  // stride
            PaddingType::Valid, // padding
            Activation::Tanh,   // activation
        ))
        .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());

    model.summary();

    // 训练模型
    let result = model.fit(&x, &y, 4);
    assert!(result.is_ok());

    // 预测
    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[3, 2, 6]);

    // 验证Tanh输出范围在[-1,1]之间
    for value in prediction.iter() {
        assert!(*value >= -1.0 && *value <= 1.0);
    }
}

#[test]
fn test_conv1d_different_strides() {
    let x = Array3::ones((1, 1, 20)).into_dyn();

    // 测试不同的stride值
    let stride_2_conv = Conv1D::new(
        1,
        3,
        vec![1, 1, 20],
        2, // stride = 2
        PaddingType::Valid,
        Activation::ReLU,
    );

    let mut model = Sequential::new();
    model
        .add(stride_2_conv)
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[1, 1, 9]);
}

#[test]
fn test_conv1d_multiple_channels() {
    // 测试多通道输入
    let x = Array3::from_shape_fn((2, 3, 15), |(b, c, l)| (b + c + l) as f32 * 0.1).into_dyn();

    let y = Array3::ones((2, 5, 13)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(Conv1D::new(
            5,                  // filters
            3,                  // kernel_size
            vec![2, 3, 15],     // input_shape (3 channels)
            1,                  // stride
            PaddingType::Valid, // padding
            Activation::ReLU,   // activation
        ))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    model.summary();

    let result = model.fit(&x, &y, 2);
    assert!(result.is_ok());

    let prediction = model.predict(&x);
    assert_eq!(prediction.shape(), &[2, 5, 13]);
}

#[test]
fn test_conv1d_activation_functions() {
    let x = Array3::from_shape_fn((1, 1, 5), |(_, _, l)| {
        l as f32 - 2.0 // 产生负值来测试激活函数
    })
    .into_dyn();

    // 测试ReLU激活函数
    let mut relu_model = Sequential::new();
    relu_model
        .add(Conv1D::new(
            1,
            3,
            vec![1, 1, 5],
            1,
            PaddingType::Valid,
            Activation::ReLU,
        ))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let relu_output = relu_model.predict(&x);
    // ReLU输出应该都是非负的
    for value in relu_output.iter() {
        assert!(*value >= 0.0);
    }

    // 测试Sigmoid激活函数
    let mut sigmoid_model = Sequential::new();
    sigmoid_model
        .add(Conv1D::new(
            1,
            3,
            vec![1, 1, 5],
            1,
            PaddingType::Valid,
            Activation::Sigmoid,
        ))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    let sigmoid_output = sigmoid_model.predict(&x);
    // Sigmoid输出应该在[0,1]之间
    for value in sigmoid_output.iter() {
        assert!(*value >= 0.0 && *value <= 1.0);
    }
}

#[test]
fn test_conv1d_parameter_count() {
    let conv1d = Conv1D::new(
        4,              // filters
        3,              // kernel_size
        vec![2, 2, 10], // input_shape (2 channels)
        1,
        PaddingType::Valid,
        Activation::ReLU,
    );

    // 参数数量 = weights + bias = (4 * 2 * 3) + (1 * 4) = 24 + 4 = 28
    assert_eq!(conv1d.param_count(), 28);
}

#[test]
#[should_panic(expected = "Cannot use Softmax activation function for convolutional layers")]
fn test_conv1d_softmax_panic() {
    let x = Array3::ones((1, 1, 5)).into_dyn();

    let mut model = Sequential::new();
    model
        .add(Conv1D::new(
            1,
            3,
            vec![1, 1, 5],
            1,
            PaddingType::Valid,
            Activation::Softmax, // 这应该导致panic
        ))
        .compile(SGD::new(0.01), MeanSquaredError::new());

    model.predict(&x); // 这里会触发panic
}
