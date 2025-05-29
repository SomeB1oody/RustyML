use super::*;

fn generate_data(batch_size: usize, channels: usize, length: usize) -> Tensor {
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // 初始化输入数据，为测试平均值计算设置特定值
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                input_data[[b, c, l]] = (b * 100 + c * 10 + l) as f32;
            }
        }
    }

    input_data
}

#[test]
fn test_global_average_pooling_1d_creation() {
    // 创建GlobalAveragePooling1D层
    let layer = GlobalAveragePooling1D::new();

    // 验证层类型
    assert_eq!(layer.layer_type(), "GlobalAveragePooling1D");

    // 验证参数数量应为0
    assert_eq!(layer.param_count(), 0);

    // 验证初始化前的输出形状
    assert_eq!(layer.output_shape(), "Unknown");
}

#[test]
fn test_global_average_pooling_1d_forward() {
    // 创建输入张量: [batch_size, channels, length]
    let batch_size = 2;
    let channels = 3;
    let length = 4;

    // 生成顺序递增的数据
    let input_data = generate_data(batch_size, channels, length);

    // 创建层并执行前向传播
    let mut layer = GlobalAveragePooling1D::new();
    let output = layer.forward(&input_data);

    // 验证输出形状 - 应为 [batch_size, channels]
    assert_eq!(output.shape(), &[batch_size, channels]);

    // 验证输出值 - 应为每个通道的平均值
    for b in 0..batch_size {
        for c in 0..channels {
            // 计算每个通道的期望平均值
            let sum: f32 = (0..length).map(|l| (b * 100 + c * 10 + l) as f32).sum();
            let expected_avg = sum / (length as f32);
            assert_relative_eq!(output[[b, c]], expected_avg);
        }
    }

    // 验证层的输出形状字符串
    assert_eq!(layer.output_shape(), format!("[batch_size, {}]", channels));
}

#[test]
fn test_global_average_pooling_1d_sequential() {
    // 创建Sequential模型
    let mut model = Sequential::new();

    // 添加GlobalAveragePooling1D层
    model.add(GlobalAveragePooling1D::new());

    // 创建测试输入数据: [batch_size, channels, length]
    let batch_size = 3;
    let channels = 4;
    let length = 8;
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // 填充数据 - 使每个通道中的所有值相同以便于验证
    for b in 0..batch_size {
        for c in 0..channels {
            let value = (b * 10 + c) as f32;
            for l in 0..length {
                input_data[[b, c, l]] = value;
            }
        }
    }

    // 前向传播
    let output = model.predict(&input_data);

    // 验证输出形状 - 应为 [batch_size, channels]
    assert_eq!(output.shape(), &[batch_size, channels]);

    // 验证输出值 - 应与输入值相同，因为每个通道中的所有值都相同
    for b in 0..batch_size {
        for c in 0..channels {
            let expected_value = (b * 10 + c) as f32;
            assert_relative_eq!(output[[b, c]], expected_value);
        }
    }
}

#[test]
fn test_global_average_pooling_1d_backward() {
    // 创建输入张量: [batch_size, channels, length]
    let batch_size = 2;
    let channels = 3;
    let length = 4;

    // 创建输入数据
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // 填充数据 - 使用随机值
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                input_data[[b, c, l]] = (b + c + l) as f32;
            }
        }
    }

    // 创建层并执行前向传播
    let mut layer = GlobalAveragePooling1D::new();
    let _output = layer.forward(&input_data);

    // 创建梯度输出 - 形状应与输出匹配 [batch_size, channels]
    let grad_output = Tensor::ones(IxDyn(&[batch_size, channels]));

    // 执行反向传播
    let grad_input = layer.backward(&grad_output).unwrap();

    // 验证梯度输入形状 - 应与原始输入匹配
    assert_eq!(grad_input.shape(), input_data.shape());

    // 验证梯度 - 应在每个位置均匀分布
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                // 梯度应该是均匀分布的，等于1.0 / length
                assert_relative_eq!(grad_input[[b, c, l]], 1.0 / (length as f32));
            }
        }
    }
}

#[test]
fn test_global_average_pooling_1d_zero_input() {
    // 创建全零输入张量
    let input_data = Tensor::zeros(IxDyn(&[2, 3, 4]));

    // 创建层并执行前向传播
    let mut layer = GlobalAveragePooling1D::new();
    let output = layer.forward(&input_data);

    // 验证输出形状
    assert_eq!(output.shape(), &[2, 3]);

    // 验证输出值 - 所有值应为0
    for b in 0..2 {
        for c in 0..3 {
            assert_relative_eq!(output[[b, c]], 0.0);
        }
    }
}
