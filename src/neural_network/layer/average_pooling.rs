use crate::neural_network::layer::LayerWeight;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::ArrayD;
use rayon::prelude::*;

/// A 2D average pooling layer for neural networks.
///
/// This layer performs average pooling operations on 4D tensors.
/// Average pooling computes the average value of each patch defined by the pool size.
///
/// # Input Shape
///
/// Input is a 4D tensor with shape \[batch_size, channels, height, width\]
///
/// # Output Shape
///
/// Output is a 4D tensor with shape \[batch_size, channels, pooled_height, pooled_width\]
/// where:
/// - pooled_height = (height - pool_size_h) / stride_h + 1
/// - pooled_width = (width - pool_size_w) / stride_w + 1
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window as (height, width)
/// - `strides` - Stride of the pooling operation as (height, width)
/// - `input_shape` - Shape of the input tensor
/// - `input_cache` - Cached input tensor from forward pass, used in backpropagation
pub struct AveragePooling {
    pool_size: (usize, usize),
    strides: (usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl AveragePooling {
    /// Creates a new AveragePooling layer.
    ///
    /// # Parameters
    ///
    /// * `pool_size` - Size of the pooling window as (height, width)
    /// * `strides` - Stride of the pooling operation as (height, width)
    /// * `input_shape` - Shape of the input tensor \[batch_size, channels, height, width\]
    ///
    /// # Returns
    ///
    /// * `Self` - A new `AveragePooling` layer instance
    pub fn new(
        pool_size: (usize, usize),
        strides: (usize, usize),
        input_shape: Vec<usize>,
    ) -> Self {
        assert_eq!(
            input_shape.len(),
            4,
            "Input shape must be 4-dimensional for AveragePooling"
        );

        AveragePooling {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
        }
    }

    /// Calculates the output shape for the pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - Shape of the output tensor after pooling
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let output_height = (input_shape[2] - self.pool_size.0) / self.strides.0 + 1;
        let output_width = (input_shape[3] - self.pool_size.1) / self.strides.1 + 1;

        vec![input_shape[0], input_shape[1], output_height, output_width]
    }

    /// Performs average pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input` - Input tensor with shape \[batch_size, channels, height, width\]
    ///
    /// # Returns
    ///
    /// * `Tensor` - Result of the pooling operation
    fn avg_pool(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // 预先分配输出数组
        let mut output = ArrayD::zeros(output_shape.clone());

        // 并行处理每个批次和通道
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                // 克隆output_shape以避免所有权移动问题
                let output_shape_clone = output_shape.clone();
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();

                    // 对每个输出位置执行池化
                    for i in 0..output_shape_clone[2] {
                        let i_start = i * self.strides.0;

                        for j in 0..output_shape_clone[3] {
                            let j_start = j * self.strides.1;

                            // 计算池化窗口内的平均值
                            let mut sum = 0.0;
                            let mut count = 0;

                            for di in 0..self.pool_size.0 {
                                let i_pos = i_start + di;
                                if i_pos >= input_shape[2] {
                                    continue;
                                }

                                for dj in 0..self.pool_size.1 {
                                    let j_pos = j_start + dj;
                                    if j_pos >= input_shape[3] {
                                        continue;
                                    }

                                    sum += input[[b, c, i_pos, j_pos]];
                                    count += 1;
                                }
                            }

                            // 计算平均值，避免除零错误
                            let avg_val = if count > 0 { sum / count as f32 } else { 0.0 };
                            batch_channel_output.push((i, j, avg_val));
                        }
                    }

                    ((b, c), batch_channel_output)
                })
            })
            .collect();

        // 将结果合并到输出张量中
        for ((b, c), outputs) in results {
            for (i, j, val) in outputs {
                output[[b, c, i, j]] = val;
            }
        }

        output
    }
}

impl Layer for AveragePooling {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 保存输入用于反向传播
        self.input_cache = Some(input.clone());

        // 执行平均池化
        self.avg_pool(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];

            // 初始化输入梯度为零
            let mut input_grad = ArrayD::zeros(input_shape.to_vec());

            // 从输出梯度形状中获取输出尺寸
            let output_shape = grad_output.shape();

            // 提前复制需要在闭包中使用的成员变量
            let pool_size = self.pool_size;
            let strides = self.strides;

            // 对每个批次和通道并行处理
            let results: Vec<_> = (0..batch_size)
                .into_par_iter()
                .flat_map(|b| {
                    (0..channels).into_par_iter().map(move |c| {
                        let mut batch_channel_grad: ArrayD<f32> =
                            ArrayD::zeros(input_shape.to_vec());

                        // 对每个输出位置
                        for i in 0..output_shape[2] {
                            let i_start = i * strides.0; // 使用复制的变量，而不是self.strides

                            for j in 0..output_shape[3] {
                                let j_start = j * strides.1; // 使用复制的变量，而不是self.strides

                                // 获取当前输出梯度
                                let grad = grad_output[[b, c, i, j]];

                                // 计算池化窗口中实际元素的数量（考虑边界）
                                let mut count = 0;
                                for di in 0..pool_size.0 {
                                    // 使用复制的变量，而不是self.pool_size
                                    let i_pos = i_start + di;
                                    if i_pos >= input_shape[2] {
                                        continue;
                                    }

                                    for dj in 0..pool_size.1 {
                                        // 使用复制的变量，而不是self.pool_size
                                        let j_pos = j_start + dj;
                                        if j_pos >= input_shape[3] {
                                            continue;
                                        }

                                        count += 1;
                                    }
                                }

                                // 将梯度平均分配给所有参与计算的输入元素
                                let grad_per_element =
                                    if count > 0 { grad / count as f32 } else { 0.0 };

                                for di in 0..pool_size.0 {
                                    // 使用复制的变量，而不是self.pool_size
                                    let i_pos = i_start + di;
                                    if i_pos >= input_shape[2] {
                                        continue;
                                    }

                                    for dj in 0..pool_size.1 {
                                        // 使用复制的变量，而不是self.pool_size
                                        let j_pos = j_start + dj;
                                        if j_pos >= input_shape[3] {
                                            continue;
                                        }

                                        batch_channel_grad[[b, c, i_pos, j_pos]] +=
                                            grad_per_element;
                                    }
                                }
                            }
                        }

                        ((b, c), batch_channel_grad)
                    })
                })
                .collect();

            // 合并所有批次和通道的梯度
            for ((b, c), grad) in results {
                for i in 0..input_shape[2] {
                    for j in 0..input_shape[3] {
                        input_grad[[b, c, i, j]] += grad[[b, c, i, j]];
                    }
                }
            }

            Ok(input_grad)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "AveragePooling"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> usize {
        // 平均池化层没有可训练参数
        0
    }

    fn update_parameters_sgd(&mut self, _lr: f32) {
        // 没有参数需要更新
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // 没有参数需要更新
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // 没有参数需要更新
    }

    fn get_weights(&self) -> LayerWeight {
        // 平均池化层没有权重
        LayerWeight::Empty
    }
}
