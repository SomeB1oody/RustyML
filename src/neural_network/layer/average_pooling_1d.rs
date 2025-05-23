use super::super::Tensor;
use crate::ModelError;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::Array3;
use rayon::prelude::*;

/// 一维平均池化层，用于神经网络。
///
/// 此层对三维张量进行平均池化操作。
/// 平均池化计算由池化大小定义的每个patch的平均值。
///
/// # 输入形状
///
/// 输入是一个三维张量，形状为 \[batch_size, channels, length\]
///
/// # 输出形状
///
/// 输出是一个三维张量，形状为 \[batch_size, channels, pooled_length\]
/// 其中:
/// - pooled_length = (length - pool_size) / stride + 1
///
/// # 字段
///
/// - `pool_size` - 池化窗口的大小
/// - `stride` - 池化操作的步长
/// - `input_shape` - 输入张量的形状
/// - `input_cache` - 前向传播中缓存的输入张量，用于反向传播
///
/// # 示例
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array3;
/// use approx::assert_relative_eq;
///
/// // 创建一个简单的输入张量: [batch_size, channels, length]
/// // 批量大小=2, 3个输入通道, 每个通道是8个元素
/// let mut input_data = Array3::zeros((2, 3, 8));
///
/// // 设置测试数据使平均池化结果可预测
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..8 {
///             input_data[[b, c, i]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // 使用Sequential模型测试AveragePooling1D
/// let mut model = Sequential::new();
/// model
///     .add(AveragePooling1D::new(
///         2,              // 池化窗口大小
///         2,              // 步长
///         vec![2, 3, 8],  // 输入形状
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // 输出形状应为 [2, 3, 4]
/// let output = model.predict(&x);
/// assert_eq!(output.shape(), &[2, 3, 4]);
///
/// // 验证池化结果的正确性
/// // 对于大小为2的窗口和步长为2，我们期望结果是窗口中元素的平均值
/// for b in 0..2 {
///     for c in 0..3 {
///         // 第一个窗口 (0,1) -> 平均应为 (0+1)/2 = 0.5
///         assert_relative_eq!(output[[b, c, 0]], 0.5);
///         // 第二个窗口 (2,3) -> 平均应为 (2+3)/2 = 2.5
///         assert_relative_eq!(output[[b, c, 1]], 2.5);
///         // 第三个窗口 (4,5) -> 平均应为 (4+5)/2 = 4.5
///         assert_relative_eq!(output[[b, c, 2]], 4.5);
///         // 第四个窗口 (6,7) -> 平均应为 (6+7)/2 = 6.5
///         assert_relative_eq!(output[[b, c, 3]], 6.5);
///     }
/// }
/// ```
pub struct AveragePooling1D {
    pool_size: usize,
    stride: usize,
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl AveragePooling1D {
    /// 创建一个新的一维平均池化层
    ///
    /// # 参数
    ///
    /// * `pool_size` - 池化窗口的大小
    /// * `stride` - 池化操作的步长
    /// * `input_shape` - 输入张量的形状 \[batch_size, channels, length\]
    pub fn new(pool_size: usize, stride: usize, input_shape: Vec<usize>) -> Self {
        AveragePooling1D {
            pool_size,
            stride,
            input_shape,
            input_cache: None,
        }
    }

    /// 计算输出形状
    ///
    /// 基于输入形状、池化窗口大小和步长计算输出形状
    fn compute_output_shape(&self) -> Vec<usize> {
        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let length = self.input_shape[2];

        let output_length = (length - self.pool_size) / self.stride + 1;

        vec![batch_size, channels, output_length]
    }
}

impl Layer for AveragePooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 缓存输入用于反向传播
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let output_length = (length - self.pool_size) / self.stride + 1;
        let mut output = Array3::<f32>::zeros((batch_size, channels, output_length)).into_dyn();

        // 从self中复制需要的值，避免在闭包中捕获self
        let pool_size = self.pool_size;
        let stride = self.stride;

        // 使用rayon并行处理批次和通道
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();

                    // 对每个输出位置执行池化
                    for i in 0..output_length {
                        let start_idx = i * stride;
                        let end_idx = start_idx + pool_size;

                        // 计算窗口中元素的平均值
                        let mut sum = 0.0;
                        for j in start_idx..end_idx {
                            sum += input[[b, c, j]];
                        }
                        batch_channel_output.push((i, sum / (pool_size as f32)));
                    }

                    ((b, c), batch_channel_output)
                })
            })
            .collect();

        // 将结果合并到输出张量
        for ((b, c), outputs) in results {
            for (i, val) in outputs {
                output[[b, c, i]] = val;
            }
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // 确保有缓存的输入
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "No cached input for AveragePooling1D".to_string(),
                ));
            }
        };

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let mut grad_input = Array3::<f32>::zeros((batch_size, channels, length)).into_dyn();

        // 计算对输入的梯度
        let scale_factor = 1.0 / (self.pool_size as f32);

        // 复制在闭包中需要的成员变量
        let pool_size = self.pool_size;
        let stride = self.stride;

        // 使用rayon并行处理批次和通道
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_grad =
                        Array3::<f32>::zeros((batch_size, channels, length)).into_dyn();

                    for i in 0..grad_output.shape()[2] {
                        let start_idx = i * stride;
                        let end_idx = start_idx + pool_size;

                        // 将梯度均匀分配给输入窗口中的每个元素
                        for j in start_idx..end_idx {
                            if j < length {
                                batch_channel_grad[[b, c, j]] +=
                                    grad_output[[b, c, i]] * scale_factor;
                            }
                        }
                    }

                    ((b, c), batch_channel_grad)
                })
            })
            .collect();

        // 合并所有批次和通道的梯度
        for ((b, c), grad) in results {
            for j in 0..length {
                grad_input[[b, c, j]] += grad[[b, c, j]];
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "AveragePooling1D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.compute_output_shape();
        format!("{:?}", output_shape)
    }

    fn update_parameters_sgd(&mut self, _lr: f32) {
        // 池化层没有可训练参数
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // 池化层没有可训练参数
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // 池化层没有可训练参数
    }

    fn get_weights(&self) -> LayerWeight {
        LayerWeight::Empty
    }
}
