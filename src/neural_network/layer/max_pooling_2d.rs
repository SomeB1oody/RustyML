use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::ArrayD;
use rayon::prelude::*;

/// 定义最大池化操作的结构体，用于在2D数据上执行最大池化。
///
/// 最大池化是CNN中常用的下采样技术，通过在每个池窗口中选择最大值来减小
/// 特征图的空间尺寸，从而减少计算量并控制过拟合。
///
/// # 字段
///
/// - `pool_size` - 池化窗口的大小，表示为(高度, 宽度)。
/// - `strides` - 池化操作的步长，表示为(垂直步长, 水平步长)。
/// - `input_shape` - 输入张量的形状。
/// - `input_cache` - 缓存的输入数据，用于反向传播。
/// - `max_positions` - 记录最大值位置的缓存，用于反向传播。
pub struct MaxPooling2D {
    pool_size: (usize, usize),
    strides: (usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize, usize)>>,
}

impl MaxPooling2D {
    /// 创建一个新的2D最大池化层。
    ///
    /// # 参数
    ///
    /// - `pool_size` - 池化窗口的大小，表示为(高度, 宽度)。
    /// - `input_shape` - 输入张量的形状，格式为\[batch_size, channels, height, width\]。
    /// - `strides` - 池化操作的步长，表示为(垂直步长, 水平步长)。如果为None，则使用与pool_size相同的值。
    ///
    /// # 返回
    ///
    /// * `Self` - 一个新的MaxPooling2D层实例。
    pub fn new(
        pool_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize)>,
    ) -> Self {
        // 如果未指定步长，则使用与池化大小相同的步长
        let strides = strides.unwrap_or(pool_size);

        MaxPooling2D {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
            max_positions: None,
        }
    }

    /// 计算最大池化层的输出形状。
    ///
    /// # 参数
    ///
    /// * `input_shape` - 输入张量的形状，格式为\[batch_size, channels, height, width\]。
    ///
    /// # 返回
    ///
    /// 一个包含计算出的输出形状的向量，格式为\[batch_size, channels, output_height, output_width\]。
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        // 计算输出的高度和宽度
        let output_height = (input_height - self.pool_size.0) / self.strides.0 + 1;
        let output_width = (input_width - self.pool_size.1) / self.strides.1 + 1;

        vec![batch_size, channels, output_height, output_width]
    }

    /// 执行最大池化操作。
    ///
    /// # 参数
    ///
    /// * `input` - 输入张量，形状为\[batch_size, channels, height, width\]。
    ///
    /// # 返回
    ///
    /// * `(Tensor, Vec<(usize, usize, usize, usize)>)` - 池化操作的结果和最大值的位置。
    fn max_pool(&self, input: &Tensor) -> (Tensor, Vec<(usize, usize, usize, usize)>) {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // 预分配输出数组
        let mut output = ArrayD::zeros(output_shape.clone());
        // 存储最大值位置的向量
        let mut max_positions = Vec::new();

        // 为每个批次和通道并行处理
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                // 在这里克隆 output_shape，避免所有权移动问题
                let output_shape_clone = output_shape.clone();
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();
                    let mut batch_channel_positions = Vec::new();

                    // 对每个输出位置执行池化
                    for i in 0..output_shape_clone[2] {
                        let i_start = i * self.strides.0;

                        for j in 0..output_shape_clone[3] {
                            let j_start = j * self.strides.1;

                            // 在池化窗口中查找最大值
                            let mut max_val = f32::NEG_INFINITY;
                            let mut max_pos = (0, 0);

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

                                    let val = input[[b, c, i_pos, j_pos]];
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = (i_pos, j_pos);
                                    }
                                }
                            }

                            batch_channel_output.push((i, j, max_val));
                            batch_channel_positions.push((b, c, max_pos.0, max_pos.1));
                        }
                    }

                    ((b, c), (batch_channel_output, batch_channel_positions))
                })
            })
            .collect();

        // 将结果合并到输出张量中
        for ((b, c), (outputs, positions)) in results {
            for (i, j, val) in outputs {
                output[[b, c, i, j]] = val;
            }
            max_positions.extend(positions);
        }

        (output, max_positions)
    }
}

impl Layer for MaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 保存输入用于反向传播
        self.input_cache = Some(input.clone());

        // 执行最大池化操作
        let (output, max_positions) = self.max_pool(input);

        // 存储最大值位置用于反向传播
        self.max_positions = Some(max_positions);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let (Some(input), Some(max_positions)) = (&self.input_cache, &self.max_positions) {
            let grad_shape = grad_output.shape();

            // 初始化输入梯度，形状与输入相同
            let mut input_gradients = ArrayD::zeros(input.dim());

            // 创建一个包含更新位置和值的向量
            let gradient_updates: Vec<_> = max_positions
                .par_iter()
                .filter_map(|&(b, c, i, j)| {
                    // 计算对应的输出梯度索引
                    let out_i = i / self.strides.0;
                    let out_j = j / self.strides.1;

                    // 确保索引在有效范围内
                    if out_i < grad_shape[2] && out_j < grad_shape[3] {
                        // 返回索引和梯度值
                        Some(((b, c, i, j), grad_output[[b, c, out_i, out_j]]))
                    } else {
                        None
                    }
                })
                .collect();

            // 顺序应用梯度更新
            for ((b, c, i, j), grad_val) in gradient_updates {
                input_gradients[[b, c, i, j]] = grad_val;
            }

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "MaxPooling2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> usize {
        // 池化层没有可训练参数
        0
    }

    // 池化层没有可训练参数，因此这些方法不做任何事情
    fn update_parameters_sgd(&mut self, _lr: f32) {
        // 最大池化层没有参数需要更新
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // 最大池化层没有参数需要更新
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // 最大池化层没有参数需要更新
    }

    fn get_weights(&self) -> super::LayerWeight {
        // 最大池化层没有权重
        super::LayerWeight::Empty
    }
}
