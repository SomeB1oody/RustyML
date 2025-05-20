use crate::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::{Array, IxDyn};

/// Global Average Pooling 2D Layer
///
/// 对输入张量的空间维度（高度和宽度）执行全局平均池化操作。
/// 输入张量形状应为 `[batch_size, channels, height, width]`，
/// 输出张量形状将为 `[batch_size, channels]`。
///
/// 该层没有可训练参数。
///
/// # 字段
///
/// * `input_shape` - 在前向传播过程中存储输入张量的形状。
/// * `input_cache` - 缓存输入张量用于反向传播。
///
/// # 示例
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // 创建一个包含多个层的Sequential模型
/// let mut model = Sequential::new();
///
/// // 添加一个GlobalAveragePooling2D层
/// model.add(GlobalAveragePooling2D::new());
///
/// // 创建测试输入张量: [batch_size, channels, height, width]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 5, 5]), 1.0);
///
/// // 前向传播
/// let output = model.predict(&input_data);
///
/// // 检查输出形状 - 应该是 [3, 4]
/// assert_eq!(output.shape(), &[3, 4]);
///
/// // 由于所有输入值均为1.0，所有输出值也应为1.0
/// for b in 0..3 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
pub struct GlobalAveragePooling2D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl GlobalAveragePooling2D {
    /// 创建全局平均池化层的新实例
    ///
    /// # 返回
    ///
    /// 一个新的 `GlobalAveragePooling2D` 实例
    pub fn new() -> Self {
        GlobalAveragePooling2D {
            input_shape: Vec::new(),
            input_cache: None,
        }
    }
}

impl Layer for GlobalAveragePooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 保存输入形状并缓存输入用于反向传播
        self.input_shape = input.shape().to_vec();
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let spatial_size = height * width;

        // 对每个样本和每个通道执行全局平均池化
        let mut result = Array::zeros(IxDyn(&[batch_size, channels]));

        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0;

                // 计算每个通道中所有值的总和
                for h in 0..height {
                    for w in 0..width {
                        sum += input[[b, c, h, w]];
                    }
                }

                // 将平均值放入结果张量
                result[[b, c]] = sum / (spatial_size as f32);
            }
        }

        result
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // 检查是否有有效的输入缓存
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run yet".to_string(),
                ));
            }
        };

        // 获取输入维度
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let spatial_size = height * width;

        // 创建一个与输入形状相同的梯度张量，初始化为零
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, channels, height, width]));

        // 将梯度均匀分布到每个空间位置
        let scale_factor = 1.0 / (spatial_size as f32);

        for b in 0..batch_size {
            for c in 0..channels {
                let grad = grad_output[[b, c]] * scale_factor;

                // 将相同的梯度值分配给每个空间位置
                for h in 0..height {
                    for w in 0..width {
                        grad_input[[b, c, h, w]] = grad;
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling2D"
    }

    fn output_shape(&self) -> String {
        if self.input_shape.is_empty() {
            return "unknown".to_string();
        }
        format!("[{}, {}]", self.input_shape[0], self.input_shape[1])
    }

    fn param_count(&self) -> usize {
        0 // GlobalAveragePooling2D层没有可训练参数
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
        // 返回空的LayerWeight，因为没有权重
        LayerWeight::Empty
    }
}
