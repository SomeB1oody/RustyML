use crate::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::{Array, IxDyn};

/// 全局最大池化层
///
/// 对输入张量在空间维度（高度和宽度）上执行全局最大池化操作。
/// 输入张量的形状应为 `[batch_size, channels, height, width]`，
/// 输出张量的形状将是 `[batch_size, channels]`。
///
/// 该层不含有可训练参数。
pub struct GlobalMaxPooling2D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize)>>,
}

impl GlobalMaxPooling2D {
    /// 创建一个新的全局最大池化层实例
    ///
    /// # 返回值
    ///
    /// 一个新的 `GlobalMaxPooling2D` 实例
    pub fn new() -> Self {
        GlobalMaxPooling2D {
            input_shape: Vec::new(),
            input_cache: None,
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 保存输入形状和输入缓存用于反向传播
        self.input_shape = input.shape().to_vec();
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];

        // 对每个样本的每个通道进行全局最大池化
        let mut result = Array::zeros(IxDyn(&[batch_size, channels]));
        let mut max_positions = Vec::with_capacity(batch_size * channels);

        for b in 0..batch_size {
            for c in 0..channels {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_h = 0;
                let mut max_w = 0;

                // 查找每个通道中的最大值及其位置
                for h in 0..height {
                    for w in 0..width {
                        let val = input[[b, c, h, w]];
                        if val > max_val {
                            max_val = val;
                            max_h = h;
                            max_w = w;
                        }
                    }
                }

                // 保存最大值位置用于反向传播
                max_positions.push((max_h, max_w));

                // 将最大值放入结果张量
                result[[b, c]] = max_val;
            }
        }

        self.max_positions = Some(max_positions);
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

        // 检查是否有有效的最大值位置记录
        let max_positions = match &self.max_positions {
            Some(positions) => positions,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run yet".to_string(),
                ));
            }
        };

        // 获取输入形状
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];

        // 创建与输入相同形状的梯度张量，初始值为 0
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, channels, height, width]));

        // 只在每个通道的最大值位置传递梯度
        let mut idx = 0;
        for b in 0..batch_size {
            for c in 0..channels {
                let (max_h, max_w) = max_positions[idx];
                // 将梯度值传递到原始输入中最大值所在的位置
                grad_input[[b, c, max_h, max_w]] = grad_output[[b, c]];
                idx += 1;
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling2D"
    }

    fn output_shape(&self) -> String {
        if self.input_shape.is_empty() {
            return "未定义".to_string();
        }
        format!("[{}, {}]", self.input_shape[0], self.input_shape[1])
    }

    fn param_count(&self) -> usize {
        0 // GlobalMaxPooling2D 层没有可训练参数
    }

    fn update_parameters_sgd(&mut self, _lr: f32) {
        // 无参数需要更新
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // 无参数需要更新
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // 无参数需要更新
    }

    fn get_weights(&self) -> LayerWeight {
        // 由于没有权重，返回空的LayerWeight
        LayerWeight::Empty
    }
}
