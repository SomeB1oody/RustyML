use crate::neural_network::layer::LayerWeight;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::IxDyn;

/// 将4D张量展平为2D张量的层。
///
/// 该层通常用于卷积网络中，将特征提取层（如卷积层或池化层）的输出转换为
/// 密集层（全连接层）可以处理的格式。
///
/// # 输入形状
///
/// 输入是一个4D张量，形状为 \[batch_size, channels, height, width\]
///
/// # 输出形状
///
/// 输出是一个2D张量，形状为 \[batch_size, channels * height * width\]
///
/// # 示例
///
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
///
/// // 创建一个4D输入张量: [batch_size, channels, height, width]
/// // 批量大小=2，3个通道，每个4x4像素
/// let x = Array4::ones((2, 3, 4, 4)).into_dyn();
///
/// // 构建包含Flatten层的模型
/// let mut model = Sequential::new();
/// model
///     .add(Flatten::new(vec![2, 3, 4, 4]))
///     .compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // 查看模型结构
/// model.summary();
///
/// // 前向传播
/// let flattened = model.predict(&x);
///
/// // 检查输出形状 - 应该是 [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
pub struct Flatten {
    output_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl Flatten {
    /// 创建一个新的Flatten层。
    ///
    /// # 参数
    ///
    /// * `input_shape` - 输入张量的形状，格式为 \[batch_size, channels, height, width\]
    ///
    /// # 返回
    ///
    /// * `Self` - 一个新的`Flatten`层实例
    pub fn new(input_shape: Vec<usize>) -> Self {
        assert_eq!(input_shape.len(), 4, "输入形状必须是4维的");

        let batch_size = input_shape[0];
        let flattened_features = input_shape[1..].iter().product();
        let output_shape = vec![batch_size, flattened_features];

        Flatten {
            output_shape,
            input_cache: None,
        }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 保存输入以便反向传播
        self.input_cache = Some(input.clone());

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        // 创建新形状
        let output = input.clone();
        output
            .into_shape_with_order(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape().to_vec();

            // 将梯度重塑回输入形状
            let reshaped_grad = grad_output
                .clone()
                .into_shape_with_order(IxDyn(&input_shape))
                .unwrap();

            Ok(reshaped_grad)
        } else {
            Err(ModelError::ProcessingError("前向传播尚未运行".to_string()))
        }
    }

    fn layer_type(&self) -> &str {
        "Flatten"
    }

    fn output_shape(&self) -> String {
        format!("({}, {})", self.output_shape[0], self.output_shape[1])
    }

    fn param_count(&self) -> usize {
        // Flatten层没有可训练参数
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
        // Flatten层没有权重
        LayerWeight::Empty
    }
}
