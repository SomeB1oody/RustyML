use super::*;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Dense 层结构体：自动初始化 weights 和 bias，同时在 backward 中计算梯度并存储
pub struct Dense {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Array2<f32>,         // 权重矩阵，形状 (input_dim, output_dim)
    pub bias: Array2<f32>,            // 偏置，形状 (1, output_dim)
    pub input_cache: Option<Array2<f32>>, // 缓存 forward 时的输入，供 backward 使用
    pub grad_weights: Option<Array2<f32>>, // 存储计算得到的权重梯度
    pub grad_bias: Option<Array2<f32>>,    // 存储计算得到的偏置梯度
    // Adam 状态：一阶矩和二阶矩
    pub m_weights: Option<Array2<f32>>,
    pub v_weights: Option<Array2<f32>>,
    pub m_bias: Option<Array2<f32>>,
    pub v_bias: Option<Array2<f32>>,

    // RMSprop 优化器所需缓存
    pub cache_weights: Option<Array2<f32>>,
    pub cache_bias: Option<Array2<f32>>,
}

impl Dense {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // 使用均匀分布 [-0.05, 0.05] 初始化 weights，bias 初始化为 0
        let weights = Array::random((input_dim, output_dim), Uniform::new(-0.05, 0.05));
        let bias = Array::zeros((1, output_dim));
        Self {
            input_dim,
            output_dim,
            weights,
            bias,
            input_cache: None,
            grad_weights: None,
            grad_bias: None,
            m_weights: None,
            v_weights: None,
            m_bias: None,
            v_bias: None,
            cache_weights: None,
            cache_bias: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // input 的形状为 [batch_size, input_dim]
        let input_2d = input.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        // 缓存输入，便于反向传播计算梯度
        self.input_cache = Some(input_2d.clone());
        // 计算线性变换： output = input.dot(weights) + bias（利用广播加偏置）
        let output = input_2d.dot(&self.weights) + &self.bias;
        output.into_dyn()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // 将梯度转换为二维数组，形状为 [batch_size, output_dim]
        let grad_output_2d = grad_output.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        // 从缓存中取出 forward 时的输入，形状为 [batch_size, input_dim]
        let input = self.input_cache.take().expect("Forward 必须先于 backward 调用");

        // 计算对权重的梯度： dW = input.T.dot(grad_output)
        let grad_w = input.t().dot(&grad_output_2d);
        // 计算对偏置的梯度： db = sum(grad_output, axis=0)，保持二维形状
        let grad_b = grad_output_2d.sum_axis(Axis(0)).insert_axis(Axis(0));
        // 将计算得到的梯度存储起来，供优化器更新参数时使用
        self.grad_weights = Some(grad_w);
        self.grad_bias = Some(grad_b);

        // 计算对输入的梯度： dX = grad_output.dot(weights.T)
        let grad_input = grad_output_2d.dot(&self.weights.t());
        grad_input.into_dyn()
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn output_shape(&self) -> String {
        // 这里只返回 (None, output_dim)
        format!("(None, {})", self.output_dim)
    }

    fn param_count(&self) -> usize {
        // 参数数量 = weights 参数数 + bias 参数数
        self.input_dim * self.output_dim + self.output_dim
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            self.weights = &self.weights - &(lr * grad_w);
            self.bias = &self.bias - &(lr * grad_b);
        }
    }

    // 使用 Adam 算法更新参数
    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // 如果 Adam 的状态尚未初始化，则进行初始化
        if self.m_weights.is_none() {
            self.m_weights = Some(Array2::<f32>::zeros((self.input_dim, self.output_dim)));
            self.v_weights = Some(Array2::<f32>::zeros((self.input_dim, self.output_dim)));
            self.m_bias = Some(Array2::<f32>::zeros((1, self.output_dim)));
            self.v_bias = Some(Array2::<f32>::zeros((1, self.output_dim)));
        }
        let m_w = self.m_weights.as_mut().unwrap();
        let v_w = self.v_weights.as_mut().unwrap();
        let m_b = self.m_bias.as_mut().unwrap();
        let v_b = self.v_bias.as_mut().unwrap();

        let grad_w = self.grad_weights.as_ref().expect("Adam: 未计算 grad_weights");
        let grad_b = self.grad_bias.as_ref().expect("Adam: 未计算 grad_bias");

        // 更新一阶矩（动量）
        *m_w = m_w.mapv(|x| x * beta1) + &(grad_w * (1.0 - beta1));
        *m_b = m_b.mapv(|x| x * beta1) + &(grad_b * (1.0 - beta1));

        // 更新二阶矩（平方梯度累积），注意使用 elementwise square
        *v_w = v_w.mapv(|x| x * beta2) + &(grad_w.mapv(|x| x * x) * (1.0 - beta2));
        *v_b = v_b.mapv(|x| x * beta2) + &(grad_b.mapv(|x| x * x) * (1.0 - beta2));

        // 计算偏差修正后的估计值
        let m_hat_w = m_w.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
        let m_hat_b = m_b.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
        let v_hat_w = v_w.mapv(|x| x / (1.0 - beta2.powi(t as i32)));
        let v_hat_b = v_b.mapv(|x| x / (1.0 - beta2.powi(t as i32)));

        // 更新参数：w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
        self.weights = &self.weights - &(lr * &m_hat_w / &(v_hat_w.mapv(f32::sqrt) + epsilon));
        self.bias = &self.bias - &(lr * &m_hat_b / &(v_hat_b.mapv(f32::sqrt) + epsilon));
    }

    // RMSprop 更新
    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            // 若未初始化 RMSprop 缓存，则以零数组初始化
            if self.cache_weights.is_none() {
                self.cache_weights = Some(Array2::<f32>::zeros((self.input_dim, self.output_dim)));
            }
            if self.cache_bias.is_none() {
                self.cache_bias = Some(Array2::<f32>::zeros((1, self.output_dim)));
            }
            if let Some(ref mut cache_w) = self.cache_weights {
                *cache_w = cache_w.mapv(|x| x * rho) + &(grad_w.mapv(|x| x * x) * (1.0 - rho));
            }
            if let Some(ref mut cache_b) = self.cache_bias {
                *cache_b = cache_b.mapv(|x| x * rho) + &(grad_b.mapv(|x| x * x) * (1.0 - rho));
            }
            if let Some(ref cache_w) = self.cache_weights {
                self.weights = &self.weights - &(lr * grad_w / &(cache_w.mapv(f32::sqrt) + epsilon));
            }
            if let Some(ref cache_b) = self.cache_bias {
                self.bias = &self.bias - &(lr * grad_b / &(cache_b.mapv(f32::sqrt) + epsilon));
            }
        }
    }
}