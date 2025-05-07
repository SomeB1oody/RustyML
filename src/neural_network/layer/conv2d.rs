use super::Conv2DLayerWeight;
use crate::neural_network::activation::Activation;
use crate::neural_network::layer::LayerWeight;
use crate::neural_network::optimizer::*;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::{Array2, Array3, Array4, ArrayD, Axis};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

pub enum PaddingType {
    Valid,
    Same,
}

pub struct Conv2D {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    weights: Array4<f32>,
    bias: Array2<f32>,
    activation: Option<Activation>,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array2<f32>>,
    optimizer_cache: Option<Array2<OptimizerCache>>,
}

impl Conv2D {
    /// 創建新的捲積層
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        activation: Option<Activation>,
    ) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // 形狀為 [batch_size, channels, height, width]
        let channels = input_shape[1];

        // 初始化權重
        let mut weights = Array4::zeros((filters, channels, kernel_size.0, kernel_size.1));
        for i in weights.iter_mut() {
            *i = normal.sample(&mut rng) as f32;
        }

        // 初始化偏置
        let bias = Array2::zeros((1, filters));

        Conv2D {
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            activation,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
            optimizer_cache: None,
        }
    }

    /// 計算輸出形狀
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = match self.padding {
            PaddingType::Valid => {
                let out_height = (input_height - self.kernel_size.0) / self.strides.0 + 1;
                let out_width = (input_width - self.kernel_size.1) / self.strides.1 + 1;
                (out_height, out_width)
            }
            PaddingType::Same => {
                let out_height = (input_height as f32 / self.strides.0 as f32).ceil() as usize;
                let out_width = (input_width as f32 / self.strides.1 as f32).ceil() as usize;
                (out_height, out_width)
            }
        };

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// 應用激活函數
    fn apply_activation(&self, x: &mut Tensor) {
        if let Some(activation) = &self.activation {
            match activation {
                Activation::ReLU => {
                    x.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                }
                Activation::Sigmoid => {
                    x.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
                }
                Activation::Tanh => {
                    x.par_mapv_inplace(|x| x.tanh());
                }
                Activation::Softmax => panic!("Cannot use Softmax for convolution"),
            }
        }
    }

    /// 計算激活函數的導數
    fn activation_derivative(&self, output: &Tensor) -> Tensor {
        let mut result = output.clone();

        if let Some(activation) = &self.activation {
            match activation {
                Activation::ReLU => {
                    result.par_mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
                }
                Activation::Sigmoid => {
                    result.par_mapv_inplace(|a| a * (1.0 - a));
                }
                Activation::Tanh => {
                    result.par_mapv_inplace(|a| 1.0 - a * a);
                }
                Activation::Softmax => panic!("Cannot use Softmax for convolution"),
            }
        }

        result
    }

    /// 執行捲積操作
    fn convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // 預先分配輸出數組
        let mut output = ArrayD::zeros(output_shape.clone());

        // 创建批次处理的结果向量
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                // 创建这个批次的输出部分
                let mut batch_output =
                    Array3::zeros((self.filters, output_shape[2], output_shape[3]));

                // 每個批次的計算
                for f in 0..self.filters {
                    for i in 0..output_shape[2] {
                        let i_base = i * self.strides.0;

                        for j in 0..output_shape[3] {
                            let j_base = j * self.strides.1;
                            let mut sum = 0.0;

                            // 捲積核的計算
                            // 預先檢查邊界條件
                            let max_ki = input_shape[2]
                                .saturating_sub(i_base)
                                .min(self.kernel_size.0);
                            let max_kj = input_shape[3]
                                .saturating_sub(j_base)
                                .min(self.kernel_size.1);

                            for c in 0..in_channels {
                                // 使用連續內存訪問模式
                                for ki in 0..max_ki {
                                    let i_pos = i_base + ki;

                                    for kj in 0..max_kj {
                                        let j_pos = j_base + kj;
                                        sum += input[[b, c, i_pos, j_pos]]
                                            * self.weights[[f, c, ki, kj]];
                                    }
                                }
                            }

                            // 更新批次输出
                            sum += self.bias[[0, f]];
                            batch_output[[f, i, j]] = sum;
                        }
                    }
                }

                (b, batch_output)
            })
            .collect();

        // 將每個批次的結果合併到最終輸出
        for (b, batch_output) in results {
            for f in 0..self.filters {
                for i in 0..output_shape[2] {
                    for j in 0..output_shape[3] {
                        output[[b, f, i, j]] = batch_output[[f, i, j]];
                    }
                }
            }
        }

        output
    }

    // 初始化优化器缓存
    fn init_optimizer_cache(&mut self) {
        if self.optimizer_cache.is_none() {
            // 创建空的优化器缓存二维数组
            let mut cache_array = Array2::default((2, 1));

            // 为每个位置设置 OptimizerCache 结构
            for i in 0..2 {
                cache_array[[i, 0]] = OptimizerCache {
                    adam_states: None,
                    rmsprop_cache: None,
                };
            }

            self.optimizer_cache = Some(cache_array);
        }
    }

    // 获取扁平化的权重和梯度
    fn get_flattened_params(
        &self,
        weight_grads: &Array4<f32>,
        bias_grads: &Array2<f32>,
    ) -> ((Array2<f32>, Array2<f32>), (Array2<f32>, Array2<f32>)) {
        // 并行扁平化权重和梯度
        let (weights_flat, weight_grads_flat) = rayon::join(
            || {
                Array2::from_shape_vec(
                    (self.weights.len(), 1),
                    self.weights.par_iter().cloned().collect(),
                )
                .unwrap()
            },
            || {
                Array2::from_shape_vec(
                    (self.weights.len(), 1),
                    weight_grads.par_iter().cloned().collect(),
                )
                .unwrap()
            },
        );

        // 并行扁平化偏置和偏置梯度
        let (bias_flat, bias_grads_flat) = rayon::join(
            || {
                Array2::from_shape_vec(
                    (self.bias.len(), 1),
                    self.bias.par_iter().cloned().collect(),
                )
                .unwrap()
            },
            || {
                Array2::from_shape_vec(
                    (self.bias.len(), 1),
                    bias_grads.par_iter().cloned().collect(),
                )
                .unwrap()
            },
        );

        (
            (weights_flat, weight_grads_flat),
            (bias_flat, bias_grads_flat),
        )
    }

    // 获取权重和偏置的缓存引用
    fn get_cache_refs(&mut self) -> (&mut OptimizerCache, &mut OptimizerCache) {
        let optimizer_cache = self.optimizer_cache.as_mut().unwrap();

        // 使用 unsafe 代码块获取不同的可变引用
        unsafe {
            let optimizer_cache_ptr = optimizer_cache.as_ptr() as *mut OptimizerCache;

            (
                &mut *optimizer_cache_ptr,        // 获取 weight_cache 指针
                &mut *optimizer_cache_ptr.add(1), // 获取 bias_cache 指针
            )
        }
    }

    // 更新原始权重和偏置数组
    fn update_original_params(&mut self, weights_flat: &Array2<f32>, bias_flat: &Array2<f32>) {
        // 将更新后的权重拷贝回原始权重数组
        for (i, &val) in weights_flat.iter().enumerate() {
            self.weights.as_slice_mut().unwrap()[i] = val;
        }

        // 将更新后的偏置拷贝回原始偏置数组
        for (i, &val) in bias_flat.iter().enumerate() {
            self.bias.as_slice_mut().unwrap()[i] = val;
        }
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // 保存輸入以便反向傳播
        self.input_cache = Some(input.clone());

        // 執行捲積操作
        let mut output = self.convolve(input);

        // 應用激活函數
        self.apply_activation(&mut output);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let grad_shape = grad_output.shape();

            // 計算激活函數的導數
            let activation_grad = self.activation_derivative(grad_output);

            // 使用 rayon 並行處理元素乘法
            let mut gradient = activation_grad.clone();
            if let (Some(grad_slice), Some(act_slice), Some(out_slice)) = (
                gradient.as_slice_mut(),
                activation_grad.as_slice(),
                grad_output.as_slice(),
            ) {
                grad_slice
                    .par_iter_mut()
                    .zip(act_slice.par_iter().zip(out_slice.par_iter()))
                    .for_each(|(g, (a, o))| {
                        *g = a * o;
                    });
            } else {
                // 回退到循環實現
                for (i, v) in activation_grad.iter().enumerate() {
                    gradient.as_slice_mut().unwrap()[i] = v * grad_output.as_slice().unwrap()[i];
                }
            }

            // 初始化權重和偏置的梯度
            let mut weight_grads = Array4::zeros(self.weights.dim());
            let mut bias_grads = Array2::zeros((1, self.filters));

            // 並行計算偏置梯度
            bias_grads
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut bias)| {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        for i in 0..grad_shape[2] {
                            for j in 0..grad_shape[3] {
                                sum += gradient[[b, f, i, j]];
                            }
                        }
                    }
                    *bias.first_mut().unwrap() = sum;
                });

            // 使用並行計算優化權重梯度計算
            weight_grads
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut filter_grad)| {
                    // 對每個過濾器並行處理
                    for c in 0..channels {
                        for h in 0..self.kernel_size.0 {
                            for w in 0..self.kernel_size.1 {
                                let mut sum = 0.0;
                                // 預先檢查邊界條件，減少條件檢查次數
                                for b in 0..batch_size {
                                    for i in 0..grad_shape[2] {
                                        let i_pos = i * self.strides.0 + h;
                                        if i_pos >= input_shape[2] {
                                            continue;
                                        }

                                        for j in 0..grad_shape[3] {
                                            let j_pos = j * self.strides.1 + w;
                                            if j_pos < input_shape[3] {
                                                sum += gradient[[b, f, i, j]]
                                                    * input[[b, c, i_pos, j_pos]];
                                            }
                                        }
                                    }
                                }
                                filter_grad[[c, h, w]] = sum;
                            }
                        }
                    }
                });

            // 保存梯度以便優化
            self.weight_gradients = Some(weight_grads);
            self.bias_gradients = Some(bias_grads);

            // 並行計算輸入梯度
            let mut input_gradients = ArrayD::zeros(input.dim());

            // 使用分批並行處理，並收集結果
            let local_results: Vec<_> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    // 為每個批次創建局部梯度
                    let mut local_gradients =
                        Array3::zeros([channels, input_shape[2], input_shape[3]]);

                    for c in 0..channels {
                        for i in 0..input_shape[2] {
                            for j in 0..input_shape[3] {
                                let mut sum = 0.0;

                                for f in 0..self.filters {
                                    for h in 0..self.kernel_size.0 {
                                        for w in 0..self.kernel_size.1 {
                                            // 檢查索引是否有效
                                            if i >= h && j >= w {
                                                let grad_i = (i - h) / self.strides.0;
                                                let grad_j = (j - w) / self.strides.1;

                                                // 檢查計算出的梯度位置是否有效
                                                if grad_i < grad_shape[2]
                                                    && grad_j < grad_shape[3]
                                                    && (i - h) % self.strides.0 == 0
                                                    && (j - w) % self.strides.1 == 0
                                                {
                                                    sum += gradient[[b, f, grad_i, grad_j]]
                                                        * self.weights[[f, c, h, w]];
                                                }
                                            }
                                        }
                                    }
                                }

                                local_gradients[[c, i, j]] = sum;
                            }
                        }
                    }

                    (b, local_gradients)
                })
                .collect();

            // 在主線程中合併結果
            for (b, local_gradients) in local_results {
                for c in 0..channels {
                    for i in 0..input_shape[2] {
                        for j in 0..input_shape[3] {
                            input_gradients[[b, c, i, j]] = local_gradients[[c, i, j]];
                        }
                    }
                }
            }

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Conv2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> usize {
        let weight_count = self.weights.len();
        let bias_count = self.bias.len();
        weight_count + bias_count
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // 更新權重
            for (w, wg) in self.weights.iter_mut().zip(weight_grads.iter()) {
                *w -= lr * wg;
            }

            // 更新偏置
            for (b, bg) in self.bias.iter_mut().zip(bias_grads.iter()) {
                *b -= lr * bg;
            }
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // 初始化 Adam 状态（如果尚未初始化）
            if self.optimizer_cache.is_none() {
                // 创建空的优化器缓存二维数组
                let mut cache_array = Array2::default((2, 1));

                // 为每个位置设置 OptimizerCache 结构
                for i in 0..2 {
                    cache_array[[i, 0]] = OptimizerCache {
                        adam_states: None,
                        rmsprop_cache: None,
                    };
                }

                self.optimizer_cache = Some(cache_array);
            }

            // 并行扁平化权重和梯度
            let (mut weights_flat, weight_grads_flat) = rayon::join(
                || {
                    Array2::from_shape_vec(
                        (self.weights.len(), 1),
                        self.weights.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
                || {
                    Array2::from_shape_vec(
                        (self.weights.len(), 1),
                        weight_grads.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
            );

            // 并行扁平化偏置和偏置梯度
            let (mut bias_flat, bias_grads_flat) = rayon::join(
                || {
                    Array2::from_shape_vec(
                        (self.bias.len(), 1),
                        self.bias.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
                || {
                    Array2::from_shape_vec(
                        (self.bias.len(), 1),
                        bias_grads.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
            );

            let optimizer_cache = self.optimizer_cache.as_mut().unwrap();

            // 使用 unsafe 代码块获取不同的可变引用
            let (bias_cache, weight_cache) = unsafe {
                let optimizer_cache_ptr = optimizer_cache.as_ptr() as *mut OptimizerCache;

                (
                    &mut *optimizer_cache_ptr.add(1), // 获取 bias_cache 指针
                    &mut *optimizer_cache_ptr,        // 获取 weight_cache 指针
                )
            };

            // 并行处理权重和偏置更新
            rayon::join(
                || {
                    // 初始化权重的 Adam 状态（如果尚未初始化）
                    if weight_cache.adam_states.is_none() {
                        let dims_w = (self.weights.len(), 1); // 扁平化权重维度
                        let dims_b = (0, 0); // 占位符，不会使用

                        weight_cache.adam_states = Some(AdamStates::new(
                            dims_w, None, // 无递归权重
                            dims_b,
                        ));
                    }

                    // 使用 AdamStates 更新参数
                    if let Some(ref mut adam_states) = weight_cache.adam_states {
                        let (w_update, _, _) = adam_states.update_parameter(
                            &weight_grads_flat,
                            None,                   // 无递归梯度
                            &Array2::zeros((0, 0)), // 占位符
                            beta1,
                            beta2,
                            epsilon,
                            t,
                            lr,
                        );

                        // 应用更新到扁平化权重
                        weights_flat = weights_flat - w_update;

                        // 将更新后的权重拷贝回原始权重数组
                        for (i, &val) in weights_flat.iter().enumerate() {
                            self.weights.as_slice_mut().unwrap()[i] = val;
                        }
                    }
                },
                || {
                    // 初始化偏置的 Adam 状态（如果尚未初始化）
                    if bias_cache.adam_states.is_none() {
                        let dims_w = (0, 0); // 占位符，不会使用
                        let dims_b = (self.bias.len(), 1); // 扁平化偏置维度

                        bias_cache.adam_states = Some(AdamStates::new(
                            dims_w, None, // 无递归权重
                            dims_b,
                        ));
                    }

                    // 使用 AdamStates 更新参数
                    if let Some(ref mut adam_states) = bias_cache.adam_states {
                        let (_, _, b_update) = adam_states.update_parameter(
                            &Array2::zeros((0, 0)), // 占位符
                            None,                   // 无递归梯度
                            &bias_grads_flat,
                            beta1,
                            beta2,
                            epsilon,
                            t,
                            lr,
                        );

                        // 应用更新到扁平化偏置
                        bias_flat = bias_flat - b_update;

                        // 将更新后的偏置拷贝回原始偏置数组
                        for (i, &val) in bias_flat.iter().enumerate() {
                            self.bias.as_slice_mut().unwrap()[i] = val;
                        }
                    }
                },
            );
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // 初始化或获取优化器缓存
            if self.optimizer_cache.is_none() {
                // 创建空的优化器缓存二维数组
                let mut cache_array = Array2::default((2, 1));

                // 为每个位置设置 OptimizerCache 结构
                for i in 0..2 {
                    cache_array[[i, 0]] = OptimizerCache {
                        adam_states: None,
                        rmsprop_cache: None,
                    };
                }

                self.optimizer_cache = Some(cache_array);
            }

            // 并行扁平化权重和梯度
            let (mut weights_flat, weight_grads_flat) = rayon::join(
                || {
                    Array2::from_shape_vec(
                        (self.weights.len(), 1),
                        self.weights.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
                || {
                    Array2::from_shape_vec(
                        (self.weights.len(), 1),
                        weight_grads.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
            );

            // 并行扁平化偏置和偏置梯度
            let (mut bias_flat, bias_grads_flat) = rayon::join(
                || {
                    Array2::from_shape_vec(
                        (self.bias.len(), 1),
                        self.bias.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
                || {
                    Array2::from_shape_vec(
                        (self.bias.len(), 1),
                        bias_grads.par_iter().cloned().collect(),
                    )
                    .unwrap()
                },
            );

            let optimizer_cache = self.optimizer_cache.as_mut().unwrap();

            // 使用 unsafe 获取不同元素的可变引用
            let (bias_cache, weight_cache) = unsafe {
                let optimizer_cache_ptr = optimizer_cache.as_ptr() as *mut OptimizerCache;
                (
                    &mut *optimizer_cache_ptr.add(1), // 获取 bias_cache 指针
                    &mut *optimizer_cache_ptr,        // 获取 weight_cache 指针
                )
            };

            // 然后并行处理
            rayon::join(
                || {
                    // 处理权重的RMSprop缓存
                    if weight_cache.rmsprop_cache.is_none() {
                        weight_cache.rmsprop_cache = Some(RMSpropCache::new(
                            (self.weights.len(), 1), // 权重维度
                            None,                    // 无递归权重
                            (1, 1),                  // 占位符
                        ));
                    }

                    // 使用RMSpropCache更新参数 (权重部分)
                    if let Some(ref mut rmsprop_cache) = weight_cache.rmsprop_cache {
                        rmsprop_cache.update_parameters(
                            &mut weights_flat,          // 主权重参数
                            None,                       // 无递归权重
                            &mut Array2::zeros((1, 1)), // 占位符，不会使用
                            &weight_grads_flat,         // 主权重梯度
                            None,                       // 无递归梯度
                            &Array2::zeros((1, 1)),     // 占位符，不会使用
                            rho,
                            lr,
                            epsilon,
                        );

                        // 将更新后的权重拷贝回原始权重数组
                        for (i, &val) in weights_flat.iter().enumerate() {
                            self.weights.as_slice_mut().unwrap()[i] = val;
                        }
                    }
                },
                || {
                    // 处理偏置的RMSprop缓存
                    if bias_cache.rmsprop_cache.is_none() {
                        bias_cache.rmsprop_cache = Some(RMSpropCache::new(
                            (1, 1),               // 占位符
                            None,                 // 无递归权重
                            (self.bias.len(), 1), // 偏置维度
                        ));
                    }

                    if let Some(ref mut rmsprop_cache) = bias_cache.rmsprop_cache {
                        rmsprop_cache.update_parameters(
                            &mut Array2::zeros((1, 1)), // 占位符，不会使用
                            None,                       // 无递归权重
                            &mut bias_flat,             // 偏置参数
                            &Array2::zeros((1, 1)),     // 占位符，不会使用
                            None,                       // 无递归梯度
                            &bias_grads_flat,           // 偏置梯度
                            rho,
                            lr,
                            epsilon,
                        );

                        // 将更新后的偏置拷贝回原始偏置数组
                        for (i, &val) in bias_flat.iter().enumerate() {
                            self.bias.as_slice_mut().unwrap()[i] = val;
                        }
                    }
                },
            );
        }
    }

    fn get_weights(&self) -> LayerWeight {
        LayerWeight::Conv2D(Conv2DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
