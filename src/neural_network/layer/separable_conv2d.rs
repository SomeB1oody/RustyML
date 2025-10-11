use super::*;

/// A 2D separable convolutional layer for neural networks.
///
/// This layer implements depthwise separable convolution, which factors a standard convolution
/// into two separate operations: a depthwise convolution followed by a pointwise convolution.
/// This approach significantly reduces the number of parameters and computational cost compared
/// to standard convolution while maintaining similar performance.
///
/// The separable convolution consists of:
/// 1. **Depthwise Convolution**: Each input channel is convolved with its own set of filters
/// 2. **Pointwise Convolution**: A 1x1 convolution that combines the outputs from the depthwise step
///
/// # Fields
///
/// - `filters` - Number of output channels from the pointwise convolution.
/// - `kernel_size` - Size of the depthwise convolution kernel as (height, width).
/// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `depth_multiplier` - Number of depthwise convolution filters per input channel.
/// - `depthwise_weights` - 4D array for depthwise filters with shape \[depth_multiplier, channels, kernel_height, kernel_width\].
/// - `pointwise_weights` - 4D array for pointwise filters with shape \[filters, channels * depth_multiplier, 1, 1\].
/// - `bias` - 2D array of bias values with shape \[1, filters\].
/// - `activation` - Optional activation function applied after the convolution.
/// - `input_cache` - Cached input from the forward pass, used during backpropagation.
/// - `depthwise_output_cache` - Cached depthwise output, used during backpropagation.
/// - `input_shape` - Shape of the input tensor.
/// - `depthwise_weight_gradients` - Gradients for the depthwise weights.
/// - `pointwise_weight_gradients` - Gradients for the pointwise weights.
/// - `bias_gradients` - Gradients for the biases.
/// - `optimizer_cache` - Cache for optimizer-specific state.
///
/// # Shape Information
///
/// Input shape: \[batch_size, channels, height, width\]
/// Intermediate shape (after depthwise): \[batch_size, channels * depth_multiplier, height', width'\]
/// Output shape: \[batch_size, filters, height', width'\]
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
///
/// // Create a simple 4D input tensor: [batch_size, channels, height, width]
/// let x = Array4::ones((2, 3, 32, 32)).into_dyn();
///
/// // Create target tensor
/// let y = Array4::ones((2, 64, 32, 32)).into_dyn();
///
/// // Build model with separable convolution
/// let mut model = Sequential::new();
/// model
///     .add(SeparableConv2D::new(
///         64,                     // Number of output filters
///         (3, 3),                 // Kernel size
///         vec![2, 3, 32, 32],     // Input shape
///         (1, 1),                 // Stride
///         PaddingType::Same,      // Same padding
///         1,                      // Depth multiplier
///         Some(Activation::ReLU), // ReLU activation
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// model.summary();
/// model.fit(&x, &y, 3).unwrap();
/// ```
pub struct SeparableConv2D {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    depth_multiplier: usize,
    depthwise_weights: Array4<f32>,
    pointwise_weights: Array4<f32>,
    bias: Array2<f32>,
    activation: Option<Activation>,
    input_cache: Option<Tensor>,
    depthwise_output_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    depthwise_weight_gradients: Option<Array4<f32>>,
    pointwise_weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array2<f32>>,
    optimizer_cache: OptimizerCacheConv2D,
}

impl SeparableConv2D {
    /// Creates a new 2D separable convolutional layer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output channels from the pointwise convolution.
    /// - `kernel_size` - Size of the depthwise convolution kernel as (height, width).
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\].
    /// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
    /// - `padding` - Type of padding to apply (`Valid` or `Same`).
    /// - `depth_multiplier` - Number of depthwise convolution filters per input channel.
    /// - `activation` - Optional activation function to apply after the convolution.
    ///
    /// # Returns
    ///
    /// * `SeparableConv2D` - A new `SeparableConv2D` layer instance.
    ///
    /// # Notes
    ///
    /// Weights are initialized using Xavier (Glorot) uniform initialization:
    /// - Depthwise weights: Based on fan-in and fan-out for depthwise filters
    /// - Pointwise weights: Based on fan-in and fan-out for 1x1 convolution
    /// - Biases are initialized to zeros
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        depth_multiplier: usize,
        activation: Option<Activation>,
    ) -> Self {
        // Verify input is 4D: [batch_size, channels, height, width]
        assert_eq!(
            input_shape.len(),
            4,
            "Input tensor must be 4-dimensional: [batch_size, channels, height, width]"
        );

        let channels = input_shape[1];

        // Initialize depthwise weights using Xavier initialization
        // For depthwise convolution, each filter only operates on one channel
        let depthwise_fan_in = kernel_size.0 * kernel_size.1;
        let depthwise_fan_out = depth_multiplier * kernel_size.0 * kernel_size.1;
        let depthwise_bound = (6.0 / (depthwise_fan_in + depthwise_fan_out) as f32).sqrt();

        let depthwise_weights = Array4::random(
            (depth_multiplier, channels, kernel_size.0, kernel_size.1),
            Uniform::new(-depthwise_bound, depthwise_bound),
        );

        // Initialize pointwise weights using Xavier initialization
        // For pointwise convolution (1x1), the kernel area is 1
        let pointwise_fan_in = channels * depth_multiplier;
        let pointwise_fan_out = filters;
        let pointwise_bound = (6.0 / (pointwise_fan_in + pointwise_fan_out) as f32).sqrt();

        let pointwise_weights = Array4::random(
            (filters, channels * depth_multiplier, 1, 1),
            Uniform::new(-pointwise_bound, pointwise_bound),
        );

        // Initialize biases to zero
        let bias = Array2::zeros((1, filters));

        SeparableConv2D {
            filters,
            kernel_size,
            strides,
            padding,
            depth_multiplier,
            depthwise_weights,
            pointwise_weights,
            bias,
            activation,
            input_cache: None,
            depthwise_output_cache: None,
            input_shape,
            depthwise_weight_gradients: None,
            pointwise_weight_gradients: None,
            bias_gradients: None,
            optimizer_cache: OptimizerCacheConv2D {
                adam_states: None,
                rmsprop_cache: None,
            },
        }
    }

    /// Calculates the output shape of the separable convolutional layer.
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// Performs depthwise convolution operation.
    fn depthwise_convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_depthwise_output_shape(input_shape);

        // Parallel processing across batches
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_output = Array3::zeros((
                    channels * self.depth_multiplier,
                    output_shape[2],
                    output_shape[3],
                ));

                for c in 0..channels {
                    for m in 0..self.depth_multiplier {
                        let output_channel = c * self.depth_multiplier + m;

                        for i in 0..output_shape[2] {
                            let i_base = i * self.strides.0;

                            for j in 0..output_shape[3] {
                                let j_base = j * self.strides.1;
                                let mut sum = 0.0;

                                let max_ki = input_shape[2]
                                    .saturating_sub(i_base)
                                    .min(self.kernel_size.0);
                                let max_kj = input_shape[3]
                                    .saturating_sub(j_base)
                                    .min(self.kernel_size.1);

                                for ki in 0..max_ki {
                                    let i_pos = i_base + ki;
                                    for kj in 0..max_kj {
                                        let j_pos = j_base + kj;
                                        sum += input[[b, c, i_pos, j_pos]]
                                            * self.depthwise_weights[[m, c, ki, kj]];
                                    }
                                }

                                batch_output[[output_channel, i, j]] = sum;
                            }
                        }
                    }
                }

                (b, batch_output)
            })
            .collect();

        // Use merge_results function to combine batch results
        merge_results(output_shape, results, channels * self.depth_multiplier)
    }

    /// Performs pointwise convolution (1x1 convolution).
    fn pointwise_convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let output_shape = vec![batch_size, self.filters, input_shape[2], input_shape[3]];

        // Parallel processing across batches
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_output =
                    Array3::zeros((self.filters, input_shape[2], input_shape[3]));

                for f in 0..self.filters {
                    for i in 0..input_shape[2] {
                        for j in 0..input_shape[3] {
                            let mut sum = 0.0;

                            for c in 0..input_shape[1] {
                                sum += input[[b, c, i, j]] * self.pointwise_weights[[f, c, 0, 0]];
                            }

                            sum += self.bias[[0, f]];
                            batch_output[[f, i, j]] = sum;
                        }
                    }
                }

                (b, batch_output)
            })
            .collect();

        // Use merge_results function to combine batch results
        merge_results(output_shape, results, self.filters)
    }

    /// Calculates the output shape after depthwise convolution.
    fn calculate_depthwise_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![
            batch_size,
            channels * self.depth_multiplier,
            output_height,
            output_width,
        ]
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `depthwise_weights` - 4D array for depthwise filters with shape [depth_multiplier, channels, kernel_height, kernel_width]
    /// - `pointwise_weights` - 4D array for pointwise filters with shape [filters, channels * depth_multiplier, 1, 1]
    /// - `bias` - 2D array of bias values with shape [1, filters]
    pub fn set_weights(
        &mut self,
        depthwise_weights: Array4<f32>,
        pointwise_weights: Array4<f32>,
        bias: Array2<f32>,
    ) {
        self.depthwise_weights = depthwise_weights;
        self.pointwise_weights = pointwise_weights;
        self.bias = bias;
    }
}

impl Layer for SeparableConv2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Step 1: Depthwise convolution - each input channel convolved independently
        let depthwise_output = self.depthwise_convolve(input);

        // Step 2: Pointwise convolution (1x1) - combines depthwise outputs
        let mut output = self.pointwise_convolve(&depthwise_output);

        // Cache depthwise output after pointwise conv to save memory if no activation
        // (activation derivative doesn't need depthwise output, only backward does)
        self.depthwise_output_cache = Some(depthwise_output);

        // Step 3: Apply activation function
        if let Some(activation) = &self.activation {
            Activation::apply_activation_inplace(activation, &mut output);
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let (Some(input), Some(depthwise_output)) =
            (&self.input_cache, &self.depthwise_output_cache)
        {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let depthwise_shape = depthwise_output.shape();

            // Apply activation derivative if needed
            let gradient = if let Some(activation) = &self.activation {
                let mut grad = grad_output.to_owned().into_dyn();
                Activation::activation_derivative_inplace(activation, &mut grad);
                grad
            } else {
                grad_output.to_owned().into_dyn()
            };

            // Initialize gradients
            let mut pointwise_weight_grads = Array4::zeros(self.pointwise_weights.dim());
            let mut depthwise_weight_grads = Array4::zeros(self.depthwise_weights.dim());
            let mut bias_grads = Array2::zeros((1, self.filters));

            // Calculate bias gradients
            for f in 0..self.filters {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    for i in 0..grad_output.shape()[2] {
                        for j in 0..grad_output.shape()[3] {
                            sum += gradient[[b, f, i, j]];
                        }
                    }
                }
                bias_grads[[0, f]] = sum;
            }

            // Calculate pointwise weight gradients
            for f in 0..self.filters {
                for c in 0..depthwise_shape[1] {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        for i in 0..depthwise_shape[2] {
                            for j in 0..depthwise_shape[3] {
                                sum += gradient[[b, f, i, j]] * depthwise_output[[b, c, i, j]];
                            }
                        }
                    }
                    pointwise_weight_grads[[f, c, 0, 0]] = sum;
                }
            }

            // Calculate gradients w.r.t. depthwise output
            let mut depthwise_grad = ArrayD::zeros(depthwise_output.dim());
            for b in 0..batch_size {
                for c in 0..depthwise_shape[1] {
                    for i in 0..depthwise_shape[2] {
                        for j in 0..depthwise_shape[3] {
                            let mut sum = 0.0;
                            for f in 0..self.filters {
                                sum +=
                                    gradient[[b, f, i, j]] * self.pointwise_weights[[f, c, 0, 0]];
                            }
                            depthwise_grad[[b, c, i, j]] = sum;
                        }
                    }
                }
            }

            // Calculate depthwise weight gradients
            for m in 0..self.depth_multiplier {
                for c in 0..channels {
                    for h in 0..self.kernel_size.0 {
                        for w in 0..self.kernel_size.1 {
                            let mut sum = 0.0;
                            let output_channel = c * self.depth_multiplier + m;

                            // 在 depthwise weight gradients 计算中替换 j 循环
                            for b in 0..batch_size {
                                for i in 0..depthwise_shape[2] {
                                    let i_pos = i * self.strides.0 + h;
                                    if i_pos < input_shape[2] {
                                        sum += compute_row_gradient_sum(
                                            &depthwise_grad,
                                            input,
                                            b,
                                            output_channel,
                                            c,
                                            i,
                                            i_pos,
                                            w,
                                            depthwise_shape,
                                            input_shape,
                                            self.strides.1,
                                        );
                                    }
                                }
                            }
                            depthwise_weight_grads[[m, c, h, w]] = sum;
                        }
                    }
                }
            }

            // Calculate input gradients
            let mut input_gradients = ArrayD::zeros(input.dim());
            for b in 0..batch_size {
                for c in 0..channels {
                    for i in 0..input_shape[2] {
                        for j in 0..input_shape[3] {
                            let mut sum = 0.0;

                            for m in 0..self.depth_multiplier {
                                let output_channel = c * self.depth_multiplier + m;

                                for h in 0..self.kernel_size.0 {
                                    for w in 0..self.kernel_size.1 {
                                        if i >= h && j >= w {
                                            let grad_i = (i - h) / self.strides.0;
                                            let grad_j = (j - w) / self.strides.1;

                                            if grad_i < depthwise_shape[2]
                                                && grad_j < depthwise_shape[3]
                                                && (i - h) % self.strides.0 == 0
                                                && (j - w) % self.strides.1 == 0
                                            {
                                                sum += depthwise_grad
                                                    [[b, output_channel, grad_i, grad_j]]
                                                    * self.depthwise_weights[[m, c, h, w]];
                                            }
                                        }
                                    }
                                }
                            }

                            input_gradients[[b, c, i, j]] = sum;
                        }
                    }
                }
            }

            // Store gradients
            self.depthwise_weight_gradients = Some(depthwise_weight_grads);
            self.pointwise_weight_gradients = Some(pointwise_weight_grads);
            self.bias_gradients = Some(bias_grads);

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "SeparableConv2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            self.depthwise_weights.len() + self.pointwise_weights.len() + self.bias.len(),
        )
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        if let (Some(depthwise_grads), Some(pointwise_grads), Some(bias_grads)) = (
            &self.depthwise_weight_gradients,
            &self.pointwise_weight_gradients,
            &self.bias_gradients,
        ) {
            // Update depthwise weights
            if let (Some(weights_slice), Some(grads_slice)) = (
                self.depthwise_weights.as_slice_mut(),
                depthwise_grads.as_slice(),
            ) {
                weights_slice
                    .par_iter_mut()
                    .zip(grads_slice.par_iter())
                    .for_each(|(weight, &grad)| {
                        *weight -= lr * grad;
                    });
            }

            // Update pointwise weights
            if let (Some(weights_slice), Some(grads_slice)) = (
                self.pointwise_weights.as_slice_mut(),
                pointwise_grads.as_slice(),
            ) {
                weights_slice
                    .par_iter_mut()
                    .zip(grads_slice.par_iter())
                    .for_each(|(weight, &grad)| {
                        *weight -= lr * grad;
                    });
            }

            // Update biases
            if let (Some(bias_slice), Some(bias_grads_slice)) =
                (self.bias.as_slice_mut(), bias_grads.as_slice())
            {
                bias_slice
                    .par_iter_mut()
                    .zip(bias_grads_slice.par_iter())
                    .for_each(|(bias, &grad)| {
                        *bias -= lr * grad;
                    });
            }
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        if let (Some(depthwise_grads), Some(pointwise_grads), Some(bias_grads)) = (
            &self.depthwise_weight_gradients,
            &self.pointwise_weight_gradients,
            &self.bias_gradients,
        ) {
            // Initialize Adam states if needed
            if self.optimizer_cache.adam_states.is_none() {
                let total_depthwise_params = self.depthwise_weights.len();
                let total_pointwise_params = self.pointwise_weights.len();
                let total_params = total_depthwise_params + total_pointwise_params;

                self.optimizer_cache.adam_states = Some(AdamStatesConv2D {
                    m: Array4::zeros((total_params, 1, 1, 1)),
                    v: Array4::zeros((total_params, 1, 1, 1)),
                    m_bias: Array2::zeros(self.bias.dim()),
                    v_bias: Array2::zeros(self.bias.dim()),
                });
            }

            let correction1 = 1.0 - beta1.powi(t as i32);
            let correction2 = 1.0 - beta2.powi(t as i32);

            if let Some(adam_states) = &mut self.optimizer_cache.adam_states {
                let depthwise_len = self.depthwise_weights.len();
                let pointwise_len = self.pointwise_weights.len();

                // Update depthwise weights
                if let (Some(weights_slice), Some(grads_slice)) = (
                    self.depthwise_weights.as_slice_mut(),
                    depthwise_grads.as_slice(),
                ) {
                    if let (Some(m_full_slice), Some(v_full_slice)) =
                        (adam_states.m.as_slice_mut(), adam_states.v.as_slice_mut())
                    {
                        if let (Some(m_slice), Some(v_slice)) = (
                            m_full_slice.get_mut(..depthwise_len),
                            v_full_slice.get_mut(..depthwise_len),
                        ) {
                            update_adam_conv(
                                weights_slice,
                                grads_slice,
                                m_slice,
                                v_slice,
                                lr,
                                beta1,
                                beta2,
                                epsilon,
                                correction1,
                                correction2,
                            );
                        }
                    }
                }

                // Update pointwise weights
                if let (Some(weights_slice), Some(grads_slice)) = (
                    self.pointwise_weights.as_slice_mut(),
                    pointwise_grads.as_slice(),
                ) {
                    if let (Some(m_full_slice), Some(v_full_slice)) =
                        (adam_states.m.as_slice_mut(), adam_states.v.as_slice_mut())
                    {
                        if let (Some(m_slice), Some(v_slice)) = (
                            m_full_slice.get_mut(depthwise_len..depthwise_len + pointwise_len),
                            v_full_slice.get_mut(depthwise_len..depthwise_len + pointwise_len),
                        ) {
                            update_adam_conv(
                                weights_slice,
                                grads_slice,
                                m_slice,
                                v_slice,
                                lr,
                                beta1,
                                beta2,
                                epsilon,
                                correction1,
                                correction2,
                            );
                        }
                    }
                }

                // Update biases
                if let (
                    Some(bias_slice),
                    Some(bias_grads_slice),
                    Some(m_bias_slice),
                    Some(v_bias_slice),
                ) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    adam_states.m_bias.as_slice_mut(),
                    adam_states.v_bias.as_slice_mut(),
                ) {
                    update_adam_conv(
                        bias_slice,
                        bias_grads_slice,
                        m_bias_slice,
                        v_bias_slice,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        correction1,
                        correction2,
                    );
                }
            }
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(depthwise_grads), Some(pointwise_grads), Some(bias_grads)) = (
            &self.depthwise_weight_gradients,
            &self.pointwise_weight_gradients,
            &self.bias_gradients,
        ) {
            // Initialize RMSprop cache if needed
            if self.optimizer_cache.rmsprop_cache.is_none() {
                let total_depthwise_params = self.depthwise_weights.len();
                let total_pointwise_params = self.pointwise_weights.len();
                let total_params = total_depthwise_params + total_pointwise_params;

                self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheConv2D {
                    cache: Array4::zeros((total_params, 1, 1, 1)),
                    bias: Array2::zeros(self.bias.dim()),
                });
            }

            if let Some(rmsprop_cache) = &mut self.optimizer_cache.rmsprop_cache {
                let depthwise_len = self.depthwise_weights.len();
                let pointwise_len = self.pointwise_weights.len();

                // Update depthwise weights
                if let (Some(weights_slice), Some(grads_slice)) = (
                    self.depthwise_weights.as_slice_mut(),
                    depthwise_grads.as_slice(),
                ) {
                    if let Some(cache_full_slice) = rmsprop_cache.cache.as_slice_mut() {
                        if let Some(cache_slice) = cache_full_slice.get_mut(..depthwise_len) {
                            update_rmsprop(
                                weights_slice,
                                grads_slice,
                                cache_slice,
                                rho,
                                epsilon,
                                lr,
                            );
                        }
                    }
                }

                // Update pointwise weights
                if let (Some(weights_slice), Some(grads_slice)) = (
                    self.pointwise_weights.as_slice_mut(),
                    pointwise_grads.as_slice(),
                ) {
                    if let Some(cache_full_slice) = rmsprop_cache.cache.as_slice_mut() {
                        if let Some(cache_slice) =
                            cache_full_slice.get_mut(depthwise_len..depthwise_len + pointwise_len)
                        {
                            update_rmsprop(
                                weights_slice,
                                grads_slice,
                                cache_slice,
                                rho,
                                epsilon,
                                lr,
                            );
                        }
                    }
                }

                // Update biases
                if let (Some(bias_slice), Some(bias_grads_slice), Some(bias_cache_slice)) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    rmsprop_cache.bias.as_slice_mut(),
                ) {
                    update_rmsprop(
                        bias_slice,
                        bias_grads_slice,
                        bias_cache_slice,
                        rho,
                        epsilon,
                        lr,
                    );
                }
            }
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::SeparableConv2DLayer(SeparableConv2DLayerWeight {
            depthwise_weight: &self.depthwise_weights,
            pointwise_weight: &self.pointwise_weights,
            bias: &self.bias,
        })
    }
}
