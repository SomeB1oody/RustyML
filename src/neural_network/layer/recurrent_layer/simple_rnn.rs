use super::*;

/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation.
///
/// SimpleRNN applies a standard recurrent operation where the output from the previous
/// timestep is used as additional input to the current timestep. This implementation
/// supports optional activation functions from the activation_layer module.
///
/// # Dimensions
///
/// - Input shape: (batch_size, timesteps, input_dim)
/// - Output shape: (batch_size, units)
///
/// # Fields
///
/// ## Core fields
/// - `input_dim` - Number of input features
/// - `units` - Number of output units (neurons)
/// - `kernel` - Weight matrix connecting inputs to the recurrent layer (shape: input_dim, units)
/// - `recurrent_kernel` - Weight matrix connecting previous hidden states to the current state (shape: units, units)
/// - `bias` - Bias vector for the layer (shape: 1, units)
/// - `activation` - Activation layer from activation_layer module
///
/// ## Cache
/// - `input_cache` - Cache of input tensors from forward pass (shape: batch, timesteps, input_dim)
/// - `hidden_state_cache` - Cache of hidden states from forward pass (length = timesteps+1)
/// - `optimizer_cache` - Cache for optimizer
///
/// ## Gradients
/// - `grad_kernel` - Gradient of the kernel weights
/// - `grad_recurrent_kernel` - Gradient of the recurrent kernel weights
/// - `grad_bias` - Gradient of the bias
///
/// # Example
/// ```rust
/// use ndarray::Array;
/// use rustyml::prelude::*;
///
/// // Create input with batch_size=2, timesteps=5, input_dim=4,
/// // and target with batch_size=2, units=3 (same dimension as the last hidden state)
/// let x = Array::ones((2, 5, 4)).into_dyn();
/// let y = Array::ones((2, 3)).into_dyn();
///
/// // Build model: one SimpleRnn layer with Tanh activation
/// let mut model = Sequential::new();
/// model
/// .add(SimpleRNN::new(4, 3, Tanh::new()))
/// .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Print structure
/// model.summary();
///
/// // Train for 1 epoch
/// model.fit(&x, &y, 1).unwrap();
///
/// // Predict
/// let pred = model.predict(&x);
/// println!("SimpleRnn prediction:\n{:#?}\n", pred);
/// ```
pub struct SimpleRNN<T: ActivationLayer> {
    input_dim: usize,
    units: usize,
    kernel: Array2<f32>,
    recurrent_kernel: Array2<f32>,
    bias: Array2<f32>,
    input_cache: Option<Array3<f32>>,
    hidden_state_cache: Option<Vec<Array2<f32>>>,
    grad_kernel: Option<Array2<f32>>,
    grad_recurrent_kernel: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,
    optimizer_cache: OptimizerCache,
    activation: T,
}

impl<T: ActivationLayer> SimpleRNN<T> {
    /// Creates a new SimpleRNN layer with the specified dimensions and optional activation layer.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - The size of each input sample
    /// - `units` - The dimensionality of the output space
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// * `SimpleRNN` - A new SimpleRNN instance with the specified activation
    pub fn new(input_dim: usize, units: usize, activation: T) -> Self {
        // Xavier/Glorot initialization for input kernel
        let limit = (6.0_f32 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random((input_dim, units), Uniform::new(-limit, limit).unwrap());

        // Orthogonal initialization for recurrent kernel to maintain gradient flow
        let recurrent_kernel = Self::orthogonal_init(units);

        let bias = Array::zeros((1, units));
        SimpleRNN {
            input_dim,
            units,
            kernel,
            recurrent_kernel,
            bias,
            input_cache: None,
            hidden_state_cache: None,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
            optimizer_cache: OptimizerCache::default(),
            activation,
        }
    }

    /// Generate an orthogonal matrix using Gram-Schmidt orthogonalization
    /// This helps prevent gradient vanishing/exploding in RNNs
    fn orthogonal_init(size: usize) -> Array2<f32> {
        // Generate a random matrix
        let mut matrix = Array::random((size, size), Uniform::new(-1.0, 1.0).unwrap());

        // Apply Gram-Schmidt orthogonalization
        for i in 0..size {
            // Orthogonalize column i against all previous columns
            for j in 0..i {
                // Compute projection: dot(col_i, col_j) / dot(col_j, col_j)
                // Since col_j is already normalized in previous iterations, denominator is 1
                let mut projection = 0.0;
                for k in 0..size {
                    projection += matrix[[k, i]] * matrix[[k, j]];
                }

                // Subtract projection from column i
                for k in 0..size {
                    matrix[[k, i]] -= projection * matrix[[k, j]];
                }
            }

            // Normalize column i
            let mut norm: f32 = 0.0;
            for k in 0..size {
                norm += matrix[[k, i]] * matrix[[k, i]];
            }
            norm = norm.sqrt();

            const EPSILON: f32 = 1e-8;

            if norm > EPSILON {
                for k in 0..size {
                    matrix[[k, i]] /= norm;
                }
            } else {
                // If norm is too small, use standard basis vector
                for k in 0..size {
                    matrix[[k, i]] = if k == i { 1.0 } else { 0.0 };
                }
            }
        }

        matrix
    }

    /// Sets the weights for this layer.
    ///
    /// # Parameters
    ///
    /// - `kernel` - Weight matrix connecting inputs to the layer with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    pub fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) {
        self.kernel = kernel;
        self.recurrent_kernel = recurrent_kernel;
        self.bias = bias;
    }
}

impl<T: ActivationLayer> Layer for SimpleRNN<T> {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 3D".to_string(),
            ));
        }

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape=(batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut linear_outputs = Vec::with_capacity(timesteps);
        hs.push(h_prev.clone());

        // Sequential timestep processing (required for RNN)
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t); // (batch, input_dim)

            // Compute: z = x_t @ W + h_{t-1} @ U + b
            let z = x_t.dot(&self.kernel) + h_prev.dot(&self.recurrent_kernel) + &self.bias;
            linear_outputs.push(z.clone());

            // Apply activation
            let h_t = self
                .activation
                .forward(&z.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            h_prev = h_t.clone();
            hs.push(h_prev.clone());
        }
        self.hidden_state_cache = Some(hs);
        Ok(h_prev.into_dyn()) // Return hidden state of the last timestep
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let grad_h_t = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        fn take_cache<T>(cache: &mut Option<T>, error_msg: &str) -> Result<T, ModelError> {
            cache
                .take()
                .ok_or_else(|| ModelError::ProcessingError(error_msg.to_string()))
        }

        let error_msg = "Forward pass has not been run";
        let x3 = take_cache(&mut self.input_cache, error_msg)?;
        let hs = take_cache(&mut self.hidden_state_cache, error_msg)?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // Initialize or reuse existing gradients for accumulation
        let mut grad_k = self
            .grad_kernel
            .take()
            .unwrap_or_else(|| Array2::<f32>::zeros((self.input_dim, self.units)));
        let mut grad_rk = self
            .grad_recurrent_kernel
            .take()
            .unwrap_or_else(|| Array2::<f32>::zeros((self.units, self.units)));
        let mut grad_b = self
            .grad_bias
            .take()
            .unwrap_or_else(|| Array2::<f32>::zeros((1, self.units)));
        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        let mut grad_h = grad_h_t;
        // Backpropagation Through Time (BPTT)
        for t in (0..timesteps).rev() {
            let h_tm1 = &hs[t];

            // Apply activation backward pass
            let d_z = {
                let grad_h_dyn = grad_h.clone().into_dyn();
                let grad_z_dyn = self.activation.backward(&grad_h_dyn)?;
                grad_z_dyn.into_dimensionality::<ndarray::Ix2>().unwrap()
            };

            // Accumulate gradients for weights
            let x_t = x3.index_axis(Axis(1), t);
            grad_k = grad_k + &x_t.t().dot(&d_z);
            grad_rk = grad_rk + &h_tm1.t().dot(&d_z);
            grad_b = grad_b + &d_z.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Gradient w.r.t. input at timestep t
            grad_x3
                .index_axis_mut(Axis(1), t)
                .assign(&d_z.dot(&self.kernel.t()));

            // Gradient w.r.t. previous hidden state (for next iteration)
            grad_h = d_z.dot(&self.recurrent_kernel.t());
        }

        // Apply gradient clipping to prevent exploding gradients
        let clip_value = 5.0;
        grad_k.mapv_inplace(|x| x.max(-clip_value).min(clip_value));
        grad_rk.mapv_inplace(|x| x.max(-clip_value).min(clip_value));
        grad_b.mapv_inplace(|x| x.max(-clip_value).min(clip_value));

        self.grad_kernel = Some(grad_k);
        self.grad_recurrent_kernel = Some(grad_rk);
        self.grad_bias = Some(grad_b);

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "SimpleRNN"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            self.input_dim * self.units + self.units * self.units + self.units,
        )
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            rayon::join(
                || {
                    rayon::join(
                        || self.kernel = &self.kernel - &(lr * gk),
                        || self.recurrent_kernel = &self.recurrent_kernel - &(lr * grk),
                    )
                },
                || self.bias = &self.bias - &(lr * gb),
            );
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Initialize Adam states (if not already initialized)
        if self.optimizer_cache.adam_states.is_none() {
            let dims_k = (self.input_dim, self.units);
            let dims_r = (self.units, self.units);
            let dims_b = (1, self.units);

            self.optimizer_cache.adam_states = Some(AdamStates::new(dims_k, Some(dims_r), dims_b));
        }

        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            let adam_states = self.optimizer_cache.adam_states.as_mut().unwrap();
            let (w_update, rk_update, b_update) =
                adam_states.update_parameter(gk, Some(grk), gb, beta1, beta2, epsilon, t, lr);

            // Apply updates
            self.kernel = &self.kernel - &w_update;
            self.recurrent_kernel = &self.recurrent_kernel - &rk_update.unwrap();
            self.bias = &self.bias - &b_update;
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, eps: f32) {
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            // Initialize RMSprop cache if it doesn't exist
            if self.optimizer_cache.rmsprop_cache.is_none() {
                self.optimizer_cache.rmsprop_cache = Some(RMSpropCache::new(
                    (self.input_dim, self.units),
                    Some((self.units, self.units)),
                    (1, self.units),
                ));
            }

            if let Some(ref mut cache) = self.optimizer_cache.rmsprop_cache {
                cache.update_parameters(
                    &mut self.kernel,
                    Some(&mut self.recurrent_kernel),
                    &mut self.bias,
                    gk,
                    Some(grk),
                    gb,
                    rho,
                    lr,
                    eps,
                );
            }
        }
    }

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        // Initialize AdaGrad cache (if not already initialized)
        if self.optimizer_cache.ada_grad_cache.is_none() {
            let dims_k = (self.input_dim, self.units);
            let dims_r = (self.units, self.units);
            let dims_b = (1, self.units);

            self.optimizer_cache.ada_grad_cache =
                Some(AdaGradStates::new(dims_k, Some(dims_r), dims_b));
        }

        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            let ada_grad_cache = self.optimizer_cache.ada_grad_cache.as_mut().unwrap();
            let (k_update, rk_update, b_update) =
                ada_grad_cache.update_parameter(gk, Some(grk), gb, epsilon, lr);

            // Apply updates
            self.kernel = &self.kernel - &k_update;
            self.recurrent_kernel = &self.recurrent_kernel - &rk_update.unwrap();
            self.bias = &self.bias - &b_update;
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::SimpleRNN(SimpleRNNLayerWeight {
            kernel: &self.kernel,
            recurrent_kernel: &self.recurrent_kernel,
            bias: &self.bias,
        })
    }
}
