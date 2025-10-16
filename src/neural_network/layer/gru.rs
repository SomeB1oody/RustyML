use super::*;

/// Threshold for using parallel computation in GRU layer.
/// When batch_size * units < this value, sequential execution is used.
/// When batch_size * units >= this value, parallel execution is used.
///
/// Value is chosen based on empirical benchmarks where rayon's thread pool
/// overhead is amortized by computational gains from parallelization.
const GRU_PARALLEL_THRESHOLD: usize = 1024;

/// Internal gate structure for GRU cell operations
///
/// This structure represents a single gate (reset or update) within a GRU cell.
/// Each gate contains the weights, biases, gradients, and optimizer cache needed for
/// forward and backward propagation through the gate.
///
/// # Fields
///
/// - `kernel` - Weight matrix for input connections with shape (input_dim, units)
/// - `recurrent_kernel` - Weight matrix for recurrent connections with shape (units, units)
/// - `bias` - Bias vector with shape (1, units)
/// - `grad_kernel` - Optional gradient for input weights, accumulated during backpropagation
/// - `grad_recurrent_kernel` - Optional gradient for recurrent weights, accumulated during backpropagation
/// - `grad_bias` - Optional gradient for bias terms, accumulated during backpropagation
/// - `optimizer_cache` - Cache for storing optimizer-specific state (Adam, RMSprop momentum, etc.)
struct Gate {
    kernel: Array2<f32>,
    recurrent_kernel: Array2<f32>,
    bias: Array2<f32>,
    grad_kernel: Option<Array2<f32>>,
    grad_recurrent_kernel: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,
    optimizer_cache: OptimizerCache,
}

impl Gate {
    /// Creates a new gate with randomly initialized weights
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of the input features
    /// - `units` - Number of units (neurons) in this gate
    ///
    /// # Returns
    ///
    /// * `Gate` - A new `Gate` instance with:
    ///     - Xavier/Glorot initialization for kernel weights
    ///     - Orthogonal initialization for recurrent_kernel weights
    ///     - Zero initialization for bias
    ///     - None gradients (will be allocated during first backward pass)
    ///     - Default optimizer cache
    pub fn new(input_dim: usize, units: usize) -> Self {
        // Xavier/Glorot initialization for input kernel
        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random((input_dim, units), Uniform::new(-limit, limit));

        // Orthogonal initialization for recurrent kernel
        let mut recurrent_kernel = Array::random((units, units), Uniform::new(-1.0, 1.0));
        if units > 0 {
            // Simplified orthogonalization using QR decomposition approximation
            // For better numerical stability, normalize each column
            for mut col in recurrent_kernel.columns_mut() {
                let norm = col.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    col /= norm;
                }
            }
        }

        // Zero initialization for bias
        let bias = Array::zeros((1, units));

        Self {
            kernel,
            recurrent_kernel,
            bias,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
            optimizer_cache: OptimizerCache::default(),
        }
    }
}

/// Gated Recurrent Unit (GRU) neural network layer
///
/// GRU is a type of recurrent neural network (RNN) architecture that is capable of learning
/// long-term dependencies in sequential data. It addresses the vanishing gradient problem
/// of traditional RNNs through the use of gating mechanisms, similar to LSTM but with a
/// simpler structure using only two gates instead of three.
///
/// The GRU cell uses two gates to control information flow:
/// - Reset Gate: Controls how much of the previous hidden state should be forgotten
/// - Update Gate: Controls how much of the new hidden state is just a copy of the old one
/// - Candidate Hidden State: Proposes new hidden state values
///
/// # Mathematical Operations
///
/// For each timestep t:
/// 1. r_t = σ(W_r · \[h_{t-1}, x_t\] + b_r)  (Reset gate)
/// 2. z_t = σ(W_z · \[h_{t-1}, x_t\] + b_z)  (Update gate)
/// 3. h̃_t = tanh(W_h · \[r_t ⊙ h_{t-1}, x_t\] + b_h)  (Candidate hidden state)
/// 4. h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  (Hidden state update)
///
/// Where σ is the sigmoid function, ⊙ is element-wise multiplication, and W, b are learned parameters.
///
/// # Fields
///
/// - `input_dim` - Dimensionality of input features
/// - `units` - Number of GRU units (neurons) in the layer
/// - `reset_gate` - Gate controlling what information to forget from previous hidden state
/// - `update_gate` - Gate controlling how much to update the hidden state
/// - `candidate_gate` - Gate proposing new candidate values for hidden state
/// - `input_cache` - Cached input tensor for backward propagation
/// - `hidden_cache` - Cached hidden states h_t for each timestep
/// - `r_cache` - Cached reset gate activations for each timestep
/// - `z_cache` - Cached update gate activations for each timestep
/// - `h_candidate_cache` - Cached candidate hidden states for each timestep
/// - `rh_cache` - Cached r_t ⊙ h_{t-1} values for each timestep
/// - `activation` - Activation function applied to final output (usually Tanh)
///
/// # Example
/// ```rust
/// use rustyml::neural_network::*;
/// use ndarray::Array;
///
/// // Create input data: batch_size=2, timesteps=5, features=4
/// let input = Array::ones((2, 5, 4)).into_dyn();
/// let target = Array::ones((2, 3)).into_dyn(); // batch_size=2, units=3
///
/// // Create GRU layer with 4 input features, 3 units, Tanh activation
/// let mut model = Sequential::new();
/// model.add(GRU::new(4, 3, Activation::Tanh))
///      .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input);
/// println!("GRU output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
pub struct GRU {
    input_dim: usize,
    units: usize,

    // Three gates: reset, update, candidate
    reset_gate: Gate,
    update_gate: Gate,
    candidate_gate: Gate,

    // Caches for forward pass
    input_cache: Option<Array3<f32>>,
    hidden_cache: Option<Vec<Array2<f32>>>, // hidden states h_t

    // Intermediate gate values cache
    r_cache: Option<Vec<Array2<f32>>>, // reset gate values (sigmoid applied)
    z_cache: Option<Vec<Array2<f32>>>, // update gate values (sigmoid applied)
    h_candidate_cache: Option<Vec<Array2<f32>>>, // candidate hidden state values (tanh applied)
    rh_cache: Option<Vec<Array2<f32>>>, // r_t * h_{t-1}

    activation: Activation,
}

impl GRU {
    /// Creates a new GRU layer with specified parameters
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of GRU units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation function applied to the final output (commonly Tanh or Linear)
    ///
    /// # Returns
    ///
    /// * `GRU` - A new `GRU` instance with:
    ///     - Three gates (reset, update, candidate) initialized with random weights
    ///     - All caches set to None (will be allocated during first forward pass)
    ///     - Specified activation function for output transformation
    pub fn new(input_dim: usize, units: usize, activation: Activation) -> Self {
        Self {
            input_dim,
            units,
            reset_gate: Gate::new(input_dim, units),
            update_gate: Gate::new(input_dim, units),
            candidate_gate: Gate::new(input_dim, units),
            input_cache: None,
            hidden_cache: None,
            r_cache: None,
            z_cache: None,
            h_candidate_cache: None,
            rh_cache: None,
            activation,
        }
    }

    /// Sets the weights for all three gates in this GRU layer.
    ///
    /// # Parameters
    ///
    /// Each gate requires three arrays:
    /// - `kernel` - Weight matrix connecting inputs with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    ///
    /// The parameters are provided for each of the three gates in order:
    /// reset gate, update gate, candidate gate
    pub fn set_weights(
        &mut self,
        reset_kernel: Array2<f32>,
        reset_recurrent_kernel: Array2<f32>,
        reset_bias: Array2<f32>,
        update_kernel: Array2<f32>,
        update_recurrent_kernel: Array2<f32>,
        update_bias: Array2<f32>,
        candidate_kernel: Array2<f32>,
        candidate_recurrent_kernel: Array2<f32>,
        candidate_bias: Array2<f32>,
    ) {
        self.reset_gate.kernel = reset_kernel;
        self.reset_gate.recurrent_kernel = reset_recurrent_kernel;
        self.reset_gate.bias = reset_bias;

        self.update_gate.kernel = update_kernel;
        self.update_gate.recurrent_kernel = update_recurrent_kernel;
        self.update_gate.bias = update_bias;

        self.candidate_gate.kernel = candidate_kernel;
        self.candidate_gate.recurrent_kernel = candidate_recurrent_kernel;
        self.candidate_gate.bias = candidate_bias;
    }
}

impl Layer for GRU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Input shape: (batch, timesteps, input_dim)
        let x3 = input.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.clone());

        // Initialize hidden state
        let mut h_prev = Array2::<f32>::zeros((batch, self.units));

        // Storage for all timesteps
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut r_vals = Vec::with_capacity(timesteps);
        let mut z_vals = Vec::with_capacity(timesteps);
        let mut h_candidate_vals = Vec::with_capacity(timesteps);
        let mut rh_vals = Vec::with_capacity(timesteps);

        hs.push(h_prev.clone());

        // Compute gate value: x_t @ kernel + h_prev @ recurrent_kernel + bias
        fn compute(gate: &Gate, x_t: &Array2<f32>, h_prev: &Array2<f32>) -> Array2<f32> {
            x_t.dot(&gate.kernel) + h_prev.dot(&gate.recurrent_kernel) + &gate.bias
        }

        // Determine whether to use parallel execution based on computational load
        let use_parallel = batch * self.units >= GRU_PARALLEL_THRESHOLD;

        // Process each timestep
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned(); // (batch, input_dim)

            // Compute reset and update gate values (parallel or sequential)
            let (r_raw, z_raw) = if use_parallel {
                rayon::join(
                    || compute(&self.reset_gate, &x_t, &h_prev),
                    || compute(&self.update_gate, &x_t, &h_prev),
                )
            } else {
                (
                    compute(&self.reset_gate, &x_t, &h_prev),
                    compute(&self.update_gate, &x_t, &h_prev),
                )
            };

            // Apply sigmoid activation to gates (parallel or sequential)
            let (r_t, z_t) = if use_parallel {
                rayon::join(
                    || Activation::apply_activation(&r_raw, &Activation::Sigmoid),
                    || Activation::apply_activation(&z_raw, &Activation::Sigmoid),
                )
            } else {
                (
                    Activation::apply_activation(&r_raw, &Activation::Sigmoid),
                    Activation::apply_activation(&z_raw, &Activation::Sigmoid),
                )
            };

            // Compute r_t * h_{t-1}
            let r_h = &r_t * &h_prev;

            // Compute candidate hidden state: tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)
            let h_candidate_raw = x_t.dot(&self.candidate_gate.kernel)
                + r_h.dot(&self.candidate_gate.recurrent_kernel)
                + &self.candidate_gate.bias;
            let h_candidate = Activation::apply_activation(&h_candidate_raw, &Activation::Tanh);

            // Update hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
            let h_t = &(1.0 - &z_t) * &h_prev + &z_t * &h_candidate;

            // Apply output activation if not Linear
            let h_t = if self.activation != Activation::Linear {
                Activation::apply_activation(&h_t, &self.activation)
            } else {
                h_t
            };

            // Cache values
            r_vals.push(r_t);
            z_vals.push(z_t);
            h_candidate_vals.push(h_candidate);
            rh_vals.push(r_h);
            hs.push(h_t.clone());

            h_prev = h_t;
        }

        // Store caches
        self.hidden_cache = Some(hs);
        self.r_cache = Some(r_vals);
        self.z_cache = Some(z_vals);
        self.h_candidate_cache = Some(h_candidate_vals);
        self.rh_cache = Some(rh_vals);

        // Return last hidden state
        h_prev.into_dyn()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let grad_h_t = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        // Helper function to extract cache
        fn take_cache<T>(cache: &mut Option<T>, error_msg: &str) -> Result<T, ModelError> {
            cache
                .take()
                .ok_or_else(|| ModelError::ProcessingError(error_msg.to_string()))
        }

        let error_msg = "Forward pass has not been run";
        let x3 = take_cache(&mut self.input_cache, error_msg)?;
        let hs = take_cache(&mut self.hidden_cache, error_msg)?;
        let r_vals = take_cache(&mut self.r_cache, error_msg)?;
        let z_vals = take_cache(&mut self.z_cache, error_msg)?;
        let h_candidate_vals = take_cache(&mut self.h_candidate_cache, error_msg)?;
        let rh_vals = take_cache(&mut self.rh_cache, error_msg)?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // Initialize gradient accumulators
        let mut grad_r_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_r_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_r_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_z_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_z_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_z_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_h_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_h_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_h_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        let mut grad_h = grad_h_t;

        // Determine whether to use parallel execution based on computational load
        let use_parallel = batch * self.units >= GRU_PARALLEL_THRESHOLD;

        // Backpropagation through time
        for t in (0..timesteps).rev() {
            let h_t = &hs[t + 1];
            let h_prev = &hs[t];
            let r_t = &r_vals[t];
            let z_t = &z_vals[t];
            let h_candidate = &h_candidate_vals[t];
            let r_h = &rh_vals[t];

            // Gradient through output activation
            let grad_h_pre_activation = if self.activation != Activation::Linear {
                if self.activation == Activation::Softmax {
                    Activation::softmax_backward(h_t, &grad_h)
                } else {
                    let d_act = Activation::activation_derivative(h_t, &self.activation);
                    d_act * &grad_h
                }
            } else {
                grad_h.clone()
            };

            // Gradient through h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
            let grad_z_t = &grad_h_pre_activation * (h_candidate - h_prev);
            let grad_h_candidate = &grad_h_pre_activation * z_t;
            let grad_h_prev_from_update = &grad_h_pre_activation * &(1.0 - z_t);

            // Gradient through h̃_t = tanh(...)
            let grad_h_candidate_raw = &grad_h_candidate * &(1.0 - h_candidate * h_candidate); // tanh derivative

            // Gradient through candidate gate computation
            let x_t = x3.index_axis(Axis(1), t).to_owned();
            let x_t_t = x_t.t();
            let h_prev_t = h_prev.t();
            let rh_t = r_h.t();

            grad_h_kernel = grad_h_kernel + &x_t_t.dot(&grad_h_candidate_raw);
            grad_h_recurrent = grad_h_recurrent + &rh_t.dot(&grad_h_candidate_raw);
            grad_h_bias =
                grad_h_bias + &grad_h_candidate_raw.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Gradient through r_h = r_t * h_{t-1}
            let grad_r_t =
                grad_h_candidate_raw.dot(&self.candidate_gate.recurrent_kernel.t()) * h_prev;
            let grad_h_prev_from_reset =
                grad_h_candidate_raw.dot(&self.candidate_gate.recurrent_kernel.t()) * r_t;

            // Gradient through gates (parallel or sequential)
            let (grad_z_raw, grad_r_raw) = if use_parallel {
                rayon::join(
                    || &grad_z_t * z_t * &(1.0 - z_t), // sigmoid derivative
                    || &grad_r_t * r_t * &(1.0 - r_t), // sigmoid derivative
                )
            } else {
                (
                    &grad_z_t * z_t * &(1.0 - z_t), // sigmoid derivative
                    &grad_r_t * r_t * &(1.0 - r_t), // sigmoid derivative
                )
            };

            // Compute gradient updates for gates
            grad_z_kernel = grad_z_kernel + &x_t_t.dot(&grad_z_raw);
            grad_z_recurrent = grad_z_recurrent + &h_prev_t.dot(&grad_z_raw);
            grad_z_bias = grad_z_bias + &grad_z_raw.sum_axis(Axis(0)).insert_axis(Axis(0));

            grad_r_kernel = grad_r_kernel + &x_t_t.dot(&grad_r_raw);
            grad_r_recurrent = grad_r_recurrent + &h_prev_t.dot(&grad_r_raw);
            grad_r_bias = grad_r_bias + &grad_r_raw.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Compute gradient with respect to input (parallel or sequential)
            let dx = if use_parallel {
                let (dx1, dx2) = rayon::join(
                    || grad_r_raw.dot(&self.reset_gate.kernel.t()),
                    || {
                        rayon::join(
                            || grad_z_raw.dot(&self.update_gate.kernel.t()),
                            || grad_h_candidate_raw.dot(&self.candidate_gate.kernel.t()),
                        )
                    },
                );
                dx1 + dx2.0 + dx2.1
            } else {
                grad_r_raw.dot(&self.reset_gate.kernel.t())
                    + grad_z_raw.dot(&self.update_gate.kernel.t())
                    + grad_h_candidate_raw.dot(&self.candidate_gate.kernel.t())
            };

            // Compute gradient with respect to previous hidden state (parallel or sequential)
            let grad_h_next = if use_parallel {
                let (dh1, dh2) = rayon::join(
                    || grad_r_raw.dot(&self.reset_gate.recurrent_kernel.t()),
                    || {
                        rayon::join(
                            || grad_z_raw.dot(&self.update_gate.recurrent_kernel.t()),
                            || grad_h_prev_from_reset + &grad_h_prev_from_update,
                        )
                    },
                );
                dh1 + dh2.0 + dh2.1
            } else {
                grad_r_raw.dot(&self.reset_gate.recurrent_kernel.t())
                    + grad_z_raw.dot(&self.update_gate.recurrent_kernel.t())
                    + grad_h_prev_from_reset
                    + grad_h_prev_from_update
            };

            grad_x3.index_axis_mut(Axis(1), t).assign(&dx);
            grad_h = grad_h_next;
        }

        // Store gradients using closure
        let store_gradients = |gate: &mut Gate, grad_kernel, grad_recurrent, grad_bias| {
            gate.grad_kernel = Some(grad_kernel);
            gate.grad_recurrent_kernel = Some(grad_recurrent);
            gate.grad_bias = Some(grad_bias);
        };

        store_gradients(
            &mut self.reset_gate,
            grad_r_kernel,
            grad_r_recurrent,
            grad_r_bias,
        );
        store_gradients(
            &mut self.update_gate,
            grad_z_kernel,
            grad_z_recurrent,
            grad_z_bias,
        );
        store_gradients(
            &mut self.candidate_gate,
            grad_h_kernel,
            grad_h_recurrent,
            grad_h_bias,
        );

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "GRU"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            3 * (self.input_dim * self.units + self.units * self.units + self.units),
        )
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        // Helper to update a single gate's parameters
        fn update_gate(gate: &mut Gate, lr: f32) {
            if let (Some(gk), Some(grk), Some(gb)) = (
                &gate.grad_kernel,
                &gate.grad_recurrent_kernel,
                &gate.grad_bias,
            ) {
                // Apply gradient clipping to prevent exploding gradients
                const CLIP_VALUE: f32 = 5.0;
                let gk_clipped = gk.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));
                let grk_clipped = grk.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));
                let gb_clipped = gb.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));

                gate.kernel = &gate.kernel - &(lr * &gk_clipped);
                gate.recurrent_kernel = &gate.recurrent_kernel - &(lr * &grk_clipped);
                gate.bias = &gate.bias - &(lr * &gb_clipped);
            }
        }

        // Update all three gates sequentially (overhead of parallelization outweighs benefits)
        update_gate(&mut self.reset_gate, lr);
        update_gate(&mut self.update_gate, lr);
        update_gate(&mut self.candidate_gate, lr);
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Helper to update a single gate's parameters with Adam
        fn update_gate_adam(
            gate: &mut Gate,
            input_dim: usize,
            units: usize,
            lr: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            t: u64,
        ) {
            // Initialize Adam states if needed
            if gate.optimizer_cache.adam_states.is_none() {
                gate.optimizer_cache.adam_states = Some(AdamStates::new(
                    (input_dim, units),
                    Some((units, units)),
                    (1, units),
                ));
            }

            if let (Some(gk), Some(grk), Some(gb)) = (
                &gate.grad_kernel,
                &gate.grad_recurrent_kernel,
                &gate.grad_bias,
            ) {
                // Apply gradient clipping before Adam update
                const CLIP_VALUE: f32 = 5.0;
                let gk_clipped = gk.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));
                let grk_clipped = grk.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));
                let gb_clipped = gb.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));

                let adam_states = gate.optimizer_cache.adam_states.as_mut().unwrap();
                let (k_update, rk_update, b_update) = adam_states.update_parameter(
                    &gk_clipped,
                    Some(&grk_clipped),
                    &gb_clipped,
                    beta1,
                    beta2,
                    epsilon,
                    t,
                    lr,
                );

                gate.kernel = &gate.kernel - &k_update;
                gate.recurrent_kernel = &gate.recurrent_kernel - &rk_update.unwrap();
                gate.bias = &gate.bias - &b_update;
            }
        }

        // Update all three gates sequentially (optimizer updates are already expensive)
        update_gate_adam(
            &mut self.reset_gate,
            self.input_dim,
            self.units,
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate_adam(
            &mut self.update_gate,
            self.input_dim,
            self.units,
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate_adam(
            &mut self.candidate_gate,
            self.input_dim,
            self.units,
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        // Helper to update a single gate's parameters with RMSprop
        fn update_gate_rmsprop(
            gate: &mut Gate,
            input_dim: usize,
            units: usize,
            lr: f32,
            rho: f32,
            epsilon: f32,
        ) {
            // Initialize RMSprop cache if needed
            if gate.optimizer_cache.rmsprop_cache.is_none() {
                gate.optimizer_cache.rmsprop_cache = Some(RMSpropCache::new(
                    (input_dim, units),
                    Some((units, units)),
                    (1, units),
                ));
            }

            if let (Some(gk), Some(grk), Some(gb)) = (
                &gate.grad_kernel,
                &gate.grad_recurrent_kernel,
                &gate.grad_bias,
            ) {
                // Apply gradient clipping before RMSprop update
                const CLIP_VALUE: f32 = 5.0;
                let gk_clipped = gk.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));
                let grk_clipped = grk.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));
                let gb_clipped = gb.mapv(|x| x.clamp(-CLIP_VALUE, CLIP_VALUE));

                if let Some(ref mut cache) = gate.optimizer_cache.rmsprop_cache {
                    cache.update_parameters(
                        &mut gate.kernel,
                        Some(&mut gate.recurrent_kernel),
                        &mut gate.bias,
                        &gk_clipped,
                        Some(&grk_clipped),
                        &gb_clipped,
                        rho,
                        lr,
                        epsilon,
                    );
                }
            }
        }

        // Update all three gates sequentially
        update_gate_rmsprop(
            &mut self.reset_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
        update_gate_rmsprop(
            &mut self.update_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
        update_gate_rmsprop(
            &mut self.candidate_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::GRU(GRULayerWeight {
            reset: GRUGateWeight {
                kernel: &self.reset_gate.kernel,
                recurrent_kernel: &self.reset_gate.recurrent_kernel,
                bias: &self.reset_gate.bias,
            },
            update: GRUGateWeight {
                kernel: &self.update_gate.kernel,
                recurrent_kernel: &self.update_gate.recurrent_kernel,
                bias: &self.update_gate.bias,
            },
            candidate: GRUGateWeight {
                kernel: &self.candidate_gate.kernel,
                recurrent_kernel: &self.candidate_gate.recurrent_kernel,
                bias: &self.candidate_gate.bias,
            },
        })
    }
}
