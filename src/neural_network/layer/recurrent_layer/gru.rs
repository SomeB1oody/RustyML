use super::*;

/// Threshold for using parallel computation in GRU layer.
/// When batch_size * units < this value, sequential execution is used.
/// When batch_size * units >= this value, parallel execution is used.
///
/// Value is chosen based on empirical benchmarks where rayon's thread pool
/// overhead is amortized by computational gains from parallelization.
const GRU_PARALLEL_THRESHOLD: usize = 1024;

/// Gated Recurrent Unit (GRU) neural network layer.
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses reset, update, and candidate
/// gates to control information flow and mitigate vanishing gradients.
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
/// - `activation` - Activation function applied to final output
///
/// # Examples
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
/// model.add(GRU::new(4, 3, Tanh::new()).unwrap())
///      .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input).unwrap();
/// println!("GRU output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
pub struct GRU<T: ActivationLayer> {
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

    activation: T,
}

impl<T: ActivationLayer> GRU<T> {
    /// Creates a GRU layer with the specified dimensions and activation.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of GRU units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new GRU layer instance
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_dim` or `units` is 0
    pub fn new(input_dim: usize, units: usize, activation: T) -> Result<Self, ModelError> {
        // Validate that dimensions are greater than 0
        validate_recurrent_dimensions(input_dim, units)?;

        Ok(Self {
            input_dim,
            units,
            reset_gate: Gate::new(input_dim, units, 0.0)?,
            update_gate: Gate::new(input_dim, units, 0.0)?,
            candidate_gate: Gate::new(input_dim, units, 0.0)?,
            input_cache: None,
            hidden_cache: None,
            r_cache: None,
            z_cache: None,
            h_candidate_cache: None,
            rh_cache: None,
            activation,
        })
    }

    /// Sets the weights for all three gates in this GRU layer.
    ///
    /// # Parameters
    ///
    /// - `reset_kernel` - Input kernel for the reset gate with shape (input_dim, units)
    /// - `reset_recurrent_kernel` - Recurrent kernel for the reset gate with shape (units, units)
    /// - `reset_bias` - Bias for the reset gate with shape (1, units)
    /// - `update_kernel` - Input kernel for the update gate with shape (input_dim, units)
    /// - `update_recurrent_kernel` - Recurrent kernel for the update gate with shape (units, units)
    /// - `update_bias` - Bias for the update gate with shape (1, units)
    /// - `candidate_kernel` - Input kernel for the candidate gate with shape (input_dim, units)
    /// - `candidate_recurrent_kernel` - Recurrent kernel for the candidate gate with shape (units, units)
    /// - `candidate_bias` - Bias for the candidate gate with shape (1, units)
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

impl<T: ActivationLayer> Layer for GRU<T> {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        // Initialize hidden state
        let mut h_prev = Array2::<f32>::zeros((batch, self.units));

        // Storage for all timesteps
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut r_vals = Vec::with_capacity(timesteps);
        let mut z_vals = Vec::with_capacity(timesteps);
        let mut h_candidate_vals = Vec::with_capacity(timesteps);
        let mut rh_vals = Vec::with_capacity(timesteps);

        hs.push(h_prev.clone());

        // Determine whether to use parallel execution based on computational load
        let use_parallel = batch * self.units >= GRU_PARALLEL_THRESHOLD;

        // Process each timestep
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned(); // (batch, input_dim)

            // Compute reset and update gate values (parallel or sequential)
            let (r_raw, z_raw) = if use_parallel {
                rayon::join(
                    || compute_gate_value(&self.reset_gate, &x_t, &h_prev),
                    || compute_gate_value(&self.update_gate, &x_t, &h_prev),
                )
            } else {
                (
                    compute_gate_value(&self.reset_gate, &x_t, &h_prev),
                    compute_gate_value(&self.update_gate, &x_t, &h_prev),
                )
            };

            // Apply sigmoid activation to gates (parallel or sequential)
            let (r_t, z_t) = if use_parallel {
                rayon::join(|| apply_sigmoid(r_raw), || apply_sigmoid(z_raw))
            } else {
                (apply_sigmoid(r_raw), apply_sigmoid(z_raw))
            };

            // Compute r_t * h_{t-1}
            let r_h = &r_t * &h_prev;

            // Compute candidate hidden state: tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)
            let h_candidate_raw = x_t.dot(&self.candidate_gate.kernel)
                + r_h.dot(&self.candidate_gate.recurrent_kernel)
                + &self.candidate_gate.bias;
            let h_candidate = h_candidate_raw.mapv(|x| {
                let clipped_x = x.clamp(-500.0, 500.0);
                clipped_x.tanh()
            });

            // Update hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
            let h_t = &(1.0 - &z_t) * &h_prev + &z_t * &h_candidate;

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

        // Apply activation
        self.activation.forward(&h_prev.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Apply activation backward pass
        let grad_upstream = self.activation.backward(grad_output)?;

        let grad_h_t = grad_upstream.into_dimensionality::<ndarray::Ix2>().unwrap();

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
            let h_prev = &hs[t];
            let r_t = &r_vals[t];
            let z_t = &z_vals[t];
            let h_candidate = &h_candidate_vals[t];
            let r_h = &rh_vals[t];

            // Gradient through h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
            let grad_z_t = &grad_h * (h_candidate - h_prev);
            let grad_h_candidate = &grad_h * z_t;
            let grad_h_prev_from_update = &grad_h * &(1.0 - z_t);

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

        // Store gradients
        store_gate_gradients(
            &mut self.reset_gate,
            grad_r_kernel,
            grad_r_recurrent,
            grad_r_bias,
        );
        store_gate_gradients(
            &mut self.update_gate,
            grad_z_kernel,
            grad_z_recurrent,
            grad_z_bias,
        );
        store_gate_gradients(
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
        // Update all three gates sequentially (overhead of parallelization outweighs benefits)
        update_gate_sgd(&mut self.reset_gate, lr);
        update_gate_sgd(&mut self.update_gate, lr);
        update_gate_sgd(&mut self.candidate_gate, lr);
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
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

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        // Update all three gates sequentially
        update_gate_ada_grad(
            &mut self.reset_gate,
            self.input_dim,
            self.units,
            lr,
            epsilon,
        );
        update_gate_ada_grad(
            &mut self.update_gate,
            self.input_dim,
            self.units,
            lr,
            epsilon,
        );
        update_gate_ada_grad(
            &mut self.candidate_gate,
            self.input_dim,
            self.units,
            lr,
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
