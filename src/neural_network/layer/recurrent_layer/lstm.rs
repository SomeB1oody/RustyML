use super::*;

/// Threshold for using parallel computation in LSTM layer.
/// When batch_size * units < this value, sequential execution is used.
/// When batch_size * units >= this value, parallel execution is used.
///
/// Value is chosen based on empirical benchmarks where rayon's thread pool
/// overhead is amortized by computational gains from parallelization.
const LSTM_PARALLEL_THRESHOLD: usize = 1024;

/// Long Short-Term Memory (LSTM) neural network layer
///
/// LSTM is a type of recurrent neural network (RNN) architecture that is capable of learning
/// long-term dependencies in sequential data. It addresses the vanishing gradient problem
/// of traditional RNNs through the use of gating mechanisms that control information flow.
///
/// The LSTM cell uses three gates to control information flow:
/// - Input Gate: Controls what new information should be stored in the cell state
/// - Forget Gate: Controls what information should be discarded from the cell state
/// - Output Gate: Controls what parts of the cell state should be output as the hidden state
/// - Cell Gate: Proposes new candidate values to be added to the cell state
///
/// The LSTM maintains two internal states:
/// - Cell State (C_t): The long-term memory that flows through the network
/// - Hidden State (h_t): The short-term memory and current output
///
/// # Mathematical Operations
///
/// For each timestep t:
/// 1. i_t = σ(W_i · \[h_{t-1}, x_t\] + b_i)  (Input gate)
/// 2. f_t = σ(W_f · \[h_{t-1}, x_t] + b_f)  (Forget gate)
/// 3. g_t = tanh(W_g · \[h_{t-1}, x_t\] + b_g)  (Cell gate)
/// 4. o_t = σ(W_o · \[h_{t-1}, x_t\] + b_o)  (Output gate)
/// 5. C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t  (Cell state update)
/// 6. h_t = o_t ⊙ tanh(C_t)  (Hidden state update)
///
/// Where σ is the sigmoid function, ⊙ is element-wise multiplication, and W, b are learned parameters.
///
/// # Fields
///
/// - `input_dim` - Dimensionality of input features
/// - `units` - Number of LSTM units (neurons) in the layer
/// - `input_gate` - Gate controlling what new information to store in cell state
/// - `forget_gate` - Gate controlling what information to discard from cell state
/// - `cell_gate` - Gate proposing new candidate values for cell state
/// - `output_gate` - Gate controlling what to output from cell state
/// - `input_cache` - Cached input tensor for backward propagation
/// - `hidden_cache` - Cached hidden states h_t for each timestep
/// - `cell_cache` - Cached cell states C_t for each timestep
/// - `cell_activated_cache` - Cached tanh(C_t) values for each timestep
/// - `i_cache` - Cached input gate activations for each timestep
/// - `f_cache` - Cached forget gate activations for each timestep
/// - `g_cache` - Cached cell gate activations for each timestep
/// - `o_cache` - Cached output gate activations for each timestep
/// - `activation` - Activation layer from activation_layer module
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
/// // Create LSTM layer with 4 input features, 3 units, Tanh activation
/// let mut model = Sequential::new();
/// model.add(LSTM::new(4, 3, Tanh::new()))
///      .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input);
/// println!("LSTM output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
pub struct LSTM<T: ActivationLayer> {
    input_dim: usize,
    units: usize,

    // Four gates: input, forget, cell, output
    input_gate: Gate,
    forget_gate: Gate,
    cell_gate: Gate,
    output_gate: Gate,

    // Caches for forward pass
    input_cache: Option<Array3<f32>>,
    hidden_cache: Option<Vec<Array2<f32>>>, // hidden states h_t
    cell_cache: Option<Vec<Array2<f32>>>,   // cell states c_t
    cell_activated_cache: Option<Vec<Array2<f32>>>, // tanh(c_t)

    // Intermediate gate values cache
    i_cache: Option<Vec<Array2<f32>>>, // input gate values (sigmoid applied)
    f_cache: Option<Vec<Array2<f32>>>, // forget gate values (sigmoid applied)
    g_cache: Option<Vec<Array2<f32>>>, // cell gate values (tanh applied)
    o_cache: Option<Vec<Array2<f32>>>, // output gate values (sigmoid applied)

    activation: T,
}

impl<T: ActivationLayer> LSTM<T> {
    /// Creates a new LSTM layer with specified parameters
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of LSTM units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// * `LSTM` - A new `LSTM` instance with:
    ///     - Four gates (input, forget, cell, output) initialized with random weights
    ///     - All caches set to None (will be allocated during first forward pass)
    ///     - Activation layer for output transformation
    pub fn new(input_dim: usize, units: usize, activation: T) -> Self {
        Self {
            input_dim,
            units,
            input_gate: Gate::new(input_dim, units, 0.0),
            forget_gate: Gate::new(input_dim, units, 1.0), // forget gate bias = 1.0
            cell_gate: Gate::new(input_dim, units, 0.0),
            output_gate: Gate::new(input_dim, units, 0.0),
            input_cache: None,
            hidden_cache: None,
            cell_cache: None,
            cell_activated_cache: None,
            i_cache: None,
            f_cache: None,
            g_cache: None,
            o_cache: None,
            activation,
        }
    }

    /// Sets the weights for all four gates in this LSTM layer.
    ///
    /// # Parameters
    ///
    /// Each gate requires three arrays:
    /// - `kernel` - Weight matrix connecting inputs with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    ///
    /// The parameters are provided for each of the four gates in order:
    /// input gate, forget gate, cell gate, output gate
    pub fn set_weights(
        &mut self,
        input_kernel: Array2<f32>,
        input_recurrent_kernel: Array2<f32>,
        input_bias: Array2<f32>,
        forget_kernel: Array2<f32>,
        forget_recurrent_kernel: Array2<f32>,
        forget_bias: Array2<f32>,
        cell_kernel: Array2<f32>,
        cell_recurrent_kernel: Array2<f32>,
        cell_bias: Array2<f32>,
        output_kernel: Array2<f32>,
        output_recurrent_kernel: Array2<f32>,
        output_bias: Array2<f32>,
    ) {
        self.input_gate.kernel = input_kernel;
        self.input_gate.recurrent_kernel = input_recurrent_kernel;
        self.input_gate.bias = input_bias;

        self.forget_gate.kernel = forget_kernel;
        self.forget_gate.recurrent_kernel = forget_recurrent_kernel;
        self.forget_gate.bias = forget_bias;

        self.cell_gate.kernel = cell_kernel;
        self.cell_gate.recurrent_kernel = cell_recurrent_kernel;
        self.cell_gate.bias = cell_bias;

        self.output_gate.kernel = output_kernel;
        self.output_gate.recurrent_kernel = output_recurrent_kernel;
        self.output_gate.bias = output_bias;
    }
}

impl<T: ActivationLayer> Layer for LSTM<T> {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 3D".to_string(),
            ));
        }

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        // Initialize hidden state and cell state
        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut c_prev = Array2::<f32>::zeros((batch, self.units));

        // Storage for all timesteps
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut cs = Vec::with_capacity(timesteps + 1);
        let mut cs_activated = Vec::with_capacity(timesteps);
        let mut i_vals = Vec::with_capacity(timesteps);
        let mut f_vals = Vec::with_capacity(timesteps);
        let mut g_vals = Vec::with_capacity(timesteps);
        let mut o_vals = Vec::with_capacity(timesteps);

        hs.push(h_prev.clone());
        cs.push(c_prev.clone());

        // Determine whether to use parallel execution based on computational load
        let use_parallel = batch * self.units >= LSTM_PARALLEL_THRESHOLD;

        // Process each timestep
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned(); // (batch, input_dim)

            // Compute all 4 gate values (parallel or sequential)
            let (i_raw, f_raw, g_raw, o_raw) = if use_parallel {
                let ((i_raw, f_raw), (g_raw, o_raw)) = rayon::join(
                    || {
                        rayon::join(
                            || compute_gate_value(&self.input_gate, &x_t, &h_prev),
                            || compute_gate_value(&self.forget_gate, &x_t, &h_prev),
                        )
                    },
                    || {
                        rayon::join(
                            || compute_gate_value(&self.cell_gate, &x_t, &h_prev),
                            || compute_gate_value(&self.output_gate, &x_t, &h_prev),
                        )
                    },
                );
                (i_raw, f_raw, g_raw, o_raw)
            } else {
                (
                    compute_gate_value(&self.input_gate, &x_t, &h_prev),
                    compute_gate_value(&self.forget_gate, &x_t, &h_prev),
                    compute_gate_value(&self.cell_gate, &x_t, &h_prev),
                    compute_gate_value(&self.output_gate, &x_t, &h_prev),
                )
            };

            // Apply activations to all 4 gates (parallel or sequential)
            let (i_t, f_t, g_t, o_t) = if use_parallel {
                let ((i_t, f_t), (g_t, o_t)) = rayon::join(
                    || rayon::join(|| apply_sigmoid(i_raw), || apply_sigmoid(f_raw)),
                    || {
                        rayon::join(
                            || {
                                g_raw.mapv(|x| {
                                    let clipped_x = x.clamp(-500.0, 500.0);
                                    clipped_x.tanh()
                                })
                            },
                            || apply_sigmoid(o_raw),
                        )
                    },
                );
                (i_t, f_t, g_t, o_t)
            } else {
                (
                    apply_sigmoid(i_raw),
                    apply_sigmoid(f_raw),
                    g_raw.mapv(|x| {
                        let clipped_x = x.clamp(-500.0, 500.0);
                        clipped_x.tanh()
                    }),
                    apply_sigmoid(o_raw),
                )
            };

            // Update cell state: c_t = f_t * c_prev + i_t * g_t
            let c_t = &f_t * &c_prev + &i_t * &g_t;

            // Apply tanh to cell state
            let c_t_activated = c_t.mapv(|x| {
                let clipped_x = x.clamp(-500.0, 500.0);
                clipped_x.tanh()
            });

            // Update hidden state: h_t = o_t * tanh(c_t)
            let h_t = &o_t * &c_t_activated;

            // Cache values
            i_vals.push(i_t);
            f_vals.push(f_t);
            g_vals.push(g_t);
            o_vals.push(o_t);
            cs.push(c_t.clone());
            cs_activated.push(c_t_activated);
            hs.push(h_t.clone());

            h_prev = h_t;
            c_prev = c_t;
        }

        // Store caches
        self.hidden_cache = Some(hs);
        self.cell_cache = Some(cs);
        self.cell_activated_cache = Some(cs_activated);
        self.i_cache = Some(i_vals);
        self.f_cache = Some(f_vals);
        self.g_cache = Some(g_vals);
        self.o_cache = Some(o_vals);

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
        let cs = take_cache(&mut self.cell_cache, error_msg)?;
        let cs_activated = take_cache(&mut self.cell_activated_cache, error_msg)?;
        let i_vals = take_cache(&mut self.i_cache, error_msg)?;
        let f_vals = take_cache(&mut self.f_cache, error_msg)?;
        let g_vals = take_cache(&mut self.g_cache, error_msg)?;
        let o_vals = take_cache(&mut self.o_cache, error_msg)?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // Initialize gradient accumulators
        let mut grad_i_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_i_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_i_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_f_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_f_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_f_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_g_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_g_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_g_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_o_kernel = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_o_recurrent = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_o_bias = Array2::<f32>::zeros((1, self.units));

        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        let mut grad_h = grad_h_t;
        let mut grad_c = Array2::<f32>::zeros((batch, self.units));

        // Determine whether to use parallel execution based on computational load
        let use_parallel = batch * self.units >= LSTM_PARALLEL_THRESHOLD;

        // Backpropagation through time
        for t in (0..timesteps).rev() {
            let h_prev = &hs[t];
            let c_prev = &cs[t];
            let c_t_activated = &cs_activated[t];
            let i_t = &i_vals[t];
            let f_t = &f_vals[t];
            let g_t = &g_vals[t];
            let o_t = &o_vals[t];

            // Gradient through h_t = o_t * tanh(c_t)
            let grad_o_t = &grad_h * c_t_activated;
            grad_c = grad_c + &(&grad_h * o_t * &(1.0 - c_t_activated * c_t_activated));

            // Gradient through c_t = f_t * c_prev + i_t * g_t
            let grad_f_t = &grad_c * c_prev;
            let grad_i_t = &grad_c * g_t;
            let grad_g_t = &grad_c * i_t;
            let grad_c_prev = &grad_c * f_t;

            // Compute gate activation derivatives (parallel or sequential)
            let (grad_o_raw, grad_f_raw, grad_i_raw, grad_g_raw) = if use_parallel {
                let ((grad_o_raw, grad_f_raw), (grad_i_raw, grad_g_raw)) = rayon::join(
                    || {
                        rayon::join(
                            || &grad_o_t * o_t * &(1.0 - o_t), // sigmoid derivative
                            || &grad_f_t * f_t * &(1.0 - f_t), // sigmoid derivative
                        )
                    },
                    || {
                        rayon::join(
                            || &grad_i_t * i_t * &(1.0 - i_t), // sigmoid derivative
                            || &grad_g_t * &(1.0 - g_t * g_t), // tanh derivative
                        )
                    },
                );
                (grad_o_raw, grad_f_raw, grad_i_raw, grad_g_raw)
            } else {
                (
                    &grad_o_t * o_t * &(1.0 - o_t), // sigmoid derivative
                    &grad_f_t * f_t * &(1.0 - f_t), // sigmoid derivative
                    &grad_i_t * i_t * &(1.0 - i_t), // sigmoid derivative
                    &grad_g_t * &(1.0 - g_t * g_t), // tanh derivative
                )
            };

            let x_t = x3.index_axis(Axis(1), t).to_owned();

            let x_t_t = x_t.t();
            let h_prev_t = h_prev.t();

            // Closure to compute gradient updates for a single gate
            let compute_gate_gradients = |grad_raw: &Array2<f32>| {
                let kernel_update = x_t_t.dot(grad_raw);
                let recurrent_update = h_prev_t.dot(grad_raw);
                let bias_update = grad_raw.sum_axis(Axis(0)).insert_axis(Axis(0));
                (kernel_update, recurrent_update, bias_update)
            };

            // Compute all gradient updates (parallel or sequential)
            let (o_updates, f_updates, i_updates, g_updates) = if use_parallel {
                let ((o_updates, f_updates), (i_updates, g_updates)) = rayon::join(
                    || {
                        rayon::join(
                            || compute_gate_gradients(&grad_o_raw),
                            || compute_gate_gradients(&grad_f_raw),
                        )
                    },
                    || {
                        rayon::join(
                            || compute_gate_gradients(&grad_i_raw),
                            || compute_gate_gradients(&grad_g_raw),
                        )
                    },
                );
                (o_updates, f_updates, i_updates, g_updates)
            } else {
                (
                    compute_gate_gradients(&grad_o_raw),
                    compute_gate_gradients(&grad_f_raw),
                    compute_gate_gradients(&grad_i_raw),
                    compute_gate_gradients(&grad_g_raw),
                )
            };

            // Apply gradient updates
            grad_o_kernel = grad_o_kernel + &o_updates.0;
            grad_o_recurrent = grad_o_recurrent + &o_updates.1;
            grad_o_bias = grad_o_bias + &o_updates.2;

            grad_f_kernel = grad_f_kernel + &f_updates.0;
            grad_f_recurrent = grad_f_recurrent + &f_updates.1;
            grad_f_bias = grad_f_bias + &f_updates.2;

            grad_i_kernel = grad_i_kernel + &i_updates.0;
            grad_i_recurrent = grad_i_recurrent + &i_updates.1;
            grad_i_bias = grad_i_bias + &i_updates.2;

            grad_g_kernel = grad_g_kernel + &g_updates.0;
            grad_g_recurrent = grad_g_recurrent + &g_updates.1;
            grad_g_bias = grad_g_bias + &g_updates.2;

            // Compute gradient with respect to input and hidden state (parallel or sequential)
            let (dx, grad_h_next) = if use_parallel {
                rayon::join(
                    || {
                        grad_o_raw.dot(&self.output_gate.kernel.t())
                            + grad_f_raw.dot(&self.forget_gate.kernel.t())
                            + grad_i_raw.dot(&self.input_gate.kernel.t())
                            + grad_g_raw.dot(&self.cell_gate.kernel.t())
                    },
                    || {
                        grad_o_raw.dot(&self.output_gate.recurrent_kernel.t())
                            + grad_f_raw.dot(&self.forget_gate.recurrent_kernel.t())
                            + grad_i_raw.dot(&self.input_gate.recurrent_kernel.t())
                            + grad_g_raw.dot(&self.cell_gate.recurrent_kernel.t())
                    },
                )
            } else {
                (
                    grad_o_raw.dot(&self.output_gate.kernel.t())
                        + grad_f_raw.dot(&self.forget_gate.kernel.t())
                        + grad_i_raw.dot(&self.input_gate.kernel.t())
                        + grad_g_raw.dot(&self.cell_gate.kernel.t()),
                    grad_o_raw.dot(&self.output_gate.recurrent_kernel.t())
                        + grad_f_raw.dot(&self.forget_gate.recurrent_kernel.t())
                        + grad_i_raw.dot(&self.input_gate.recurrent_kernel.t())
                        + grad_g_raw.dot(&self.cell_gate.recurrent_kernel.t()),
                )
            };

            grad_x3.index_axis_mut(Axis(1), t).assign(&dx);
            grad_h = grad_h_next;

            // Gradient with respect to previous cell state
            grad_c = grad_c_prev;
        }

        // Store gradients
        store_gate_gradients(
            &mut self.input_gate,
            grad_i_kernel,
            grad_i_recurrent,
            grad_i_bias,
        );
        store_gate_gradients(
            &mut self.forget_gate,
            grad_f_kernel,
            grad_f_recurrent,
            grad_f_bias,
        );
        store_gate_gradients(
            &mut self.cell_gate,
            grad_g_kernel,
            grad_g_recurrent,
            grad_g_bias,
        );
        store_gate_gradients(
            &mut self.output_gate,
            grad_o_kernel,
            grad_o_recurrent,
            grad_o_bias,
        );

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "LSTM"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            4 * (self.input_dim * self.units + self.units * self.units + self.units),
        )
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        // Update all four gates sequentially (overhead of parallelization outweighs benefits)
        update_gate_sgd(&mut self.input_gate, lr);
        update_gate_sgd(&mut self.forget_gate, lr);
        update_gate_sgd(&mut self.cell_gate, lr);
        update_gate_sgd(&mut self.output_gate, lr);
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Update all four gates sequentially (optimizer updates are already expensive)
        update_gate_adam(
            &mut self.input_gate,
            self.input_dim,
            self.units,
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate_adam(
            &mut self.forget_gate,
            self.input_dim,
            self.units,
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate_adam(
            &mut self.cell_gate,
            self.input_dim,
            self.units,
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate_adam(
            &mut self.output_gate,
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
        // Update all four gates sequentially
        update_gate_rmsprop(
            &mut self.input_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
        update_gate_rmsprop(
            &mut self.forget_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
        update_gate_rmsprop(
            &mut self.cell_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
        update_gate_rmsprop(
            &mut self.output_gate,
            self.input_dim,
            self.units,
            lr,
            rho,
            epsilon,
        );
    }

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        // Update all four gates sequentially
        update_gate_ada_grad(
            &mut self.input_gate,
            self.input_dim,
            self.units,
            lr,
            epsilon,
        );
        update_gate_ada_grad(
            &mut self.forget_gate,
            self.input_dim,
            self.units,
            lr,
            epsilon,
        );
        update_gate_ada_grad(&mut self.cell_gate, self.input_dim, self.units, lr, epsilon);
        update_gate_ada_grad(
            &mut self.output_gate,
            self.input_dim,
            self.units,
            lr,
            epsilon,
        );
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::LSTM(LSTMLayerWeight {
            input: LSTMGateWeight {
                kernel: &self.input_gate.kernel,
                recurrent_kernel: &self.input_gate.recurrent_kernel,
                bias: &self.input_gate.bias,
            },
            forget: LSTMGateWeight {
                kernel: &self.forget_gate.kernel,
                recurrent_kernel: &self.forget_gate.recurrent_kernel,
                bias: &self.forget_gate.bias,
            },
            cell: LSTMGateWeight {
                kernel: &self.cell_gate.kernel,
                recurrent_kernel: &self.cell_gate.recurrent_kernel,
                bias: &self.cell_gate.bias,
            },
            output: LSTMGateWeight {
                kernel: &self.output_gate.kernel,
                recurrent_kernel: &self.output_gate.recurrent_kernel,
                bias: &self.output_gate.bias,
            },
        })
    }
}
