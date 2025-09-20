use super::super::*;
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::distributions::uniform::Uniform;
use rayon::prelude::*;

/// LSTM (Long Short-Term Memory) neural network layer implementation.
///
/// A Long Short-Term Memory layer is a type of recurrent neural network (RNN) layer
/// that is capable of learning long-term dependencies. It uses gates to control
/// the flow of information and mitigate the vanishing gradient problem.
///
/// # Structure
/// The LSTM cell contains four gates:
/// - Input gate (i): Controls when new information is added to cell state
/// - Forget gate (f): Controls what information is discarded from cell state
/// - Cell gate (c): Generates candidate values to add to the cell state
/// - Output gate (o): Controls what part of the cell state is output
///
/// # Dimensions
///
/// - Input shape: (batch_size, timesteps, input_dim)
/// - Output shape: (batch_size, timesteps, units)
///
/// # Fields
/// - `input_dim`: The dimensionality of the input features.
/// - `units`: The number of LSTM units (output dimensionality).
/// - `input_gate`: Gate controlling when new information is added to cell state.
/// - `forget_gate`: Gate controlling what information is discarded from cell state.
/// - `cell_gate`: Gate generating candidate values to add to the cell state.
/// - `output_gate`: Gate controlling what part of the cell state is output.
/// - `input_cache`: Cache for storing input during forward pass for use in backward pass.
/// - `hidden_cache`: Cache for storing hidden states during forward pass.
/// - `cell_cache`: Cache for storing cell states during forward pass.
/// - `cell_activated_cache`: Cache for storing activated cell states.
/// - `activation`: Activation function applied to the cell state and gates.
///
/// # Example
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::*;
///
/// // Create input with batch_size=2, timesteps=5, input_dim=4,
/// // and target with batch_size=2, units=3 (same dimension as the last hidden state)
/// let x = Array::ones((2, 5, 4)).into_dyn();
/// let y = Array::ones((2, 3)).into_dyn();
///
/// // Build model: one SimpleRnn layer with tanh activation
/// let mut model = Sequential::new();
/// model
/// .add(LSTM::new(4, 3, Activation::Tanh))
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
/// println!("LSTM prediction:\n{:#?}\n", pred);
/// ```
pub struct LSTM {
    input_dim: usize,
    units: usize,

    // Using four Gate structures instead of separate weight matrices
    input_gate: Gate,  // i - input gate
    forget_gate: Gate, // f - forget gate
    cell_gate: Gate,   // c - cell state update gate
    output_gate: Gate, // o - output gate

    // Cache
    input_cache: Option<Array3<f32>>,
    hidden_cache: Option<Vec<Array2<f32>>>,
    cell_cache: Option<Vec<Array2<f32>>>,
    cell_activated_cache: Option<Vec<Array2<f32>>>,

    activation: Activation,
}

/// Represents a gate unit in a Long Short-Term Memory (LSTM) neural network layer.
///
/// In LSTM architecture, gates control the flow of information through the cell state.
/// Each gate consists of weight matrices, biases, and various caches for training and
/// optimization purposes.
///
/// # Gates in LSTM
/// The LSTM implementation uses four types of gates:
/// - Input gate: Controls when new information is added to cell state
/// - Forget gate: Controls what information is discarded from cell state
/// - Cell gate: Generates candidate values to add to the cell state
/// - Output gate: Controls what part of the cell state is output
///
/// # Fields
/// - `kernel` - Weight matrix applied to the input
/// - `recurrent_kernel` - Weight matrix applied to the previous hidden state
/// - `bias` - Bias term added to the weighted inputs
/// - `gate_cache` - Forward propagation cache
/// - `grad_*` - Gradients
/// - `adam_states` - Adam optimizer states
/// - `rmsprop_cache` - RMSprop optimizer cache
/// - `optimizer_cache` - Cache for optimizer
struct Gate {
    kernel: Array2<f32>,
    recurrent_kernel: Array2<f32>,
    bias: Array2<f32>,
    gate_cache: Option<Vec<Array2<f32>>>,
    grad_kernel: Option<Array2<f32>>,
    grad_recurrent_kernel: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,
    optimizer_cache: OptimizerCache,
}

impl Gate {
    /// Creates a new Gate instance for use in an LSTM layer.
    ///
    /// This function initializes a gate unit with weight matrices, biases, and various caches
    /// needed for forward/backward propagation and optimization during training.
    ///
    /// # Arguments
    ///
    /// - `input_dim` - The dimensionality of the input features. Determines the number of rows in the kernel matrix.
    /// - `units` - The number of LSTM units (output dimensionality). Determines the number of columns in weight matrices and the size of the bias vector.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new `Gate` instance with:
    ///     - Randomly initialized weight matrices using a uniform distribution between -0.05 and 0.05
    ///     - Bias vector initialized to zeros
    ///     - All caches and optimization states initialized to `None`
    pub fn new(input_dim: usize, units: usize) -> Self {
        fn rand_mat(r: usize, c: usize) -> Array2<f32> {
            Array::random((r, c), Uniform::new(-0.05, 0.05))
        }
        let z = Array2::zeros((1, units));

        Self {
            kernel: rand_mat(input_dim, units),
            recurrent_kernel: rand_mat(units, units),
            bias: z.clone(),
            gate_cache: None,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
            optimizer_cache: OptimizerCache {
                adam_states: None,
                rmsprop_cache: None,
            },
        }
    }
}

impl LSTM {
    /// Creates a new Long Short-Term Memory (LSTM) layer.
    ///
    /// # Arguments
    ///
    /// - `input_dim` - The dimensionality of the input features.
    /// - `units` - The number of LSTM units (output dimensionality).
    /// - `activation` - The activation function to use for the output.
    ///
    /// # Returns
    ///
    /// * `Self` - A new LSTM instance with initialized weights and biases. All weights are randomly initialized using a uniform distribution between -0.05 and 0.05, and all biases are initialized to zeros.
    pub fn new(input_dim: usize, units: usize, activation: Activation) -> Self {
        LSTM {
            input_dim,
            units,
            input_gate: Gate::new(input_dim, units),
            forget_gate: Gate::new(input_dim, units),
            cell_gate: Gate::new(input_dim, units),
            output_gate: Gate::new(input_dim, units),
            input_cache: None,
            hidden_cache: None,
            cell_cache: None,
            cell_activated_cache: None,
            activation,
        }
    }
}

impl Layer for LSTM {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let x3 = input.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.clone());

        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut cs = Vec::with_capacity(timesteps + 1);
        let mut c_activateds = Vec::with_capacity(timesteps);

        let mut h_t = Array2::<f32>::zeros((batch, self.units));
        let mut c_t = Array2::<f32>::zeros((batch, self.units));
        hs.push(h_t.clone());
        cs.push(c_t.clone());

        // Clear previous gate caches
        self.input_gate.gate_cache = Some(Vec::with_capacity(timesteps));
        self.forget_gate.gate_cache = Some(Vec::with_capacity(timesteps));
        self.cell_gate.gate_cache = Some(Vec::with_capacity(timesteps));
        self.output_gate.gate_cache = Some(Vec::with_capacity(timesteps));

        // Define a closure function to compute gate activation values
        let compute_gate = |gate: &mut Gate,
                            x_t: &Array2<f32>,
                            h_t: &Array2<f32>,
                            activation: &Activation|
         -> Array2<f32> {
            let pre = x_t.dot(&gate.kernel) + h_t.dot(&gate.recurrent_kernel) + &gate.bias;
            let activated = Activation::apply_activation(&pre, activation);

            // Save gate state to gate_cache
            if let Some(cache) = &mut gate.gate_cache {
                cache.push(activated.clone());
            }
            activated
        };

        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned();

            // Use rayon to compute the four gate values in parallel
            let ((i_t, f_t), (c_bar, o_t)) = rayon::join(
                || {
                    rayon::join(
                        || compute_gate(&mut self.input_gate, &x_t, &h_t, &self.activation),
                        || compute_gate(&mut self.forget_gate, &x_t, &h_t, &self.activation),
                    )
                },
                || {
                    rayon::join(
                        || compute_gate(&mut self.cell_gate, &x_t, &h_t, &self.activation),
                        || compute_gate(&mut self.output_gate, &x_t, &h_t, &self.activation),
                    )
                },
            );

            // Update cell state
            c_t = &f_t * &c_t + &i_t * &c_bar;

            // Apply activation function to cell state
            let c_activated = Activation::apply_activation(&c_t, &self.activation);
            c_activateds.push(c_activated.clone());

            // Calculate new hidden state
            let h_new = &o_t * &c_activated;

            // Save state caches
            hs.push(h_new.clone());
            cs.push(c_t.clone());
            h_t = h_new;
        }

        // Save caches for backward propagation
        self.hidden_cache = Some(hs);
        self.cell_cache = Some(cs);
        self.cell_activated_cache = Some(c_activateds);

        // Return the final hidden state
        h_t.into_dyn()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let mut grad_h = grad_output
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
        let hs = take_cache(&mut self.hidden_cache, error_msg)?;
        let cs = take_cache(&mut self.cell_cache, error_msg)?;
        let c_activateds = take_cache(&mut self.cell_activated_cache, error_msg)?;

        // Retrieve caches from each gate
        let gis = take_cache(&mut self.input_gate.gate_cache, error_msg)?;
        let gfs = take_cache(&mut self.forget_gate.gate_cache, error_msg)?;
        let gcs = take_cache(&mut self.cell_gate.gate_cache, error_msg)?;
        let gos = take_cache(&mut self.output_gate.gate_cache, error_msg)?;

        // Process the gradient of the last time step, if it's softmax, special handling is needed
        if self.activation == Activation::Softmax {
            grad_h = Activation::softmax_backward(&hs[hs.len() - 1], &grad_h);
        }

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // Initialize gradient matrices for each gate
        let mut grad_kernel_i = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_kernel_f = grad_kernel_i.clone();
        let mut grad_kernel_c = grad_kernel_i.clone();
        let mut grad_kernel_o = grad_kernel_i.clone();
        let mut grad_rk_i = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_rk_f = grad_rk_i.clone();
        let mut grad_rk_c = grad_rk_i.clone();
        let mut grad_rk_o = grad_rk_i.clone();
        let mut grad_b_i = Array2::<f32>::zeros((1, self.units));
        let mut grad_b_f = grad_b_i.clone();
        let mut grad_b_c = grad_b_i.clone();
        let mut grad_b_o = grad_b_i.clone();

        let mut grad_c = Array2::<f32>::zeros((batch, self.units));
        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        for t in (0..timesteps).rev() {
            let i_t = &gis[t];
            let f_t = &gfs[t];
            let c_bar = &gcs[t];
            let o_t = &gos[t];
            let c_prev = &cs[t];
            let h_prev = &hs[t];
            let x_t = x3.index_axis(Axis(1), t).to_owned();
            let c_activated = &c_activateds[t];

            // Apply activation function derivative to c_prev of the current time step
            let c_act_deriv = if self.activation == Activation::Softmax {
                Activation::softmax_backward(c_activated, &Array::ones(c_activated.dim()))
            } else {
                Activation::activation_derivative(c_activated, &self.activation)
            };

            // o
            let d_o =
                &grad_h * c_activated * Activation::activation_derivative(o_t, &self.activation);
            // cell
            let d_c = &grad_h * o_t * c_act_deriv + &grad_c;

            // Calculate gradient components in parallel
            let (d_f, (d_i, d_cbar)) = rayon::join(
                || &d_c * c_prev * Activation::activation_derivative(f_t, &self.activation),
                || {
                    rayon::join(
                        || &d_c * c_bar * Activation::activation_derivative(i_t, &self.activation),
                        || &d_c * i_t * Activation::activation_derivative(c_bar, &self.activation),
                    )
                },
            );

            // Define a function to compute gradients
            fn compute_gradients(
                x_t: &Array2<f32>,
                h_prev: &Array2<f32>,
                delta: &Array2<f32>,
            ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
                let kernel_grad = x_t.t().dot(delta);
                let rk_grad = h_prev.t().dot(delta);
                let b_grad = delta.sum_axis(Axis(0)).insert_axis(Axis(0));
                (kernel_grad, rk_grad, b_grad)
            }

            // Compute gradients for all gates in parallel
            let gates_data = [
                (&d_i, &mut grad_kernel_i, &mut grad_rk_i, &mut grad_b_i),
                (&d_f, &mut grad_kernel_f, &mut grad_rk_f, &mut grad_b_f),
                (&d_cbar, &mut grad_kernel_c, &mut grad_rk_c, &mut grad_b_c),
                (&d_o, &mut grad_kernel_o, &mut grad_rk_o, &mut grad_b_o),
            ];

            // Process all gates in parallel using rayon
            let results = gates_data
                .iter()
                .map(|(delta, _, _, _)| *delta)
                .collect::<Vec<_>>()
                .par_iter()
                .map(|delta| compute_gradients(&x_t, &h_prev, *delta))
                .collect::<Vec<_>>();

            // Accumulate results
            for (i, (kernel_grad, rk_grad, b_grad)) in results.iter().enumerate() {
                *gates_data[i].1 = &*gates_data[i].1 + kernel_grad;
                *gates_data[i].2 = &*gates_data[i].2 + rk_grad;
                *gates_data[i].3 = &*gates_data[i].3 + b_grad;
            }

            // Propagate gradients using the core weights of each gate
            let dx = d_i.dot(&self.input_gate.kernel.t())
                + d_f.dot(&self.forget_gate.kernel.t())
                + d_cbar.dot(&self.cell_gate.kernel.t())
                + d_o.dot(&self.output_gate.kernel.t());

            grad_x3.index_axis_mut(Axis(1), t).assign(&dx);

            // Propagate gradients using the recurrent weights of each gate
            grad_h = d_i.dot(&self.input_gate.recurrent_kernel.t())
                + d_f.dot(&self.forget_gate.recurrent_kernel.t())
                + d_cbar.dot(&self.cell_gate.recurrent_kernel.t())
                + d_o.dot(&self.output_gate.recurrent_kernel.t());

            grad_c = &d_c * f_t;
        }

        // Save gradients for each gate
        self.input_gate.grad_kernel = Some(grad_kernel_i);
        self.forget_gate.grad_kernel = Some(grad_kernel_f);
        self.cell_gate.grad_kernel = Some(grad_kernel_c);
        self.output_gate.grad_kernel = Some(grad_kernel_o);

        self.input_gate.grad_recurrent_kernel = Some(grad_rk_i);
        self.forget_gate.grad_recurrent_kernel = Some(grad_rk_f);
        self.cell_gate.grad_recurrent_kernel = Some(grad_rk_c);
        self.output_gate.grad_recurrent_kernel = Some(grad_rk_o);

        self.input_gate.grad_bias = Some(grad_b_i);
        self.forget_gate.grad_bias = Some(grad_b_f);
        self.cell_gate.grad_bias = Some(grad_b_c);
        self.output_gate.grad_bias = Some(grad_b_o);

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "LSTM"
    }
    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }
    fn param_count(&self) -> usize {
        4 * (self.input_dim * self.units + self.units * self.units + self.units)
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        // Create a helper function to update gate parameters
        fn update_gate_parameters(gate: &mut Gate, lr: f32) {
            if let (Some(gk), Some(grk), Some(gb)) = (
                &gate.grad_kernel,
                &gate.grad_recurrent_kernel,
                &gate.grad_bias,
            ) {
                SGD::update_sgd_parameters_rnn(
                    &mut gate.kernel,
                    gk,
                    &mut gate.recurrent_kernel,
                    grk,
                    &mut gate.bias,
                    gb,
                    lr,
                );
            }
        }

        // Use the helper function to update parameters for each gate
        rayon::join(
            || {
                rayon::join(
                    || update_gate_parameters(&mut self.input_gate, lr),
                    || update_gate_parameters(&mut self.forget_gate, lr),
                )
            },
            || {
                rayon::join(
                    || update_gate_parameters(&mut self.cell_gate, lr),
                    || update_gate_parameters(&mut self.output_gate, lr),
                )
            },
        );
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Initialize AdamStates structure if not already initialized
        if self.input_gate.optimizer_cache.adam_states.is_none() {
            let dk = (self.input_dim, self.units);
            let dr = (self.units, self.units);
            let db = (1, self.units);

            // Initialize Adam states for all gates
            self.input_gate.optimizer_cache.adam_states = Some(AdamStates::new(dk, Some(dr), db));
            self.forget_gate.optimizer_cache.adam_states = Some(AdamStates::new(dk, Some(dr), db));
            self.cell_gate.optimizer_cache.adam_states = Some(AdamStates::new(dk, Some(dr), db));
            self.output_gate.optimizer_cache.adam_states = Some(AdamStates::new(dk, Some(dr), db));
        }

        // Helper function to update individual gate parameters
        fn update_gate(gate: &mut Gate, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
            if let (Some(gk), Some(grk), Some(gb)) = (
                &gate.grad_kernel,
                &gate.grad_recurrent_kernel,
                &gate.grad_bias,
            ) {
                let adam = gate.optimizer_cache.adam_states.as_mut().unwrap();
                let (w_update, rk_update, b_update) =
                    adam.update_parameter(gk, Some(grk), gb, beta1, beta2, epsilon, t, lr);

                // Apply parameter updates
                gate.kernel = &gate.kernel - &w_update;
                gate.recurrent_kernel = &gate.recurrent_kernel - &rk_update.unwrap();
                gate.bias = &gate.bias - &b_update;
            }
        }

        // Apply parameter updates for each gate
        update_gate(&mut self.input_gate, lr, beta1, beta2, epsilon, t);
        update_gate(&mut self.forget_gate, lr, beta1, beta2, epsilon, t);
        update_gate(&mut self.cell_gate, lr, beta1, beta2, epsilon, t);
        update_gate(&mut self.output_gate, lr, beta1, beta2, epsilon, t);
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        // Initialize RMSprop cache if it doesn't exist
        if self.input_gate.optimizer_cache.rmsprop_cache.is_none() {
            let dk = (self.input_dim, self.units);
            let dr = (self.units, self.units);
            let db = (1, self.units);

            // Initialize RMSprop cache for all gates
            self.input_gate.optimizer_cache.rmsprop_cache =
                Some(RMSpropCache::new(dk, Some(dr), db));
            self.forget_gate.optimizer_cache.rmsprop_cache =
                Some(RMSpropCache::new(dk, Some(dr), db));
            self.cell_gate.optimizer_cache.rmsprop_cache =
                Some(RMSpropCache::new(dk, Some(dr), db));
            self.output_gate.optimizer_cache.rmsprop_cache =
                Some(RMSpropCache::new(dk, Some(dr), db));
        }

        // Update parameters for each gate
        fn update_gate_rms(gate: &mut Gate, lr: f32, rho: f32, epsilon: f32) {
            if let (Some(gk), Some(grk), Some(gb)) = (
                &gate.grad_kernel,
                &gate.grad_recurrent_kernel,
                &gate.grad_bias,
            ) {
                if let Some(ref mut cache) = gate.optimizer_cache.rmsprop_cache {
                    cache.update_parameters(
                        &mut gate.kernel,
                        Some(&mut gate.recurrent_kernel),
                        &mut gate.bias,
                        gk,
                        Some(grk),
                        gb,
                        rho,
                        lr,
                        epsilon,
                    );
                }
            }
        }

        // Apply RMSprop updates to all gates
        rayon::join(
            || {
                rayon::join(
                    || update_gate_rms(&mut self.input_gate, lr, rho, epsilon),
                    || update_gate_rms(&mut self.forget_gate, lr, rho, epsilon),
                )
            },
            || {
                rayon::join(
                    || update_gate_rms(&mut self.cell_gate, lr, rho, epsilon),
                    || update_gate_rms(&mut self.output_gate, lr, rho, epsilon),
                )
            },
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
