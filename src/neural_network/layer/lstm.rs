use crate::ModelError;
pub use crate::neural_network::Tensor;
pub use crate::neural_network::activation::Activation;
use crate::traits::Layer;
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::distributions::uniform::Uniform;

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
/// ## Core fields
/// - `input_dim` - Dimensionality of the input features
/// - `units` - Number of LSTM units/neurons in the layer
/// - `activation` - Activation function applied to the output
/// - `cell_activated_cache` - Cache for storing activated cell states
///
/// ## Weight matrices
/// - `kernel_*` - Weight matrices for input connections (shape: input_dim × units)
/// - `recurrent_kernel_*` - Weight matrices for recurrent connections (shape: units × units)
/// - `bias_*` - Bias vectors (shape: 1 × units)
///
/// ## Cache fields
/// These fields store intermediate values needed for backpropagation:
/// - `input_cache` - Cached input tensor (shape: batch × timesteps × input_dim)
/// - `hidden_cache` - Cached hidden states for each timestep (length: timesteps+1)
/// - `cell_cache` - Cached cell states for each timestep (length: timesteps+1)
/// - `gate_*_cache` - Cached gate activations for each timestep (length: timesteps)
///
/// ## Gradient fields
/// These fields store gradients for parameter updates:
/// - `grad_kernel_*` - Gradients for kernel weights
/// - `grad_recurrent_kernel_*` - Gradients for recurrent weights
/// - `grad_bias_*` - Gradients for bias values
///
/// ## Optimizer-specific fields
/// - Adam optimizer fields (`m_*`, `v_*`): First and second moment estimates
/// - RMSprop optimizer fields (`cache_*`): Moving average of squared gradients
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
    kernel_i: Array2<f32>, // (input_dim, units)
    kernel_f: Array2<f32>,
    kernel_c: Array2<f32>,
    kernel_o: Array2<f32>,
    recurrent_kernel_i: Array2<f32>, // (units, units)
    recurrent_kernel_f: Array2<f32>,
    recurrent_kernel_c: Array2<f32>,
    recurrent_kernel_o: Array2<f32>,
    bias_i: Array2<f32>, // (1, units)
    bias_f: Array2<f32>,
    bias_c: Array2<f32>,
    bias_o: Array2<f32>,

    input_cache: Option<Array3<f32>>, // (batch, timesteps, input_dim)
    hidden_cache: Option<Vec<Array2<f32>>>, // len = timesteps+1
    cell_cache: Option<Vec<Array2<f32>>>, // len = timesteps+1
    gate_i_cache: Option<Vec<Array2<f32>>>, // len = timesteps
    gate_f_cache: Option<Vec<Array2<f32>>>,
    gate_c_cache: Option<Vec<Array2<f32>>>,
    gate_o_cache: Option<Vec<Array2<f32>>>,

    grad_kernel_i: Option<Array2<f32>>,
    grad_kernel_f: Option<Array2<f32>>,
    grad_kernel_c: Option<Array2<f32>>,
    grad_kernel_o: Option<Array2<f32>>,
    grad_recurrent_kernel_i: Option<Array2<f32>>,
    grad_recurrent_kernel_f: Option<Array2<f32>>,
    grad_recurrent_kernel_c: Option<Array2<f32>>,
    grad_recurrent_kernel_o: Option<Array2<f32>>,
    grad_bias_i: Option<Array2<f32>>,
    grad_bias_f: Option<Array2<f32>>,
    grad_bias_c: Option<Array2<f32>>,
    grad_bias_o: Option<Array2<f32>>,

    m_kernel_i: Option<Array2<f32>>,
    v_kernel_i: Option<Array2<f32>>,
    m_kernel_f: Option<Array2<f32>>,
    v_kernel_f: Option<Array2<f32>>,
    m_kernel_c: Option<Array2<f32>>,
    v_kernel_c: Option<Array2<f32>>,
    m_kernel_o: Option<Array2<f32>>,
    v_kernel_o: Option<Array2<f32>>,
    m_recurrent_kernel_i: Option<Array2<f32>>,
    v_recurrent_kernel_i: Option<Array2<f32>>,
    m_recurrent_kernel_f: Option<Array2<f32>>,
    v_recurrent_kernel_f: Option<Array2<f32>>,
    m_recurrent_kernel_c: Option<Array2<f32>>,
    v_recurrent_kernel_c: Option<Array2<f32>>,
    m_recurrent_kernel_o: Option<Array2<f32>>,
    v_recurrent_kernel_o: Option<Array2<f32>>,
    m_bias_i: Option<Array2<f32>>,
    v_bias_i: Option<Array2<f32>>,
    m_bias_f: Option<Array2<f32>>,
    v_bias_f: Option<Array2<f32>>,
    m_bias_c: Option<Array2<f32>>,
    v_bias_c: Option<Array2<f32>>,
    m_bias_o: Option<Array2<f32>>,
    v_bias_o: Option<Array2<f32>>,

    cache_kernel_i: Option<Array2<f32>>,
    cache_kernel_f: Option<Array2<f32>>,
    cache_kernel_c: Option<Array2<f32>>,
    cache_kernel_o: Option<Array2<f32>>,
    cache_recurrent_kernel_i: Option<Array2<f32>>,
    cache_recurrent_kernel_f: Option<Array2<f32>>,
    cache_recurrent_kernel_c: Option<Array2<f32>>,
    cache_recurrent_kernel_o: Option<Array2<f32>>,
    cache_bias_i: Option<Array2<f32>>,
    cache_bias_f: Option<Array2<f32>>,
    cache_bias_c: Option<Array2<f32>>,
    cache_bias_o: Option<Array2<f32>>,

    activation: Activation,
    cell_activated_cache: Option<Vec<Array2<f32>>>,
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
        fn rand_mat(r: usize, c: usize) -> Array2<f32> {
            Array::random((r, c), Uniform::new(-0.05, 0.05))
        }
        let z = Array2::zeros((1, units));
        LSTM {
            input_dim,
            units,
            kernel_i: rand_mat(input_dim, units),
            kernel_f: rand_mat(input_dim, units),
            kernel_c: rand_mat(input_dim, units),
            kernel_o: rand_mat(input_dim, units),
            recurrent_kernel_i: rand_mat(units, units),
            recurrent_kernel_f: rand_mat(units, units),
            recurrent_kernel_c: rand_mat(units, units),
            recurrent_kernel_o: rand_mat(units, units),
            bias_i: z.clone(),
            bias_f: z.clone(),
            bias_c: z.clone(),
            bias_o: z,
            input_cache: None,
            hidden_cache: None,
            cell_cache: None,
            gate_i_cache: None,
            gate_f_cache: None,
            gate_c_cache: None,
            gate_o_cache: None,
            grad_kernel_i: None,
            grad_kernel_f: None,
            grad_kernel_c: None,
            grad_kernel_o: None,
            grad_recurrent_kernel_i: None,
            grad_recurrent_kernel_f: None,
            grad_recurrent_kernel_c: None,
            grad_recurrent_kernel_o: None,
            grad_bias_i: None,
            grad_bias_f: None,
            grad_bias_c: None,
            grad_bias_o: None,
            m_kernel_i: None,
            v_kernel_i: None,
            m_kernel_f: None,
            v_kernel_f: None,
            m_kernel_c: None,
            v_kernel_c: None,
            m_kernel_o: None,
            v_kernel_o: None,
            m_recurrent_kernel_i: None,
            v_recurrent_kernel_i: None,
            m_recurrent_kernel_f: None,
            v_recurrent_kernel_f: None,
            m_recurrent_kernel_c: None,
            v_recurrent_kernel_c: None,
            m_recurrent_kernel_o: None,
            v_recurrent_kernel_o: None,
            m_bias_i: None,
            v_bias_i: None,
            m_bias_f: None,
            v_bias_f: None,
            m_bias_c: None,
            v_bias_c: None,
            m_bias_o: None,
            v_bias_o: None,
            cache_kernel_i: None,
            cache_kernel_f: None,
            cache_kernel_c: None,
            cache_kernel_o: None,
            cache_recurrent_kernel_i: None,
            cache_recurrent_kernel_f: None,
            cache_recurrent_kernel_c: None,
            cache_recurrent_kernel_o: None,
            cache_bias_i: None,
            cache_bias_f: None,
            cache_bias_c: None,
            cache_bias_o: None,
            activation,
            cell_activated_cache: None,
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
        let mut gis = Vec::with_capacity(timesteps);
        let mut gfs = Vec::with_capacity(timesteps);
        let mut gcs = Vec::with_capacity(timesteps);
        let mut gos = Vec::with_capacity(timesteps);
        let mut c_activateds = Vec::with_capacity(timesteps);

        let mut h_t = Array2::<f32>::zeros((batch, self.units));
        let mut c_t = Array2::<f32>::zeros((batch, self.units));
        hs.push(h_t.clone());
        cs.push(c_t.clone());

        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned();

            // Calculate gate intermediate values
            let i_pre = x_t.dot(&self.kernel_i) + h_t.dot(&self.recurrent_kernel_i) + &self.bias_i;
            let f_pre = x_t.dot(&self.kernel_f) + h_t.dot(&self.recurrent_kernel_f) + &self.bias_f;
            let c_pre = x_t.dot(&self.kernel_c) + h_t.dot(&self.recurrent_kernel_c) + &self.bias_c;
            let o_pre = x_t.dot(&self.kernel_o) + h_t.dot(&self.recurrent_kernel_o) + &self.bias_o;

            // Apply activation functions
            let i_t = Activation::apply_activation(&i_pre, &self.activation);
            let f_t = Activation::apply_activation(&f_pre, &self.activation);
            let c_bar = Activation::apply_activation(&c_pre, &self.activation);
            let o_t = Activation::apply_activation(&o_pre, &self.activation);

            // Update cell state and hidden state
            c_t = &f_t * &c_t + &i_t * &c_bar;

            // 计算激活后的cell状态并缓存
            let c_activated = Activation::apply_activation(&c_t, &self.activation);
            c_activateds.push(c_activated.clone());

            let h_new = &o_t * &c_activated;

            // Cache
            gis.push(i_t.clone());
            gfs.push(f_t.clone());
            gcs.push(c_bar.clone());
            gos.push(o_t.clone());
            hs.push(h_new.clone());
            cs.push(c_t.clone());
            h_t = h_new;
        }

        self.hidden_cache = Some(hs);
        self.cell_cache = Some(cs);
        self.gate_i_cache = Some(gis);
        self.gate_f_cache = Some(gfs);
        self.gate_c_cache = Some(gcs);
        self.gate_o_cache = Some(gos);
        self.cell_activated_cache = Some(c_activateds);

        h_t.into_dyn()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let mut grad_h = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let x3 = match self.input_cache.take() {
            Some(x3) => x3,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };
        let hs = match self.hidden_cache.take() {
            Some(hs) => hs,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };
        let cs = match self.cell_cache.take() {
            Some(cs) => cs,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };
        let gis = match self.gate_i_cache.take() {
            Some(gis) => gis,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };
        let gfs = match self.gate_f_cache.take() {
            Some(gfs) => gfs,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };
        let gcs = match self.gate_c_cache.take() {
            Some(gcs) => gcs,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };
        let gos = match self.gate_o_cache.take() {
            Some(gos) => gos,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };

        let c_activateds = match self.cell_activated_cache.take() {
            Some(c_activateds) => c_activateds,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run".to_string(),
                ));
            }
        };

        // Process the gradient of the last time step, if it's softmax, special handling is needed
        if self.activation == Activation::Softmax {
            grad_h = Activation::softmax_backward(&hs[hs.len() - 1], &grad_h);
        }

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

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
            // f
            let d_f = &d_c * c_prev * Activation::activation_derivative(f_t, &self.activation);
            // i
            let d_i = &d_c * c_bar * Activation::activation_derivative(i_t, &self.activation);
            // c_bar
            let d_cbar = &d_c * i_t * Activation::activation_derivative(c_bar, &self.activation);

            // accumulate
            grad_kernel_i = grad_kernel_i + &x_t.t().dot(&d_i);
            grad_rk_i = grad_rk_i + &h_prev.t().dot(&d_i);
            grad_b_i = grad_b_i + &d_i.sum_axis(Axis(0)).insert_axis(Axis(0));

            grad_kernel_f = grad_kernel_f + &x_t.t().dot(&d_f);
            grad_rk_f = grad_rk_f + &h_prev.t().dot(&d_f);
            grad_b_f = grad_b_f + &d_f.sum_axis(Axis(0)).insert_axis(Axis(0));

            grad_kernel_c = grad_kernel_c + &x_t.t().dot(&d_cbar);
            grad_rk_c = grad_rk_c + &h_prev.t().dot(&d_cbar);
            grad_b_c = grad_b_c + &d_cbar.sum_axis(Axis(0)).insert_axis(Axis(0));

            grad_kernel_o = grad_kernel_o + &x_t.t().dot(&d_o);
            grad_rk_o = grad_rk_o + &h_prev.t().dot(&d_o);
            grad_b_o = grad_b_o + &d_o.sum_axis(Axis(0)).insert_axis(Axis(0));

            // propagate to x and h_prev and c_prev
            let dx = d_i.dot(&self.kernel_i.t())
                + d_f.dot(&self.kernel_f.t())
                + d_cbar.dot(&self.kernel_c.t())
                + d_o.dot(&self.kernel_o.t());
            grad_x3.index_axis_mut(Axis(1), t).assign(&dx);

            grad_h = d_i.dot(&self.recurrent_kernel_i.t())
                + d_f.dot(&self.recurrent_kernel_f.t())
                + d_cbar.dot(&self.recurrent_kernel_c.t())
                + d_o.dot(&self.recurrent_kernel_o.t());
            grad_c = &d_c * f_t;
        }

        self.grad_kernel_i = Some(grad_kernel_i);
        self.grad_kernel_f = Some(grad_kernel_f);
        self.grad_kernel_c = Some(grad_kernel_c);
        self.grad_kernel_o = Some(grad_kernel_o);
        self.grad_recurrent_kernel_i = Some(grad_rk_i);
        self.grad_recurrent_kernel_f = Some(grad_rk_f);
        self.grad_recurrent_kernel_c = Some(grad_rk_c);
        self.grad_recurrent_kernel_o = Some(grad_rk_o);
        self.grad_bias_i = Some(grad_b_i);
        self.grad_bias_f = Some(grad_b_f);
        self.grad_bias_c = Some(grad_b_c);
        self.grad_bias_o = Some(grad_b_o);

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

    fn update_parameters(&mut self, lr: f32) {
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel_i,
            &self.grad_recurrent_kernel_i,
            &self.grad_bias_i,
        ) {
            self.kernel_i = &self.kernel_i - &(lr * gk);
            self.recurrent_kernel_i = &self.recurrent_kernel_i - &(lr * grk);
            self.bias_i = &self.bias_i - &(lr * gb);
        }
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel_f,
            &self.grad_recurrent_kernel_f,
            &self.grad_bias_f,
        ) {
            self.kernel_f = &self.kernel_f - &(lr * gk);
            self.recurrent_kernel_f = &self.recurrent_kernel_f - &(lr * grk);
            self.bias_f = &self.bias_f - &(lr * gb);
        }
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel_c,
            &self.grad_recurrent_kernel_c,
            &self.grad_bias_c,
        ) {
            self.kernel_c = &self.kernel_c - &(lr * gk);
            self.recurrent_kernel_c = &self.recurrent_kernel_c - &(lr * grk);
            self.bias_c = &self.bias_c - &(lr * gb);
        }
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel_o,
            &self.grad_recurrent_kernel_o,
            &self.grad_bias_o,
        ) {
            self.kernel_o = &self.kernel_o - &(lr * gk);
            self.recurrent_kernel_o = &self.recurrent_kernel_o - &(lr * grk);
            self.bias_o = &self.bias_o - &(lr * gb);
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Initialize first moment (m_*) and second moment (v_*) to zeros
        if self.m_kernel_i.is_none() {
            let dk = (self.input_dim, self.units);
            let dr = (self.units, self.units);
            let db = (1, self.units);
            self.m_kernel_i = Some(Array2::zeros(dk));
            self.v_kernel_i = Some(Array2::zeros(dk));
            self.m_kernel_f = Some(Array2::zeros(dk));
            self.v_kernel_f = Some(Array2::zeros(dk));
            self.m_kernel_c = Some(Array2::zeros(dk));
            self.v_kernel_c = Some(Array2::zeros(dk));
            self.m_kernel_o = Some(Array2::zeros(dk));
            self.v_kernel_o = Some(Array2::zeros(dk));

            self.m_recurrent_kernel_i = Some(Array2::zeros(dr));
            self.v_recurrent_kernel_i = Some(Array2::zeros(dr));
            self.m_recurrent_kernel_f = Some(Array2::zeros(dr));
            self.v_recurrent_kernel_f = Some(Array2::zeros(dr));
            self.m_recurrent_kernel_c = Some(Array2::zeros(dr));
            self.v_recurrent_kernel_c = Some(Array2::zeros(dr));
            self.m_recurrent_kernel_o = Some(Array2::zeros(dr));
            self.v_recurrent_kernel_o = Some(Array2::zeros(dr));

            self.m_bias_i = Some(Array2::zeros(db));
            self.v_bias_i = Some(Array2::zeros(db));
            self.m_bias_f = Some(Array2::zeros(db));
            self.v_bias_f = Some(Array2::zeros(db));
            self.m_bias_c = Some(Array2::zeros(db));
            self.v_bias_c = Some(Array2::zeros(db));
            self.m_bias_o = Some(Array2::zeros(db));
            self.v_bias_o = Some(Array2::zeros(db));
        }

        // Helper function to update a single gate
        fn update_gate(
            w: &mut Array2<f32>,  // kernel
            rk: &mut Array2<f32>, // recurrent kernel
            b: &mut Array2<f32>,  // bias
            gw: &Array2<f32>,
            gr: &Array2<f32>,
            gb: &Array2<f32>,
            m_w: &mut Array2<f32>,
            v_w: &mut Array2<f32>,
            m_r: &mut Array2<f32>,
            v_r: &mut Array2<f32>,
            m_b: &mut Array2<f32>,
            v_b: &mut Array2<f32>,
            lr: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            t: u64,
        ) {
            // Update first and second moments
            *m_w = m_w.mapv(|x| x * beta1) + &(gw * (1.0 - beta1));
            *v_w = v_w.mapv(|x| x * beta2) + &(gw.mapv(|x| x * x) * (1.0 - beta2));
            *m_r = m_r.mapv(|x| x * beta1) + &(gr * (1.0 - beta1));
            *v_r = v_r.mapv(|x| x * beta2) + &(gr.mapv(|x| x * x) * (1.0 - beta2));
            *m_b = m_b.mapv(|x| x * beta1) + &(gb * (1.0 - beta1));
            *v_b = v_b.mapv(|x| x * beta2) + &(gb.mapv(|x| x * x) * (1.0 - beta2));

            // Bias correction
            let m_hat_w = m_w.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
            let v_hat_w = v_w.mapv(|x| x / (1.0 - beta2.powi(t as i32)));
            let m_hat_r = m_r.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
            let v_hat_r = v_r.mapv(|x| x / (1.0 - beta2.powi(t as i32)));
            let m_hat_b = m_b.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
            let v_hat_b = v_b.mapv(|x| x / (1.0 - beta2.powi(t as i32)));

            // Update parameters
            *w = &*w - &(lr * &m_hat_w / &(v_hat_w.mapv(f32::sqrt) + epsilon));
            *rk = &*rk - &(lr * &m_hat_r / &(v_hat_r.mapv(f32::sqrt) + epsilon));
            *b = &*b - &(lr * &m_hat_b / &(v_hat_b.mapv(f32::sqrt) + epsilon));
        }

        // Update each of the four gates separately
        update_gate(
            &mut self.kernel_i,
            &mut self.recurrent_kernel_i,
            &mut self.bias_i,
            self.grad_kernel_i.as_ref().unwrap(),
            self.grad_recurrent_kernel_i.as_ref().unwrap(),
            self.grad_bias_i.as_ref().unwrap(),
            self.m_kernel_i.as_mut().unwrap(),
            self.v_kernel_i.as_mut().unwrap(),
            self.m_recurrent_kernel_i.as_mut().unwrap(),
            self.v_recurrent_kernel_i.as_mut().unwrap(),
            self.m_bias_i.as_mut().unwrap(),
            self.v_bias_i.as_mut().unwrap(),
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate(
            &mut self.kernel_f,
            &mut self.recurrent_kernel_f,
            &mut self.bias_f,
            self.grad_kernel_f.as_ref().unwrap(),
            self.grad_recurrent_kernel_f.as_ref().unwrap(),
            self.grad_bias_f.as_ref().unwrap(),
            self.m_kernel_f.as_mut().unwrap(),
            self.v_kernel_f.as_mut().unwrap(),
            self.m_recurrent_kernel_f.as_mut().unwrap(),
            self.v_recurrent_kernel_f.as_mut().unwrap(),
            self.m_bias_f.as_mut().unwrap(),
            self.v_bias_f.as_mut().unwrap(),
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate(
            &mut self.kernel_c,
            &mut self.recurrent_kernel_c,
            &mut self.bias_c,
            self.grad_kernel_c.as_ref().unwrap(),
            self.grad_recurrent_kernel_c.as_ref().unwrap(),
            self.grad_bias_c.as_ref().unwrap(),
            self.m_kernel_c.as_mut().unwrap(),
            self.v_kernel_c.as_mut().unwrap(),
            self.m_recurrent_kernel_c.as_mut().unwrap(),
            self.v_recurrent_kernel_c.as_mut().unwrap(),
            self.m_bias_c.as_mut().unwrap(),
            self.v_bias_c.as_mut().unwrap(),
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
        update_gate(
            &mut self.kernel_o,
            &mut self.recurrent_kernel_o,
            &mut self.bias_o,
            self.grad_kernel_o.as_ref().unwrap(),
            self.grad_recurrent_kernel_o.as_ref().unwrap(),
            self.grad_bias_o.as_ref().unwrap(),
            self.m_kernel_o.as_mut().unwrap(),
            self.v_kernel_o.as_mut().unwrap(),
            self.m_recurrent_kernel_o.as_mut().unwrap(),
            self.v_recurrent_kernel_o.as_mut().unwrap(),
            self.m_bias_o.as_mut().unwrap(),
            self.v_bias_o.as_mut().unwrap(),
            lr,
            beta1,
            beta2,
            epsilon,
            t,
        );
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        // Initialize cache_* to zeros
        if self.cache_kernel_i.is_none() {
            let dk = (self.input_dim, self.units);
            let dr = (self.units, self.units);
            let db = (1, self.units);
            self.cache_kernel_i = Some(Array2::zeros(dk));
            self.cache_kernel_f = Some(Array2::zeros(dk));
            self.cache_kernel_c = Some(Array2::zeros(dk));
            self.cache_kernel_o = Some(Array2::zeros(dk));
            self.cache_recurrent_kernel_i = Some(Array2::zeros(dr));
            self.cache_recurrent_kernel_f = Some(Array2::zeros(dr));
            self.cache_recurrent_kernel_c = Some(Array2::zeros(dr));
            self.cache_recurrent_kernel_o = Some(Array2::zeros(dr));
            self.cache_bias_i = Some(Array2::zeros(db));
            self.cache_bias_f = Some(Array2::zeros(db));
            self.cache_bias_c = Some(Array2::zeros(db));
            self.cache_bias_o = Some(Array2::zeros(db));
        }

        // Helper function for update
        fn update_gate_rms(
            w: &mut Array2<f32>,
            rk: &mut Array2<f32>,
            b: &mut Array2<f32>,
            gw: &Array2<f32>,
            gr: &Array2<f32>,
            gb: &Array2<f32>,
            cache_w: &mut Array2<f32>,
            cache_r: &mut Array2<f32>,
            cache_b: &mut Array2<f32>,
            lr: f32,
            rho: f32,
            epsilon: f32,
        ) {
            *cache_w = cache_w.mapv(|x| x * rho) + &(gw.mapv(|x| x * x) * (1.0 - rho));
            *cache_r = cache_r.mapv(|x| x * rho) + &(gr.mapv(|x| x * x) * (1.0 - rho));
            *cache_b = cache_b.mapv(|x| x * rho) + &(gb.mapv(|x| x * x) * (1.0 - rho));

            *w = &*w - &(lr * gw / &(cache_w.mapv(f32::sqrt) + epsilon));
            *rk = &*rk - &(lr * gr / &(cache_r.mapv(f32::sqrt) + epsilon));
            *b = &*b - &(lr * gb / &(cache_b.mapv(f32::sqrt) + epsilon));
        }

        update_gate_rms(
            &mut self.kernel_i,
            &mut self.recurrent_kernel_i,
            &mut self.bias_i,
            self.grad_kernel_i.as_ref().unwrap(),
            self.grad_recurrent_kernel_i.as_ref().unwrap(),
            self.grad_bias_i.as_ref().unwrap(),
            self.cache_kernel_i.as_mut().unwrap(),
            self.cache_recurrent_kernel_i.as_mut().unwrap(),
            self.cache_bias_i.as_mut().unwrap(),
            lr,
            rho,
            epsilon,
        );
        update_gate_rms(
            &mut self.kernel_f,
            &mut self.recurrent_kernel_f,
            &mut self.bias_f,
            self.grad_kernel_f.as_ref().unwrap(),
            self.grad_recurrent_kernel_f.as_ref().unwrap(),
            self.grad_bias_f.as_ref().unwrap(),
            self.cache_kernel_f.as_mut().unwrap(),
            self.cache_recurrent_kernel_f.as_mut().unwrap(),
            self.cache_bias_f.as_mut().unwrap(),
            lr,
            rho,
            epsilon,
        );
        update_gate_rms(
            &mut self.kernel_c,
            &mut self.recurrent_kernel_c,
            &mut self.bias_c,
            self.grad_kernel_c.as_ref().unwrap(),
            self.grad_recurrent_kernel_c.as_ref().unwrap(),
            self.grad_bias_c.as_ref().unwrap(),
            self.cache_kernel_c.as_mut().unwrap(),
            self.cache_recurrent_kernel_c.as_mut().unwrap(),
            self.cache_bias_c.as_mut().unwrap(),
            lr,
            rho,
            epsilon,
        );
        update_gate_rms(
            &mut self.kernel_o,
            &mut self.recurrent_kernel_o,
            &mut self.bias_o,
            self.grad_kernel_o.as_ref().unwrap(),
            self.grad_recurrent_kernel_o.as_ref().unwrap(),
            self.grad_bias_o.as_ref().unwrap(),
            self.cache_kernel_o.as_mut().unwrap(),
            self.cache_recurrent_kernel_o.as_mut().unwrap(),
            self.cache_bias_o.as_mut().unwrap(),
            lr,
            rho,
            epsilon,
        );
    }
}
