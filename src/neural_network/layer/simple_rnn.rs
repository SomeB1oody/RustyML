use crate::ModelError;
use crate::neural_network::optimizer::*;
use crate::neural_network::{Activation, Layer, Tensor};
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation.
///
/// SimpleRNN applies a standard recurrent operation where the output from the previous
/// timestep is used as additional input to the current timestep. This implementation
/// supports various activation functions, with tanh as the default.
///
/// # Dimensions
///
/// - Input shape: (batch_size, timesteps, input_dim)
/// - Output shape: (batch_size, units)
///
/// # Fields
/// ## Core fields
/// - `input_dim` - Number of input features
/// - `units` - Number of output units (neurons)
/// - `kernel` - Weight matrix connecting inputs to the recurrent layer (shape: input_dim, units)
/// - `recurrent_kernel` - Weight matrix connecting previous hidden states to the current state (shape: units, units)
/// - `bias` - Bias vector for the layer (shape: 1, units)
/// - `activation` - Activation function to use (default: tanh)
///
/// ## Cache
/// - `input_cache` - Cache of input tensors from forward pass (shape: batch, timesteps, input_dim)
/// - `hidden_state_cache` - Cache of hidden states from forward pass (length = timesteps+1)
///
/// ## Gradients
/// - `grad_kernel` - Gradient of the kernel weights
/// - `grad_recurrent_kernel` - Gradient of the recurrent kernel weights
/// - `grad_bias` - Gradient of the bias
///
/// ## Adam states
/// - `adam_states` - AdamStates containing moment vectors for all parameters
///
/// ## RMSprop cache
/// - `rmsprop_cache` - RMSprop cache for optimization
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
/// // Build model: one SimpleRnn layer with tanh activation
/// let mut model = Sequential::new();
/// model
/// .add(SimpleRNN::new(4, 3, Activation::Tanh))
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
pub struct SimpleRNN {
    /// Number of input features
    input_dim: usize,
    /// Number of output units (neurons)
    units: usize,
    /// Weight matrix connecting inputs to the recurrent layer (shape: input_dim, units)
    kernel: Array2<f32>,
    /// Weight matrix connecting previous hidden states to the current state (shape: units, units)
    recurrent_kernel: Array2<f32>,
    /// Bias vector for the layer (shape: 1, units)
    bias: Array2<f32>,

    // Cache
    /// Cache of input tensors from forward pass (shape: batch, timesteps, input_dim)
    input_cache: Option<Array3<f32>>,
    /// Cache of hidden states from forward pass (length = timesteps+1)
    hidden_state_cache: Option<Vec<Array2<f32>>>,

    // Gradients
    /// Gradient of the kernel weights
    grad_kernel: Option<Array2<f32>>,
    /// Gradient of the recurrent kernel weights
    grad_recurrent_kernel: Option<Array2<f32>>,
    /// Gradient of the bias
    grad_bias: Option<Array2<f32>>,

    /// AdamStates containing moment vectors for all parameters
    adam_states: Option<AdamStates>,

    /// RMSprop cache for optimization
    rmsprop_cache: Option<RMSpropCache>,

    /// Activation function to use
    activation: Activation,
}

impl SimpleRNN {
    /// Creates a new SimpleRNN layer with the specified dimensions and activation function.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - The size of each input sample
    /// - `units` - The dimensionality of the output space
    /// - `activation` - The activation function to use
    ///
    /// # Returns
    ///
    /// * `Self` - A new SimpleRNN instance with the specified activation
    pub fn new(input_dim: usize, units: usize, activation: Activation) -> Self {
        let kernel = Array::random((input_dim, units), Uniform::new(-0.05, 0.05));
        let recurrent_kernel = Array::random((units, units), Uniform::new(-0.05, 0.05));
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
            adam_states: None,
            rmsprop_cache: None,
            activation,
        }
    }
}

impl Layer for SimpleRNN {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Input shape=(batch, timesteps, input_dim)
        let x3 = input.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.clone());

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut hs = Vec::with_capacity(timesteps + 1);
        hs.push(h_prev.clone());

        // Time steps loop is still sequential
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned(); // (batch, input_dim)

            // Matrix multiplication here can be parallelized internally
            let z = x_t.dot(&self.kernel) + h_prev.dot(&self.recurrent_kernel) + &self.bias;

            let h_t = Activation::apply_activation(&z, &self.activation);
            hs.push(h_t.clone());
            h_prev = h_t;
        }
        self.hidden_state_cache = Some(hs);
        h_prev.into_dyn() // Return hidden state of the last timestep
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

        let mut grad_k = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_rk = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_b = Array2::<f32>::zeros((1, self.units));
        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        let mut grad_h = grad_h_t;
        // BPTT
        for t in (0..timesteps).rev() {
            let h_t = hs[t + 1].clone();
            let h_tm1 = hs[t].clone();
            let d_z = if self.activation == Activation::Softmax {
                Activation::softmax_backward(&h_t, &grad_h)
            } else {
                let d_act = Activation::activation_derivative(&h_t, &self.activation);
                d_act * &grad_h
            };
            let x_t = x3.index_axis(Axis(1), t).to_owned();
            grad_k = grad_k + &x_t.t().dot(&d_z);
            grad_rk = grad_rk + &h_tm1.t().dot(&d_z);
            let db = d_z.sum_axis(Axis(0)).insert_axis(Axis(0));
            grad_b = grad_b + &db;
            let dx = d_z.dot(&self.kernel.t());
            grad_x3.index_axis_mut(Axis(1), t).assign(&dx);
            grad_h = d_z.dot(&self.recurrent_kernel.t());
        }

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

    fn param_count(&self) -> usize {
        self.input_dim * self.units + self.units * self.units + self.units
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
        if self.adam_states.is_none() {
            let dims_k = (self.input_dim, self.units);
            let dims_r = (self.units, self.units);
            let dims_b = (1, self.units);

            self.adam_states = Some(AdamStates::new(dims_k, Some(dims_r), dims_b));
        }

        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            let adam_states = self.adam_states.as_mut().unwrap();
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
            if self.rmsprop_cache.is_none() {
                self.rmsprop_cache = Some(RMSpropCache::new(
                    (self.input_dim, self.units),
                    Some((self.units, self.units)),
                    (1, self.units),
                ));
            }

            if let Some(ref mut cache) = self.rmsprop_cache {
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
}
