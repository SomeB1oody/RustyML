use super::*;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Dense layer struct: automatically initializes weights and bias,
/// and calculates and stores gradients during backward pass.
pub struct Dense {
    /// Input dimension size
    pub input_dim: usize,
    /// Output dimension size
    pub output_dim: usize,
    /// Weight matrix with shape (input_dim, output_dim)
    pub weights: Array2<f32>,
    /// Bias vector with shape (1, output_dim)
    pub bias: Array2<f32>,
    /// Cache of the input from forward pass for use in backward pass
    pub input_cache: Option<Array2<f32>>,
    /// Stored weight gradients
    pub grad_weights: Option<Array2<f32>>,
    /// Stored bias gradients
    pub grad_bias: Option<Array2<f32>>,
    /// Adam optimizer state: first moment for weights
    pub m_weights: Option<Array2<f32>>,
    /// Adam optimizer state: second moment for weights
    pub v_weights: Option<Array2<f32>>,
    /// Adam optimizer state: first moment for bias
    pub m_bias: Option<Array2<f32>>,
    /// Adam optimizer state: second moment for bias
    pub v_bias: Option<Array2<f32>>,
    /// RMSprop optimizer cache for weights
    pub cache_weights: Option<Array2<f32>>,
    /// RMSprop optimizer cache for bias
    pub cache_bias: Option<Array2<f32>>,
}

impl Dense {
    /// Creates a new Dense layer with the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features
    /// * `output_dim` - Number of output features
    ///
    /// # Returns
    ///
    /// A new Dense layer with randomly initialized weights and zero bias
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights with uniform distribution [-0.05, 0.05], bias initialized to 0
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
    /// Performs forward propagation through the dense layer.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor after linear transformation
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Input shape is [batch_size, input_dim]
        let input_2d = input.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        // Cache the input for backpropagation gradient calculation
        self.input_cache = Some(input_2d.clone());
        // Compute linear transformation: output = input.dot(weights) + bias (using broadcasting for bias)
        let output = input_2d.dot(&self.weights) + &self.bias;
        output.into_dyn()
    }

    /// Performs backward propagation through the dense layer.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient tensor from the next layer with shape [batch_size, output_dim]
    ///
    /// # Returns
    ///
    /// Gradient tensor to be passed to the previous layer
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Convert gradient to 2D array with shape [batch_size, output_dim]
        let grad_output_2d = grad_output.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        // Get the input from cache with shape [batch_size, input_dim]
        let input = self.input_cache.take().expect("Forward must be called before backward");

        // Calculate gradient with respect to weights: dW = input.T.dot(grad_output)
        let grad_w = input.t().dot(&grad_output_2d);
        // Calculate gradient with respect to bias: db = sum(grad_output, axis=0), keep 2D shape
        let grad_b = grad_output_2d.sum_axis(Axis(0)).insert_axis(Axis(0));
        // Store the calculated gradients for parameter updates
        self.grad_weights = Some(grad_w);
        self.grad_bias = Some(grad_b);

        // Calculate gradient with respect to input: dX = grad_output.dot(weights.T)
        let grad_input = grad_output_2d.dot(&self.weights.t());
        grad_input.into_dyn()
    }

    /// Returns the type of this layer.
    ///
    /// # Returns
    ///
    /// String slice indicating the layer type
    fn layer_type(&self) -> &str {
        "Dense"
    }

    /// Returns a string describing the output shape of this layer.
    ///
    /// # Returns
    ///
    /// String representation of the output shape as "(None, output_dim)"
    fn output_shape(&self) -> String {
        // Returns only (None, output_dim)
        format!("(None, {})", self.output_dim)
    }

    /// Returns the total number of trainable parameters in the layer.
    ///
    /// # Returns
    ///
    /// Sum of all weights and bias parameters
    fn param_count(&self) -> usize {
        // Parameter count = number of weight parameters + number of bias parameters
        self.input_dim * self.output_dim + self.output_dim
    }

    /// Updates layer parameters using Stochastic Gradient Descent.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for parameter updates
    fn update_parameters_sgd(&mut self, lr: f32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            self.weights = &self.weights - &(lr * grad_w);
            self.bias = &self.bias - &(lr * grad_b);
        }
    }

    /// Updates layer parameters using the Adam optimization algorithm.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    /// * `t` - Current iteration count
    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Initialize Adam state if not already initialized
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

        let grad_w = self.grad_weights.as_ref().expect("Adam: grad_weights not calculated");
        let grad_b = self.grad_bias.as_ref().expect("Adam: grad_bias not calculated");

        // Update first moment (momentum)
        *m_w = m_w.mapv(|x| x * beta1) + &(grad_w * (1.0 - beta1));
        *m_b = m_b.mapv(|x| x * beta1) + &(grad_b * (1.0 - beta1));

        // Update second moment (accumulated squared gradients), using elementwise square
        *v_w = v_w.mapv(|x| x * beta2) + &(grad_w.mapv(|x| x * x) * (1.0 - beta2));
        *v_b = v_b.mapv(|x| x * beta2) + &(grad_b.mapv(|x| x * x) * (1.0 - beta2));

        // Calculate bias-corrected estimates
        let m_hat_w = m_w.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
        let m_hat_b = m_b.mapv(|x| x / (1.0 - beta1.powi(t as i32)));
        let v_hat_w = v_w.mapv(|x| x / (1.0 - beta2.powi(t as i32)));
        let v_hat_b = v_b.mapv(|x| x / (1.0 - beta2.powi(t as i32)));

        // Update parameters: w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
        self.weights = &self.weights - &(lr * &m_hat_w / &(v_hat_w.mapv(f32::sqrt) + epsilon));
        self.bias = &self.bias - &(lr * &m_hat_b / &(v_hat_b.mapv(f32::sqrt) + epsilon));
    }

    /// Updates layer parameters using the RMSprop optimization algorithm.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for parameter updates
    /// * `rho` - Decay rate for moving average of squared gradients
    /// * `epsilon` - Small constant for numerical stability
    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            // Initialize RMSprop caches with zeros if not already initialized
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