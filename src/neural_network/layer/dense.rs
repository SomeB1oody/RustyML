use crate::ModelError;
use crate::neural_network::layer::{DenseLayerWeight, LayerWeight};
use crate::neural_network::optimizer::*;
use crate::neural_network::{Activation, Layer, Tensor};
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;

/// Dense (Fully Connected) layer implementation for neural networks.
///
/// This layer performs a linear transformation of the input data using a weight matrix and bias vector,
/// optionally followed by an activation function. The transformation is defined as: output = activation(input * weights + bias).
///
/// The Dense layer automatically initializes weights using Xavier/Glorot initialization to maintain proper variance
/// across network layers, helping prevent vanishing/exploding gradients problems. Bias values are initialized to zeros.
///
/// During training, this layer stores necessary intermediate values for backpropagation and supports
/// multiple optimization algorithms including SGD, Adam, and RMSprop.
///
/// # Dimensions
///
/// - Input shape: (batch_size, input_dim)
/// - Output shape: (batch_size, output_dim)
///
/// # Fields
/// ## Core fields
/// - `input_dim` - Input dimension size
/// - `output_dim` - Output dimension size
/// - `weights` - Weight matrix with shape (input_dim, output_dim)
/// - `bias` - Bias vector with shape (1, output_dim)
/// - `activation` - Activation function for the layer
/// - `activation_output` - Cached output after activation for use in backward pass
///
/// ## Cache
/// - `input_cache` - Cache of the input from forward pass for use in backward pass
/// - `grad_weights` - Stored weight gradients
/// - `grad_bias` - Stored bias gradients
/// - `optimizer_cache` - Cache for optimizer
///
/// # Example
/// ```rust
/// use ndarray::Array;
/// use rustyml::prelude::*;
///
/// // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 3, Activation::ReLU)).add(Dense::new(3, 1, Activation::ReLU));
/// model.compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // Print model structure (summary)
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("Prediction results: {:?}", prediction);
/// ```
pub struct Dense {
    /// Input dimension size
    input_dim: usize,
    /// Output dimension size
    output_dim: usize,
    /// Weight matrix with shape (input_dim, output_dim)
    weights: Array2<f32>,
    /// Bias vector with shape (1, output_dim)
    bias: Array2<f32>,
    /// Cache of the input from forward pass for use in backward pass
    input_cache: Option<Array2<f32>>,
    /// Stored weight gradients
    grad_weights: Option<Array2<f32>>,
    /// Stored bias gradients
    grad_bias: Option<Array2<f32>>,
    /// Cache for optimizer
    optimizer_cache: OptimizerCache,
    /// Activation function for the layer
    activation: Activation,
    /// Cached output after activation for use in backward pass
    activation_output: Option<Array2<f32>>,
}

impl Dense {
    /// Creates a new dense layer without activation function.
    ///
    /// # Returns
    ///
    /// * `Self` - A new Dense layer instance with specified dimensions
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
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
            activation,
            activation_output: None,
            optimizer_cache: OptimizerCache {
                adam_states: None,
                rmsprop_cache: None,
            },
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Input shape is [batch_size, input_dim]
        let input_2d = input.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        self.input_cache = Some(input_2d.clone());

        // Use dot operation (ndarray itself will use parallel computation when rayon feature is enabled)
        let z = input_2d.dot(&self.weights) + &self.bias;

        let a = Activation::apply_activation(&z, &self.activation);
        self.activation_output = Some(a.clone());
        a.into_dyn()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Convert gradient to 2D array with shape [batch_size, output_dim]
        let mut grad_upstream = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        if self.activation == Activation::Softmax {
            let a = match self.activation_output.take() {
                Some(a) => a,
                None => Err(ModelError::ProcessingError(String::from(
                    "Forward pass has not been run",
                )))?,
            };
            grad_upstream = Activation::softmax_backward(&a, &grad_upstream);
        } else {
            let a = match self.activation_output.take() {
                Some(a) => a,
                None => Err(ModelError::ProcessingError(String::from(
                    "Forward pass has not been run",
                )))?,
            };

            // calculation of element-wise multiplication of activation function derivative and gradient
            let deriv = Activation::activation_derivative(&a, &self.activation);
            let mut result = deriv.clone();

            // Use parallel iterator for element-wise multiplication
            result.par_mapv_inplace(|v| v * grad_upstream[[0, 0]]);
            for ((i, j), val) in result.indexed_iter_mut() {
                *val = deriv[[i, j]] * grad_upstream[[i, j]];
            }

            grad_upstream = result;
        }

        // Get input cache
        let input = match self.input_cache.take() {
            Some(input) => input,
            None => Err(ModelError::ProcessingError(String::from(
                "Forward pass has not been run",
            )))?,
        };

        // Use parallel matrix multiplication to calculate weight gradients for large matrices
        let batch_size = input.nrows();
        let input_dim = input.ncols();
        let output_dim = grad_upstream.ncols();

        // Create weight gradient matrix
        let mut grad_w = Array2::<f32>::zeros((input_dim, output_dim));

        // Parallel calculation of weight gradients
        grad_w
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..output_dim {
                    let mut sum = 0.0;
                    for k in 0..batch_size {
                        sum += input[[k, i]] * grad_upstream[[k, j]];
                    }
                    row[j] = sum;
                }
            });

        // Calculate bias gradients
        let grad_b = grad_upstream.sum_axis(Axis(0)).insert_axis(Axis(0));

        self.grad_weights = Some(grad_w);
        self.grad_bias = Some(grad_b);

        // Calculate gradients to be passed to previous layer
        // Use parallel matrix multiplication for large matrices
        let mut grad_input = Array2::<f32>::zeros((batch_size, input_dim));

        // Parallel calculation of input gradients
        grad_input
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..input_dim {
                    let mut sum = 0.0;
                    for k in 0..output_dim {
                        sum += grad_upstream[[i, k]] * self.weights[[j, k]];
                    }
                    row[j] = sum;
                }
            });

        Ok(grad_input.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn output_shape(&self) -> String {
        // Returns only (None, output_dim)
        format!("(None, {})", self.output_dim)
    }

    fn param_count(&self) -> usize {
        // Parameter count = number of weight parameters + number of bias parameters
        self.input_dim * self.output_dim + self.output_dim
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            SGD::update_sgd_parameters(
                self.weights.as_slice_mut().unwrap(),
                grad_w.as_slice().unwrap(),
                self.bias.as_slice_mut().unwrap(),
                grad_b.as_slice().unwrap(),
                lr,
            )
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Initialize Adam states (if not already initialized)
        if self.optimizer_cache.adam_states.is_none() {
            let dims_w = (self.input_dim, self.output_dim);
            let dims_b = (1, self.output_dim);

            self.optimizer_cache.adam_states = Some(AdamStates::new(
                dims_w, None, // No recurrent weights
                dims_b,
            ));
        }

        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            let adam_states = self.optimizer_cache.adam_states.as_mut().unwrap();
            let (w_update, _, b_update) = adam_states.update_parameter(
                grad_w, None, // No recurrent weights
                grad_b, beta1, beta2, epsilon, t, lr,
            );

            // Apply updates
            self.weights = &self.weights - &w_update;
            self.bias = &self.bias - &b_update;
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            // Initialize RMSprop cache if it doesn't exist
            if self.optimizer_cache.rmsprop_cache.is_none() {
                self.optimizer_cache.rmsprop_cache = Some(RMSpropCache::new(
                    (self.input_dim, self.output_dim),
                    None,
                    (1, self.output_dim),
                ));
            }

            if let Some(ref mut cache) = self.optimizer_cache.rmsprop_cache {
                cache.update_parameters(
                    &mut self.weights,
                    None,
                    &mut self.bias,
                    grad_w,
                    None,
                    grad_b,
                    rho,
                    lr,
                    epsilon,
                );
            }
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Dense(DenseLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
