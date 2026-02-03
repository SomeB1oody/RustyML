use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::{DenseLayerWeight, LayerWeight};
use crate::neural_network::neural_network_trait::{ActivationLayer, Layer};
use crate::neural_network::optimizer::{
    OptimizerCache, ada_grad::AdaGradStates, adam::AdamStates, rms_prop::RMSpropCache, sgd::SGD,
};
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distr::Uniform;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Threshold for determining when to use parallel computation in dense layer operations.
/// When the total number of operations (input_dim * output_dim) is below this threshold,
/// sequential computation is used to avoid parallelization overhead.
const DENSE_PARALLEL_THRESHOLD: usize = 512;

/// Dense (Fully Connected) layer implementation for neural networks.
///
/// This layer performs a linear transformation using a weight matrix and bias vector, optionally
/// followed by an activation function: output = activation(input * weights + bias). Input shape
/// is (batch_size, input_dim) and output shape is (batch_size, output_dim).
///
/// Weights are initialized with Xavier/Glorot initialization and biases start at zeros. During
/// training, the layer stores intermediate values for backpropagation and supports multiple
/// optimization algorithms including SGD, Adam, and RMSprop.
///
/// # Fields
///
/// - `input_dim` - Input dimension size
/// - `output_dim` - Output dimension size
/// - `weights` - Weight matrix with shape (input_dim, output_dim)
/// - `bias` - Bias vector with shape (1, output_dim)
/// - `input_cache` - Cache of the input from forward pass for use in backward pass
/// - `grad_weights` - Stored weight gradients
/// - `grad_bias` - Stored bias gradients
/// - `optimizer_cache` - Cache for optimizer
/// - `activation` - Activation layer from activation_layer module
///
/// # Examples
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::{Dense, ReLU};
/// use rustyml::neural_network::optimizer::SGD;
/// use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
///
/// // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 3, ReLU::new()).unwrap())
///     .add(Dense::new(3, 1, ReLU::new()).unwrap());
/// model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
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
pub struct Dense<T: ActivationLayer> {
    input_dim: usize,
    output_dim: usize,
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_cache: Option<Array2<f32>>,
    grad_weights: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,
    optimizer_cache: OptimizerCache,
    activation: T,
}

impl<T: ActivationLayer> Dense<T> {
    /// Creates a new dense layer with an activation layer.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New `Dense` layer instance with initialized parameters
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_dim` or `units` is zero
    pub fn new(input_dim: usize, units: usize, activation: T) -> Result<Self, ModelError> {
        // Validate that dimensions are greater than zero
        if input_dim == 0 {
            return Err(ModelError::InputValidationError(
                "input_dim must be greater than 0".to_string(),
            ));
        }
        if units == 0 {
            return Err(ModelError::InputValidationError(
                "units must be greater than 0".to_string(),
            ));
        }

        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let weights = Array::random((input_dim, units), Uniform::new(-limit, limit).unwrap());
        let bias = Array::zeros((1, units));
        Ok(Self {
            input_dim,
            output_dim: units,
            weights,
            bias,
            input_cache: None,
            grad_weights: None,
            grad_bias: None,
            activation,
            optimizer_cache: OptimizerCache {
                adam_states: None,
                rmsprop_cache: None,
                ada_grad_cache: None,
            },
        })
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `weights` - Weight matrix with shape (input_dim, output_dim)
    /// - `bias` - Bias vector with shape (1, output_dim)
    pub fn set_weights(&mut self, weights: Array2<f32>, bias: Array2<f32>) {
        self.weights = weights;
        self.bias = bias;
    }
}

impl<T: ActivationLayer> Layer for Dense<T> {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 2D
        if input.ndim() != 2 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 2D".to_string(),
            ));
        }

        let input_2d = input.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        // Input shape is [batch_size, input_dim]
        self.input_cache = Some(input_2d.to_owned());

        // Use adaptive parallel/sequential matrix multiplication
        let total_ops = self.input_dim * self.output_dim;
        let z = if total_ops < DENSE_PARALLEL_THRESHOLD {
            // Sequential computation using dot
            input_2d.dot(&self.weights) + &self.bias
        } else {
            // Parallel computation using par_iter
            let batch_size = input_2d.nrows();
            let mut result = Array2::<f32>::zeros((batch_size, self.output_dim));

            result
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(input_2d.axis_iter(Axis(0)))
                .for_each(|(mut out_row, in_row)| {
                    for j in 0..self.output_dim {
                        let mut sum = self.bias[[0, j]];
                        for k in 0..self.input_dim {
                            sum += in_row[k] * self.weights[[k, j]];
                        }
                        out_row[j] = sum;
                    }
                });

            result
        };

        // Apply activation
        self.activation.forward(&z.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Apply activation backward pass if activation layer exists
        let grad_upstream = self.activation.backward(grad_output)?;

        // Convert gradient to 2D array with shape [batch_size, output_dim]
        let grad_upstream_2d = grad_upstream.into_dimensionality::<ndarray::Ix2>().unwrap();

        // Get input cache
        let input = self.input_cache.take().ok_or_else(|| {
            ModelError::ProcessingError(String::from("Forward pass has not been run"))
        })?;

        let total_ops = self.input_dim * self.output_dim;

        // Calculate weight gradients: grad_w = input^T * grad_upstream
        let grad_w = if total_ops < DENSE_PARALLEL_THRESHOLD {
            // Sequential computation using dot
            input.t().dot(&grad_upstream_2d)
        } else {
            // Parallel computation using par_iter
            let mut result = Array2::<f32>::zeros((self.input_dim, self.output_dim));

            result
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut out_col)| {
                    for i in 0..self.input_dim {
                        let mut sum = 0.0;
                        for k in 0..input.nrows() {
                            sum += input[[k, i]] * grad_upstream_2d[[k, j]];
                        }
                        out_col[i] = sum;
                    }
                });

            result
        };

        // Calculate bias gradients by summing over batch dimension
        let grad_b = grad_upstream_2d.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Ensure arrays are contiguous before storing (as_standard_layout() ensures contiguous memory layout)
        self.grad_weights = Some(grad_w.as_standard_layout().to_owned());
        self.grad_bias = Some(grad_b.as_standard_layout().to_owned());

        // Calculate gradients to be passed to previous layer: grad_input = grad_upstream * weights^T
        let grad_input = if total_ops < DENSE_PARALLEL_THRESHOLD {
            // Sequential computation using dot
            grad_upstream_2d.dot(&self.weights.t())
        } else {
            // Parallel computation using par_iter
            let batch_size = grad_upstream_2d.nrows();
            let mut result = Array2::<f32>::zeros((batch_size, self.input_dim));

            result
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(grad_upstream_2d.axis_iter(Axis(0)))
                .for_each(|(mut out_row, grad_row)| {
                    for j in 0..self.input_dim {
                        let mut sum = 0.0;
                        for k in 0..self.output_dim {
                            sum += grad_row[k] * self.weights[[j, k]];
                        }
                        out_row[j] = sum;
                    }
                });

            result
        };

        Ok(grad_input.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn output_shape(&self) -> String {
        // Returns only (None, output_dim)
        format!("(None, {})", self.output_dim)
    }

    fn param_count(&self) -> TrainingParameters {
        // Parameter count = number of weight parameters + number of bias parameters
        TrainingParameters::Trainable(self.input_dim * self.output_dim + self.output_dim)
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

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        // Initialize AdaGrad cache (if not already initialized)
        if self.optimizer_cache.ada_grad_cache.is_none() {
            let dims_w = (self.input_dim, self.output_dim);
            let dims_b = (1, self.output_dim);

            self.optimizer_cache.ada_grad_cache = Some(AdaGradStates::new(
                dims_w, None, // No recurrent weights
                dims_b,
            ));
        }

        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            let ada_grad_cache = self.optimizer_cache.ada_grad_cache.as_mut().unwrap();
            let (w_update, _, b_update) = ada_grad_cache.update_parameter(
                grad_w, None, // No recurrent weights
                grad_b, epsilon, lr,
            );

            // Apply updates
            self.weights = &self.weights - &w_update;
            self.bias = &self.bias - &b_update;
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Dense(DenseLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
