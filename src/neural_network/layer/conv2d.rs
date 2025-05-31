use super::*;
use crate::neural_network::activation::Activation;
use crate::neural_network::optimizer::*;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::{Array2, Array3, Array4, ArrayD, Axis};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// A 2D convolutional layer for neural networks.
///
/// This layer applies a convolution operation to input data, which is a fundamental
/// operation in convolutional neural networks (CNNs). It's particularly effective for
/// processing data with grid-like topology, such as images.
///
/// # Fields
///
/// - `filters` - Number of convolution filters (output channels).
/// - `kernel_size` - Size of the convolution kernel as (height, width).
/// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `weights` - 4D array of filter weights with shape \[filters, channels, kernel_height, kernel_width\].
/// - `bias` - 2D array of bias values with shape \[1, filters\].
/// - `activation` - Optional activation function applied after the convolution.
/// - `input_cache` - Cached input from the forward pass, used during backpropagation.
/// - `input_shape` - Shape of the input tensor.
/// - `weight_gradients` - Gradients for the weights, computed during backpropagation.
/// - `bias_gradients` - Gradients for the biases, computed during backpropagation.
/// - `optimizer_cache` - Cache for optimizer-specific state (e.g., momentum values for Adam).
///
/// # Shape Information
///
/// Input shape: \[batch_size, channels, height, width\]
/// Output shape: \[batch_size, filters, output_height, output_width\]
///
/// The output dimensions (output_height, output_width) depend on:
/// - Input dimensions
/// - Kernel size
/// - Stride values
/// - Padding type
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
///
/// // Create a simple 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 1 input channel, 5x5 pixels
/// let x = Array4::ones((2, 1, 5, 5)).into_dyn();
///
/// // Create target tensor - assuming we'll have 3 filters with output size 3x3
/// let y = Array4::ones((2, 3, 3, 3)).into_dyn();
///
/// // Build model: add a Conv2D layer with 3 filters and 3x3 kernel
/// let mut model = Sequential::new();
/// model
///     .add(Conv2D::new(
///         3,                      // Number of filters
///         (3, 3),                 // Kernel size
///         vec![2, 1, 5, 5],       // Input shape
///         (1, 1),                 // Stride
///         PaddingType::Valid,     // No padding
///         Some(Activation::ReLU), // ReLU activation function
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("Convolution layer prediction results: {:?}", prediction);
///
/// // Check if output shape is correct - should be [2, 3, 3, 3]
/// assert_eq!(prediction.shape(), &[2, 3, 3, 3]);
/// ```
pub struct Conv2D {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    weights: Array4<f32>,
    bias: Array2<f32>,
    activation: Option<Activation>,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array2<f32>>,
    optimizer_cache: OptimizerCacheConv2D,
}

impl Conv2D {
    /// Creates a new 2D convolutional layer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of convolution filters (output channels).
    /// - `kernel_size` - Size of the convolution kernel as (height, width).
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\].
    /// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
    /// - `padding` - Type of padding to apply (`Valid` or `Same`).
    /// - `activation` - Optional activation function to apply after the convolution.
    ///
    /// # Returns
    ///
    /// * `Self` - A new `Conv2D` layer instance with randomly initialized weights.
    ///
    /// # Notes
    ///
    /// Weights are initialized from a normal distribution with mean 0.0 and standard
    /// deviation 0.1. Biases are initialized to zeros.
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        activation: Option<Activation>,
    ) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Shape is [batch_size, channels, height, width]
        let channels = input_shape[1];

        // Initialize weights
        let mut weights = Array4::zeros((filters, channels, kernel_size.0, kernel_size.1));
        for i in weights.iter_mut() {
            *i = normal.sample(&mut rng) as f32;
        }

        // Initialize biases
        let bias = Array2::zeros((1, filters));

        Conv2D {
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            activation,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
            optimizer_cache: OptimizerCacheConv2D {
                adam_states: None,
                rmsprop_cache: None,
            },
        }
    }

    /// Calculates the output shape of the convolutional layer based on input dimensions.
    ///
    /// This function determines the spatial dimensions of the output tensor based on:
    /// - Input dimensions
    /// - Kernel size
    /// - Stride values
    /// - Padding type
    ///
    /// # Parameters
    ///
    /// * `input_shape` - A slice containing the shape of the input tensor in the format
    ///   \[batch_size, channels, height, width\].
    ///
    /// # Returns
    ///
    /// A vector containing the calculated output shape in the format
    /// \[batch_size, filters, output_height, output_width\].
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = match self.padding {
            PaddingType::Valid => {
                let out_height = (input_height - self.kernel_size.0) / self.strides.0 + 1;
                let out_width = (input_width - self.kernel_size.1) / self.strides.1 + 1;
                (out_height, out_width)
            }
            PaddingType::Same => {
                let out_height = (input_height as f32 / self.strides.0 as f32).ceil() as usize;
                let out_width = (input_width as f32 / self.strides.1 as f32).ceil() as usize;
                (out_height, out_width)
            }
        };

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// Applies the activation function to the tensor in-place.
    ///
    /// If an activation function is specified for this layer, this method applies it
    /// element-wise to the input tensor using parallel processing.
    ///
    /// # Parameters
    ///
    /// * `x` - A mutable reference to the tensor to which the activation function will be applied.
    ///
    /// # Panics
    ///
    /// Panics if Softmax activation is used, as it's not suitable for convolutional layers.
    fn apply_activation(&self, x: &mut Tensor) {
        if let Some(activation) = &self.activation {
            match activation {
                Activation::ReLU => {
                    x.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                }
                Activation::Sigmoid => {
                    x.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
                }
                Activation::Tanh => {
                    x.par_mapv_inplace(|x| x.tanh());
                }
                Activation::Softmax => panic!("Cannot use Softmax for convolution"),
            }
        }
    }

    /// Calculates the derivative of the activation function at the given output values.
    ///
    /// This function is used during backpropagation to compute gradients.
    ///
    /// # Parameters
    ///
    /// * `output` - A reference to the tensor containing the output values of the forward pass.
    ///
    /// # Returns
    ///
    /// * `Tensor` - A new tensor containing the activation function derivatives calculated element-wise.
    ///
    /// # Panics
    ///
    /// Panics if Softmax activation is used, as it's not suitable for convolutional layers.
    fn activation_derivative(&self, output: &Tensor) -> Tensor {
        let mut result = output.clone();

        if let Some(activation) = &self.activation {
            match activation {
                Activation::ReLU => {
                    result.par_mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
                }
                Activation::Sigmoid => {
                    result.par_mapv_inplace(|a| a * (1.0 - a));
                }
                Activation::Tanh => {
                    result.par_mapv_inplace(|a| 1.0 - a * a);
                }
                Activation::Softmax => panic!("Cannot use Softmax for convolution"),
            }
        }

        result
    }

    /// Performs the convolution operation on the input tensor.
    ///
    /// This method implements the core convolution algorithm with optimizations:
    /// - Parallel batch processing using Rayon
    /// - Boundary condition pre-checking
    /// - Memory access pattern optimization
    ///
    /// # Parameters
    ///
    /// * `input` - A reference to the input tensor with shape \[batch_size, channels, height, width\].
    ///
    /// # Returns
    ///
    /// * `Tensor` - A new tensor containing the result of the convolution operation with shape \[batch_size, filters, output_height, output_width\].
    fn convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // Pre-allocate output array
        let mut output = ArrayD::zeros(output_shape.clone());

        // Create vector for batch processing results
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                // Create output portion for this batch
                let mut batch_output =
                    Array3::zeros((self.filters, output_shape[2], output_shape[3]));

                // Computation for each batch
                for f in 0..self.filters {
                    for i in 0..output_shape[2] {
                        let i_base = i * self.strides.0;

                        for j in 0..output_shape[3] {
                            let j_base = j * self.strides.1;
                            let mut sum = 0.0;

                            // Convolution kernel calculation
                            // Pre-check boundary conditions
                            let max_ki = input_shape[2]
                                .saturating_sub(i_base)
                                .min(self.kernel_size.0);
                            let max_kj = input_shape[3]
                                .saturating_sub(j_base)
                                .min(self.kernel_size.1);

                            for c in 0..in_channels {
                                // Use contiguous memory access pattern
                                for ki in 0..max_ki {
                                    let i_pos = i_base + ki;

                                    for kj in 0..max_kj {
                                        let j_pos = j_base + kj;
                                        sum += input[[b, c, i_pos, j_pos]]
                                            * self.weights[[f, c, ki, kj]];
                                    }
                                }
                            }

                            // Update batch output
                            sum += self.bias[[0, f]];
                            batch_output[[f, i, j]] = sum;
                        }
                    }
                }

                (b, batch_output)
            })
            .collect();

        // Merge results from each batch into final output
        for (b, batch_output) in results {
            for f in 0..self.filters {
                for i in 0..output_shape[2] {
                    for j in 0..output_shape[3] {
                        output[[b, f, i, j]] = batch_output[[f, i, j]];
                    }
                }
            }
        }

        output
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform convolution operation
        let mut output = self.convolve(input);

        // Apply activation function
        self.apply_activation(&mut output);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let grad_shape = grad_output.shape();

            // Calculate derivatives of the activation function
            let activation_grad = self.activation_derivative(grad_output);

            // Use rayon for parallel element-wise multiplication
            let mut gradient = activation_grad.clone();
            if let (Some(grad_slice), Some(act_slice), Some(out_slice)) = (
                gradient.as_slice_mut(),
                activation_grad.as_slice(),
                grad_output.as_slice(),
            ) {
                grad_slice
                    .par_iter_mut()
                    .zip(act_slice.par_iter().zip(out_slice.par_iter()))
                    .for_each(|(g, (a, o))| {
                        *g = a * o;
                    });
            } else {
                // Fallback to loop implementation
                for (i, v) in activation_grad.iter().enumerate() {
                    gradient.as_slice_mut().unwrap()[i] = v * grad_output.as_slice().unwrap()[i];
                }
            }

            // Initialize gradients for weights and biases
            let mut weight_grads = Array4::zeros(self.weights.dim());
            let mut bias_grads = Array2::zeros((1, self.filters));

            // Calculate bias gradients in parallel
            bias_grads
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut bias)| {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        for i in 0..grad_shape[2] {
                            for j in 0..grad_shape[3] {
                                sum += gradient[[b, f, i, j]];
                            }
                        }
                    }
                    *bias.first_mut().unwrap() = sum;
                });

            // Optimize weight gradient calculation using parallel computation
            weight_grads
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut filter_grad)| {
                    // Process each filter in parallel
                    for c in 0..channels {
                        for h in 0..self.kernel_size.0 {
                            for w in 0..self.kernel_size.1 {
                                let mut sum = 0.0;
                                // Pre-check boundary conditions to reduce conditional checks
                                for b in 0..batch_size {
                                    for i in 0..grad_shape[2] {
                                        let i_pos = i * self.strides.0 + h;
                                        if i_pos >= input_shape[2] {
                                            continue;
                                        }

                                        for j in 0..grad_shape[3] {
                                            let j_pos = j * self.strides.1 + w;
                                            if j_pos < input_shape[3] {
                                                sum += gradient[[b, f, i, j]]
                                                    * input[[b, c, i_pos, j_pos]];
                                            }
                                        }
                                    }
                                }
                                filter_grad[[c, h, w]] = sum;
                            }
                        }
                    }
                });

            // Save gradients for optimization
            self.weight_gradients = Some(weight_grads);
            self.bias_gradients = Some(bias_grads);

            // Calculate input gradients in parallel
            let mut input_gradients = ArrayD::zeros(input.dim());

            // Use batch-wise parallel processing and collect results
            let local_results: Vec<_> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    // Create local gradients for each batch
                    let mut local_gradients =
                        Array3::zeros([channels, input_shape[2], input_shape[3]]);

                    for c in 0..channels {
                        for i in 0..input_shape[2] {
                            for j in 0..input_shape[3] {
                                let mut sum = 0.0;

                                for f in 0..self.filters {
                                    for h in 0..self.kernel_size.0 {
                                        for w in 0..self.kernel_size.1 {
                                            // Check if indices are valid
                                            if i >= h && j >= w {
                                                let grad_i = (i - h) / self.strides.0;
                                                let grad_j = (j - w) / self.strides.1;

                                                // Check if calculated gradient position is valid
                                                if grad_i < grad_shape[2]
                                                    && grad_j < grad_shape[3]
                                                    && (i - h) % self.strides.0 == 0
                                                    && (j - w) % self.strides.1 == 0
                                                {
                                                    sum += gradient[[b, f, grad_i, grad_j]]
                                                        * self.weights[[f, c, h, w]];
                                                }
                                            }
                                        }
                                    }
                                }

                                local_gradients[[c, i, j]] = sum;
                            }
                        }
                    }

                    (b, local_gradients)
                })
                .collect();

            // Merge results in the main thread
            for (b, local_gradients) in local_results {
                for c in 0..channels {
                    for i in 0..input_shape[2] {
                        for j in 0..input_shape[3] {
                            input_gradients[[b, c, i, j]] = local_gradients[[c, i, j]];
                        }
                    }
                }
            }

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Conv2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> usize {
        let weight_count = self.weights.len();
        let bias_count = self.bias.len();
        weight_count + bias_count
    }

    update_sgd_conv!();

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize momentum and variance (if not initialized)
            if self.optimizer_cache.adam_states.is_none() {
                self.optimizer_cache.adam_states = Some(AdamStatesConv2D {
                    m: Array4::zeros(self.weights.dim()),
                    v: Array4::zeros(self.weights.dim()),
                    m_bias: Array2::zeros(self.bias.dim()),
                    v_bias: Array2::zeros(self.bias.dim()),
                });
            }

            let correction1 = 1.0 - beta1.powi(t as i32);
            let correction2 = 1.0 - beta2.powi(t as i32);

            // Update weights and biases using a closure to avoid code duplication
            if let Some(adam_states) = &mut self.optimizer_cache.adam_states {
                // Update weights using parallel computation
                if let (Some(weight_slice), Some(weight_grad_slice), Some(m_slice), Some(v_slice)) = (
                    self.weights.as_slice_mut(),
                    weight_grads.as_slice(),
                    adam_states.m.as_slice_mut(),
                    adam_states.v.as_slice_mut(),
                ) {
                    update_adam_conv(
                        weight_slice,
                        weight_grad_slice,
                        m_slice,
                        v_slice,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        correction1,
                        correction2,
                    );
                } else {
                    // If slices cannot be obtained, use original sequential implementation as fallback
                    for i in 0..self.weights.len() {
                        let grad = weight_grads.as_slice().unwrap()[i];
                        let m = &mut adam_states.m.as_slice_mut().unwrap()[i];
                        let v = &mut adam_states.v.as_slice_mut().unwrap()[i];

                        *m = beta1 * *m + (1.0 - beta1) * grad;
                        *v = beta2 * *v + (1.0 - beta2) * grad * grad;

                        let m_corrected = *m / correction1;
                        let v_corrected = *v / correction2;

                        self.weights.as_slice_mut().unwrap()[i] -=
                            lr * m_corrected / (v_corrected.sqrt() + epsilon);
                    }
                }

                // Use parallel computation to update biases
                if let (
                    Some(bias_slice),
                    Some(bias_grad_slice),
                    Some(m_bias_slice),
                    Some(v_bias_slice),
                ) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    adam_states.m_bias.as_slice_mut(),
                    adam_states.v_bias.as_slice_mut(),
                ) {
                    update_adam_conv(
                        bias_slice,
                        bias_grad_slice,
                        m_bias_slice,
                        v_bias_slice,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        correction1,
                        correction2,
                    );
                } else {
                    // If slices cannot be obtained, use original sequential implementation as fallback
                    for i in 0..self.bias.len() {
                        let grad = bias_grads.as_slice().unwrap()[i];
                        let m = &mut adam_states.m_bias.as_slice_mut().unwrap()[i];
                        let v = &mut adam_states.v_bias.as_slice_mut().unwrap()[i];

                        *m = beta1 * *m + (1.0 - beta1) * grad;
                        *v = beta2 * *v + (1.0 - beta2) * grad * grad;

                        let m_corrected = *m / correction1;
                        let v_corrected = *v / correction2;

                        self.bias.as_slice_mut().unwrap()[i] -=
                            lr * m_corrected / (v_corrected.sqrt() + epsilon);
                    }
                }
            }
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize cache (if not initialized yet)
            if self.optimizer_cache.rmsprop_cache.is_none() {
                self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheConv2D {
                    cache: Array4::zeros(self.weights.dim()),
                    bias: Array2::zeros(self.bias.dim()),
                });
            }

            // Create a generic RMSprop update closure
            let update_rmsprop = |params: &mut [f32], grads: &[f32], cache: &mut [f32]| {
                params
                    .par_iter_mut()
                    .zip(grads.par_iter())
                    .zip(cache.par_iter_mut())
                    .for_each(|((param, &grad), cache_val)| {
                        // Update cache
                        *cache_val = rho * *cache_val + (1.0 - rho) * grad * grad;
                        // Update parameters
                        *param -= lr * grad / (cache_val.sqrt() + epsilon);
                    });
            };

            if let Some(rmsprop_cache) = &mut self.optimizer_cache.rmsprop_cache {
                // Update weights
                if let (Some(weight_slice), Some(weight_grad_slice), Some(cache_slice)) = (
                    self.weights.as_slice_mut(),
                    weight_grads.as_slice(),
                    rmsprop_cache.cache.as_slice_mut(),
                ) {
                    // Use closure to update weights
                    update_rmsprop(weight_slice, weight_grad_slice, cache_slice);
                } else {
                    // Fallback to original loop implementation
                    for i in 0..self.weights.len() {
                        let grad = weight_grads.as_slice().unwrap()[i];
                        let cache = &mut rmsprop_cache.cache.as_slice_mut().unwrap()[i];

                        *cache = rho * *cache + (1.0 - rho) * grad * grad;
                        self.weights.as_slice_mut().unwrap()[i] -=
                            lr * grad / (cache.sqrt() + epsilon);
                    }
                }

                // Update biases
                if let (Some(bias_slice), Some(bias_grad_slice), Some(bias_cache_slice)) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    rmsprop_cache.bias.as_slice_mut(),
                ) {
                    // Use the same closure to update biases
                    update_rmsprop(bias_slice, bias_grad_slice, bias_cache_slice);
                } else {
                    // Fallback to original loop implementation
                    for i in 0..self.bias.len() {
                        let grad = bias_grads.as_slice().unwrap()[i];
                        let cache = &mut rmsprop_cache.bias.as_slice_mut().unwrap()[i];

                        *cache = rho * *cache + (1.0 - rho) * grad * grad;
                        self.bias.as_slice_mut().unwrap()[i] -=
                            lr * grad / (cache.sqrt() + epsilon);
                    }
                }
            }
        }
    }

    fn get_weights(&self) -> LayerWeight {
        LayerWeight::Conv2D(Conv2DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
