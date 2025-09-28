use super::*;

/// A 1D convolutional layer for neural networks.
///
/// This layer applies a convolution operation to input data, which is particularly
/// effective for processing sequential data, such as time series, audio signals,
/// or text data.
///
/// # Fields
///
/// - `filters` - Number of convolution filters (output channels).
/// - `kernel_size` - Size of the convolution kernel.
/// - `stride` - Stride value for the convolution operation.
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `weights` - 3D array of filter weights with shape \[filters, channels, kernel_size\].
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
/// Input shape: \[batch_size, channels, length\]
/// Output shape: \[batch_size, filters, output_length\]
///
/// The output dimension (output_length) depends on:
/// - Input length
/// - Kernel size
/// - Stride value
/// - Padding type
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array3;
///
/// // Create a simple 3D input tensor: [batch_size, channels, length]
/// // Batch size=2, 1 input channel, 10 time steps
/// let x = Array3::ones((2, 1, 10)).into_dyn();
///
/// // Create target tensor - assuming we'll have 3 filters with output length 8
/// let y = Array3::ones((2, 3, 8)).into_dyn();
///
/// // Build model: add a Conv1D layer with 3 filters and kernel size 3
/// let mut model = Sequential::new();
/// model
///     .add(Conv1D::new(
///         3,                      // Number of filters
///         3,                      // Kernel size
///         vec![2, 1, 10],         // Input shape
///         1,                      // Stride
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
/// // Check if output shape is correct - should be [2, 3, 8]
/// assert_eq!(prediction.shape(), &[2, 3, 8]);
/// ```
pub struct Conv1D {
    filters: usize,
    kernel_size: usize,
    stride: usize,
    padding: PaddingType,
    weights: Array3<f32>,
    bias: Array2<f32>,
    activation: Option<Activation>,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array3<f32>>,
    bias_gradients: Option<Array2<f32>>,
    optimizer_cache: OptimizerCacheConv1D,
}

impl Conv1D {
    /// Creates a new Conv1D layer.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (channels)
    /// - `kernel_size` - Size of the convolution kernel
    /// - `input_shape` - Shape of input tensor \[batch_size, channels, length\]
    /// - `stride` - Stride for the convolution operation
    /// - `padding` - Padding type (Valid or Same)
    /// - `activation` - Optional activation function
    ///
    /// # Returns
    ///
    /// * `Conv1D` - A new `Conv1D` layer instance
    pub fn new(
        filters: usize,
        kernel_size: usize,
        input_shape: Vec<usize>,
        stride: usize,
        padding: PaddingType,
        activation: Option<Activation>,
    ) -> Self {
        // verify input is 3D: [batch_size, channels, length]
        assert_eq!(
            input_shape.len(),
            3,
            "Input tensor must be 5-dimensional: [batch_size, channels, length]"
        );
        let input_channels = input_shape[1];

        // Initialize weights using Xavier initialization
        let weight_bound = (6.0 / (input_channels + filters) as f32).sqrt();

        let weights = Array3::random(
            (filters, input_channels, kernel_size),
            ndarray_rand::rand_distr::Uniform::new(-weight_bound, weight_bound),
        );

        // Initialize bias to zero
        let bias = Array2::zeros((1, filters));

        Self {
            filters,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
            activation,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
            optimizer_cache: OptimizerCacheConv1D {
                adam_states: None,
                rmsprop_cache: None,
            },
        }
    }

    /// Calculates the output shape after convolution.
    ///
    /// # Parameters
    ///
    /// * `input_length` - Length of the input sequence
    ///
    /// # Returns
    ///
    /// * `usize` - Output length after convolution
    fn calculate_output_length(&self, input_length: usize) -> usize {
        match self.padding {
            PaddingType::Valid => (input_length - self.kernel_size) / self.stride + 1,
            PaddingType::Same => (input_length + self.stride - 1) / self.stride,
        }
    }

    /// Applies padding to the input tensor.
    ///
    /// # Parameters
    ///
    /// * `input` - Input tensor to pad
    ///
    /// # Returns
    ///
    /// * `Tensor` - Padded tensor
    fn apply_padding(&self, input: &Tensor) -> Tensor {
        match self.padding {
            PaddingType::Valid => input.clone(),
            PaddingType::Same => {
                let input_shape = input.shape();
                let batch_size = input_shape[0];
                let channels = input_shape[1];
                let input_length = input_shape[2];

                let output_length = (input_length + self.stride - 1) / self.stride;
                let pad_total = ((output_length - 1) * self.stride + self.kernel_size)
                    .saturating_sub(input_length);
                let pad_left = pad_total / 2;

                let mut padded = Array3::zeros((batch_size, channels, input_length + pad_total));
                padded
                    .slice_mut(ndarray::s![.., .., pad_left..input_length + pad_left])
                    .assign(&input.clone().into_dimensionality::<ndarray::Ix3>().unwrap());

                padded.into_dyn()
            }
        }
    }

    /// Performs 1D convolution operation with parallel processing.
    ///
    /// # Parameters
    ///
    /// - `input` - Input tensor with shape \[batch_size, channels, length\]
    ///
    /// # Returns
    ///
    /// Output tensor after convolution
    fn conv1d(&self, input: &Tensor) -> Tensor {
        let padded_input = self.apply_padding(input);
        let input_shape = padded_input.shape();
        let batch_size = input_shape[0];
        let input_length = input_shape[2];

        let output_length = self.calculate_output_length(input_length);
        let mut output = Array3::zeros((batch_size, self.filters, output_length));

        let input_3d = padded_input
            .clone()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();

        // Parallel processing of batches and filters
        output
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(batch, mut batch_output)| {
                batch_output
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(filter, mut filter_output)| {
                        filter_output.indexed_iter_mut().par_bridge().for_each(
                            |(out_pos, output_val)| {
                                let start_pos = out_pos * self.stride;
                                let mut sum = 0.0;

                                // Convolution operation
                                for in_channel in 0..self.input_shape[1] {
                                    for kernel_pos in 0..self.kernel_size {
                                        let input_pos = start_pos + kernel_pos;
                                        if input_pos < input_length {
                                            sum += input_3d[[batch, in_channel, input_pos]]
                                                * self.weights[[filter, in_channel, kernel_pos]];
                                        }
                                    }
                                }

                                // Add bias
                                sum += self.bias[[0, filter]];
                                *output_val = sum;
                            },
                        );
                    });
            });

        output.into_dyn()
    }
}

impl Layer for Conv1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Cache input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform convolution
        let mut output = self.conv1d(input);

        // Apply activation function
        if let Some(activation) = &self.activation {
            Activation::apply_activation_inplace(activation, &mut output);
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Retrieve cached input from forward pass
        let input = self.input_cache.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("No cached input for backward pass".to_string())
        })?;

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_channels = input_shape[1];
        let input_length = input_shape[2];

        // Apply activation function derivatives
        let mut grad_output = grad_output.clone();
        if let Some(ref activation) = self.activation {
            Activation::activation_derivative_inplace(activation, &mut grad_output);
        }

        let grad_output_3d = grad_output
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| {
                ModelError::ProcessingError(format!("Failed to convert gradient output: {}", e))
            })?;

        // Initialize gradients
        let mut weight_gradients = Array3::zeros(self.weights.dim());
        let mut bias_gradients = Array2::zeros(self.bias.dim());
        let mut input_gradients = Array3::zeros((batch_size, input_channels, input_length));

        let padded_input = self.apply_padding(input);
        let input_3d = padded_input
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| ModelError::ProcessingError(format!("Failed to convert input: {}", e)))?;

        let output_length = grad_output_3d.shape()[2];

        // Use parallel processing to compute gradients
        use std::sync::Mutex;
        let weight_gradients_mutex = Mutex::new(&mut weight_gradients);
        let bias_gradients_mutex = Mutex::new(&mut bias_gradients);
        let input_gradients_mutex = Mutex::new(&mut input_gradients);

        // Parallelize over batches
        (0..batch_size).into_par_iter().for_each(|batch| {
            // Create local gradient accumulators for each thread
            let mut local_weight_gradients = Array3::zeros(self.weights.dim());
            let mut local_bias_gradients = Array2::zeros(self.bias.dim());
            let mut local_input_gradients = Array3::zeros((1, input_channels, input_length));

            for filter in 0..self.filters {
                for out_pos in 0..output_length {
                    let grad_val = grad_output_3d[[batch, filter, out_pos]];
                    let start_pos = out_pos * self.stride;

                    // Bias gradients
                    local_bias_gradients[[0, filter]] += grad_val;

                    // Weight and input gradients
                    for in_channel in 0..input_channels {
                        for kernel_pos in 0..self.kernel_size {
                            let input_pos = start_pos + kernel_pos;
                            if input_pos < input_3d.shape()[2] {
                                // Weight gradients
                                local_weight_gradients[[filter, in_channel, kernel_pos]] +=
                                    grad_val * input_3d[[batch, in_channel, input_pos]];

                                // Input gradients (considering padding)
                                let original_input_pos = match self.padding {
                                    PaddingType::Valid => input_pos,
                                    PaddingType::Same => {
                                        let pad_left = ((output_length - 1) * self.stride
                                            + self.kernel_size)
                                            .saturating_sub(input_length)
                                            / 2;
                                        if input_pos >= pad_left
                                            && input_pos < pad_left + input_length
                                        {
                                            input_pos - pad_left
                                        } else {
                                            continue;
                                        }
                                    }
                                };

                                if original_input_pos < input_length {
                                    local_input_gradients[[0, in_channel, original_input_pos]] +=
                                        grad_val * self.weights[[filter, in_channel, kernel_pos]];
                                }
                            }
                        }
                    }
                }
            }

            // Accumulate local gradients into global gradients
            {
                let mut global_weight_gradients = weight_gradients_mutex.lock().unwrap();
                **global_weight_gradients += &local_weight_gradients;
            }
            {
                let mut global_bias_gradients = bias_gradients_mutex.lock().unwrap();
                **global_bias_gradients += &local_bias_gradients;
            }
            {
                let mut global_input_gradients = input_gradients_mutex.lock().unwrap();

                global_input_gradients
                    .slice_mut(ndarray::s![batch, .., ..])
                    .assign(&local_input_gradients.slice(ndarray::s![0, .., ..]));
            }
        });

        // Store gradients
        self.weight_gradients = Some(weight_gradients);
        self.bias_gradients = Some(bias_gradients);

        Ok(input_gradients.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "Conv1D"
    }

    fn output_shape(&self) -> String {
        let input_length = self.input_shape[2];
        let output_length = self.calculate_output_length(input_length);
        format!(
            "({}, {}, {})",
            self.input_shape[0], self.filters, output_length
        )
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.weights.len() + self.bias.len())
    }

    update_sgd_conv!();

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Ensure gradients are available
        if let (Some(weight_gradients), Some(bias_gradients)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize Adam states (if not already initialized)
            if self.optimizer_cache.adam_states.is_none() {
                use crate::neural_network::optimizer::AdamStatesConv1D;

                self.optimizer_cache.adam_states = Some(AdamStatesConv1D {
                    m: Array3::zeros(self.weights.dim()),
                    v: Array3::zeros(self.weights.dim()),
                    m_bias: Array2::zeros(self.bias.dim()),
                    v_bias: Array2::zeros(self.bias.dim()),
                });
            }

            if let Some(adam_states) = &mut self.optimizer_cache.adam_states {
                // Compute bias correction factors
                let bias_correction1 = 1.0 - beta1.powi(t as i32);
                let bias_correction2 = 1.0 - beta2.powi(t as i32);

                // Update weight parameters
                update_adam_conv(
                    self.weights.as_slice_mut().unwrap(),
                    weight_gradients.as_slice().unwrap(),
                    adam_states.m.as_slice_mut().unwrap(),
                    adam_states.v.as_slice_mut().unwrap(),
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    bias_correction1,
                    bias_correction2,
                );

                // Update bias parameters
                update_adam_conv(
                    self.bias.as_slice_mut().unwrap(),
                    bias_gradients.as_slice().unwrap(),
                    adam_states.m_bias.as_slice_mut().unwrap(),
                    adam_states.v_bias.as_slice_mut().unwrap(),
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    bias_correction1,
                    bias_correction2,
                );
            }
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        // Ensure gradients are available
        if let (Some(weight_gradients), Some(bias_gradients)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize RMSprop cache (if not already initialized)
            if self.optimizer_cache.rmsprop_cache.is_none() {
                use crate::neural_network::optimizer::RMSpropCacheConv1D;

                self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheConv1D {
                    cache: Some(Array3::zeros(self.weights.dim())),
                    bias: Some(Array2::zeros(self.bias.dim())),
                });
            }

            if let Some(rmsprop_cache) = &mut self.optimizer_cache.rmsprop_cache {
                // Define a generic parameter update closure, also handling cache updates
                let update_parameters = |params: &mut [f32], cache: &mut [f32], grads: &[f32]| {
                    // Update cache (moving average of squared gradients) in parallel
                    cache
                        .par_iter_mut()
                        .zip(grads.par_iter())
                        .for_each(|(c, &grad)| {
                            *c = rho * *c + (1.0 - rho) * grad * grad;
                        });

                    // Update parameters in parallel
                    params
                        .par_iter_mut()
                        .zip(grads.par_iter())
                        .zip(cache.par_iter())
                        .for_each(|((param, &grad), &cache_val)| {
                            *param -= lr * grad / (cache_val.sqrt() + epsilon);
                        });
                };

                // Update weight parameters
                if let Some(weight_cache) = &mut rmsprop_cache.cache {
                    update_parameters(
                        self.weights.as_slice_mut().unwrap(),
                        weight_cache.as_slice_mut().unwrap(),
                        weight_gradients.as_slice().unwrap(),
                    );
                }

                // Update bias parameters
                if let Some(bias_cache) = &mut rmsprop_cache.bias {
                    update_parameters(
                        self.bias.as_slice_mut().unwrap(),
                        bias_cache.as_slice_mut().unwrap(),
                        bias_gradients.as_slice().unwrap(),
                    );
                }
            }
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv1D(Conv1DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
