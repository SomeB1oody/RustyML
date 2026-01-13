use super::*;

/// Threshold for deciding when to use parallel computation in Conv2D operations.
/// If batch_size * filters * output_area < threshold, use sequential processing.
/// Otherwise, use parallel processing with Rayon.
const CONV_2D_PARALLEL_THRESHOLD: usize = 10000;

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
/// - `activation` - Activation layer from activation_layer module
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
///         ReLU::new(), // ReLU activation layer
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
pub struct Conv2D<T: ActivationLayer> {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    weights: Array4<f32>,
    bias: Array2<f32>,
    activation: T,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array2<f32>>,
    optimizer_cache: OptimizerCacheConv2D,
}

impl<T: ActivationLayer> Conv2D<T> {
    /// Creates a new 2D convolutional layer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of convolution filters (output channels).
    /// - `kernel_size` - Size of the convolution kernel as (height, width).
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\].
    /// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
    /// - `padding` - Type of padding to apply (`Valid` or `Same`).
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// * `Conv2D` - A new `Conv2D` layer instance with randomly initialized weights.
    ///
    /// # Notes
    ///
    /// Weights are initialized using Xavier (Glorot) uniform initialization.
    /// Biases are initialized to zeros.
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        activation: T,
    ) -> Self {
        // Shape is [batch_size, channels, height, width]
        let channels = input_shape[1];

        // Initialize weights using Xavier initialization for convolutional layers
        // Formula: sqrt(6 / (input_channels * kernel_area + filters * kernel_area))
        let fan_in = channels * kernel_size.0 * kernel_size.1;
        let fan_out = filters * kernel_size.0 * kernel_size.1;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let weights = Array4::random(
            (filters, channels, kernel_size.0, kernel_size.1),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
        );

        // Initialize biases to zero
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
                ada_grad_cache: None,
            },
        }
    }

    /// Calculates the output shape of the convolutional layer based on input dimensions.
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
                let out_height = (input_height + self.strides.0 - 1) / self.strides.0;
                let out_width = (input_width + self.strides.1 - 1) / self.strides.1;
                (out_height, out_width)
            }
        };

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `weights` - 4D array of filter weights with shape \[filters, channels, kernel_height, kernel_width\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    pub fn set_weights(&mut self, weights: Array4<f32>, bias: Array2<f32>) {
        self.weights = weights;
        self.bias = bias;
    }

    /// Applies padding to the input tensor for PaddingType::Same.
    fn apply_padding(&self, input: &Tensor) -> Tensor {
        match self.padding {
            PaddingType::Valid => input.clone(),
            PaddingType::Same => {
                let input_shape = input.shape();
                let batch_size = input_shape[0];
                let channels = input_shape[1];
                let input_height = input_shape[2];
                let input_width = input_shape[3];

                // Calculate padding amounts
                let out_height = (input_height + self.strides.0 - 1) / self.strides.0;
                let out_width = (input_width + self.strides.1 - 1) / self.strides.1;

                let pad_height = ((out_height - 1) * self.strides.0 + self.kernel_size.0)
                    .saturating_sub(input_height);
                let pad_width = ((out_width - 1) * self.strides.1 + self.kernel_size.1)
                    .saturating_sub(input_width);

                let pad_top = pad_height / 2;
                let pad_left = pad_width / 2;

                let padded_height = input_height + pad_height;
                let padded_width = input_width + pad_width;

                let mut padded = Array4::zeros((batch_size, channels, padded_height, padded_width));
                let input_4d = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();

                padded
                    .slice_mut(s![
                        ..,
                        ..,
                        pad_top..pad_top + input_height,
                        pad_left..pad_left + input_width
                    ])
                    .assign(&input_4d);

                padded.into_dyn()
            }
        }
    }

    /// Computes convolution for a single batch.
    fn compute_batch_convolution(
        &self,
        b: usize,
        padded_input: &Tensor,
        in_channels: usize,
        output_shape: &[usize],
    ) -> (usize, Array3<f32>) {
        // Create output portion for this batch
        let mut batch_output = Array3::zeros((self.filters, output_shape[2], output_shape[3]));

        // Computation for each batch
        for f in 0..self.filters {
            for i in 0..output_shape[2] {
                let i_base = i * self.strides.0;

                for j in 0..output_shape[3] {
                    let j_base = j * self.strides.1;
                    let mut sum = 0.0;

                    // Convolution kernel calculation
                    for c in 0..in_channels {
                        for ki in 0..self.kernel_size.0 {
                            let i_pos = i_base + ki;

                            for kj in 0..self.kernel_size.1 {
                                let j_pos = j_base + kj;
                                sum += padded_input[[b, c, i_pos, j_pos]]
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
    }

    /// Performs the convolution operation on the input tensor.
    ///
    /// This method implements the core convolution algorithm with optimizations:
    /// - Adaptive parallel/sequential processing based on workload size
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
        // Apply padding if needed
        let padded_input = self.apply_padding(input);
        let input_shape = padded_input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input.shape());

        // Calculate workload size to decide between parallel and sequential execution
        let workload_size = batch_size * self.filters * output_shape[2] * output_shape[3];

        // Choose execution strategy based on workload
        let results: Vec<_> = if workload_size >= CONV_2D_PARALLEL_THRESHOLD {
            // Use parallel processing for large workloads
            (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    self.compute_batch_convolution(b, &padded_input, in_channels, &output_shape)
                })
                .collect()
        } else {
            // Use sequential processing for small workloads
            (0..batch_size)
                .map(|b| {
                    self.compute_batch_convolution(b, &padded_input, in_channels, &output_shape)
                })
                .collect()
        };

        // Merge results from each batch into final output
        merge_results(output_shape, results, self.filters)
    }
}

impl<T: ActivationLayer> Layer for Conv2D<T> {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 4D".to_string(),
            ));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform convolution operation
        let output = self.convolve(input);

        // Apply activation
        self.activation.forward(&output.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Apply activation backward pass
        let grad_upstream = self.activation.backward(grad_output)?;

        if let Some(input) = &self.input_cache {
            let original_input_shape = input.shape();

            // Apply padding to input (same as in forward pass)
            let padded_input = self.apply_padding(input);
            let input_shape = padded_input.shape();

            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let grad_shape = grad_upstream.shape();

            let gradient = grad_upstream.clone();

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

                                        sum += compute_row_gradient_sum(
                                            &gradient,
                                            &padded_input,
                                            b,
                                            f,
                                            c,
                                            i,
                                            i_pos,
                                            w,
                                            grad_shape,
                                            input_shape,
                                            self.strides.1,
                                        );
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

            // Use batch-wise parallel processing and collect results
            let local_results: Vec<_> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    // Create local gradients for padded input shape
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

            // Merge padded gradients
            let padded_grad = merge_results(
                vec![batch_size, channels, input_shape[2], input_shape[3]],
                local_results,
                channels,
            );

            // Remove padding from gradients if PaddingType::Same was used
            let final_grad = match self.padding {
                PaddingType::Valid => padded_grad,
                PaddingType::Same => {
                    let pad_height = input_shape[2].saturating_sub(original_input_shape[2]);
                    let pad_width = input_shape[3].saturating_sub(original_input_shape[3]);
                    let pad_top = pad_height / 2;
                    let pad_left = pad_width / 2;

                    let padded_4d = padded_grad.into_dimensionality::<ndarray::Ix4>().unwrap();
                    padded_4d
                        .slice(s![
                            ..,
                            ..,
                            pad_top..pad_top + original_input_shape[2],
                            pad_left..pad_left + original_input_shape[3]
                        ])
                        .to_owned()
                        .into_dyn()
                }
            };

            Ok(final_grad)
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

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.weights.len() + self.bias.len())
    }

    update_sgd_conv!();

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize Adam states (if not already initialized)
            if self.optimizer_cache.adam_states.is_none() {
                self.optimizer_cache.adam_states = Some(AdamStatesConv2D {
                    m: Array4::zeros(self.weights.dim()),
                    v: Array4::zeros(self.weights.dim()),
                    m_bias: Array2::zeros(self.bias.dim()),
                    v_bias: Array2::zeros(self.bias.dim()),
                });
            }

            if let Some(adam_states) = &mut self.optimizer_cache.adam_states {
                // Compute bias correction factors
                let bias_correction1 = 1.0 - beta1.powi(t as i32);
                let bias_correction2 = 1.0 - beta2.powi(t as i32);

                // Update weight parameters
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
                        bias_correction1,
                        bias_correction2,
                    );
                }

                // Update bias parameters
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
                        bias_correction1,
                        bias_correction2,
                    );
                }
            }
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize RMSprop cache (if not already initialized)
            if self.optimizer_cache.rmsprop_cache.is_none() {
                self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheConv2D {
                    cache: Array4::zeros(self.weights.dim()),
                    bias: Array2::zeros(self.bias.dim()),
                });
            }

            if let Some(rmsprop_cache) = &mut self.optimizer_cache.rmsprop_cache {
                // Update weights
                if let (Some(weight_slice), Some(weight_grad_slice), Some(cache_slice)) = (
                    self.weights.as_slice_mut(),
                    weight_grads.as_slice(),
                    rmsprop_cache.cache.as_slice_mut(),
                ) {
                    update_rmsprop(
                        weight_slice,
                        weight_grad_slice,
                        cache_slice,
                        rho,
                        epsilon,
                        lr,
                    );
                }

                // Update biases
                if let (Some(bias_slice), Some(bias_grad_slice), Some(bias_cache_slice)) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    rmsprop_cache.bias.as_slice_mut(),
                ) {
                    update_rmsprop(
                        bias_slice,
                        bias_grad_slice,
                        bias_cache_slice,
                        rho,
                        epsilon,
                        lr,
                    );
                }
            }
        }
    }

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        if let (Some(weight_gradients), Some(bias_gradients)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize AdaGrad cache (if not already initialized)
            if self.optimizer_cache.ada_grad_cache.is_none() {
                use crate::neural_network::optimizer::AdaGradStatesConv2D;

                self.optimizer_cache.ada_grad_cache = Some(AdaGradStatesConv2D {
                    accumulator: Array4::zeros(self.weights.dim()),
                    accumulator_bias: Array2::zeros(self.bias.dim()),
                });
            }

            if let Some(ada_grad_cache) = &mut self.optimizer_cache.ada_grad_cache {
                // Define a generic parameter update closure for AdaGrad
                let update_parameters =
                    |params: &mut [f32], accumulator: &mut [f32], grads: &[f32]| {
                        // Update accumulator (accumulated squared gradients) in parallel
                        accumulator.par_iter_mut().zip(grads.par_iter()).for_each(
                            |(acc, &grad)| {
                                *acc += grad * grad;
                            },
                        );

                        // Update parameters in parallel
                        params
                            .par_iter_mut()
                            .zip(grads.par_iter())
                            .zip(accumulator.par_iter())
                            .for_each(|((param, &grad), &acc_val)| {
                                *param -= lr * grad / (acc_val.sqrt() + epsilon);
                            });
                    };

                // Update weight parameters
                update_parameters(
                    self.weights.as_slice_mut().unwrap(),
                    ada_grad_cache.accumulator.as_slice_mut().unwrap(),
                    weight_gradients.as_slice().unwrap(),
                );

                // Update bias parameters
                update_parameters(
                    self.bias.as_slice_mut().unwrap(),
                    ada_grad_cache.accumulator_bias.as_slice_mut().unwrap(),
                    bias_gradients.as_slice().unwrap(),
                );
            }
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv2D(Conv2DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
