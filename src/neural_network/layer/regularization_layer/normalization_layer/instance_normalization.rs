use super::*;

/// Threshold for switching between sequential and parallel instance normalization computation.
/// Based on total elements in the tensor.
const INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD: usize = 1024;

/// Instance Normalization layer for neural networks, which normalizes the inputs
/// independently for each sample and each channel to improve training stability.
///
/// Instance Normalization computes statistics (mean and variance) for each channel
/// in each sample independently, making it particularly useful for style transfer
/// and generative models where batch statistics may not be meaningful.
///
/// # Fields
///
/// - `epsilon` - Small constant for numerical stability in normalization.
/// - `channel_axis` - The axis representing channels (typically 1 for [batch, channels, spatial...]).
/// - `input_shape` - Shape of the input tensor.
/// - `gamma` - Scale parameter (trainable).
/// - `beta` - Shift parameter (trainable).
/// - `training` - Whether the layer is in training mode or inference mode.
/// - `x_normalized` - Normalized input (used in backward pass).
/// - `x_centered` - Centered input (used in backward pass).
/// - `mean` - Mean computed during forward pass (used in backward pass).
/// - `std_dev` - Standard deviation computed during forward pass (used in backward pass).
/// - `grad_gamma` - Gradient for gamma parameter.
/// - `grad_beta` - Gradient for beta parameter.
/// - `optimizer_cache` - Cache for optimizer states.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array3;
///
/// // Create an InstanceNormalization layer for input shape [batch, channels, spatial]
/// // with channels at axis 1 (default)
/// let mut in_layer = InstanceNormalization::new(vec![4, 3, 32], 1, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array3::ones((4, 3, 32)).into_dyn();
///
/// // During training, normalizes each channel of each sample independently
/// let output = in_layer.forward(&input).unwrap();
/// ```
pub struct InstanceNormalization {
    epsilon: f32,
    channel_axis: usize,
    input_shape: Vec<usize>,
    gamma: Tensor,
    beta: Tensor,
    training: bool,
    // Cache for backward pass
    x_normalized: Option<Tensor>,
    x_centered: Option<Tensor>,
    mean: Option<Tensor>,
    std_dev: Option<Tensor>,
    // Gradients
    grad_gamma: Option<Tensor>,
    grad_beta: Option<Tensor>,
    // Optimizer cache
    optimizer_cache: OptimizerCacheNormalizationLayer,
}

impl InstanceNormalization {
    /// Creates a new InstanceNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor.
    /// - `channel_axis` - The axis representing channels. For standard input format \[batch, channels, ...\],
    ///   this should be 1. The normalization will be computed across all axes except axis 0 (batch) and
    ///   `channel_axis`.
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5).
    ///
    /// # Returns
    ///
    /// * `Result<Self, ModelError>` - A new instance of the InstanceNormalization layer, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::InputValidationError` if:
    /// - `input_shape` is empty
    /// - `channel_axis` is out of bounds or is 0 (batch axis)
    /// - `epsilon` is not positive or not finite
    pub fn new(
        input_shape: Vec<usize>,
        channel_axis: usize,
        epsilon: f32,
    ) -> Result<Self, ModelError> {
        validate_input_shape_not_empty(&input_shape)?;
        validate_channel_axis_with_shape(channel_axis, &input_shape)?;
        validate_epsilon(epsilon)?;

        // For instance normalization, parameters have the shape of the channel dimension
        let param_shape = if input_shape.len() > channel_axis {
            vec![input_shape[channel_axis]]
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(InstanceNormalization {
            epsilon,
            channel_axis,
            input_shape,
            gamma: Tensor::ones(param_shape_ndarray),
            beta: Tensor::zeros(param_shape_ndarray),
            training: true,
            x_normalized: None,
            x_centered: None,
            mean: None,
            std_dev: None,
            grad_gamma: None,
            grad_beta: None,
            optimizer_cache: OptimizerCacheNormalizationLayer::default(),
        })
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the InstanceNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable).
    /// - `beta` - Shift parameter (trainable).
    pub fn set_weights(&mut self, gamma: Tensor, beta: Tensor) {
        self.gamma = gamma;
        self.beta = beta;
    }
}

impl Layer for InstanceNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Instance normalization")?;
        validate_channel_axis(self.channel_axis, input.ndim())?;

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let num_channels = input_shape[self.channel_axis];

        // Calculate spatial size (all dimensions except batch and channel)
        let spatial_size: usize = input_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 0 && i != self.channel_axis)
            .map(|(_, &dim)| dim)
            .product();

        let total_elements = input.len();

        // Build mean shape: keep batch and channel dimensions, set others to 1
        let mut mean_shape = vec![1; input.ndim()];
        mean_shape[0] = batch_size;
        mean_shape[self.channel_axis] = num_channels;

        let mean = if total_elements >= INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut mean_flat = vec![0.0f32; batch_size * num_channels];

            mean_flat
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, mean_val)| {
                    let batch_idx = idx / num_channels;
                    let channel_idx = idx % num_channels;

                    let start = (batch_idx * num_channels + channel_idx) * spatial_size;
                    let end = start + spatial_size;

                    let sum: f32 = input.as_slice().unwrap()[start..end].iter().sum();
                    *mean_val = sum / spatial_size as f32;
                });

            Tensor::from_shape_vec(mean_shape.as_slice(), mean_flat).unwrap()
        } else {
            // Sequential computation
            let mut mean_flat = vec![0.0f32; batch_size * num_channels];

            for batch_idx in 0..batch_size {
                for channel_idx in 0..num_channels {
                    let start = (batch_idx * num_channels + channel_idx) * spatial_size;
                    let end = start + spatial_size;

                    let sum: f32 = input.as_slice().unwrap()[start..end].iter().sum();
                    mean_flat[batch_idx * num_channels + channel_idx] = sum / spatial_size as f32;
                }
            }

            Tensor::from_shape_vec(mean_shape.as_slice(), mean_flat).unwrap()
        };

        // Center the data and compute variance
        let (x_centered, var) = if total_elements >= INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel centering and variance computation
            let mut x_centered = Tensor::zeros(input.raw_dim());
            let mut squared_diff = Tensor::zeros(input.raw_dim());

            x_centered
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(squared_diff.as_slice_mut().unwrap().par_iter_mut())
                .zip(input.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((centered, sq_diff), &val))| {
                    let instance_idx = i / spatial_size;
                    let mean_val = mean.as_slice().unwrap()[instance_idx];
                    let diff = val - mean_val;
                    *centered = diff;
                    *sq_diff = diff * diff;
                });

            // Compute variance for each (batch, channel) pair
            let mut var_flat = vec![0.0f32; batch_size * num_channels];

            var_flat
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, var_val)| {
                    let start = idx * spatial_size;
                    let end = start + spatial_size;

                    let sum: f32 = squared_diff.as_slice().unwrap()[start..end].iter().sum();
                    *var_val = sum / spatial_size as f32;
                });

            let var = Tensor::from_shape_vec(mean_shape.as_slice(), var_flat).unwrap();
            (x_centered, var)
        } else {
            // Sequential computation
            let x_centered = input - &mean;

            let mut var_flat = vec![0.0f32; batch_size * num_channels];

            for batch_idx in 0..batch_size {
                for channel_idx in 0..num_channels {
                    let start = (batch_idx * num_channels + channel_idx) * spatial_size;
                    let end = start + spatial_size;

                    let sum: f32 = x_centered.as_slice().unwrap()[start..end]
                        .iter()
                        .map(|&x| x * x)
                        .sum();
                    var_flat[batch_idx * num_channels + channel_idx] = sum / spatial_size as f32;
                }
            }

            let var = Tensor::from_shape_vec(mean_shape.as_slice(), var_flat).unwrap();
            (x_centered, var)
        };

        // Normalize
        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = if total_elements >= INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel normalization
            let mut x_normalized = Tensor::zeros(x_centered.raw_dim());

            x_normalized
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (norm, &centered))| {
                    let instance_idx = i / spatial_size;
                    let std_val = std_dev.as_slice().unwrap()[instance_idx];
                    *norm = centered / std_val;
                });

            x_normalized
        } else {
            // Sequential normalization
            &x_centered / &std_dev
        };

        // Scale and shift
        // Reshape gamma and beta to match the input shape for broadcasting
        let mut gamma_shape = vec![1; input.ndim()];
        gamma_shape[self.channel_axis] = num_channels;
        let mut beta_shape = vec![1; input.ndim()];
        beta_shape[self.channel_axis] = num_channels;

        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();
        let beta_broadcast = self
            .beta
            .clone()
            .into_shape_with_order(beta_shape.as_slice())
            .unwrap();

        let output = if total_elements >= INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel scale and shift
            let mut output = Tensor::zeros(x_normalized.raw_dim());

            output
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_normalized.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (out, &norm))| {
                    let channel_idx = (i / spatial_size) % num_channels;
                    let gamma_val = self.gamma.as_slice().unwrap()[channel_idx];
                    let beta_val = self.beta.as_slice().unwrap()[channel_idx];
                    *out = norm * gamma_val + beta_val;
                });

            output
        } else {
            // Sequential scale and shift
            &x_normalized * &gamma_broadcast + &beta_broadcast
        };

        // Cache values for backward pass
        self.x_normalized = Some(x_normalized);
        self.x_centered = Some(x_centered);
        self.mean = Some(mean);
        self.std_dev = Some(std_dev);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        let input_shape = grad_output.shape();
        let batch_size = input_shape[0];
        let num_channels = input_shape[self.channel_axis];

        let spatial_size: usize = input_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 0 && i != self.channel_axis)
            .map(|(_, &dim)| dim)
            .product();
        let spatial_size_f32 = spatial_size as f32;

        let x_normalized = self.x_normalized.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let x_centered = self.x_centered.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let std_dev = self.std_dev.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let total_elements = grad_output.len();

        // Compute gradients for gamma and beta
        // Sum over all spatial dimensions and batch dimension for each channel
        let mut grad_gamma = Tensor::zeros(self.gamma.raw_dim());
        let mut grad_beta = Tensor::zeros(self.beta.raw_dim());

        compute_normalization_layer_parameter_gradients!(
            grad_gamma,
            grad_beta,
            grad_output,
            x_normalized,
            batch_size,
            num_channels,
            spatial_size,
            total_elements,
            INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD
        );

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        // Compute gradient with respect to normalized input
        let grad_x_normalized = if total_elements >= INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());

            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    let channel_idx = (i / spatial_size) % num_channels;
                    let gamma_val = self.gamma.as_slice().unwrap()[channel_idx];
                    *g_norm = g_out * gamma_val;
                });

            grad_x_norm
        } else {
            // Sequential computation
            let mut gamma_shape = vec![1; grad_output.ndim()];
            gamma_shape[self.channel_axis] = num_channels;
            let gamma_broadcast = self
                .gamma
                .clone()
                .into_shape_with_order(gamma_shape.as_slice())
                .unwrap();
            grad_output * &gamma_broadcast
        };

        // Compute inverse standard deviation
        let inv_std = std_dev.mapv(|x| 1.0 / x);

        // Compute gradient with respect to variance and mean for each (batch, channel) instance
        let grad_input = if total_elements >= INSTANCE_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_var_flat = vec![0.0f32; batch_size * num_channels];
            let mut grad_mean_flat = vec![0.0f32; batch_size * num_channels];

            // Compute grad_var and grad_mean for each instance
            grad_var_flat
                .par_iter_mut()
                .zip(grad_mean_flat.par_iter_mut())
                .enumerate()
                .for_each(|(idx, (g_var, g_mean))| {
                    let start = idx * spatial_size;
                    let end = start + spatial_size;
                    let inv_std_val = inv_std.as_slice().unwrap()[idx];

                    let grad_var_sum: f32 = (start..end)
                        .map(|i| {
                            grad_x_normalized.as_slice().unwrap()[i]
                                * x_centered.as_slice().unwrap()[i]
                        })
                        .sum();

                    *g_var = grad_var_sum * -0.5 * inv_std_val * inv_std_val * inv_std_val;

                    let grad_mean_1: f32 = (start..end)
                        .map(|i| -grad_x_normalized.as_slice().unwrap()[i])
                        .sum();

                    let x_sum: f32 = x_centered.as_slice().unwrap()[start..end].iter().sum();
                    let grad_mean_2 = *g_var * (-2.0 * x_sum / spatial_size_f32);

                    *g_mean = grad_mean_1 * inv_std_val + grad_mean_2;
                });

            // Compute gradient with respect to input
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());

            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_x_normalized.as_slice().unwrap().par_iter())
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((g_inp, &g_norm), &x_cent))| {
                    let instance_idx = i / spatial_size;
                    let inv_std_val = inv_std.as_slice().unwrap()[instance_idx];
                    let grad_var_val = grad_var_flat[instance_idx];
                    let grad_mean_val = grad_mean_flat[instance_idx];

                    *g_inp = g_norm * inv_std_val
                        + grad_var_val * x_cent * 2.0 / spatial_size_f32
                        + grad_mean_val / spatial_size_f32;
                });

            grad_inp
        } else {
            // Sequential computation
            let mut mean_shape = vec![1; grad_output.ndim()];
            mean_shape[0] = batch_size;
            mean_shape[self.channel_axis] = num_channels;

            let mut grad_var_flat = vec![0.0f32; batch_size * num_channels];
            let mut grad_mean_flat = vec![0.0f32; batch_size * num_channels];

            for idx in 0..(batch_size * num_channels) {
                let start = idx * spatial_size;
                let end = start + spatial_size;
                let inv_std_val = inv_std.as_slice().unwrap()[idx];

                let grad_var_sum: f32 = (start..end)
                    .map(|i| {
                        grad_x_normalized.as_slice().unwrap()[i] * x_centered.as_slice().unwrap()[i]
                    })
                    .sum();

                grad_var_flat[idx] = grad_var_sum * -0.5 * inv_std_val * inv_std_val * inv_std_val;

                let grad_mean_1: f32 = (start..end)
                    .map(|i| -grad_x_normalized.as_slice().unwrap()[i])
                    .sum();

                let x_sum: f32 = x_centered.as_slice().unwrap()[start..end].iter().sum();
                let grad_mean_2 = grad_var_flat[idx] * (-2.0 * x_sum / spatial_size_f32);

                grad_mean_flat[idx] = grad_mean_1 * inv_std_val + grad_mean_2;
            }

            let grad_var = Tensor::from_shape_vec(mean_shape.as_slice(), grad_var_flat).unwrap();
            let grad_mean = Tensor::from_shape_vec(mean_shape.as_slice(), grad_mean_flat).unwrap();

            &grad_x_normalized * &inv_std
                + &grad_var * (x_centered * 2.0 / spatial_size_f32)
                + &grad_mean / spatial_size_f32
        };

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "InstanceNormalization"
    }

    fn output_shape(&self) -> String {
        normalization_layer_output_shape!(self)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.gamma.len() + self.beta.len())
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        normalization_layer_update_parameters_sgd!(self, lr)
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        normalization_layer_update_parameters_adam!(self, lr, beta1, beta2, epsilon, t)
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        normalization_layer_update_parameters_rmsprop!(self, lr, rho, epsilon)
    }

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        normalization_layer_update_parameters_ada_grad!(self, lr, epsilon)
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::InstanceNormalizationLayer(InstanceNormalizationLayerWeight {
            gamma: &self.gamma,
            beta: &self.beta,
        })
    }

    mode_dependent_layer_trait!();
}
