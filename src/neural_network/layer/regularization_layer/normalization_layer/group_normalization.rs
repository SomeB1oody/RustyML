use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::{GroupNormalizationLayerWeight, LayerWeight};
use crate::neural_network::layer::regularization_layer::input_validation_function::{
    validate_channel_axis, validate_channel_axis_with_shape, validate_epsilon,
    validate_input_shape, validate_input_shape_not_empty, validate_min_input_ndim,
    validate_num_groups, validate_num_groups_positive,
};
use crate::neural_network::neural_network_trait::Layer;
use crate::neural_network::optimizer::OptimizerCacheNormalizationLayer;
use crate::neural_network::optimizer::{
    ada_grad::AdaGradStatesNormalizationLayer, adam::AdamStatesNormalizationLayer,
    rms_prop::RMSpropCacheNormalizationLayer,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

/// Threshold for switching between sequential and parallel group normalization computation.
/// Based on total elements in the tensor.
const GROUP_NORMALIZATION_PARALLEL_THRESHOLD: usize = 1024;

/// Macro to compute start and end indices for a specific channel within a batch
macro_rules! channel_range {
    ($batch_idx:expr, $channel_idx:expr, $num_channels:expr, $spatial_size:expr) => {{
        let start = ($batch_idx * $num_channels + $channel_idx) * $spatial_size;
        let end = start + $spatial_size;
        (start, end)
    }};
}

/// Group Normalization layer for neural networks.
///
/// Divides channels into groups and normalizes within each group per sample,
/// reducing dependence on batch size. Channel divisibility is validated during
/// the forward pass.
///
/// # Fields
///
/// - `num_groups` - Number of groups to divide channels into
/// - `epsilon` - Small constant for numerical stability in normalization
/// - `channel_axis` - Axis representing channels (typically 1 for \[batch, channels, spatial...\])
/// - `input_shape` - Shape of the input tensor
/// - `gamma` - Scale parameter (trainable)
/// - `beta` - Shift parameter (trainable)
/// - `training` - Whether the layer is in training mode or inference mode
/// - `x_normalized` - Normalized input (used in backward pass)
/// - `x_centered` - Centered input (used in backward pass)
/// - `mean` - Mean computed during forward pass (used in backward pass)
/// - `std_dev` - Standard deviation computed during forward pass (used in backward pass)
/// - `grad_gamma` - Gradient for gamma parameter
/// - `grad_beta` - Gradient for beta parameter
/// - `optimizer_cache` - Cache for optimizer states
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::neural_network_trait::Layer;
/// use ndarray::Array3;
///
/// // Create a GroupNormalization layer for input shape [batch, channels, spatial]
/// // with 4 groups dividing 8 channels
/// let mut gn_layer = GroupNormalization::new(vec![4, 8, 32], 4, 1, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array3::ones((4, 8, 32)).into_dyn();
///
/// // During training, normalizes within each group independently
/// let output = gn_layer.forward(&input).unwrap();
/// ```
pub struct GroupNormalization {
    num_groups: usize,
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

impl GroupNormalization {
    /// Creates a new GroupNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor
    /// - `num_groups` - Number of groups to divide channels into
    /// - `channel_axis` - Axis representing channels for inputs like \[batch, channels, ...\]
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New GroupNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_shape` is empty
    /// - `ModelError::InputValidationError` - If `num_groups` is 0
    /// - `ModelError::InputValidationError` - If `epsilon` is not positive
    /// - `ModelError::InputValidationError` - If `channel_axis` is out of bounds or is 0 (batch axis)
    pub fn new(
        input_shape: Vec<usize>,
        num_groups: usize,
        channel_axis: usize,
        epsilon: f32,
    ) -> Result<Self, ModelError> {
        validate_input_shape_not_empty(&input_shape)?;
        validate_num_groups_positive(num_groups)?;
        validate_epsilon(epsilon)?;
        validate_channel_axis_with_shape(channel_axis, &input_shape)?;

        // For group normalization, parameters have the shape of the channel dimension
        let param_shape = if input_shape.len() > channel_axis {
            let num_channels = input_shape[channel_axis];
            // Note: We don't validate divisibility here to allow the layer to be created.
            // Validation happens in forward() to match the error handling pattern of other layers.
            vec![num_channels]
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(GroupNormalization {
            num_groups,
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

    /// Sets the weights for the GroupNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable)
    /// - `beta` - Shift parameter (trainable)
    pub fn set_weights(&mut self, gamma: Tensor, beta: Tensor) {
        self.gamma = gamma;
        self.beta = beta;
    }
}

impl Layer for GroupNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_min_input_ndim(input.ndim(), 3, "Group normalization")?;
        validate_channel_axis(self.channel_axis, input.ndim())?;

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let num_channels = input_shape[self.channel_axis];

        validate_num_groups(num_channels, self.num_groups)?;

        let channels_per_group = num_channels / self.num_groups;

        // Calculate spatial size (all dimensions except batch and channel)
        let spatial_size: usize = input_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 0 && i != self.channel_axis)
            .map(|(_, &dim)| dim)
            .product();

        let group_size = channels_per_group * spatial_size;
        let total_elements = input.len();

        // Build mean shape: keep batch and group dimensions
        let num_instances = batch_size * self.num_groups;
        let mut mean_shape = vec![1; input.ndim()];
        mean_shape[0] = batch_size;
        mean_shape[self.channel_axis] = self.num_groups;

        // Closure to compute mean for a single (batch, group) instance
        let compute_group_mean = |batch_idx: usize, group_idx: usize| -> f32 {
            let mut sum = 0.0f32;
            for ch in 0..channels_per_group {
                let channel_idx = group_idx * channels_per_group + ch;
                let start = (batch_idx * num_channels + channel_idx) * spatial_size;
                let end = start + spatial_size;
                sum += input.as_slice().unwrap()[start..end].iter().sum::<f32>();
            }
            sum / group_size as f32
        };

        // Compute mean for each (batch, group) pair
        let mean = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mean_flat: Vec<f32> = (0..num_instances)
                .into_par_iter()
                .map(|idx| {
                    let batch_idx = idx / self.num_groups;
                    let group_idx = idx % self.num_groups;
                    compute_group_mean(batch_idx, group_idx)
                })
                .collect();

            Tensor::from_shape_vec(mean_shape.as_slice(), mean_flat).unwrap()
        } else {
            // Sequential computation
            let mut mean_flat = vec![0.0f32; num_instances];

            for batch_idx in 0..batch_size {
                for group_idx in 0..self.num_groups {
                    mean_flat[batch_idx * self.num_groups + group_idx] =
                        compute_group_mean(batch_idx, group_idx);
                }
            }

            Tensor::from_shape_vec(mean_shape.as_slice(), mean_flat).unwrap()
        };

        // Expand mean to match input shape for broadcasting
        let mut mean_expanded_shape = input_shape.to_vec();
        mean_expanded_shape[self.channel_axis] = self.num_groups;

        // Map each channel to its group mean
        let x_centered = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut x_cent = Tensor::zeros(input.raw_dim());

            x_cent
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(input.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (cent, &val))| {
                    let flat_idx = i / spatial_size;
                    let batch_idx = flat_idx / num_channels;
                    let channel_idx = flat_idx % num_channels;
                    let group_idx = channel_idx / channels_per_group;
                    let mean_idx = batch_idx * self.num_groups + group_idx;
                    let mean_val = mean.as_slice().unwrap()[mean_idx];
                    *cent = val - mean_val;
                });

            x_cent
        } else {
            // Sequential computation
            let mut x_cent = Tensor::zeros(input.raw_dim());

            for i in 0..total_elements {
                let flat_idx = i / spatial_size;
                let batch_idx = flat_idx / num_channels;
                let channel_idx = flat_idx % num_channels;
                let group_idx = channel_idx / channels_per_group;
                let mean_idx = batch_idx * self.num_groups + group_idx;
                let mean_val = mean.as_slice().unwrap()[mean_idx];
                x_cent.as_slice_mut().unwrap()[i] = input.as_slice().unwrap()[i] - mean_val;
            }

            x_cent
        };

        // Closure to compute variance for a single (batch, group) instance
        let compute_group_variance = |batch_idx: usize, group_idx: usize| -> f32 {
            let mut sum_sq = 0.0f32;
            for ch in 0..channels_per_group {
                let channel_idx = group_idx * channels_per_group + ch;
                let (start, end) =
                    channel_range!(batch_idx, channel_idx, num_channels, spatial_size);

                for i in start..end {
                    let val = x_centered.as_slice().unwrap()[i];
                    sum_sq += val * val;
                }
            }
            sum_sq / group_size as f32
        };

        // Compute variance for each (batch, group) pair
        let var = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let var_flat: Vec<f32> = (0..num_instances)
                .into_par_iter()
                .map(|idx| {
                    let batch_idx = idx / self.num_groups;
                    let group_idx = idx % self.num_groups;
                    compute_group_variance(batch_idx, group_idx)
                })
                .collect();

            Tensor::from_shape_vec(mean_shape.as_slice(), var_flat).unwrap()
        } else {
            // Sequential computation
            let mut var_flat = vec![0.0f32; num_instances];

            for batch_idx in 0..batch_size {
                for group_idx in 0..self.num_groups {
                    var_flat[batch_idx * self.num_groups + group_idx] =
                        compute_group_variance(batch_idx, group_idx);
                }
            }

            Tensor::from_shape_vec(mean_shape.as_slice(), var_flat).unwrap()
        };

        // Normalize
        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel normalization
            let mut x_norm = Tensor::zeros(x_centered.raw_dim());

            x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (norm, &centered))| {
                    let flat_idx = i / spatial_size;
                    let batch_idx = flat_idx / num_channels;
                    let channel_idx = flat_idx % num_channels;
                    let group_idx = channel_idx / channels_per_group;
                    let std_idx = batch_idx * self.num_groups + group_idx;
                    let std_val = std_dev.as_slice().unwrap()[std_idx];
                    *norm = centered / std_val;
                });

            x_norm
        } else {
            // Sequential normalization
            let mut x_norm = Tensor::zeros(x_centered.raw_dim());

            for i in 0..total_elements {
                let flat_idx = i / spatial_size;
                let batch_idx = flat_idx / num_channels;
                let channel_idx = flat_idx % num_channels;
                let group_idx = channel_idx / channels_per_group;
                let std_idx = batch_idx * self.num_groups + group_idx;
                let std_val = std_dev.as_slice().unwrap()[std_idx];
                x_norm.as_slice_mut().unwrap()[i] = x_centered.as_slice().unwrap()[i] / std_val;
            }

            x_norm
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

        let output = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel scale and shift
            let mut out = Tensor::zeros(x_normalized.raw_dim());

            out.as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_normalized.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (output_val, &norm))| {
                    let flat_idx = i / spatial_size;
                    let channel_idx = flat_idx % num_channels;
                    let gamma_val = self.gamma.as_slice().unwrap()[channel_idx];
                    let beta_val = self.beta.as_slice().unwrap()[channel_idx];
                    *output_val = norm * gamma_val + beta_val;
                });

            out
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
        let channels_per_group = num_channels / self.num_groups;

        let spatial_size: usize = input_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 0 && i != self.channel_axis)
            .map(|(_, &dim)| dim)
            .product();
        let group_size = channels_per_group * spatial_size;
        let group_size_f32 = group_size as f32;

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
        let num_instances = batch_size * self.num_groups;

        // Compute gradients for gamma and beta
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
            GROUP_NORMALIZATION_PARALLEL_THRESHOLD
        );

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        // Compute gradient with respect to normalized input
        let grad_x_normalized = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());

            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    let flat_idx = i / spatial_size;
                    let channel_idx = flat_idx % num_channels;
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

        // Closure to compute gradient statistics for a single (batch, group) instance
        let compute_group_grad_stats =
            |batch_idx: usize, group_idx: usize, inv_std_val: f32| -> (f32, f32) {
                let mut grad_var_sum = 0.0f32;
                let mut grad_mean_1 = 0.0f32;
                let mut x_sum = 0.0f32;

                for ch in 0..channels_per_group {
                    let channel_idx = group_idx * channels_per_group + ch;
                    let (start, end) =
                        channel_range!(batch_idx, channel_idx, num_channels, spatial_size);

                    for i in start..end {
                        grad_var_sum += grad_x_normalized.as_slice().unwrap()[i]
                            * x_centered.as_slice().unwrap()[i];
                        grad_mean_1 += -grad_x_normalized.as_slice().unwrap()[i];
                        x_sum += x_centered.as_slice().unwrap()[i];
                    }
                }

                let grad_var = grad_var_sum * -0.5 * inv_std_val * inv_std_val * inv_std_val;
                let grad_mean_2 = grad_var * (-2.0 * x_sum / group_size_f32);
                let grad_mean = grad_mean_1 * inv_std_val + grad_mean_2;

                (grad_var, grad_mean)
            };

        // Compute gradient with respect to variance and mean for each (batch, group) instance
        let grad_input = if total_elements >= GROUP_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let (grad_var_flat, grad_mean_flat): (Vec<f32>, Vec<f32>) = (0..num_instances)
                .into_par_iter()
                .map(|idx| {
                    let batch_idx = idx / self.num_groups;
                    let group_idx = idx % self.num_groups;
                    let inv_std_val = inv_std.as_slice().unwrap()[idx];
                    compute_group_grad_stats(batch_idx, group_idx, inv_std_val)
                })
                .unzip();

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
                    let flat_idx = i / spatial_size;
                    let batch_idx = flat_idx / num_channels;
                    let channel_idx = flat_idx % num_channels;
                    let group_idx = channel_idx / channels_per_group;
                    let group_instance_idx = batch_idx * self.num_groups + group_idx;

                    let inv_std_val = inv_std.as_slice().unwrap()[group_instance_idx];
                    let grad_var_val = grad_var_flat[group_instance_idx];
                    let grad_mean_val = grad_mean_flat[group_instance_idx];

                    *g_inp = g_norm * inv_std_val
                        + grad_var_val * x_cent * 2.0 / group_size_f32
                        + grad_mean_val / group_size_f32;
                });

            grad_inp
        } else {
            // Sequential computation
            let mut grad_var_flat = vec![0.0f32; num_instances];
            let mut grad_mean_flat = vec![0.0f32; num_instances];

            for idx in 0..num_instances {
                let batch_idx = idx / self.num_groups;
                let group_idx = idx % self.num_groups;
                let inv_std_val = inv_std.as_slice().unwrap()[idx];

                let (grad_var, grad_mean) =
                    compute_group_grad_stats(batch_idx, group_idx, inv_std_val);
                grad_var_flat[idx] = grad_var;
                grad_mean_flat[idx] = grad_mean;
            }

            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());

            for i in 0..total_elements {
                let flat_idx = i / spatial_size;
                let batch_idx = flat_idx / num_channels;
                let channel_idx = flat_idx % num_channels;
                let group_idx = channel_idx / channels_per_group;
                let group_instance_idx = batch_idx * self.num_groups + group_idx;

                let inv_std_val = inv_std.as_slice().unwrap()[group_instance_idx];
                let grad_var_val = grad_var_flat[group_instance_idx];
                let grad_mean_val = grad_mean_flat[group_instance_idx];

                grad_inp.as_slice_mut().unwrap()[i] = grad_x_normalized.as_slice().unwrap()[i]
                    * inv_std_val
                    + grad_var_val * x_centered.as_slice().unwrap()[i] * 2.0 / group_size_f32
                    + grad_mean_val / group_size_f32;
            }

            grad_inp
        };

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GroupNormalization"
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
        LayerWeight::GroupNormalizationLayer(GroupNormalizationLayerWeight {
            gamma: &self.gamma,
            beta: &self.beta,
        })
    }

    mode_dependent_layer_trait!();
}
