use super::*;

/// Threshold for switching between sequential and parallel layer normalization computation.
/// Based on total elements in the tensor.
const LAYER_NORMALIZATION_PARALLEL_THRESHOLD: usize = 1024;

/// Enum to specify which axis to normalize along in Layer Normalization
///
/// # Variants
///
/// - `Default` - Normalize along the last dimension (feature dimension) - most common case
/// - `Custom(usize)` - Normalize along a custom specified axis
#[derive(Debug, Clone, Copy)]
pub enum LayerNormalizationAxis {
    Default,
    Custom(usize),
}

/// Layer Normalization layer for neural networks, which normalizes the inputs
/// across the features for each data sample to improve training stability and speed.
///
/// Unlike Batch Normalization which normalizes across the batch dimension, Layer Normalization
/// normalizes across the feature dimensions for each individual sample. This makes it particularly
/// useful for recurrent neural networks and situations where batch sizes are small or variable.
///
/// # Fields
///
/// - `epsilon` - Small constant for numerical stability in normalization.
/// - `normalized_axis` - The axis along which to normalize (Default = last axis, Custom = specified axis).
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
/// - `m_gamma` - First moment estimate for gamma (Adam optimizer).
/// - `v_gamma` - Second moment estimate for gamma (Adam optimizer).
/// - `m_beta` - First moment estimate for beta (Adam optimizer).
/// - `v_beta` - Second moment estimate for beta (Adam optimizer).
/// - `cache_gamma` - Cache for gamma (RMSprop optimizer).
/// - `cache_beta` - Cache for beta (RMSprop optimizer).
/// - `acc_grad_gamma` - Accumulated gradient for gamma (AdaGrad optimizer).
/// - `acc_grad_beta` - Accumulated gradient for beta (AdaGrad optimizer).
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a LayerNormalization layer
/// let mut ln = LayerNormalization::new(vec![32, 128], LayerNormalizationAxis::Default, 1e-5);
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, normalizes the input
/// let output = ln.forward(&input).unwrap();
/// ```
pub struct LayerNormalization {
    epsilon: f32,
    normalized_axis: LayerNormalizationAxis,
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

impl LayerNormalization {
    /// Creates a new LayerNormalization layer.
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor.
    /// - `normalized_axis` - The axis along which to normalize (Default for last axis, Custom(usize) for specific axis).    /// - `epsilon` - Small constant for numerical stability (typically 1e-5).    ///    /// # Returns
    ///
    /// * `Self` - A new instance of the LayerNormalization layer.
    pub fn new(
        input_shape: Vec<usize>,
        normalized_axis: LayerNormalizationAxis,
        epsilon: f32,
    ) -> Self {
        // For layer normalization, we normalize across specified feature dimensions
        // The parameters should have the shape of the normalized dimensions
        let axis = match normalized_axis {
            LayerNormalizationAxis::Default => {
                if input_shape.is_empty() {
                    0
                } else {
                    input_shape.len() - 1
                }
            }
            LayerNormalizationAxis::Custom(axis) => axis,
        };

        let param_shape = if input_shape.len() > axis {
            input_shape[axis..].to_vec()
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        LayerNormalization {
            epsilon,
            normalized_axis,
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
        }
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the LayerNormalization layer.
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

impl Layer for LayerNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input shape matches expected shape
        if !self.input_shape.is_empty() && input.shape() != self.input_shape.as_slice() {
            return Err(ModelError::InputValidationError(format!(
                "Input shape mismatch: expected {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )));
        }

        // Determine the axis to normalize along
        let axis_idx = match self.normalized_axis {
            LayerNormalizationAxis::Default => {
                if input.ndim() == 0 {
                    return Err(ModelError::InputValidationError(
                        "Cannot normalize a scalar tensor".to_string(),
                    ));
                }
                input.ndim() - 1
            }
            LayerNormalizationAxis::Custom(axis) => {
                if axis >= input.ndim() {
                    return Err(ModelError::InputValidationError(format!(
                        "Normalization axis {} is out of bounds for input with {} dimensions",
                        axis,
                        input.ndim()
                    )));
                }
                axis
            }
        };

        let total_elements = input.len();

        // Compute mean along the specified axis
        let mean = input.mean_axis(Axis(axis_idx)).unwrap();

        // Insert the axis back to make broadcasting work
        let mean = mean.insert_axis(Axis(axis_idx));

        // Center the data and compute variance
        let (x_centered, var) = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel centering and variance computation
            let mut x_centered = Tensor::zeros(input.raw_dim());
            let mut squared_diff = Tensor::zeros(input.raw_dim());

            let norm_dim_size: usize = input.shape()[axis_idx..].iter().product();

            x_centered
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(squared_diff.as_slice_mut().unwrap().par_iter_mut())
                .zip(input.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((centered, sq_diff), &val))| {
                    let norm_idx = i % norm_dim_size;
                    let mean_val = mean.as_slice().unwrap()[norm_idx];
                    let diff = val - mean_val;
                    *centered = diff;
                    *sq_diff = diff * diff;
                });

            let var = squared_diff
                .mean_axis(Axis(axis_idx))
                .unwrap()
                .insert_axis(Axis(axis_idx));
            (x_centered, var)
        } else {
            // Sequential computation
            let x_centered = input - &mean;
            let var = (&x_centered * &x_centered)
                .mean_axis(Axis(axis_idx))
                .unwrap();
            let var = var.insert_axis(Axis(axis_idx));
            (x_centered, var)
        };

        // Normalize
        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel normalization
            let mut x_normalized = Tensor::zeros(x_centered.raw_dim());
            let norm_dim_size: usize = input.shape()[axis_idx..].iter().product();

            x_normalized
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (norm, &centered))| {
                    let norm_idx = i % norm_dim_size;
                    let std_val = std_dev.as_slice().unwrap()[norm_idx];
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
        let mut beta_shape = vec![1; input.ndim()];

        // Set the dimensions from axis_idx onwards to match gamma/beta shape
        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
            beta_shape[axis_idx + i] = dim;
        }

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

        let output = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel scale and shift
            let mut output = Tensor::zeros(x_normalized.raw_dim());
            let norm_dim_size: usize = input.shape()[axis_idx..].iter().product();

            output
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(x_normalized.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (out, &norm))| {
                    let norm_idx = i % norm_dim_size;
                    let gamma_val = gamma_broadcast.as_slice().unwrap()[norm_idx];
                    let beta_val = beta_broadcast.as_slice().unwrap()[norm_idx];
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

        let axis_idx = match self.normalized_axis {
            LayerNormalizationAxis::Default => grad_output.ndim() - 1,
            LayerNormalizationAxis::Custom(axis) => axis,
        };

        let x_normalized = self.x_normalized.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let x_centered = self.x_centered.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let std_dev = self.std_dev.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        // Compute gradients for gamma and beta
        // Sum over all axes except the normalized ones
        let mut grad_gamma = grad_output * x_normalized;
        let mut grad_beta = grad_output.clone();

        // Sum over all axes before the normalized axis
        for i in (0..axis_idx).rev() {
            grad_gamma = grad_gamma.sum_axis(Axis(i));
            grad_beta = grad_beta.sum_axis(Axis(i));
        }

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        let total_elements = grad_output.len();

        // Compute gradient with respect to normalized input
        // Reshape gamma for broadcasting
        let mut gamma_shape = vec![1; grad_output.ndim()];
        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
        }
        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();

        let grad_x_normalized = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());
            let norm_dim_size: usize = grad_output.shape()[axis_idx..].iter().product();

            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    let norm_idx = i % norm_dim_size;
                    let gamma_val = gamma_broadcast.as_slice().unwrap()[norm_idx];
                    *g_norm = g_out * gamma_val;
                });
            grad_x_norm
        } else {
            // Sequential computation
            grad_output * &gamma_broadcast
        };

        // Compute inverse standard deviation
        let inv_std = std_dev.mapv(|x| 1.0 / x);

        // Get the size of the normalization dimension
        let norm_size = grad_output.shape()[axis_idx] as f32;

        // Compute gradient with respect to variance
        let grad_var = (&grad_x_normalized * x_centered * -0.5).sum_axis(Axis(axis_idx));
        let grad_var = grad_var.insert_axis(Axis(axis_idx));
        let grad_var = &grad_var * &inv_std * &inv_std * &inv_std;

        // Compute gradient with respect to mean
        let grad_mean_1 = (&grad_x_normalized * -1.0).sum_axis(Axis(axis_idx));
        let grad_mean_1 = grad_mean_1.insert_axis(Axis(axis_idx));
        let grad_mean_1 = &grad_mean_1 * &inv_std;

        let x_sum = x_centered
            .sum_axis(Axis(axis_idx))
            .insert_axis(Axis(axis_idx));
        let grad_mean_2 = &grad_var * (&x_sum * -2.0 / norm_size);
        let grad_mean = grad_mean_1 + grad_mean_2;

        // Compute gradient with respect to input
        let grad_input = if total_elements >= LAYER_NORMALIZATION_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());
            let norm_dim_size: usize = grad_output.shape()[axis_idx..].iter().product();

            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_x_normalized.as_slice().unwrap().par_iter())
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((g_inp, &g_norm), &x_cent))| {
                    let norm_idx = i % norm_dim_size;
                    let inv_std_val = inv_std.as_slice().unwrap()[norm_idx];
                    let grad_var_val = grad_var.as_slice().unwrap()[norm_idx];
                    let grad_mean_val = grad_mean.as_slice().unwrap()[norm_idx];

                    *g_inp = g_norm * inv_std_val
                        + grad_var_val * x_cent * 2.0 / norm_size
                        + grad_mean_val / norm_size;
                });
            grad_inp
        } else {
            // Sequential computation
            &grad_x_normalized * &inv_std
                + &grad_var * (x_centered * 2.0 / norm_size)
                + &grad_mean / norm_size
        };

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "LayerNormalization"
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
        LayerWeight::LayerNormalizationLayer(LayerNormalizationLayerWeight {
            gamma: &self.gamma,
            beta: &self.beta,
        })
    }

    mode_dependent_layer_trait!();
}
