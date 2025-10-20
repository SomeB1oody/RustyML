use super::*;

/// Threshold for switching between sequential and parallel batch normalization computation.
/// Based on batch_size * feature_size (total elements in batch).
const BATCH_NORM_PARALLEL_THRESHOLD: usize = 1024;

/// Batch Normalization layer for neural networks, which normalizes the inputs
/// of each mini-batch to improve training stability and speed.
///
/// Batch normalization normalizes the activations of the previous layer at each batch,
/// i.e., applies a transformation that maintains the mean activation close to 0 and
/// the activation standard deviation close to 1.
///
/// # Fields
///
/// - `epsilon` - Small constant for numerical stability in normalization.
/// - `momentum` - Momentum for the moving average of mean and variance.
/// - `input_shape` - Shape of the input tensor.
/// - `gamma` - Scale parameter (trainable).
/// - `beta` - Shift parameter (trainable).
/// - `running_mean` - Running mean for inference.
/// - `running_var` - Running variance for inference.
/// - `training` - Whether the layer is in training mode or inference mode.
/// - `batch_mean` - Mean computed during forward pass (used in backward pass).
/// - `batch_var` - Variance computed during forward pass (used in backward pass).
/// - `x_normalized` - Normalized input (used in backward pass).
/// - `x_centered` - Centered input (used in backward pass).
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
/// // Create a BatchNormalization layer
/// let mut bn = BatchNormalization::new(vec![32, 128], 0.99, 1e-5);
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, normalizes the input
/// let output = bn.forward(&input).unwrap();
/// ```
pub struct BatchNormalization {
    epsilon: f32,
    momentum: f32,
    input_shape: Vec<usize>,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    // Cache for backward pass
    batch_mean: Option<Tensor>,
    batch_var: Option<Tensor>,
    x_normalized: Option<Tensor>,
    x_centered: Option<Tensor>,
    // Gradients
    grad_gamma: Option<Tensor>,
    grad_beta: Option<Tensor>,
    // Adam optimizer state
    m_gamma: Option<Tensor>,
    v_gamma: Option<Tensor>,
    m_beta: Option<Tensor>,
    v_beta: Option<Tensor>,
    // RMSprop optimizer state
    cache_gamma: Option<Tensor>,
    cache_beta: Option<Tensor>,
    // AdaGrad optimizer state
    acc_grad_gamma: Option<Tensor>,
    acc_grad_beta: Option<Tensor>,
}

impl BatchNormalization {
    /// Creates a new BatchNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor.
    /// - `momentum` - Momentum for the moving average of mean and variance (typically 0.9 or 0.99).
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5).
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the BatchNormalization layer.
    pub fn new(input_shape: Vec<usize>, momentum: f32, epsilon: f32) -> Self {
        // For batch normalization, we normalize across the batch dimension (first dimension)
        // So the parameters should have the shape of the feature dimensions
        let param_shape = if input_shape.len() > 1 {
            input_shape[1..].to_vec()
        } else {
            vec![1]
        };

        let param_shape_ndarray = param_shape.as_slice();

        BatchNormalization {
            epsilon,
            momentum,
            input_shape,
            gamma: Tensor::ones(param_shape_ndarray),
            beta: Tensor::zeros(param_shape_ndarray),
            running_mean: Tensor::zeros(param_shape_ndarray),
            running_var: Tensor::ones(param_shape_ndarray),
            training: true,
            batch_mean: None,
            batch_var: None,
            x_normalized: None,
            x_centered: None,
            grad_gamma: None,
            grad_beta: None,
            m_gamma: None,
            v_gamma: None,
            m_beta: None,
            v_beta: None,
            cache_gamma: None,
            cache_beta: None,
            acc_grad_gamma: None,
            acc_grad_beta: None,
        }
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the BatchNormalization layer.
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable).
    /// - `beta` - Shift parameter (trainable).
    /// - `running_mean` - Running mean for inference.
    /// - `running_var` - Running variance for inference.
    pub fn set_weights(
        &mut self,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
    ) {
        self.gamma = gamma;
        self.beta = beta;
        self.running_mean = running_mean;
        self.running_var = running_var;
    }
}

impl Layer for BatchNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input shape matches expected shape
        if !self.input_shape.is_empty() && input.shape() != self.input_shape.as_slice() {
            return Err(ModelError::InputValidationError(format!(
                "Input shape mismatch: expected {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )));
        }

        if self.training {
            let total_elements = input.len();
            // Compute mean across batch dimension (axis 0)
            let batch_mean = input.mean_axis(Axis(0)).unwrap();

            // Center the data and compute variance
            let (x_centered, batch_var) = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
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
                        let feature_size = batch_mean.len();
                        let feature_idx = i % feature_size;
                        let mean_val = batch_mean.as_slice().unwrap()[feature_idx];
                        let diff = val - mean_val;
                        *centered = diff;
                        *sq_diff = diff * diff;
                    });

                let batch_var = squared_diff.mean_axis(Axis(0)).unwrap();
                (x_centered, batch_var)
            } else {
                // Sequential computation for small batches
                let x_centered = input - &batch_mean;
                let batch_var = (&x_centered * &x_centered).mean_axis(Axis(0)).unwrap();
                (x_centered, batch_var)
            };

            // Normalize
            let std_dev = (&batch_var + self.epsilon).mapv(|x| x.sqrt());
            let x_normalized = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
                // Parallel normalization
                let mut x_normalized = Tensor::zeros(x_centered.raw_dim());

                x_normalized
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .zip(x_centered.as_slice().unwrap().par_iter())
                    .enumerate()
                    .for_each(|(i, (norm, &centered))| {
                        let feature_size = std_dev.len();
                        let feature_idx = i % feature_size;
                        let std_val = std_dev.as_slice().unwrap()[feature_idx];
                        *norm = centered / std_val;
                    });

                x_normalized
            } else {
                // Sequential normalization
                &x_centered / &std_dev
            };

            // Scale and shift
            let output = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
                // Parallel scale and shift
                let mut output = Tensor::zeros(x_normalized.raw_dim());

                output
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .zip(x_normalized.as_slice().unwrap().par_iter())
                    .enumerate()
                    .for_each(|(i, (out, &norm))| {
                        let feature_size = self.gamma.len();
                        let feature_idx = i % feature_size;
                        let gamma_val = self.gamma.as_slice().unwrap()[feature_idx];
                        let beta_val = self.beta.as_slice().unwrap()[feature_idx];
                        *out = norm * gamma_val + beta_val;
                    });

                output
            } else {
                // Sequential scale and shift
                &x_normalized * &self.gamma + &self.beta
            };

            // Update running statistics
            self.running_mean =
                &self.running_mean * self.momentum + &batch_mean * (1.0 - self.momentum);
            self.running_var =
                &self.running_var * self.momentum + &batch_var * (1.0 - self.momentum);

            // Cache values for backward pass
            self.batch_mean = Some(batch_mean);
            self.batch_var = Some(batch_var);
            self.x_normalized = Some(x_normalized);
            self.x_centered = Some(x_centered);

            Ok(output)
        } else {
            // Inference mode: use running statistics
            let std_dev = (&self.running_var + self.epsilon).mapv(|x| x.sqrt());
            let x_normalized = (input - &self.running_mean) / &std_dev;
            let output = &x_normalized * &self.gamma + &self.beta;

            Ok(output)
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        let batch_size = grad_output.shape()[0] as f32;
        let total_elements = grad_output.len();

        let x_normalized = self.x_normalized.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let x_centered = self.x_centered.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let batch_var = self.batch_var.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        // Compute gradients for gamma and beta
        let grad_gamma = (grad_output * x_normalized).sum_axis(Axis(0));
        let grad_beta = grad_output.sum_axis(Axis(0));

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        // Compute gradient with respect to normalized input
        let grad_x_normalized = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_x_norm = Tensor::zeros(grad_output.raw_dim());
            grad_x_norm
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_output.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, (g_norm, &g_out))| {
                    let feature_size = self.gamma.len();
                    let feature_idx = i % feature_size;
                    let gamma_val = self.gamma.as_slice().unwrap()[feature_idx];
                    *g_norm = g_out * gamma_val;
                });
            grad_x_norm
        } else {
            // Sequential computation
            grad_output * &self.gamma
        };

        // Compute gradient with respect to variance
        let std_dev = (batch_var + self.epsilon).mapv(|x| x.sqrt());
        let inv_std = std_dev.mapv(|x| 1.0 / x);

        let grad_var = (&grad_x_normalized * x_centered * -0.5).sum_axis(Axis(0))
            * &inv_std
            * &inv_std
            * &inv_std;

        // Compute gradient with respect to mean
        let grad_mean_1 = (&grad_x_normalized * -1.0).sum_axis(Axis(0)) * &inv_std;
        let grad_mean_2 = &grad_var * (x_centered.sum_axis(Axis(0)) * -2.0 / batch_size);
        let grad_mean = grad_mean_1 + grad_mean_2;

        // Compute gradient with respect to input
        let grad_input = if total_elements >= BATCH_NORM_PARALLEL_THRESHOLD {
            // Parallel computation
            let mut grad_inp = Tensor::zeros(grad_output.raw_dim());
            grad_inp
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .zip(grad_x_normalized.as_slice().unwrap().par_iter())
                .zip(x_centered.as_slice().unwrap().par_iter())
                .enumerate()
                .for_each(|(i, ((g_inp, &g_norm), &x_cent))| {
                    let feature_size = inv_std.len();
                    let feature_idx = i % feature_size;
                    let inv_std_val = inv_std.as_slice().unwrap()[feature_idx];
                    let grad_var_val = grad_var.as_slice().unwrap()[feature_idx];
                    let grad_mean_val = grad_mean.as_slice().unwrap()[feature_idx];

                    *g_inp = g_norm * inv_std_val
                        + grad_var_val * x_cent * 2.0 / batch_size
                        + grad_mean_val / batch_size;
                });
            grad_inp
        } else {
            // Sequential computation
            &grad_x_normalized * &inv_std
                + &grad_var * (x_centered * 2.0 / batch_size)
                + &grad_mean / batch_size
        };

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "BatchNormalization"
    }

    fn output_shape(&self) -> String {
        if !self.input_shape.is_empty() {
            format!(
                "({})",
                self.input_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::from("Unknown")
        }
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.gamma.len() + self.beta.len())
    }

    fn update_parameters_sgd(&mut self, lr: f32) {
        if let Some(grad_gamma) = &self.grad_gamma {
            self.gamma = &self.gamma - &(grad_gamma * lr);
        }
        if let Some(grad_beta) = &self.grad_beta {
            self.beta = &self.beta - &(grad_beta * lr);
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        // Initialize moment estimates if not already done
        if self.m_gamma.is_none() {
            self.m_gamma = Some(Tensor::zeros(self.gamma.raw_dim()));
            self.v_gamma = Some(Tensor::zeros(self.gamma.raw_dim()));
            self.m_beta = Some(Tensor::zeros(self.beta.raw_dim()));
            self.v_beta = Some(Tensor::zeros(self.beta.raw_dim()));
        }

        if let Some(grad_gamma) = &self.grad_gamma {
            let m_gamma = self.m_gamma.as_mut().unwrap();
            let v_gamma = self.v_gamma.as_mut().unwrap();

            // Update biased first moment estimate
            *m_gamma = m_gamma.clone() * beta1 + grad_gamma * (1.0 - beta1);

            // Update biased second raw moment estimate
            *v_gamma = v_gamma.clone() * beta2 + &(grad_gamma * grad_gamma) * (1.0 - beta2);

            // Compute bias-corrected first moment estimate
            let m_hat = m_gamma.clone() / (1.0 - beta1.powi(t as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = v_gamma.clone() / (1.0 - beta2.powi(t as i32));

            // Update parameters
            self.gamma = &self.gamma - &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + epsilon) * lr);
        }

        if let Some(grad_beta) = &self.grad_beta {
            let m_beta = self.m_beta.as_mut().unwrap();
            let v_beta = self.v_beta.as_mut().unwrap();

            *m_beta = m_beta.clone() * beta1 + grad_beta * (1.0 - beta1);
            *v_beta = v_beta.clone() * beta2 + &(grad_beta * grad_beta) * (1.0 - beta2);

            let m_hat = m_beta.clone() / (1.0 - beta1.powi(t as i32));
            let v_hat = v_beta.clone() / (1.0 - beta2.powi(t as i32));

            self.beta = &self.beta - &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + epsilon) * lr);
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        // Initialize cache if not already done
        if self.cache_gamma.is_none() {
            self.cache_gamma = Some(Tensor::zeros(self.gamma.raw_dim()));
            self.cache_beta = Some(Tensor::zeros(self.beta.raw_dim()));
        }

        if let Some(grad_gamma) = &self.grad_gamma {
            let cache_gamma = self.cache_gamma.as_mut().unwrap();

            // Update cache
            *cache_gamma = cache_gamma.clone() * rho + &(grad_gamma * grad_gamma) * (1.0 - rho);

            // Update parameters
            self.gamma =
                &self.gamma - &(grad_gamma / &(cache_gamma.mapv(|x| x.sqrt()) + epsilon) * lr);
        }

        if let Some(grad_beta) = &self.grad_beta {
            let cache_beta = self.cache_beta.as_mut().unwrap();

            *cache_beta = cache_beta.clone() * rho + &(grad_beta * grad_beta) * (1.0 - rho);

            self.beta = &self.beta - &(grad_beta / &(cache_beta.mapv(|x| x.sqrt()) + epsilon) * lr);
        }
    }

    fn update_parameters_ada_grad(&mut self, lr: f32, epsilon: f32) {
        // Initialize accumulated gradient if not already done
        if self.acc_grad_gamma.is_none() {
            self.acc_grad_gamma = Some(Tensor::zeros(self.gamma.raw_dim()));
            self.acc_grad_beta = Some(Tensor::zeros(self.beta.raw_dim()));
        }

        if let Some(grad_gamma) = &self.grad_gamma {
            let acc_grad_gamma = self.acc_grad_gamma.as_mut().unwrap();

            // Accumulate squared gradients
            *acc_grad_gamma = acc_grad_gamma.clone() + &(grad_gamma * grad_gamma);

            // Update parameters
            self.gamma =
                &self.gamma - &(grad_gamma / &(acc_grad_gamma.mapv(|x| x.sqrt()) + epsilon) * lr);
        }

        if let Some(grad_beta) = &self.grad_beta {
            let acc_grad_beta = self.acc_grad_beta.as_mut().unwrap();

            *acc_grad_beta = acc_grad_beta.clone() + &(grad_beta * grad_beta);

            self.beta =
                &self.beta - &(grad_beta / &(acc_grad_beta.mapv(|x| x.sqrt()) + epsilon) * lr);
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::BatchNormalization(BatchNormalizationLayerWeight {
            gamma: &self.gamma,
            beta: &self.beta,
            running_mean: &self.running_mean,
            running_var: &self.running_var,
        })
    }

    mode_dependent_layer_trait!();
}
