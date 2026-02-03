/// Common implementation for `output_shape` method in normalization layers
macro_rules! normalization_layer_output_shape {
    ($self:expr) => {
        if !$self.input_shape.is_empty() {
            format!(
                "({})",
                $self
                    .input_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::from("Unknown")
        }
    };
}

/// Common implementation for `update_parameters_sgd` method in normalization layers
macro_rules! normalization_layer_update_parameters_sgd {
    ($self:expr, $lr:expr) => {{
        if let Some(grad_gamma) = &$self.grad_gamma {
            $self.gamma = &$self.gamma - &(grad_gamma * $lr);
        }
        if let Some(grad_beta) = &$self.grad_beta {
            $self.beta = &$self.beta - &(grad_beta * $lr);
        }
    }};
}

/// Common implementation for `update_parameters_adam` method in normalization layers
macro_rules! normalization_layer_update_parameters_adam {
    ($self:expr, $lr:expr, $beta1:expr, $beta2:expr, $epsilon:expr, $t:expr) => {{
        // Initialize moment estimates if not already done
        if $self.optimizer_cache.adam_states.is_none() {
            $self.optimizer_cache.adam_states = Some(AdamStatesNormalizationLayer {
                m_gamma: Tensor::zeros($self.gamma.raw_dim()),
                v_gamma: Tensor::zeros($self.gamma.raw_dim()),
                m_beta: Tensor::zeros($self.beta.raw_dim()),
                v_beta: Tensor::zeros($self.beta.raw_dim()),
            });
        }

        if let Some(grad_gamma) = &$self.grad_gamma {
            let adam_states = $self.optimizer_cache.adam_states.as_mut().unwrap();
            let m_gamma = &mut adam_states.m_gamma;
            let v_gamma = &mut adam_states.v_gamma;

            // Update biased first moment estimate
            *m_gamma = m_gamma.clone() * $beta1 + grad_gamma * (1.0 - $beta1);

            // Update biased second raw moment estimate
            *v_gamma = v_gamma.clone() * $beta2 + &(grad_gamma * grad_gamma) * (1.0 - $beta2);

            // Compute bias-corrected first moment estimate
            let m_hat = m_gamma.clone() / (1.0 - $beta1.powi($t as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = v_gamma.clone() / (1.0 - $beta2.powi($t as i32));

            // Update parameters
            $self.gamma = &$self.gamma - &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + $epsilon) * $lr);
        }

        if let Some(grad_beta) = &$self.grad_beta {
            let adam_states = $self.optimizer_cache.adam_states.as_mut().unwrap();
            let m_beta = &mut adam_states.m_beta;
            let v_beta = &mut adam_states.v_beta;

            *m_beta = m_beta.clone() * $beta1 + grad_beta * (1.0 - $beta1);
            *v_beta = v_beta.clone() * $beta2 + &(grad_beta * grad_beta) * (1.0 - $beta2);

            let m_hat = m_beta.clone() / (1.0 - $beta1.powi($t as i32));
            let v_hat = v_beta.clone() / (1.0 - $beta2.powi($t as i32));

            $self.beta = &$self.beta - &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + $epsilon) * $lr);
        }
    }};
}

/// Common implementation for `update_parameters_rmsprop` method in normalization layers
macro_rules! normalization_layer_update_parameters_rmsprop {
    ($self:expr, $lr:expr, $rho:expr, $epsilon:expr) => {{
        // Initialize cache if not already done
        if $self.optimizer_cache.rmsprop_cache.is_none() {
            $self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheNormalizationLayer {
                cache_gamma: Tensor::zeros($self.gamma.raw_dim()),
                cache_beta: Tensor::zeros($self.beta.raw_dim()),
            });
        }

        if let Some(grad_gamma) = &$self.grad_gamma {
            let rmsprop_cache = $self.optimizer_cache.rmsprop_cache.as_mut().unwrap();
            let cache_gamma = &mut rmsprop_cache.cache_gamma;

            // Update cache
            *cache_gamma = cache_gamma.clone() * $rho + &(grad_gamma * grad_gamma) * (1.0 - $rho);

            // Update parameters
            $self.gamma =
                &$self.gamma - &(grad_gamma / &(cache_gamma.mapv(|x| x.sqrt()) + $epsilon) * $lr);
        }

        if let Some(grad_beta) = &$self.grad_beta {
            let rmsprop_cache = $self.optimizer_cache.rmsprop_cache.as_mut().unwrap();
            let cache_beta = &mut rmsprop_cache.cache_beta;

            *cache_beta = cache_beta.clone() * $rho + &(grad_beta * grad_beta) * (1.0 - $rho);

            $self.beta =
                &$self.beta - &(grad_beta / &(cache_beta.mapv(|x| x.sqrt()) + $epsilon) * $lr);
        }
    }};
}

/// Common implementation for `update_parameters_ada_grad` method in normalization layers
macro_rules! normalization_layer_update_parameters_ada_grad {
    ($self:expr, $lr:expr, $epsilon:expr) => {{
        // Initialize accumulated gradient if not already done
        if $self.optimizer_cache.ada_grad_cache.is_none() {
            $self.optimizer_cache.ada_grad_cache = Some(AdaGradStatesNormalizationLayer {
                acc_grad_gamma: Tensor::zeros($self.gamma.raw_dim()),
                acc_grad_beta: Tensor::zeros($self.beta.raw_dim()),
            });
        }

        if let Some(grad_gamma) = &$self.grad_gamma {
            let ada_grad_cache = $self.optimizer_cache.ada_grad_cache.as_mut().unwrap();
            let acc_grad_gamma = &mut ada_grad_cache.acc_grad_gamma;

            // Accumulate squared gradients
            *acc_grad_gamma = acc_grad_gamma.clone() + &(grad_gamma * grad_gamma);

            // Update parameters
            $self.gamma = &$self.gamma
                - &(grad_gamma / &(acc_grad_gamma.mapv(|x| x.sqrt()) + $epsilon) * $lr);
        }

        if let Some(grad_beta) = &$self.grad_beta {
            let ada_grad_cache = $self.optimizer_cache.ada_grad_cache.as_mut().unwrap();
            let acc_grad_beta = &mut ada_grad_cache.acc_grad_beta;

            *acc_grad_beta = acc_grad_beta.clone() + &(grad_beta * grad_beta);

            $self.beta =
                &$self.beta - &(grad_beta / &(acc_grad_beta.mapv(|x| x.sqrt()) + $epsilon) * $lr);
        }
    }};
}

/// Common implementation for computing gamma and beta gradients in normalization layers
///
/// This macro computes the gradients for gamma and beta parameters by summing over
/// all spatial dimensions and batch dimensions for each channel. It handles both
/// parallel and sequential computation based on the total number of elements.
///
/// # Parameters
///
/// - `$grad_gamma:expr` - Mutable tensor to store gamma gradients
/// - `$grad_beta:expr` - Mutable tensor to store beta gradients
/// - `$grad_output:expr` - Gradient from the next layer
/// - `$x_normalized:expr` - Normalized input from forward pass
/// - `$batch_size:expr` - Number of samples in the batch
/// - `$num_channels:expr` - Number of channels
/// - `$spatial_size:expr` - Size of spatial dimensions (product of all spatial dims)
/// - `$total_elements:expr` - Total number of elements in the tensor
/// - `$parallel_threshold:expr` - Threshold for switching to parallel computation
macro_rules! compute_normalization_layer_parameter_gradients {
    (
        $grad_gamma:expr,
        $grad_beta:expr,
        $grad_output:expr,
        $x_normalized:expr,
        $batch_size:expr,
        $num_channels:expr,
        $spatial_size:expr,
        $total_elements:expr,
        $parallel_threshold:expr
    ) => {{
        if $total_elements >= $parallel_threshold {
            // Parallel computation of parameter gradients
            // Compute both gamma and beta gradients in a single pass
            let (grad_gamma_vec, grad_beta_vec): (Vec<f32>, Vec<f32>) = (0..$num_channels)
                .into_par_iter()
                .map(|channel_idx| {
                    let mut gamma_sum = 0.0f32;
                    let mut beta_sum = 0.0f32;

                    for batch_idx in 0..$batch_size {
                        let start = (batch_idx * $num_channels + channel_idx) * $spatial_size;
                        let end = start + $spatial_size;

                        for i in start..end {
                            let grad_out = $grad_output.as_slice().unwrap()[i];
                            gamma_sum += grad_out * $x_normalized.as_slice().unwrap()[i];
                            beta_sum += grad_out;
                        }
                    }

                    (gamma_sum, beta_sum)
                })
                .unzip();

            $grad_gamma
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&grad_gamma_vec);
            $grad_beta
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&grad_beta_vec);
        } else {
            // Sequential computation
            for channel_idx in 0..$num_channels {
                let mut gamma_sum = 0.0f32;
                let mut beta_sum = 0.0f32;

                for batch_idx in 0..$batch_size {
                    let start = (batch_idx * $num_channels + channel_idx) * $spatial_size;
                    let end = start + $spatial_size;

                    for i in start..end {
                        gamma_sum += $grad_output.as_slice().unwrap()[i]
                            * $x_normalized.as_slice().unwrap()[i];
                        beta_sum += $grad_output.as_slice().unwrap()[i];
                    }
                }

                $grad_gamma.as_slice_mut().unwrap()[channel_idx] = gamma_sum;
                $grad_beta.as_slice_mut().unwrap()[channel_idx] = beta_sum;
            }
        }
    }};
}

/// Batch Normalization layer for neural networks
pub mod batch_normalization;
/// Group Normalization layer for neural networks
pub mod group_normalization;
/// Instance Normalization layer for neural networks
pub mod instance_normalization;
/// Layer Normalization layer for neural networks
pub mod layer_normalization;

pub use batch_normalization::BatchNormalization;
pub use group_normalization::GroupNormalization;
pub use instance_normalization::InstanceNormalization;
pub use layer_normalization::{LayerNormalization, LayerNormalizationAxis};
