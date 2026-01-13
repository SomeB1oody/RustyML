use super::*;

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

/// Batch Normalization layer for neural networks
pub mod batch_normalization;
/// Instance Normalization layer for neural networks
mod instance_normalization;
/// Layer Normalization layer for neural networks
pub mod layer_normalization;

pub use batch_normalization::*;
pub use instance_normalization::*;
pub use layer_normalization::*;
