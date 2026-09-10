//! AdaGrad optimizer that adapts per-parameter learning rates using accumulated squared gradients

use crate::error::Error;
use crate::neural_network::ctx::Grads;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_global_clipnorm, validate_non_negative_finite, validate_positive_finite,
};
use crate::neural_network::traits::{LayerBase, Optimizer, ParamId};
use ahash::AHashMap;

/// AdaGrad (Adaptive Gradient Algorithm) optimizer
///
/// Adapts learning rates per parameter using accumulated squared gradients
#[derive(Debug)]
pub struct AdaGrad {
    /// Initial learning rate controlling the size of parameter updates
    learning_rate: f32,
    /// Small constant added for numerical stability
    epsilon: f32,
    /// Per-parameter accumulated squared gradients, keyed by the address of the parameter. An
    /// accumulator therefore follows its own tensor, whatever else the model does
    accumulators: AHashMap<ParamId, Vec<f32>>,
    /// Optional clip-by-global-norm threshold. `None` disables gradient clipping
    global_clipnorm: Option<f32>,
    /// Decoupled (AdamW-style) weight decay coefficient. `0.0` disables it
    weight_decay: f32,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the specified hyperparameters
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - initial step size for parameter updates (typically 0.01)
    /// - `epsilon` - small constant for numerical stability (typically 1e-8)
    /// - `weight_decay` - decoupled (AdamW-style) weight-decay coefficient applied directly to the
    ///   parameters. `0.0` disables it
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - a new AdaGrad optimizer instance or an error
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`AdaGrad::with_global_clipnorm`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if `learning_rate` or `epsilon` is not positive and finite,
    ///   or `weight_decay` is negative or not finite
    pub fn new(learning_rate: f32, epsilon: f32, weight_decay: f32) -> Result<Self, Error> {
        validate_positive_finite(learning_rate, "learning_rate")?;
        validate_positive_finite(epsilon, "epsilon")?;
        validate_non_negative_finite(weight_decay, "weight_decay")?;

        Ok(Self {
            learning_rate,
            epsilon,
            accumulators: AHashMap::new(),
            global_clipnorm: None,
            weight_decay,
        })
    }

    /// Enables clip-by-global-norm gradient clipping (disabled by default)
    ///
    /// `global_clipnorm` scales every gradient so the global L2 norm never exceeds it, preserving
    /// the gradient direction
    ///
    /// # Parameters
    ///
    /// - `global_clipnorm` - clip-by-global-norm threshold. Must be positive and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated optimizer, or an error if `global_clipnorm` is not
    ///   positive and finite
    pub fn with_global_clipnorm(mut self, global_clipnorm: f32) -> Result<Self, Error> {
        validate_global_clipnorm(Some(global_clipnorm))?;
        self.global_clipnorm = Some(global_clipnorm);
        Ok(self)
    }
}

impl Optimizer for AdaGrad {
    fn global_clipnorm(&self) -> Option<f32> {
        self.global_clipnorm
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn update(&mut self, scope: usize, layer: &mut dyn LayerBase, grads: &Grads, grad_scale: f32) {
        for pg in layer.parameters_mut() {
            let Some(grad) = grads.get(ParamId::new(scope, pg.name)) else {
                continue;
            };
            let grad = grad
                .as_slice()
                .expect("a stored gradient is in the standard memory order");
            debug_assert_eq!(grad.len(), pg.value.len());
            let accumulator = self
                .accumulators
                .entry(ParamId::new(scope, pg.name))
                .or_default();
            if accumulator.len() != pg.value.len() {
                // The tensor was resized under its own name: start the accumulator again
                *accumulator = vec![0.0; pg.value.len()];
            }
            let grad = kernels::scaled_grad(grad, grad_scale);
            // Decoupled weight decay shrinks the parameter before the adaptive step (weights
            // only, biases and normalization gamma/beta excluded)
            if pg.decays {
                kernels::apply_weight_decay(pg.value, self.learning_rate, self.weight_decay);
            }
            kernels::adagrad_step(
                pg.value,
                &grad,
                accumulator,
                self.learning_rate,
                self.epsilon,
            );
        }
    }
}
