//! RMSprop (Root Mean Square Propagation) optimizer

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_decay_rate, validate_epsilon, validate_global_clipnorm, validate_learning_rate,
    validate_non_negative_finite,
};
use crate::neural_network::traits::{Layer, Optimizer};

/// RMSprop (Root Mean Square Propagation) optimizer
///
/// Adapts per-parameter learning rates using a moving average of squared gradients
#[derive(Debug)]
pub struct RMSprop {
    /// Learning rate controlling the size of parameter updates
    learning_rate: f32,
    /// Decay rate for the moving average of squared gradients
    rho: f32,
    /// Small constant added for numerical stability
    epsilon: f32,
    /// Per-parameter squared-gradient running averages, indexed by parameter order each step
    caches: Vec<Vec<f32>>,
    /// Position within `caches` for the parameter currently being updated. Reset each `step`
    cursor: usize,
    /// Optional clip-by-global-norm threshold. `None` disables gradient clipping
    global_clipnorm: Option<f32>,
    /// Decoupled (AdamW-style) weight decay coefficient. `0.0` disables it
    weight_decay: f32,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer with the specified hyperparameters
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `rho` - Decay rate for moving average of squared gradients (typically 0.9)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    /// - `weight_decay` - Decoupled (AdamW-style) weight-decay coefficient applied directly to the
    ///   parameters. `0.0` disables it
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new RMSprop optimizer instance or an error
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`RMSprop::with_global_clipnorm`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` or `epsilon` is not positive and finite,
    ///   `rho` is outside [0, 1), or `weight_decay` is negative or not finite
    pub fn new(
        learning_rate: f32,
        rho: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_decay_rate(rho, "rho")?;
        validate_epsilon(epsilon)?;
        validate_non_negative_finite(weight_decay, "weight_decay")?;

        Ok(Self {
            learning_rate,
            rho,
            epsilon,
            caches: Vec::new(),
            cursor: 0,
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
    /// - `global_clipnorm` - Clip-by-global-norm threshold. Must be positive and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated optimizer, or an error if `global_clipnorm` is not
    ///   positive and finite
    pub fn with_global_clipnorm(mut self, global_clipnorm: f32) -> Result<Self, Error> {
        validate_global_clipnorm(Some(global_clipnorm))?;
        self.global_clipnorm = Some(global_clipnorm);
        Ok(self)
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        // Rewind to the first parameter. Layers yield parameters in the same order every step
        self.cursor = 0;
    }

    fn global_clipnorm(&self) -> Option<f32> {
        self.global_clipnorm
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn update(&mut self, layer: &mut dyn Layer, grad_scale: f32) {
        for pg in layer.parameters() {
            if self.cursor >= self.caches.len() {
                self.caches.push(vec![0.0; pg.value.len()]);
            } else if self.caches[self.cursor].len() != pg.value.len() {
                // Reset the cache to match
                self.caches[self.cursor] = vec![0.0; pg.value.len()];
            }
            let grad = kernels::scaled_grad(pg.grad, grad_scale);
            // Decoupled weight decay shrinks the parameter before the adaptive step (weights
            // only, biases and normalization gamma/beta excluded)
            if pg.decays {
                kernels::apply_weight_decay(pg.value, self.learning_rate, self.weight_decay);
            }
            kernels::rmsprop_step(
                pg.value,
                &grad,
                &mut self.caches[self.cursor],
                self.rho,
                self.learning_rate,
                self.epsilon,
            );
            self.cursor += 1;
        }
    }
}
