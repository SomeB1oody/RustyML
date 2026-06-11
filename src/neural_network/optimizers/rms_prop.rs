//! RMSprop (Root Mean Square Propagation) optimizer

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_clip_norm, validate_decay_rate, validate_epsilon, validate_learning_rate,
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
    /// Position within `caches` for the parameter currently being updated; reset each `step`
    cursor: usize,
    /// Optional clip-by-global-norm threshold; `None` disables gradient clipping
    clip_norm: Option<f32>,
    /// Decoupled (AdamW-style) weight decay coefficient; `0.0` disables it
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
    /// - `clip_norm` - Optional clip-by-global-norm threshold; `Some(max_norm)` scales every
    ///   gradient so the global L2 norm never exceeds `max_norm` (preserving direction), `None`
    ///   disables clipping
    /// - `weight_decay` - Decoupled (AdamW-style) weight-decay coefficient applied directly to the
    ///   parameters; `0.0` disables it
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new RMSprop optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` or `epsilon` is not positive and finite,
    ///   `rho` is outside [0, 1), `clip_norm` is `Some` value that is not positive and finite, or
    ///   `weight_decay` is negative or not finite
    pub fn new(
        learning_rate: f32,
        rho: f32,
        epsilon: f32,
        clip_norm: Option<f32>,
        weight_decay: f32,
    ) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_decay_rate(rho, "rho")?;
        validate_epsilon(epsilon)?;
        validate_clip_norm(clip_norm)?;
        validate_non_negative_finite(weight_decay, "weight_decay")?;

        Ok(Self {
            learning_rate,
            rho,
            epsilon,
            caches: Vec::new(),
            cursor: 0,
            clip_norm,
            weight_decay,
        })
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        // Rewind to the first parameter; layers yield parameters in the same order every step
        self.cursor = 0;
    }

    fn clip_norm(&self) -> Option<f32> {
        self.clip_norm
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
            // Decoupled weight decay shrinks the parameter before the adaptive step
            kernels::apply_weight_decay(pg.value, self.learning_rate, self.weight_decay);
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
