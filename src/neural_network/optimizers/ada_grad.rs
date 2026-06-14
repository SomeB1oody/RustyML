//! AdaGrad optimizer that adapts per-parameter learning rates using accumulated squared gradients

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_clip_norm, validate_non_negative_finite, validate_positive_finite,
};
use crate::neural_network::traits::{Layer, Optimizer};

/// AdaGrad (Adaptive Gradient Algorithm) optimizer
///
/// Adapts learning rates per parameter using accumulated squared gradients
#[derive(Debug)]
pub struct AdaGrad {
    /// Initial learning rate controlling the size of parameter updates
    learning_rate: f32,
    /// Small constant added for numerical stability
    epsilon: f32,
    /// Per-parameter accumulated squared gradients, indexed by parameter order each step
    accumulators: Vec<Vec<f32>>,
    /// Position within `accumulators` for the parameter currently being updated; reset each `step`
    cursor: usize,
    /// Optional clip-by-global-norm threshold; `None` disables gradient clipping
    clip_norm: Option<f32>,
    /// Decoupled (AdamW-style) weight decay coefficient; `0.0` disables it
    weight_decay: f32,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the specified hyperparameters
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Initial step size for parameter updates (typically 0.01)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    /// - `weight_decay` - Decoupled (AdamW-style) weight-decay coefficient applied directly to the
    ///   parameters; `0.0` disables it
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`AdaGrad::with_clip_norm`].
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new AdaGrad optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` or `epsilon` is not positive and finite,
    ///   or `weight_decay` is negative or not finite
    pub fn new(learning_rate: f32, epsilon: f32, weight_decay: f32) -> Result<Self, Error> {
        validate_positive_finite(learning_rate, "learning_rate")?;
        validate_positive_finite(epsilon, "epsilon")?;
        validate_non_negative_finite(weight_decay, "weight_decay")?;

        Ok(Self {
            learning_rate,
            epsilon,
            accumulators: Vec::new(),
            cursor: 0,
            clip_norm: None,
            weight_decay,
        })
    }

    /// Enables clip-by-global-norm gradient clipping (disabled by default)
    ///
    /// `max_norm` scales every gradient so the global L2 norm never exceeds it, preserving the
    /// gradient direction.
    ///
    /// # Parameters
    ///
    /// - `clip_norm` - Clip-by-global-norm threshold; must be positive and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated optimizer, or an error if `clip_norm` is not positive
    ///   and finite
    pub fn with_clip_norm(mut self, clip_norm: f32) -> Result<Self, Error> {
        validate_clip_norm(Some(clip_norm))?;
        self.clip_norm = Some(clip_norm);
        Ok(self)
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self) {
        // Rewind to the first parameter
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
            if self.cursor >= self.accumulators.len() {
                self.accumulators.push(vec![0.0; pg.value.len()]);
            } else if self.accumulators[self.cursor].len() != pg.value.len() {
                // Reset the accumulator to match
                self.accumulators[self.cursor] = vec![0.0; pg.value.len()];
            }
            let grad = kernels::scaled_grad(pg.grad, grad_scale);
            // Decoupled weight decay shrinks the parameter before the adaptive step
            kernels::apply_weight_decay(pg.value, self.learning_rate, self.weight_decay);
            kernels::adagrad_step(
                pg.value,
                &grad,
                &mut self.accumulators[self.cursor],
                self.learning_rate,
                self.epsilon,
            );
            self.cursor += 1;
        }
    }
}
