//! Stochastic Gradient Descent (SGD) optimizer

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{validate_clip_norm, validate_learning_rate};
use crate::neural_network::traits::{Layer, Optimizer};

/// SGD (Stochastic Gradient Descent) optimizer
///
/// Updates parameters in the direction of the negative gradient, scaled by the learning rate
#[derive(Debug)]
pub struct SGD {
    /// Learning rate controlling the size of parameter updates
    learning_rate: f32,
    /// Optional clip-by-global-norm threshold; `None` disables gradient clipping
    clip_norm: Option<f32>,
}

impl SGD {
    /// Creates a new SGD optimizer with the specified learning rate
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `clip_norm` - Optional clip-by-global-norm threshold; `Some(max_norm)` scales every
    ///   gradient so the global L2 norm never exceeds `max_norm` (preserving direction), `None`
    ///   disables clipping
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SGD optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` is not positive and finite, or `clip_norm`
    ///   is `Some` value that is not positive and finite
    pub fn new(learning_rate: f32, clip_norm: Option<f32>) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_clip_norm(clip_norm)?;

        Ok(Self {
            learning_rate,
            clip_norm,
        })
    }
}

impl Optimizer for SGD {
    fn clip_norm(&self) -> Option<f32> {
        self.clip_norm
    }

    fn update(&mut self, layer: &mut dyn Layer, grad_scale: f32) {
        // SGD is stateless: apply `param -= lr * grad` to every trainable tensor
        for pg in layer.parameters() {
            let grad = kernels::scaled_grad(pg.grad, grad_scale);
            kernels::sgd_step(pg.value, &grad, self.learning_rate);
        }
    }
}
