//! AdaGrad optimizer that adapts per-parameter learning rates using accumulated squared gradients

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{validate_clip_norm, validate_positive_finite};
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
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the specified hyperparameters
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Initial step size for parameter updates (typically 0.01)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    /// - `clip_norm` - Optional clip-by-global-norm threshold; `Some(max_norm)` scales every
    ///   gradient so the global L2 norm never exceeds `max_norm` (preserving direction), `None`
    ///   disables clipping
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new AdaGrad optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` or `epsilon` is not positive and finite, or
    ///   `clip_norm` is `Some` value that is not positive and finite
    pub fn new(learning_rate: f32, epsilon: f32, clip_norm: Option<f32>) -> Result<Self, Error> {
        validate_positive_finite(learning_rate, "learning_rate")?;
        validate_positive_finite(epsilon, "epsilon")?;
        validate_clip_norm(clip_norm)?;

        Ok(Self {
            learning_rate,
            epsilon,
            accumulators: Vec::new(),
            cursor: 0,
            clip_norm,
        })
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

    fn update(&mut self, layer: &mut dyn Layer, grad_scale: f32) {
        for pg in layer.parameters() {
            if self.cursor >= self.accumulators.len() {
                self.accumulators.push(vec![0.0; pg.value.len()]);
            }
            let grad = kernels::scaled_grad(pg.grad, grad_scale);
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
