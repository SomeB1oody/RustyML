//! Stochastic Gradient Descent (SGD) optimizer

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_clip_norm, validate_learning_rate, validate_non_negative_finite,
};
use crate::neural_network::traits::{Layer, Optimizer};

/// SGD (Stochastic Gradient Descent) optimizer
///
/// Updates parameters in the direction of the negative gradient, scaled by the learning rate, with
/// optional momentum / Nesterov acceleration and decoupled weight decay
#[derive(Debug)]
pub struct SGD {
    /// Learning rate controlling the size of parameter updates
    learning_rate: f32,
    /// Momentum coefficient; `0.0` disables momentum (plain SGD)
    momentum: f32,
    /// Whether to use Nesterov-accelerated momentum (only meaningful when `momentum > 0`)
    nesterov: bool,
    /// Decoupled (SGDW-style) weight decay coefficient; `0.0` disables it
    weight_decay: f32,
    /// Optional clip-by-global-norm threshold; `None` disables gradient clipping
    clip_norm: Option<f32>,
    /// Per-parameter momentum buffers, allocated lazily when `momentum > 0`, indexed by the order
    /// layers yield parameters each step
    velocities: Vec<Vec<f32>>,
    /// Position within `velocities` for the parameter currently being updated; reset each `step`
    cursor: usize,
}

impl SGD {
    /// Creates a new SGD optimizer
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `momentum` - Momentum coefficient (typically 0.9); `0.0` is plain SGD
    /// - `nesterov` - Use Nesterov-accelerated momentum (only meaningful when `momentum > 0`)
    /// - `weight_decay` - Decoupled (SGDW-style) weight-decay coefficient applied directly to the
    ///   parameters; `0.0` disables it
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`SGD::with_clip_norm`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SGD optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` is not positive and finite, or
    ///   `momentum`/`weight_decay` is negative or not finite
    pub fn new(
        learning_rate: f32,
        momentum: f32,
        nesterov: bool,
        weight_decay: f32,
    ) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_non_negative_finite(momentum, "momentum")?;
        validate_non_negative_finite(weight_decay, "weight_decay")?;

        Ok(Self {
            learning_rate,
            momentum,
            nesterov,
            weight_decay,
            clip_norm: None,
            velocities: Vec::new(),
            cursor: 0,
        })
    }

    /// Enables clip-by-global-norm gradient clipping (disabled by default)
    ///
    /// Scales every gradient so the global L2 norm never exceeds `clip_norm`, preserving the
    /// gradient direction
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

impl Optimizer for SGD {
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
            let grad = kernels::scaled_grad(pg.grad, grad_scale);
            // Decoupled weight decay shrinks the parameter before the gradient step (weights only;
            // biases and normalization gamma/beta excluded)
            if pg.decays {
                kernels::apply_weight_decay(pg.value, self.learning_rate, self.weight_decay);
            }

            if self.momentum == 0.0 {
                kernels::sgd_step(pg.value, &grad, self.learning_rate);
            } else {
                if self.cursor >= self.velocities.len() {
                    self.velocities.push(vec![0.0; pg.value.len()]);
                } else if self.velocities[self.cursor].len() != pg.value.len() {
                    // Parameter length changed at this position: reset the buffer to match
                    self.velocities[self.cursor] = vec![0.0; pg.value.len()];
                }
                kernels::sgd_momentum_step(
                    pg.value,
                    &grad,
                    &mut self.velocities[self.cursor],
                    self.learning_rate,
                    self.momentum,
                    self.nesterov,
                );
                self.cursor += 1;
            }
        }
    }
}
