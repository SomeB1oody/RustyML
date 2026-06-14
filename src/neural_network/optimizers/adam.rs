//! Adam (Adaptive Moment Estimation) optimizer with classic coupled L2 weight decay

use crate::error::Error;
use crate::neural_network::optimizers::adam_core::AdamCore;
use crate::neural_network::traits::{Layer, Optimizer};

/// Adam (Adaptive Moment Estimation) optimizer
///
/// Computes adaptive learning rates from running estimates of the gradient's first and second
/// moments, with bias correction
///
/// Its `weight_decay` is **classic coupled L2 regularization**: `weight_decay * param` is added to
/// the gradient, so the penalty flows through the moment estimates and is rescaled by the adaptive
/// `1 / (sqrt(v_hat) + epsilon)` denominator. For *decoupled* weight decay, applied straight to
/// the parameter and generally the better choice with adaptive optimizers, use
/// [`AdamW`](crate::neural_network::optimizers::AdamW) instead. With `weight_decay == 0.0` the two
/// are identical. As with the other optimizers, weight decay (when non-zero) applies to weight
/// tensors only, never to biases or normalization scale/shift parameters
#[derive(Debug)]
pub struct Adam {
    core: AdamCore,
}

impl Adam {
    /// Creates a new Adam optimizer with the specified hyperparameters
    ///
    /// Validates hyperparameters and initializes internal timestep tracking
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `beta1` - Decay rate for the first moment estimates (typically 0.9)
    /// - `beta2` - Decay rate for the second moment estimates (typically 0.999)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    /// - `weight_decay` - Classic coupled L2 weight-decay coefficient folded into the gradient;
    ///   `0.0` disables it. For decoupled decay use [`AdamW`](crate::neural_network::optimizers::AdamW)
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`Adam::with_clip_norm`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new Adam optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any hyperparameter is out of range or `weight_decay` is
    ///   negative or not finite
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<Self, Error> {
        Ok(Self {
            core: AdamCore::new(learning_rate, beta1, beta2, epsilon, weight_decay, false)?,
        })
    }

    /// Enables clip-by-global-norm gradient clipping (disabled by default)
    ///
    /// `max_norm` scales every gradient so the global L2 norm never exceeds it, preserving the
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
    pub fn with_clip_norm(self, clip_norm: f32) -> Result<Self, Error> {
        Ok(Self {
            core: self.core.with_clip_norm(clip_norm)?,
        })
    }
}

impl Optimizer for Adam {
    fn clip_norm(&self) -> Option<f32> {
        self.core.clip_norm()
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.core.set_learning_rate(learning_rate);
    }

    fn step(&mut self) {
        self.core.step();
    }

    fn update(&mut self, layer: &mut dyn Layer, grad_scale: f32) {
        self.core.update(layer, grad_scale);
    }
}
