//! AdamW optimizer: Adam with decoupled weight decay

use crate::error::Error;
use crate::neural_network::optimizers::adam_core::AdamCore;
use crate::neural_network::traits::{Layer, Optimizer};

/// AdamW (Adam with decoupled weight decay) optimizer
///
/// Identical adaptive moment math to [`Adam`](crate::neural_network::optimizers::Adam), but its
/// `weight_decay` is **decoupled**: the parameter is shrunk directly by the factor
/// `(1 - learning_rate * weight_decay)` before the gradient step, rather than folding an L2 term
/// into the gradient. The decay therefore does not flow through the moment estimates and is not
/// rescaled by the adaptive denominator — the formulation of Loshchilov & Hutter, and generally
/// the better-behaved choice with adaptive optimizers. With `weight_decay == 0.0` it is identical
/// to `Adam`. Weight decay is applied to weight tensors only, never to biases or normalization
/// scale/shift parameters
#[derive(Debug)]
pub struct AdamW {
    core: AdamCore,
}

impl AdamW {
    /// Creates a new AdamW optimizer with the specified hyperparameters
    ///
    /// Validates hyperparameters and initializes internal timestep tracking
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `beta1` - Decay rate for the first moment estimates (typically 0.9)
    /// - `beta2` - Decay rate for the second moment estimates (typically 0.999)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    /// - `weight_decay` - Decoupled weight-decay coefficient applied directly to the parameters;
    ///   `0.0` disables it. For classic coupled L2 decay use
    ///   [`Adam`](crate::neural_network::optimizers::Adam)
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`AdamW::with_clip_norm`].
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new AdamW optimizer instance or an error
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
            core: AdamCore::new(learning_rate, beta1, beta2, epsilon, weight_decay, true)?,
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
    pub fn with_clip_norm(self, clip_norm: f32) -> Result<Self, Error> {
        Ok(Self {
            core: self.core.with_clip_norm(clip_norm)?,
        })
    }
}

impl Optimizer for AdamW {
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
