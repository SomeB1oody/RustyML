//! Adam (Adaptive Moment Estimation) optimizer with classic coupled L2 weight decay

use crate::error::Error;
use crate::neural_network::optimizers::adam_core::AdamCore;
use crate::neural_network::traits::{Layer, Optimizer};

/// Adam (Adaptive Moment Estimation) optimizer
///
/// Computes adaptive learning rates from running estimates of the gradient's first and second
/// moments, with bias correction
///
/// Its `weight_decay` implements classic coupled L2 regularization. It adds `weight_decay * param`
/// to the gradient. The penalty then flows through the moment estimates, and the adaptive
/// `1 / (sqrt(v_hat) + epsilon)` denominator rescales it. For decoupled weight decay, applied
/// straight to the parameter and generally the better choice with adaptive optimizers, use
/// [`AdamW`](crate::neural_network::optimizers::AdamW) instead. With `weight_decay == 0.0` the 2
/// are identical. As with the other optimizers, weight decay, when non-zero, applies to weight
/// tensors only, never to biases or normalization scale/shift parameters
#[derive(Debug)]
pub struct Adam {
    /// The shared Adam-family optimizer state
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
    /// - `weight_decay` - Classic coupled L2 weight-decay coefficient folded into the gradient.
    ///   `0.0` disables it. For decoupled decay use
    ///   [`AdamW`](crate::neural_network::optimizers::AdamW)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new Adam optimizer instance or an error
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`Adam::with_global_clipnorm`]
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
    pub fn with_global_clipnorm(self, global_clipnorm: f32) -> Result<Self, Error> {
        Ok(Self {
            core: self.core.with_global_clipnorm(global_clipnorm)?,
        })
    }
}

impl Optimizer for Adam {
    fn global_clipnorm(&self) -> Option<f32> {
        self.core.global_clipnorm()
    }

    fn learning_rate(&self) -> f32 {
        self.core.learning_rate()
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
