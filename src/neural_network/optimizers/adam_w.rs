//! AdamW optimizer: Adam with decoupled weight decay

use crate::error::Error;
use crate::neural_network::optimizers::adam_core::AdamCore;
use crate::neural_network::traits::{Layer, Optimizer};

/// AdamW (Adam with decoupled weight decay) optimizer
///
/// Identical adaptive moment math to [`Adam`](crate::neural_network::optimizers::Adam), but its
/// `weight_decay` is decoupled. AdamW shrinks the parameter directly by the factor
/// `(1 - learning_rate * weight_decay)` before the gradient step. This replaces folding an L2
/// term into the gradient. The decay therefore does not flow through the moment estimates, and
/// the adaptive denominator does not rescale it. This is the Loshchilov and Hutter formulation, and
/// the better-behaved choice with adaptive optimizers. With `weight_decay == 0.0` it matches
/// `Adam`. Weight decay applies to weight tensors only, never to biases or normalization
/// scale/shift parameters
#[derive(Debug)]
pub struct AdamW {
    /// The shared Adam-family optimizer state
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
    /// - `weight_decay` - Decoupled weight-decay coefficient applied directly to the parameters.
    ///   `0.0` disables it. For classic coupled L2 decay use
    ///   [`Adam`](crate::neural_network::optimizers::Adam)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new AdamW optimizer instance or an error
    ///
    /// # Notes
    ///
    /// Gradient clipping is disabled by default. Enable it with [`AdamW::with_global_clipnorm`]
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

impl Optimizer for AdamW {
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
