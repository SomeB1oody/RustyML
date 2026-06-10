//! Stochastic Gradient Descent (SGD) optimizer

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::validate_learning_rate;
use crate::neural_network::traits::{Layer, Optimizer};

/// SGD (Stochastic Gradient Descent) optimizer
///
/// Updates parameters in the direction of the negative gradient, scaled by the learning rate
#[derive(Debug)]
pub struct SGD {
    /// Learning rate controlling the size of parameter updates
    learning_rate: f32,
}

impl SGD {
    /// Creates a new SGD optimizer with the specified learning rate
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SGD optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `learning_rate` is not positive or not finite
    pub fn new(learning_rate: f32) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;

        Ok(Self { learning_rate })
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // SGD is stateless: apply `param -= lr * grad` to every trainable tensor
        for pg in layer.parameters() {
            kernels::sgd_step(pg.value, pg.grad, self.learning_rate);
        }
    }
}
