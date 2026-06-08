use crate::error::Error;
use crate::neural_network::traits::{Layer, Optimizer};
use crate::neural_network::optimizers::validation::validate_learning_rate;
use crate::neural_network::optimizers::kernels;

/// SGD (Stochastic Gradient Descent) optimizer.
///
/// Updates parameters in the direction of the negative gradient, scaled by the learning rate.
///
/// # Fields
///
/// - `learning_rate` - Learning rate controlling the size of parameter updates
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    /// Creates a new SGD optimizer with the specified learning rate.
    ///
    /// Initializes the optimizer and validates the learning rate.
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
    /// - `Error::InvalidParameter` - If `learning_rate` is not positive
    pub fn new(learning_rate: f32) -> Result<Self, Error> {
        // input validation
        validate_learning_rate(learning_rate)?;

        Ok(Self { learning_rate })
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // SGD is stateless: apply `param -= lr * grad` to every trainable tensor.
        for pg in layer.parameters() {
            kernels::sgd_step(pg.value, pg.grad, self.learning_rate);
        }
    }
}
