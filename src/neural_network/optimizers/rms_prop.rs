use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_decay_rate, validate_epsilon, validate_learning_rate,
};
use crate::neural_network::traits::{Layer, Optimizer};

/// RMSprop (Root Mean Square Propagation) optimizer.
///
/// Adapts per-parameter learning rates using a moving average of squared gradients.
///
/// # Fields
///
/// - `learning_rate` - Learning rate controlling the size of parameter updates
/// - `rho` - Decay rate for the moving average of squared gradients
/// - `epsilon` - Small constant added for numerical stability
#[derive(Debug)]
pub struct RMSprop {
    learning_rate: f32,
    rho: f32,
    epsilon: f32,
    /// Per-parameter squared-gradient running averages, indexed by parameter order each step.
    caches: Vec<Vec<f32>>,
    /// Position within `caches` for the parameter currently being updated; reset each `step`.
    cursor: usize,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer with the specified parameters.
    ///
    /// Validates hyperparameters and initializes the optimizer.
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `rho` - Decay rate for moving average of squared gradients (typically 0.9)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new RMSprop optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any hyperparameter is out of range
    pub fn new(learning_rate: f32, rho: f32, epsilon: f32) -> Result<Self, Error> {
        // input validation
        validate_learning_rate(learning_rate)?;
        validate_decay_rate(rho, "rho")?;
        validate_epsilon(epsilon)?;

        Ok(Self {
            learning_rate,
            rho,
            epsilon,
            caches: Vec::new(),
            cursor: 0,
        })
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        // Rewind to the first parameter; layers yield parameters in the same order every step.
        self.cursor = 0;
    }

    fn update(&mut self, layer: &mut dyn Layer) {
        for pg in layer.parameters() {
            if self.cursor >= self.caches.len() {
                self.caches.push(vec![0.0; pg.value.len()]);
            }
            kernels::rmsprop_step(
                pg.value,
                pg.grad,
                &mut self.caches[self.cursor],
                self.rho,
                self.learning_rate,
                self.epsilon,
            );
            self.cursor += 1;
        }
    }
}
