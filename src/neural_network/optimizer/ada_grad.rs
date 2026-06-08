use crate::error::ModelError;
use crate::neural_network::neural_network_trait::{Layer, Optimizer};
use crate::neural_network::optimizer::validation::validate_positive_finite;
use crate::neural_network::optimizer::kernels;

/// AdaGrad (Adaptive Gradient Algorithm) optimizer.
///
/// Adapts learning rates per parameter using accumulated squared gradients.
///
/// # Fields
///
/// - `learning_rate` - Initial learning rate controlling the size of parameter updates
/// - `epsilon` - Small constant added for numerical stability
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
    /// Per-parameter accumulated squared gradients, indexed by parameter order each step.
    accumulators: Vec<Vec<f32>>,
    /// Position within `accumulators` for the parameter currently being updated; reset each `step`.
    cursor: usize,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the specified parameters.
    ///
    /// Validates hyperparameters and initializes the optimizer.
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Initial step size for parameter updates (typically 0.01)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new AdaGrad optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `learning_rate` or `epsilon` is not positive and finite
    pub fn new(learning_rate: f32, epsilon: f32) -> Result<Self, ModelError> {
        // input validation
        validate_positive_finite(learning_rate, "learning_rate")?;
        validate_positive_finite(epsilon, "epsilon")?;

        Ok(Self {
            learning_rate,
            epsilon,
            accumulators: Vec::new(),
            cursor: 0,
        })
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self) {
        // Rewind to the first parameter; layers yield parameters in the same order every step.
        self.cursor = 0;
    }

    fn update(&mut self, layer: &mut dyn Layer) {
        for pg in layer.parameters() {
            if self.cursor >= self.accumulators.len() {
                self.accumulators.push(vec![0.0; pg.value.len()]);
            }
            kernels::adagrad_step(
                pg.value,
                pg.grad,
                &mut self.accumulators[self.cursor],
                self.learning_rate,
                self.epsilon,
            );
            self.cursor += 1;
        }
    }
}
