use crate::neural_network::{Layer, Optimizer};

/// Adam optimizer implementation.
///
/// An optimization algorithm that computes individual adaptive learning
/// rates for different parameters from estimates of first and second moments
/// of the gradients.
pub struct Adam {
    /// Learning rate controlling the size of parameter updates.
    learning_rate: f32,
    /// Exponential decay rate for the first moment estimates.
    beta1: f32,
    /// Exponential decay rate for the second moment estimates.
    beta2: f32,
    /// Small constant added for numerical stability.
    epsilon: f32,
    /// Current timestep, incremented with each update.
    t: u64,
}

impl Adam {
    /// Creates a new Adam optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `beta1` - Decay rate for the first moment estimates (typically 0.9)
    /// - `beta2` - Decay rate for the second moment estimates (typically 0.999)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// * `Self` - A new Adam optimizer instance
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, layer: &mut dyn Layer) {
        self.t += 1; // Increment step count with each update
        layer.update_parameters_adam(
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.t,
        );
    }
}
