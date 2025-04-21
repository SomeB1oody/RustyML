use crate::neural_network::{Layer, Optimizer};

/// RMSprop optimizer implementation.
///
/// An optimization algorithm that adapts the learning rate for each parameter
/// using a moving average of squared gradients.
pub struct RMSprop {
    /// Learning rate controlling the size of parameter updates.
    learning_rate: f32,
    /// Decay rate for the moving average of squared gradients.
    rho: f32,
    /// Small constant added for numerical stability.
    epsilon: f32,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `rho` - Decay rate for moving average of squared gradients (typically 0.9)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// A new RMSprop optimizer instance
    pub fn new(learning_rate: f32, rho: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon,
        }
    }
}

impl Optimizer for RMSprop {
    /// Updates the layer parameters using the RMSprop algorithm.
    ///
    /// # Parameters
    ///
    /// * `layer` - The layer whose parameters will be updated
    fn update(&mut self, layer: &mut dyn Layer) {
        layer.update_parameters_rmsprop(self.learning_rate, self.rho, self.epsilon);
    }
}
