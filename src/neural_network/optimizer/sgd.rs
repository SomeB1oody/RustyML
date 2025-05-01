use crate::neural_network::{Layer, Optimizer};

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// A simple optimization algorithm that updates parameters in the direction
/// of the negative gradient, scaled by the learning rate.
pub struct SGD {
    /// Learning rate controlling the size of parameter updates.
    learning_rate: f32,
}

impl SGD {
    /// Creates a new SGD optimizer with the specified learning rate.
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Step size for parameter updates
    ///
    /// # Returns
    ///
    /// * `Self` - A new SGD optimizer instance
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // Directly call the layer's parameter update method
        layer.update_parameters_sgd(self.learning_rate);
    }
}
