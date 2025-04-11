use super::*;

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
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    ///
    /// # Returns
    ///
    /// A new SGD optimizer instance
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    /// Updates the layer parameters using the SGD algorithm.
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer whose parameters will be updated
    fn update(&mut self, layer: &mut dyn Layer) {
        // Directly call the layer's parameter update method
        layer.update_parameters_sgd(self.learning_rate);
    }
}

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
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    /// * `beta1` - Decay rate for the first moment estimates (typically 0.9)
    /// * `beta2` - Decay rate for the second moment estimates (typically 0.999)
    /// * `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// A new Adam optimizer instance
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
    /// Updates the layer parameters using the Adam algorithm.
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer whose parameters will be updated
    fn update(&mut self, layer: &mut dyn Layer) {
        self.t += 1; // Increment step count with each update
        layer.update_parameters_adam(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.t);
    }
}

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
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    /// * `rho` - Decay rate for moving average of squared gradients (typically 0.9)
    /// * `epsilon` - Small constant for numerical stability (typically 1e-8)
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
    /// # Arguments
    ///
    /// * `layer` - The layer whose parameters will be updated
    fn update(&mut self, layer: &mut dyn Layer) {
        layer.update_parameters_rmsprop(self.learning_rate, self.rho, self.epsilon);
    }
}