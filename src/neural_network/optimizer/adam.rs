use crate::neural_network::{Layer, Optimizer};
use ndarray::{Array2, Array3, Array4};

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

/// Stores and manages optimization state for the Adam optimizer algorithm.
///
/// Adam (Adaptive Moment Estimation) is an optimization algorithm that combines
/// the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSprop.
/// This struct maintains the first and second moment estimates (moving averages of gradients and
/// squared gradients) for weights, recurrent weights (optional), and biases.
///
/// # Fields
///
/// - `m` - First moment vector (moving average of gradients) for main parameters
/// - `v` - Second moment vector (moving average of squared gradients) for main parameters
/// - `m_recurrent` - First moment vector for recurrent parameters (if applicable)
/// - `v_recurrent` - Second moment vector for recurrent parameters (if applicable)
/// - `m_bias` - First moment vector for bias parameters
/// - `v_bias` - Second moment vector for bias parameters
#[derive(Debug, Clone, Default)]
pub struct AdamStates {
    pub m: Array2<f32>,
    pub v: Array2<f32>,
    pub m_recurrent: Option<Array2<f32>>,
    pub v_recurrent: Option<Array2<f32>>,
    pub m_bias: Array2<f32>,
    pub v_bias: Array2<f32>,
}

impl AdamStates {
    /// Creates a new Adam state object, initialized to zero
    ///
    /// # Parameters
    ///
    /// - `dims_param` - Tuple containing dimensions (rows, columns) for the main parameter matrices m and v
    /// - `dims_recurrent` - Optional tuple containing dimensions for recurrent parameter matrices; None if not using recurrent parameters
    /// - `dims_bias` - Tuple containing dimensions (rows, columns) for the bias parameter matrices
    ///
    /// # Returns
    ///
    /// - `Self` - A new AdamStates instance with all moment vectors initialized to zero matrices of appropriate dimensions
    pub fn new(
        dims_param: (usize, usize),
        dims_recurrent: Option<(usize, usize)>,
        dims_bias: (usize, usize),
    ) -> Self {
        let m_recurrent = dims_recurrent.map(|dims| Array2::zeros(dims));
        let v_recurrent = dims_recurrent.map(|dims| Array2::zeros(dims));

        Self {
            m: Array2::zeros(dims_param),
            v: Array2::zeros(dims_param),
            m_recurrent,
            v_recurrent,
            m_bias: Array2::zeros(dims_bias),
            v_bias: Array2::zeros(dims_bias),
        }
    }

    /// Updates Adam state for a single parameter and calculates update values
    ///
    /// # Parameters
    ///
    /// - `grad_param` - Gradient of the main parameter matrix
    /// - `grad_recurrent` - Optional gradient of the recurrent parameter matrix; None if not using recurrent parameters
    /// - `grad_bias` - Gradient of the bias parameter matrix
    /// - `beta1` - Exponential decay rate for first moment estimates (typically 0.9)
    /// - `beta2` - Exponential decay rate for second moment estimates (typically 0.999)
    /// - `epsilon` - Small constant added for numerical stability (typically 1e-8)
    /// - `t` - Current timestep (iteration number)
    /// - `lr` - Learning rate for parameter updates
    ///
    /// # Returns
    ///
    /// - Tuple containing:
    ///   - `Array2<f32>` - Update values for main parameter matrix
    ///   - `Option<Array2<f32>>` - Optional update values for recurrent parameter matrix; None if not using recurrent parameters
    ///   - `Array2<f32>` - Update values for bias parameter matrix
    pub fn update_parameter(
        &mut self,
        grad_param: &Array2<f32>,
        grad_recurrent: Option<&Array2<f32>>,
        grad_bias: &Array2<f32>,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: u64,
        lr: f32,
    ) -> (Array2<f32>, Option<Array2<f32>>, Array2<f32>) {
        // Update main parameter state
        Self::update_adam_param(&mut self.m, &mut self.v, grad_param, beta1, beta2);

        // Update recurrent parameter state (if exists)
        let recurrent_update = if let (Some(m_r), Some(v_r), Some(g_r)) = (
            self.m_recurrent.as_mut(),
            self.v_recurrent.as_mut(),
            grad_recurrent,
        ) {
            Self::update_adam_param(m_r, v_r, g_r, beta1, beta2);
            Some((m_r, v_r))
        } else {
            None
        };

        // Update bias parameter state
        Self::update_adam_param(&mut self.m_bias, &mut self.v_bias, grad_bias, beta1, beta2);

        // Calculate bias-corrected states
        let (m_hat, v_hat) = rayon::join(
            || self.m.mapv(|x| x / (1.0 - beta1.powi(t as i32))),
            || self.v.mapv(|x| x / (1.0 - beta2.powi(t as i32))),
        );

        let (m_hat_bias, v_hat_bias) = rayon::join(
            || self.m_bias.mapv(|x| x / (1.0 - beta1.powi(t as i32))),
            || self.v_bias.mapv(|x| x / (1.0 - beta2.powi(t as i32))),
        );

        // Calculate final updates
        let (param_update, bias_update) = rayon::join(
            || lr * &m_hat / &(v_hat.mapv(f32::sqrt) + epsilon),
            || lr * &m_hat_bias / &(v_hat_bias.mapv(f32::sqrt) + epsilon),
        );

        // Calculate recurrent parameter update (if exists)
        let recurrent_update = recurrent_update.map(|(m_r, v_r)| {
            let (m_hat_r, v_hat_r) = rayon::join(
                || m_r.mapv(|x| x / (1.0 - beta1.powi(t as i32))),
                || v_r.mapv(|x| x / (1.0 - beta2.powi(t as i32))),
            );
            lr * &m_hat_r / &(v_hat_r.mapv(f32::sqrt) + epsilon)
        });

        (param_update, recurrent_update, bias_update)
    }

    /// Helper function: Update Adam state variables
    ///
    /// # Parameters
    ///
    /// - `m` - First moment vector (moving average of gradients)
    /// - `v` - Second moment vector (moving average of squared gradients)
    /// - `g` - Current gradient
    /// - `beta1` - Exponential decay rate for first moment estimates
    /// - `beta2` - Exponential decay rate for second moment estimates
    ///
    /// # Effects
    ///
    /// - Updates `m` in-place with new first moment values: m = beta1*m + (1-beta1)*g
    /// - Updates `v` in-place with new second moment values: v = beta2*v + (1-beta2)*g²
    fn update_adam_param(
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
        g: &Array2<f32>,
        beta1: f32,
        beta2: f32,
    ) {
        // Parallel update computation
        let (m_updated, v_updated) = rayon::join(
            || m.mapv(|x| x * beta1) + &(g * (1.0 - beta1)),
            || v.mapv(|x| x * beta2) + &(g.mapv(|x| x * x) * (1.0 - beta2)),
        );

        *m = m_updated;
        *v = v_updated;
    }
}

/// Stores and manages optimization state for the Adam optimizer algorithm for Conv2D layer.
///
/// This struct is specifically designed to handle the optimization state for layers involved in feature extraction,
/// which typically deal with 4D tensors (e.g., convolutional layers). It maintains the first and second moment
/// estimates (moving averages of gradients and squared gradients) for weights and biases used in the Adam
/// optimization algorithm.
///
/// # Fields
///
/// - `m` - First moment tensor (moving average of gradients) for main parameters, stored as a 4D array
///   to accommodate convolutional filter dimensions
/// - `v` - Second moment tensor (moving average of squared gradients) for main parameters, stored as a 4D array
/// - `m_bias` - First moment matrix for bias parameters
/// - `v_bias` - Second moment matrix for bias parameters
#[derive(Debug, Clone, Default)]
pub struct AdamStatesConv2D {
    pub m: Array4<f32>,
    pub v: Array4<f32>,
    pub m_bias: Array2<f32>,
    pub v_bias: Array2<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct AdamStatesConv1D {
    pub m: Array3<f32>,
    pub v: Array3<f32>,
    pub m_bias: Array2<f32>,
    pub v_bias: Array2<f32>,
}
