use super::*;

/// Threshold for switching between sequential and parallel computation.
/// For arrays smaller than this threshold, sequential computation is used
/// to avoid parallelization overhead.
const ADA_GRAD_PARALLEL_THRESHOLD: usize = 1024;

/// AdaGrad (Adaptive Gradient Algorithm) optimizer
///
/// An adaptive learning rate optimization algorithm that adjusts the learning rate
/// for each parameter based on the historical sum of squared gradients. AdaGrad
/// performs larger updates for infrequent parameters and smaller updates for frequent ones.
///
/// # Fields
///
/// - `learning_rate` - Initial learning rate controlling the size of parameter updates
/// - `epsilon` - Small constant added for numerical stability
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Initial step size for parameter updates (typically 0.01)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// * `Result<Self, ModelError>` - A new AdaGrad optimizer instance or an error
    pub fn new(learning_rate: f32, epsilon: f32) -> Result<Self, ModelError> {
        // input validation
        validate_positive_finite(learning_rate, "learning_rate")?;
        validate_positive_finite(epsilon, "epsilon")?;

        Ok(Self {
            learning_rate,
            epsilon,
        })
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, layer: &mut dyn Layer) {
        layer.update_parameters_ada_grad(self.learning_rate, self.epsilon);
    }
}

/// Stores and manages optimization state for the AdaGrad optimizer algorithm.
///
/// AdaGrad maintains accumulated squared gradients for each parameter, which are
/// used to adapt the learning rate per parameter. This struct stores the accumulation
/// state for weights, recurrent weights (optional), and biases.
///
/// # Fields
///
/// - `accumulator` - Accumulated squared gradients for main parameters
/// - `accumulator_recurrent` - Accumulated squared gradients for recurrent parameters (if applicable)
/// - `accumulator_bias` - Accumulated squared gradients for bias parameters
#[derive(Debug, Clone, Default)]
pub struct AdaGradStates {
    pub accumulator: Array2<f32>,
    pub accumulator_recurrent: Option<Array2<f32>>,
    pub accumulator_bias: Array2<f32>,
}

impl AdaGradStates {
    /// Creates a new AdaGrad state object, initialized to zero
    ///
    /// # Parameters
    ///
    /// - `dims_param` - Tuple containing dimensions (rows, columns) for the main parameter accumulator
    /// - `dims_recurrent` - Optional tuple containing dimensions for recurrent parameter accumulator; None if not using recurrent parameters
    /// - `dims_bias` - Tuple containing dimensions (rows, columns) for the bias parameter accumulator
    ///
    /// # Returns
    ///
    /// - `AdaGradStates` - A new AdaGradStates instance with all accumulators initialized to zero matrices of appropriate dimensions
    pub fn new(
        dims_param: (usize, usize),
        dims_recurrent: Option<(usize, usize)>,
        dims_bias: (usize, usize),
    ) -> Self {
        let accumulator_recurrent = dims_recurrent.map(|dims| Array2::zeros(dims));

        Self {
            accumulator: Array2::zeros(dims_param),
            accumulator_recurrent,
            accumulator_bias: Array2::zeros(dims_bias),
        }
    }

    /// Updates AdaGrad state for parameters and calculates update values
    ///
    /// # Parameters
    ///
    /// - `grad_param` - Gradient of the main parameter matrix
    /// - `grad_recurrent` - Optional gradient of the recurrent parameter matrix; None if not using recurrent parameters
    /// - `grad_bias` - Gradient of the bias parameter matrix
    /// - `epsilon` - Small constant added for numerical stability (typically 1e-8)
    /// - `lr` - Learning rate for parameter updates
    ///
    /// # Returns
    ///
    /// * Tuple containing:
    ///   - `Array2<f32>` - Update values for main parameter matrix
    ///   - `Option<Array2<f32>>` - Optional update values for recurrent parameter matrix; None if not using recurrent parameters
    ///   - `Array2<f32>` - Update values for bias parameter matrix
    pub fn update_parameter(
        &mut self,
        grad_param: &Array2<f32>,
        grad_recurrent: Option<&Array2<f32>>,
        grad_bias: &Array2<f32>,
        epsilon: f32,
        lr: f32,
    ) -> (Array2<f32>, Option<Array2<f32>>, Array2<f32>) {
        // Determine whether to use parallel computation
        let use_parallel = self.accumulator.len() >= ADA_GRAD_PARALLEL_THRESHOLD;

        if use_parallel {
            // Parallel update for accumulators
            rayon::join(
                || Self::update_ada_grad_param(&mut self.accumulator, grad_param),
                || Self::update_ada_grad_param(&mut self.accumulator_bias, grad_bias),
            );
        } else {
            // Sequential update for accumulators
            Self::update_ada_grad_param(&mut self.accumulator, grad_param);
            Self::update_ada_grad_param(&mut self.accumulator_bias, grad_bias);
        }

        // Update recurrent parameter accumulator (if exists)
        let recurrent_accumulator = if let (Some(acc_r), Some(g_r)) =
            (self.accumulator_recurrent.as_mut(), grad_recurrent)
        {
            Self::update_ada_grad_param(acc_r, g_r);
            Some(acc_r)
        } else {
            None
        };

        // Calculate final updates
        let (param_update, bias_update) = if use_parallel {
            rayon::join(
                || lr * grad_param / &(self.accumulator.mapv(f32::sqrt) + epsilon),
                || lr * grad_bias / &(self.accumulator_bias.mapv(f32::sqrt) + epsilon),
            )
        } else {
            (
                lr * grad_param / &(self.accumulator.mapv(f32::sqrt) + epsilon),
                lr * grad_bias / &(self.accumulator_bias.mapv(f32::sqrt) + epsilon),
            )
        };

        // Calculate recurrent parameter update (if exists)
        let recurrent_update = recurrent_accumulator
            .map(|acc_r| lr * grad_recurrent.unwrap() / &(acc_r.mapv(f32::sqrt) + epsilon));

        (param_update, recurrent_update, bias_update)
    }

    /// Update AdaGrad accumulator:
    /// - Updates `accumulator` in-place with accumulated squared gradients: accumulator += g^2
    ///
    /// # Parameters
    ///
    /// - `accumulator` - Accumulated squared gradients
    /// - `g` - Current gradient
    fn update_ada_grad_param(accumulator: &mut Array2<f32>, g: &Array2<f32>) {
        *accumulator = &*accumulator + &g.mapv(|x| x * x);
    }
}

/// Stores and manages optimization state for the AdaGrad optimizer algorithm for Conv1D layer.
///
/// This struct is specifically designed to handle the optimization state for 1D convolutional layers
/// that process sequential data (e.g., time series, text sequences). It maintains the accumulated
/// squared gradients for weights and biases used in the AdaGrad optimization algorithm.
///
/// # Fields
///
/// - `accumulator` - Accumulated squared gradients for main parameters, stored as a 3D array
///   to accommodate 1D convolutional filter dimensions \[output_channels, input_channels, kernel_size\]
/// - `accumulator_bias` - Accumulated squared gradients for bias parameters
#[derive(Debug, Clone, Default)]
pub struct AdaGradStatesConv1D {
    pub accumulator: Array3<f32>,
    pub accumulator_bias: Array2<f32>,
}

/// Stores and manages optimization state for the AdaGrad optimizer algorithm for Conv2D layer.
///
/// This struct is specifically designed to handle the optimization state for 2D convolutional layers,
/// which typically deal with 4D tensors (e.g., image processing layers). It maintains the accumulated
/// squared gradients for weights and biases used in the AdaGrad optimization algorithm.
///
/// # Fields
///
/// - `accumulator` - Accumulated squared gradients for main parameters, stored as a 4D array
///   to accommodate convolutional filter dimensions
/// - `accumulator_bias` - Accumulated squared gradients for bias parameters
#[derive(Debug, Clone, Default)]
pub struct AdaGradStatesConv2D {
    pub accumulator: Array4<f32>,
    pub accumulator_bias: Array2<f32>,
}

/// AdaGrad optimizer state variables for 3D convolutional layers
///
/// This structure stores the accumulated squared gradients required by the AdaGrad optimizer
/// for updating weights and biases in 3D convolutional neural network layers. AdaGrad maintains
/// accumulated squared gradients to adapt the learning rate for each parameter.
///
/// # Fields
///
/// - `accumulator` - Accumulated squared gradients for the 5D convolution weights with shape
///   (output_channels, input_channels, kernel_depth, kernel_height, kernel_width)
/// - `accumulator_bias` - Accumulated squared gradients for the bias tensor with shape (1, output_channels)
#[derive(Debug, Clone, Default)]
pub struct AdaGradStatesConv3D {
    pub accumulator: Array5<f32>,
    pub accumulator_bias: Array2<f32>,
}

/// Stores and manages optimization state for the AdaGrad optimizer algorithm for Normalization layers.
///
/// This struct is specifically designed to handle the optimization state for normalization layers
/// (e.g., BatchNormalization, LayerNormalization) that have gamma (scale) and beta (shift) parameters.
/// AdaGrad maintains accumulated squared gradients for each parameter, which are used to adapt
/// the learning rate per parameter.
///
/// # Fields
///
/// - `acc_grad_gamma` - Accumulated squared gradients for gamma (scale) parameter
/// - `acc_grad_beta` - Accumulated squared gradients for beta (shift) parameter
#[derive(Debug, Clone, Default)]
pub struct AdaGradStatesNormalizationLayer {
    pub acc_grad_gamma: Tensor,
    pub acc_grad_beta: Tensor,
}
