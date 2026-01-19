use super::*;

/// Threshold for switching between sequential and parallel computation.
/// For arrays smaller than this threshold, sequential computation is used
/// to avoid parallelization overhead.
const SGD_PARALLEL_THRESHOLD: usize = 1024;

/// SGD (Stochastic Gradient Descent) optimizer
///
/// A simple optimization algorithm that updates parameters in the direction
/// of the negative gradient, scaled by the learning rate.
///
/// # Fields
///
/// * `learning_rate` - Learning rate controlling the size of parameter updates
pub struct SGD {
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
    /// * `Result<Self, ModelError>` - A new SGD optimizer instance or an error
    pub fn new(learning_rate: f32) -> Result<Self, ModelError> {
        // input validation
        validate_learning_rate(learning_rate)?;

        Ok(Self { learning_rate })
    }

    /// Simultaneously update two sets of parameters (with automatic parallel/sequential selection)
    ///
    /// # Parameters
    ///
    /// - `weights` - Mutable reference to weights array to be updated
    /// - `weight_grads` - Reference to weight gradients array
    /// - `bias` - Mutable reference to bias array to be updated
    /// - `bias_grads` - Reference to bias gradients array
    /// - `lr` - Learning rate
    pub fn update_sgd_parameters(
        weights: &mut [f32],
        weight_grads: &[f32],
        bias: &mut [f32],
        bias_grads: &[f32],
        lr: f32,
    ) {
        let use_parallel =
            weights.len() >= SGD_PARALLEL_THRESHOLD || bias.len() >= SGD_PARALLEL_THRESHOLD;

        let update_fn = |params: &mut [f32], grads: &[f32]| {
            if use_parallel {
                params
                    .par_iter_mut()
                    .zip(grads.par_iter())
                    .for_each(|(p, g)| *p -= *g * lr);
            } else {
                params
                    .iter_mut()
                    .zip(grads.iter())
                    .for_each(|(p, g)| *p -= *g * lr);
            }
        };

        if use_parallel {
            rayon::join(
                || update_fn(weights, weight_grads),
                || update_fn(bias, bias_grads),
            );
        } else {
            update_fn(weights, weight_grads);
            update_fn(bias, bias_grads);
        }
    }

    /// Update three sets of RNN parameters: kernel, recurrent_kernel and bias (with automatic parallel/sequential selection)
    ///
    /// # Parameters
    ///
    /// - `kernel` - Mutable reference to kernel matrix to be updated
    /// - `grad_kernel` - Reference to kernel gradient matrix
    /// - `recurrent_kernel` - Mutable reference to recurrent_kernel matrix to be updated
    /// - `grad_recurrent_kernel` - Reference to recurrent_kernel gradient matrix
    /// - `bias` - Mutable reference to bias matrix to be updated
    /// - `grad_bias` - Reference to bias gradient matrix
    /// - `lr` - Learning rate
    pub fn update_sgd_parameters_rnn(
        kernel: &mut Array2<f32>,
        grad_kernel: &Array2<f32>,
        recurrent_kernel: &mut Array2<f32>,
        grad_recurrent_kernel: &Array2<f32>,
        bias: &mut Array2<f32>,
        grad_bias: &Array2<f32>,
        lr: f32,
    ) {
        let use_parallel = kernel.len() >= SGD_PARALLEL_THRESHOLD
            || recurrent_kernel.len() >= SGD_PARALLEL_THRESHOLD
            || bias.len() >= SGD_PARALLEL_THRESHOLD;

        if use_parallel {
            rayon::join(
                || {
                    rayon::join(
                        || *kernel = kernel.clone() - (grad_kernel.clone() * lr),
                        || {
                            *recurrent_kernel =
                                recurrent_kernel.clone() - (grad_recurrent_kernel.clone() * lr)
                        },
                    )
                },
                || *bias = bias.clone() - (grad_bias.clone() * lr),
            );
        } else {
            *kernel = kernel.clone() - (grad_kernel.clone() * lr);
            *recurrent_kernel = recurrent_kernel.clone() - (grad_recurrent_kernel.clone() * lr);
            *bias = bias.clone() - (grad_bias.clone() * lr);
        }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // Directly call the layer's parameter update method
        layer.update_parameters_sgd(self.learning_rate);
    }
}
