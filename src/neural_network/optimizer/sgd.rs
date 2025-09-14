use crate::neural_network::{Layer, Optimizer};
use rayon::prelude::*;

/// Stochastic Gradient Descent (SGD) optimizer.
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
    /// * `Self` - A new SGD optimizer instance
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Simultaneously update two sets of parameters in parallel
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
        rayon::join(
            || {
                // Update weights
                weights
                    .par_iter_mut()
                    .zip(weight_grads.par_iter())
                    .for_each(|(w, wg)| {
                        *w -= *wg * lr;
                    });
            },
            || {
                // Update bias
                bias.par_iter_mut()
                    .zip(bias_grads.par_iter())
                    .for_each(|(b, bg)| {
                        *b -= *bg * lr;
                    });
            },
        );
    }

    /// Update three sets of RNN parameters in parallel: kernel, recurrent_kernel and bias
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
        kernel: &mut ndarray::Array2<f32>,
        grad_kernel: &ndarray::Array2<f32>,
        recurrent_kernel: &mut ndarray::Array2<f32>,
        grad_recurrent_kernel: &ndarray::Array2<f32>,
        bias: &mut ndarray::Array2<f32>,
        grad_bias: &ndarray::Array2<f32>,
        lr: f32,
    ) {
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
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // Directly call the layer's parameter update method
        layer.update_parameters_sgd(self.learning_rate);
    }
}
