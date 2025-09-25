use ndarray::{Array2, ArrayD, Axis, Zip};
use rayon::prelude::*;

/// Activation function enum, supporting ReLU, Tanh, Sigmoid, Softmax, and Linear
///
/// # Variants
///
/// - `ReLU` - Rectified Linear Unit activation function.
///   Returns max(0, x) for each element. Commonly used in hidden layers
///   due to its computational efficiency and ability to mitigate vanishing gradients.
///
/// - `Tanh` - Hyperbolic tangent activation function.
///   Maps input values to the range (-1, 1). Often preferred over sigmoid
///   as it is zero-centered, which can help with gradient flow during training.
///
/// - `Sigmoid` - Logistic sigmoid activation function.
///   Maps input values to the range (0, 1) using the formula 1/(1 + e^(-x)).
///   Traditionally used for binary classification output layers.
///
/// - `Softmax` - Softmax activation function for multi-class classification.
///   Converts a vector of real numbers into a probability distribution
///   where all values sum to 1. Applied row-wise for batch processing.
///
/// - `Linear` - Linear (identity) activation function.
///   Returns the input unchanged: f(x) = x. Commonly used in regression
///   output layers where the network needs to predict continuous values.
#[derive(Debug, PartialEq)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    Linear,
}
impl Activation {
    /// Forward application of activation functions
    ///
    /// Applies the specified activation function to the input tensor.
    ///
    /// # Parameters
    ///
    /// - `z` - Input tensor to apply activation function to
    /// - `activation` - The activation function to apply
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - A new tensor with the activation function applied
    pub fn apply_activation(z: &Array2<f32>, activation: &Activation) -> Array2<f32> {
        match activation {
            Activation::ReLU => {
                let mut result = z.clone();
                result.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                result
            }
            Activation::Sigmoid => {
                let mut result = z.clone();
                result.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
                result
            }
            Activation::Tanh => {
                let mut result = z.clone();
                result.par_mapv_inplace(|x| x.tanh());
                result
            }
            Activation::Linear => {
                // Linear activation: f(x) = x (identity function)
                z.clone()
            }
            Activation::Softmax => {
                let mut out = z.clone();

                if out.nrows() > 8 {
                    out.axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .for_each(|mut row| {
                            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            row.mapv_inplace(|x| (x - max_val).exp());
                            let sum = row.sum();
                            row.mapv_inplace(|x| x / sum);
                        });
                } else {
                    for mut row in out.outer_iter_mut() {
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        row.map_inplace(|x| *x = (*x - max_val).exp());
                        let sum = row.sum();
                        row.map_inplace(|x| *x /= sum);
                    }
                }
                out
            }
        }
    }

    /// Applies the activation function to the tensor in-place (for conv layers and general use).
    ///
    /// If an activation function is specified for this layer, this method applies it
    /// element-wise to the input tensor using parallel processing.
    ///
    /// # Parameters
    ///
    /// - `activation` - Activation type used
    /// - `x` - A mutable reference to the tensor to which the activation function will be applied
    pub fn apply_activation_inplace(activation: &Activation, x: &mut ArrayD<f32>) {
        match activation {
            Activation::ReLU => {
                x.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
            }
            Activation::Sigmoid => {
                x.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            }
            Activation::Tanh => {
                x.par_mapv_inplace(|x| x.tanh());
            }
            Activation::Linear => {
                // Linear activation: f(x) = x (identity function)
                // No operation needed as input remains unchanged
            }
            Activation::Softmax => {
                panic!("Cannot use Softmax for convolution layers");
            }
        }
    }

    /// Computes derivatives for ReLU, Sigmoid, and Tanh activation functions
    ///
    /// Returns the derivative of the activation function given the activated output.
    /// For Softmax, the gradient is handled separately in backward propagation.
    ///
    /// # Parameters
    ///
    /// - `activation_output` - The output after activation function has been applied
    /// - `activation` - The activation function whose derivative to compute
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - A tensor containing the derivative values
    pub fn activation_derivative(
        activation_output: &Array2<f32>,
        activation: &Activation,
    ) -> Array2<f32> {
        let mut result = activation_output.clone();

        match activation {
            Activation::ReLU => result.par_mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => result.par_mapv_inplace(|a| a * (1.0 - a)),
            Activation::Tanh => result.par_mapv_inplace(|a| 1.0 - a * a),
            Activation::Linear => {
                // Linear activation derivative: f'(x) = 1
                result.par_mapv_inplace(|_| 1.0);
            }
            Activation::Softmax => return Array2::ones(activation_output.dim()),
        }

        result
    }

    /// Calculates the derivative of the activation function at the given output values (in-place).
    ///
    /// This function is used during backpropagation to compute gradients.
    ///
    /// # Parameters
    ///
    /// - `activation` - The activation function whose derivative to compute
    /// - `output` - A mutable reference to the tensor containing the output values of the forward pass.
    pub fn activation_derivative_inplace(activation: &Activation, output: &mut ArrayD<f32>) {
        match activation {
            Activation::ReLU => {
                output.par_mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
            }
            Activation::Sigmoid => {
                output.par_mapv_inplace(|a| a * (1.0 - a));
            }
            Activation::Tanh => {
                output.par_mapv_inplace(|a| 1.0 - a * a);
            }
            Activation::Linear => {
                // Linear activation derivative: f'(x) = 1
                output.par_mapv_inplace(|_| 1.0);
            }
            Activation::Softmax => {
                panic!("Cannot use Softmax for convolution layers");
            }
        }
    }

    /// Backward propagation for Softmax activation
    ///
    /// For each row, computes:
    /// new_grad\[i\] = a\[i\] * (upstream\[i\] - sum_j(a\[j\]*upstream\[j\]))
    ///
    /// # Parameters
    ///
    /// - `a` - The output from the softmax activation
    /// - `upstream` - The gradient flowing from the next layer
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - The gradient with respect to the input of the softmax function
    pub fn softmax_backward(a: &Array2<f32>, upstream: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::<f32>::zeros(a.raw_dim());

        Zip::from(result.axis_iter_mut(Axis(0)))
            .and(a.axis_iter(Axis(0)))
            .and(upstream.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, a_row, up_row| {
                let dot = a_row
                    .iter()
                    .zip(up_row.iter())
                    .map(|(&ai, &gi)| ai * gi)
                    .sum::<f32>();

                for (j, r) in out_row.iter_mut().enumerate() {
                    *r = a_row[j] * (up_row[j] - dot);
                }
            });

        result
    }
}
