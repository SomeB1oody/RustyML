use ndarray::{Array2, Axis, Zip};

/// Activation function enum, supporting ReLU, Tanh, Sigmoid, and Softmax
#[derive(Debug, PartialEq)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
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
    /// * `Array2<f32>` - A new tensor with the activation function applied
    pub fn apply_activation(z: &Array2<f32>, activation: &Activation) -> Array2<f32> {
        use rayon::prelude::*;

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
    /// * `Array2<f32>` - A tensor containing the derivative values
    pub fn activation_derivative(
        activation_output: &Array2<f32>,
        activation: &Activation,
    ) -> Array2<f32> {
        match activation {
            Activation::ReLU => activation_output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => activation_output.mapv(|a| a * (1.0 - a)),
            Activation::Tanh => activation_output.mapv(|a| 1.0 - a * a),
            Activation::Softmax => Array2::ones(activation_output.dim()),
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
