//! Sigmoid activation layer for neural networks

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Sigmoid activation layer
///
/// Applies `1 / (1 + e^(-x))` elementwise to the input tensor, squashing values to (0, 1)
/// while preserving the input shape
///
/// The activation math is provided by [`Activation::Sigmoid`]; this layer only adds
/// boundary validation and the caching required for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::sigmoid::Sigmoid;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Sigmoid activation
/// let mut model = Sequential::new();
/// model
///     .add(Sigmoid::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be approximately: [[0.27, 0.88, 0.05], [0.98, 0.007, 0.998]]
/// ```
#[derive(Debug)]
pub struct Sigmoid {
    /// Cached output tensor from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl Sigmoid {
    /// Creates a new Sigmoid activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Sigmoid` layer instance
    pub fn new() -> Self {
        Sigmoid { output_cache: None }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Sigmoid clips the input internally for numerical stability
        let output = Activation::Sigmoid.forward(input)?;

        // Cache the output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Sigmoid clips the input internally for numerical stability
        Activation::Sigmoid.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
            Activation::Sigmoid.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("Sigmoid"))
        }
    }

    fn layer_type(&self) -> &str {
        "Sigmoid"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
