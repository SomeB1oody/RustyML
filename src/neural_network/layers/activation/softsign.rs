//! Softsign activation layer that applies `x / (1 + |x|)` elementwise and caches the output
//! for backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Softsign activation layer
///
/// Applies `f(x) = x / (1 + |x|)` elementwise to the input tensor, keeping the original shape.
/// Common inputs include 2D tensors for dense layers and 4D tensors for convolutional layers
///
/// Softsign is a bounded, zero-centered activation with the range `(-1, 1)`, like tanh. It
/// approaches its limits polynomially and not exponentially, so the tails saturate more slowly
///
/// [`Activation::Softsign`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::softsign::Softsign;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Softsign activation
/// let mut model = Sequential::new();
/// model
///     .add(Softsign::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-0.5, 0.66666669, -0.75], [0.80000001, -0.83333331, 0.85714287]]
/// ```
#[derive(Debug)]
pub struct Softsign {
    /// Cached activated output from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl Softsign {
    /// Creates a new Softsign activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Softsign` layer
    pub fn new() -> Self {
        Softsign { output_cache: None }
    }
}

impl Default for Softsign {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Softsign {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let output = Activation::Softsign.forward(input)?;

        // Cache activated output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        Activation::Softsign.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // Softsign preserves shape, so gradient must match the cached output
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Softsign derivative is 1 / (1 + |x|)^2
            Activation::Softsign.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("Softsign"))
        }
    }

    fn layer_type(&self) -> &str {
        "Softsign"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
