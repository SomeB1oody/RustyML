//! Softplus activation layer that applies `ln(1 + e^x)` elementwise and caches the output
//! for backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Softplus activation layer
///
/// Applies `ln(1 + e^x)` elementwise to the input tensor, keeping the original shape.
/// Common inputs include 2D tensors for dense layers and 4D tensors for convolutional layers
///
/// Softplus is a smooth approximation of ReLU. The output is strictly positive, and the
/// gradient never reaches 0, so there is no dead-unit failure mode
///
/// [`Activation::Softplus`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::softplus::Softplus;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Softplus activation
/// let mut model = Sequential::new();
/// model
///     .add(Softplus::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.31326169, 2.1269281, 0.048587352], [4.0181499, 0.0067153485, 6.0024757]]
/// ```
#[derive(Debug)]
pub struct Softplus {
    /// Cached activated output from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl Softplus {
    /// Creates a new Softplus activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Softplus` layer
    pub fn new() -> Self {
        Softplus { output_cache: None }
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Softplus {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let output = Activation::Softplus.forward(input)?;

        // Cache activated output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        Activation::Softplus.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // Softplus preserves shape, so gradient must match the cached output
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Softplus derivative is the logistic sigmoid of the input
            Activation::Softplus.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("Softplus"))
        }
    }

    fn layer_type(&self) -> &str {
        "Softplus"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
