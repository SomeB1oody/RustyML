//! Exponential activation layer that applies `e^x` elementwise and caches the output for
//! backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Exponential activation layer
///
/// Applies `f(x) = e^x` elementwise to the input tensor, keeping the original shape.
/// Common inputs include 2D tensors for dense layers and 4D tensors for convolutional layers
///
/// The output is strictly positive and unbounded. It suits a head that must give a positive
/// quantity, such as a rate or a variance. It overflows to infinity for a large input
///
/// [`Activation::Exponential`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::exponential::Exponential;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Exponential activation
/// let mut model = Sequential::new();
/// model
///     .add(Exponential::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.36787945, 7.3890562, 0.049787067], [54.598148, 0.006737947, 403.4288]]
/// ```
#[derive(Debug)]
pub struct Exponential {
    /// Cached activated output from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl Exponential {
    /// Creates a new Exponential activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Exponential` layer
    pub fn new() -> Self {
        Exponential { output_cache: None }
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Exponential {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let output = Activation::Exponential.forward(input)?;

        // Cache activated output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        Activation::Exponential.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // Exponential preserves shape, so gradient must match the cached output
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Exponential derivative is e^x, which equals the cached output
            Activation::Exponential.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("Exponential"))
        }
    }

    fn layer_type(&self) -> &str {
        "Exponential"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
