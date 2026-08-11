//! Hard sigmoid activation layer that applies `clip(x/6 + 0.5, 0, 1)` elementwise and caches
//! the output for backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Hard sigmoid activation layer
///
/// Applies `f(x) = clip(x/6 + 0.5, 0, 1)` elementwise to the input tensor, keeping the
/// original shape. Common inputs include 2D tensors for dense layers and 4D tensors for
/// convolutional layers
///
/// This function is a piecewise-linear approximation of the logistic sigmoid that needs no
/// exponential. The output is exactly 0 at `x <= -3` and exactly 1 at `x >= 3`, where the
/// gradient is 0
///
/// [`Activation::HardSigmoid`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::hard_sigmoid::HardSigmoid;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with hard sigmoid activation
/// let mut model = Sequential::new();
/// model
///     .add(HardSigmoid::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.33333334, 0.83333337, 0.0], [1.0, 0.0, 1.0]]
/// ```
#[derive(Debug)]
pub struct HardSigmoid {
    /// Cached activated output from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl HardSigmoid {
    /// Creates a new hard sigmoid activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `HardSigmoid` layer
    pub fn new() -> Self {
        HardSigmoid { output_cache: None }
    }
}

impl Default for HardSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for HardSigmoid {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let output = Activation::HardSigmoid.forward(input)?;

        // Cache activated output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        Activation::HardSigmoid.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // Hard sigmoid preserves shape, so gradient must match the cached output
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Hard sigmoid derivative is 1/6 on the linear segment, and 0 on both saturated ends
            Activation::HardSigmoid.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("HardSigmoid"))
        }
    }

    fn layer_type(&self) -> &str {
        "HardSigmoid"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
