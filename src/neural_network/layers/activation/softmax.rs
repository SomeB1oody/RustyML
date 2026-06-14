//! Softmax activation layer that converts logits into per-row probability distributions

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Softmax activation layer
///
/// Applies softmax along the last axis, converting logits into a probability distribution
/// that sums to 1 for each row while preserving the input shape
///
/// The activation math is provided by [`Activation::Softmax`]; this layer only adds
/// boundary validation and the caching required for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::softmax::Softmax;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor with logits
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Softmax activation
/// let mut model = Sequential::new();
/// model
///     .add(Softmax::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output is a probability distribution that sums to 1.0 for each batch
/// ```
#[derive(Debug)]
pub struct Softmax {
    /// Cached output tensor from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl Softmax {
    /// Creates a new Softmax activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Softmax` layer instance
    pub fn new() -> Self {
        Softmax { output_cache: None }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Apply softmax over the last axis (input must be at least 2D)
        let output = Activation::Softmax.forward(input)?;

        // Cache output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Apply softmax over the last axis (input must be at least 2D)
        Activation::Softmax.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        match &self.output_cache {
            Some(output) => {
                // Softmax preserves shape, so the gradient must match the cached output
                if grad_output.shape() != output.shape() {
                    return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
                }

                Activation::Softmax.backward(output, grad_output)
            }
            None => Err(Error::forward_pass_not_run("Softmax")),
        }
    }

    fn layer_type(&self) -> &str {
        "Softmax"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
