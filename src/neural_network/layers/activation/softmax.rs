//! Softmax activation layer that converts logits into per-lane probability distributions

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{
    Activation, DEFAULT_SOFTMAX_AXIS, format_output_shape,
};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Softmax activation layer
///
/// Applies softmax along 1 axis, which the `axis` field selects. The lanes along that axis
/// each become a probability distribution that sums to 1. The tensor keeps its shape
///
/// The default axis is `-1`, the last axis. A negative axis counts back from the end, and the
/// layer resolves it against the rank of the input on each call. The same layer therefore
/// normalizes the last axis of a rank-2 input and the last axis of a rank-4 input
///
/// [`Activation::Softmax`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
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
///
/// Normalize a different axis with the builder:
///
/// ```rust
/// use rustyml::neural_network::layers::activation::softmax::Softmax;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array3;
///
/// // Each of the 3 channels of a position becomes a distribution over the 2 batch items
/// let x = Array3::<f32>::zeros((2, 4, 3)).into_dyn();
/// let mut layer = Softmax::new().with_axis(0);
/// let output = layer.forward(&x).unwrap();
/// assert_eq!(output.shape(), &[2, 4, 3]);
/// ```
#[derive(Debug)]
pub struct Softmax {
    /// Axis to normalize. A negative value counts back from the end
    pub(super) axis: i32,
    /// Cached output tensor from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
}

impl Softmax {
    /// Creates a new Softmax activation layer over the last axis
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Softmax` layer with `axis` set to `-1`
    pub fn new() -> Self {
        Softmax {
            axis: DEFAULT_SOFTMAX_AXIS,
            output_cache: None,
        }
    }

    /// Sets the axis that the layer normalizes
    ///
    /// The layer keeps the value as given. A negative value counts back from the end, and the
    /// layer resolves it against the rank of the input on each forward pass. A layer that held
    /// a resolved index would reduce the wrong axis as soon as the input rank changed. An
    /// out-of-range axis therefore fails the forward pass, not this call
    ///
    /// # Parameters
    ///
    /// - `axis` - Axis to normalize, which can be negative
    ///
    /// # Returns
    ///
    /// - `Self` - The layer with the new axis
    pub fn with_axis(mut self, axis: i32) -> Self {
        self.axis = axis;
        self
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

        // The axis resolves against the rank of this input, and an out-of-range axis fails here
        let output = Activation::Softmax { axis: self.axis }.forward(input)?;

        // Cache output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        Activation::Softmax { axis: self.axis }.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        match &self.output_cache {
            Some(output) => {
                // Softmax preserves shape, so the gradient must match the cached output
                if grad_output.shape() != output.shape() {
                    return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
                }

                Activation::Softmax { axis: self.axis }.backward(output, grad_output)
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
