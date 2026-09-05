//! Softmax activation layer that converts logits into per-lane probability distributions

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::{Activation, DEFAULT_SOFTMAX_AXIS};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

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
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(Softmax::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
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
/// use rustyml::neural_network::traits::UnaryLayer;
/// use rustyml::neural_network::Ctx;
/// use ndarray::Array3;
///
/// // Each of the 3 channels of a position becomes a distribution over the 2 batch items
/// let x = Array3::<f32>::zeros((2, 4, 3)).into_dyn();
/// let mut layer = Softmax::new().with_axis(0);
/// let mut ctx = Ctx::inference();
/// let output = layer.forward_mut(&x, &mut ctx).unwrap();
/// assert_eq!(output.shape(), &[2, 4, 3]);
/// ```
#[derive(Debug)]
pub struct Softmax {
    /// Axis to normalize. A negative value counts back from the end
    pub(super) axis: i32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
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
            built: None,
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

impl LayerBase for Softmax {
    fn layer_type(&self) -> &str {
        "Softmax"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Softmax {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Softmax", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // The axis resolves against the rank of this input, and an out-of-range axis fails here
        let output = Activation::Softmax { axis: self.axis }.forward(input)?;

        // Cache output for backpropagation
        if ctx.is_training() {
            ctx.push_cache(output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("Softmax")?;

        // Softmax preserves shape, so the gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        Activation::Softmax { axis: self.axis }.backward(&output, grad_output)
    }
}
