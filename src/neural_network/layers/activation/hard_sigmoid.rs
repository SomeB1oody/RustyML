//! Hard sigmoid activation layer that applies `clip(x/6 + 0.5, 0, 1)` elementwise and parks
//! the output for backpropagation

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

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
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(HardSigmoid::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.33333334, 0.83333337, 0.0], [1.0, 0.0, 1.0]]
/// ```
#[derive(Debug)]
pub struct HardSigmoid {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl HardSigmoid {
    /// Creates a new hard sigmoid activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `HardSigmoid` layer
    pub fn new() -> Self {
        HardSigmoid { built: None }
    }
}

impl Default for HardSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for HardSigmoid {
    fn layer_type(&self) -> &str {
        "HardSigmoid"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for HardSigmoid {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "HardSigmoid", input)? else {
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

        let output = Activation::HardSigmoid.forward(input)?;

        // Cache activated output for backpropagation
        if ctx.is_training() {
            ctx.push_cache("HardSigmoid", output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("HardSigmoid")?;

        // Hard sigmoid preserves shape, so gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // Hard sigmoid derivative is 1/6 on the linear segment, and 0 on both saturated ends
        Activation::HardSigmoid.backward(&output, grad_output)
    }
}
