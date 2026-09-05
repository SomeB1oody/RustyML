//! Softsign activation layer that applies `x / (1 + |x|)` elementwise and parks the output
//! for backpropagation

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

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
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(Softsign::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-0.5, 0.66666669, -0.75], [0.80000001, -0.83333331, 0.85714287]]
/// ```
#[derive(Debug)]
pub struct Softsign {
    /// Shape the layer was built for. `None` before the build
    built: Option<Shape>,
}

impl Softsign {
    /// Creates a new Softsign activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Softsign` layer
    pub fn new() -> Self {
        Softsign { built: None }
    }
}

impl Default for Softsign {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for Softsign {
    fn layer_type(&self) -> &str {
        "Softsign"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Softsign {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Softsign", input)? else {
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

        let output = Activation::Softsign.forward(input)?;

        // Cache activated output for backpropagation
        if ctx.is_training() {
            ctx.push_cache("Softsign", output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("Softsign")?;

        // Softsign preserves shape, so gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // Softsign derivative is 1 / (1 + |x|)^2
        Activation::Softsign.backward(&output, grad_output)
    }
}
