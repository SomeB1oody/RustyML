//! Tanh (hyperbolic tangent) activation layer

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Tanh (hyperbolic tangent) activation layer
///
/// Applies `tanh(x)` element-wise to the input tensor, mapping values to (-1, 1) while
/// preserving the input shape
///
/// [`Activation::Tanh`] provides the activation math. This layer adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::tanh::Tanh;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Tanh activation
/// let mut model = SequentialBuilder::new()
///     .add(Tanh::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be approximately: [[-0.76, 0.96, -0.99], [0.99, -0.99, 1.0]]
/// ```
#[derive(Debug)]
pub struct Tanh {
    /// Shape the layer was built for. `None` before the build
    built: Option<Shape>,
}

impl Tanh {
    /// Creates a new Tanh activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Tanh` layer
    pub fn new() -> Self {
        Tanh { built: None }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for Tanh {
    fn layer_type(&self) -> &str {
        "Tanh"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Tanh {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Tanh", input)? else {
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

        // Apply tanh. Large-magnitude inputs saturate toward -1/+1 by construction
        let output = Activation::Tanh.forward(input)?;

        // Cache the activated output for backpropagation
        if ctx.is_training() {
            ctx.push_cache(output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("Tanh")?;

        // tanh preserves shape, so the gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // Derivative: d/dx tanh(x) = 1 - tanh^2(x)
        Activation::Tanh.backward(&output, grad_output)
    }
}
