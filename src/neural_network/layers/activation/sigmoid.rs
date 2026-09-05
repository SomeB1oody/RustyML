//! Sigmoid activation layer for neural networks

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Sigmoid activation layer
///
/// Applies `1 / (1 + e^(-x))` elementwise to the input tensor, squashing values to (0, 1)
/// while preserving the input shape
///
/// [`Activation::Sigmoid`] provides the activation math. This layer adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::sigmoid::Sigmoid;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Sigmoid activation
/// let mut model = SequentialBuilder::new()
///     .add(Sigmoid::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be approximately: [[0.27, 0.88, 0.05], [0.98, 0.007, 0.998]]
/// ```
#[derive(Debug)]
pub struct Sigmoid {
    /// Shape the layer was built for. `None` before the build
    built: Option<Shape>,
}

impl Sigmoid {
    /// Creates a new Sigmoid activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Sigmoid` layer
    pub fn new() -> Self {
        Sigmoid { built: None }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for Sigmoid {
    fn layer_type(&self) -> &str {
        "Sigmoid"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Sigmoid {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Sigmoid", input)? else {
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

        // Large-magnitude inputs saturate to 0/1 by construction
        let output = Activation::Sigmoid.forward(input)?;

        if ctx.is_training() {
            ctx.push_cache(output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("Sigmoid")?;

        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
        Activation::Sigmoid.backward(&output, grad_output)
    }
}
