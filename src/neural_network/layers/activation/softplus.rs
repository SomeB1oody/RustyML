//! Softplus activation layer that applies `ln(1 + e^x)` elementwise and parks the output
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
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(Softplus::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.31326169, 2.1269281, 0.048587352], [4.0181499, 0.0067153485, 6.0024757]]
/// ```
#[derive(Debug)]
pub struct Softplus {
    /// Shape the layer was built for. `None` before the build
    built: Option<Shape>,
}

impl Softplus {
    /// Creates a new Softplus activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Softplus` layer
    pub fn new() -> Self {
        Softplus { built: None }
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for Softplus {
    fn layer_type(&self) -> &str {
        "Softplus"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Softplus {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Softplus", input)? else {
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

        let output = Activation::Softplus.forward(input)?;

        // Cache activated output for backpropagation
        if ctx.is_training() {
            ctx.push_cache(output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("Softplus")?;

        // Softplus preserves shape, so gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // Softplus derivative is the logistic sigmoid of the input
        Activation::Softplus.backward(&output, grad_output)
    }
}
