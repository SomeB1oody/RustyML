//! Linear (identity) activation layer

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Linear (identity) activation layer
///
/// Applies the identity function `f(x) = x` elementwise, preserving the input shape
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::linear::Linear;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Linear activation
/// let mut model = SequentialBuilder::new()
///     .add(Linear::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]
/// ```
#[derive(Debug)]
pub struct Linear {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl Linear {
    /// Creates a new Linear activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Linear` layer
    pub fn new() -> Self {
        Linear { built: None }
    }
}

impl Default for Linear {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for Linear {
    fn layer_type(&self) -> &str {
        "Linear"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Linear {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Linear", input)? else {
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

        // Save the input shape for backward-pass validation only
        if ctx.is_training() {
            ctx.push_cache(input.shape().to_vec());
        }

        // Identity: f(x) = x
        Ok(input.clone())
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("Linear")?;

        if grad_output.shape() != input_shape.as_slice() {
            return Err(Error::shape_mismatch(input_shape, grad_output.shape()));
        }

        // Derivative is 1, so the gradient passes through unchanged
        Ok(grad_output.clone())
    }
}
