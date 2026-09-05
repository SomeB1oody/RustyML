//! ELU activation layer that applies `x` above 0 and `alpha * (e^x - 1)` at 0 and below, and
//! caches the output for backpropagation

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// ELU (Exponential Linear Unit) activation layer
///
/// Applies `f(x) = x` for `x > 0` and `f(x) = alpha * (e^x - 1)` at 0 and below to the input
/// tensor, keeping the original shape. Common inputs include 2D tensors for dense layers and
/// 4D tensors for convolutional layers
///
/// The negative branch saturates at `-alpha` instead of at 0, which moves the mean activation
/// toward 0. Reference: Clevert et al. (2016)
///
/// [`Activation::ELU`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::elu::ELU;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with ELU activation
/// let mut model = SequentialBuilder::new()
///     .add(ELU::new(1.0).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-0.63212055, 2.0, -0.95021296], [4.0, -0.99326205, 6.0]]
/// ```
#[derive(Debug)]
pub struct ELU {
    /// Scale of the saturating negative branch
    pub(super) alpha: f32,
    /// Shape the layer was built for. `None` before the build
    built: Option<Shape>,
}

impl ELU {
    /// Creates a new ELU activation layer
    ///
    /// # Parameters
    ///
    /// - `alpha` - Scale of the saturating negative branch. Must be finite and greater than 0
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `ELU` layer, or an error if `alpha` is not valid
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - `alpha` is not finite and greater than 0
    pub fn new(alpha: f32) -> Result<Self, Error> {
        Activation::ELU { alpha }.validate()?;
        Ok(ELU { alpha, built: None })
    }
}

impl Default for ELU {
    /// Creates a new ELU layer with the default negative-branch scale
    ///
    /// # Default Values
    ///
    /// - `alpha` - `1.0`
    ///
    /// # Returns
    ///
    /// - `Self` - A new `ELU` layer with the default scale
    fn default() -> Self {
        ELU {
            alpha: 1.0,
            built: None,
        }
    }
}

impl LayerBase for ELU {
    fn layer_type(&self) -> &str {
        "ELU"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for ELU {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "ELU", input)? else {
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

        let output = Activation::ELU { alpha: self.alpha }.forward(input)?;

        // Cache activated output for backpropagation
        if ctx.is_training() {
            ctx.push_cache("ELU", output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("ELU")?;

        // ELU preserves shape, so gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // ELU derivative is 1 for x > 0, and alpha * e^x, which is `a + alpha`, below it
        Activation::ELU { alpha: self.alpha }.backward(&output, grad_output)
    }
}
