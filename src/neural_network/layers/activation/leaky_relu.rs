//! Leaky ReLU activation layer that scales the negative side by `negative_slope` and parks the
//! output for backpropagation

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Leaky ReLU (Leaky Rectified Linear Unit) activation layer
///
/// Applies `f(x) = x` for `x >= 0`, and `f(x) = negative_slope * x` below 0. The transform is
/// elementwise and keeps the original shape. Common inputs include 2D tensors for dense layers
/// and 4D tensors for convolutional layers
///
/// Unlike [`Activation::ReLU`], the negative side keeps a non-zero gradient. A unit whose
/// pre-activation stays negative for the whole batch still receives gradient, so it can
/// recover. This is the standard answer to the dead-unit failure mode of ReLU
///
/// [`Activation::LeakyReLU`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::leaky_relu::LeakyReLU;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Leaky ReLU activation
/// let mut model = SequentialBuilder::new()
///     .add(LeakyReLU::new(0.3).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-0.3, 2.0, -0.9], [4.0, -1.5, 6.0]]
/// ```
#[derive(Debug)]
pub struct LeakyReLU {
    /// Slope applied below 0. Must be finite and greater than 0
    pub(super) negative_slope: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl LeakyReLU {
    /// Creates a new Leaky ReLU activation layer
    ///
    /// # Parameters
    ///
    /// - `negative_slope` - Slope applied below 0. Must be finite and greater than 0
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `LeakyReLU` layer, or an error if the slope is invalid
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - `negative_slope` is not finite and greater than 0
    pub fn new(negative_slope: f32) -> Result<Self, Error> {
        Activation::LeakyReLU { negative_slope }.validate()?;
        Ok(LeakyReLU {
            negative_slope,
            built: None,
        })
    }
}

impl Default for LeakyReLU {
    /// Creates a new LeakyReLU layer with the default slope
    ///
    /// # Default Values
    ///
    /// - `negative_slope` - `0.3`
    ///
    /// # Returns
    ///
    /// - `Self` - A new `LeakyReLU` layer with the default slope
    fn default() -> Self {
        LeakyReLU {
            negative_slope: 0.3,
            built: None,
        }
    }
}

impl LayerBase for LeakyReLU {
    fn layer_type(&self) -> &str {
        "LeakyReLU"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for LeakyReLU {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "LeakyReLU", input)? else {
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

        let output = Activation::LeakyReLU {
            negative_slope: self.negative_slope,
        }
        .forward(input)?;

        if ctx.is_training() {
            ctx.push_cache("LeakyReLU", output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("LeakyReLU")?;

        // Leaky ReLU preserves shape, so gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // Leaky ReLU derivative is 1 for x >= 0, and `negative_slope` below 0
        Activation::LeakyReLU {
            negative_slope: self.negative_slope,
        }
        .backward(&output, grad_output)
    }
}
