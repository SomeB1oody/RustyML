//! SELU activation layer that applies the scaled exponential linear unit elementwise and parks
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

/// SELU (Scaled Exponential Linear Unit) activation layer
///
/// Applies `f(x) = scale * x` for `x > 0`, and `f(x) = scale * alpha * (e^x - 1)` at 0 and
/// below, keeping the original shape. The values `alpha = 1.6732632` and `scale = 1.0507010`
/// are fixed. Common inputs include 2D tensors for dense layers and 4D tensors for
/// convolutional layers
///
/// Klambauer et al. (2017) picked the 2 constants so that the activations keep a mean of 0
/// and a variance of 1 across layers. That self-normalizing property needs Lecun-normal
/// initialization and a wide enough layer
///
/// [`Activation::SELU`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::selu::SELU;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with SELU activation
/// let mut model = SequentialBuilder::new()
///     .add(SELU::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-1.1113307, 2.101402, -1.6705688], [4.2028041, -1.7462534, 6.3042059]]
/// ```
#[derive(Debug)]
pub struct SELU {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl SELU {
    /// Creates a new SELU activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `SELU` layer
    pub fn new() -> Self {
        SELU { built: None }
    }
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for SELU {
    fn layer_type(&self) -> &str {
        "SELU"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for SELU {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "SELU", input)? else {
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

        let output = Activation::SELU.forward(input)?;

        if ctx.is_training() {
            ctx.push_cache("SELU", output.clone());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output: Tensor = ctx.pop_cache("SELU")?;

        // SELU preserves shape, so gradient must match the cached output
        if grad_output.shape() != output.shape() {
            return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
        }

        // SELU derivative is scale for x > 0, and scale * alpha * e^x for x <= 0
        Activation::SELU.backward(&output, grad_output)
    }
}
