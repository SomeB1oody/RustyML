//! SiLU (Sigmoid Linear Unit) activation layer, also named Swish
//!
//! The `SiLU` struct holds only the shape recorded at build time, because the layer takes no
//! parameter. `UnaryLayer::forward` computes `Activation::SiLU` and caches the input during
//! training. `UnaryLayer::backward` reads that cache to compute the gradient

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::{Activation, ActivationCache};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// SiLU (Sigmoid Linear Unit) activation layer
///
/// Applies `x * sigmoid(x)` elementwise to the input tensor, keeping the original shape.
/// Common inputs include 2D tensors for dense layers and 4D tensors for convolutional layers
///
/// Elfwing et al. (2018) introduced this activation, and Ramachandran et al. (2017) named it
/// Swish. The output is smooth and not monotonic. Its minimum is about -0.2785 at x = -1.2785
///
/// [`Activation::SiLU`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation. The derivative has no closed form in
/// the output, so the layer caches the input
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::silu::SiLU;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with SiLU activation
/// let mut model = SequentialBuilder::new()
///     .add(SiLU::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-0.26894143, 1.761594, -0.14227763], [3.928055, -0.033464253, 5.9851646]]
/// ```
#[derive(Debug)]
pub struct SiLU {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl SiLU {
    /// Creates a new SiLU activation layer
    ///
    /// # Returns
    ///
    /// - `Self` - A new `SiLU` layer
    pub fn new() -> Self {
        SiLU { built: None }
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for SiLU {
    fn layer_type(&self) -> &str {
        "SiLU"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for SiLU {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "SiLU", input)? else {
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

        if !ctx.is_training() {
            return Activation::SiLU.forward(input);
        }

        // The backward pass reads the input, so the cache holds a copy of it
        let (output, cache) = Activation::SiLU.forward_train(input.clone())?;
        ctx.push_cache("SiLU", cache);

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: ActivationCache = ctx.pop_cache("SiLU")?;

        // SiLU derivative is sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        Activation::SiLU.backward(&cache, grad_output)
    }
}
