//! GELU (Gaussian Error Linear Unit) activation layer
//!
//! The `GELU` struct holds the `approximate` flag and the shape recorded at build time. The
//! `with_approximate` method changes that flag after construction. `UnaryLayer::forward`
//! computes `Activation::GELU` and caches the input during training. `UnaryLayer::backward`
//! reads that cache to compute the gradient

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::{Activation, ActivationCache};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// GELU (Gaussian Error Linear Unit) activation layer
///
/// Applies `x * Phi(x)` elementwise to the input tensor, keeping the original shape. `Phi` is
/// the cumulative distribution function of the standard normal distribution. Common inputs
/// include 2D tensors for dense layers and 4D tensors for convolutional layers
///
/// Hendrycks and Gimpel (2016) introduced this activation. The layer uses the exact `Phi` by
/// default. [`GELU::with_approximate`] selects the tanh approximation
/// `x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) / 2`
///
/// [`Activation::GELU`] provides the activation math. This layer only adds boundary
/// validation and the caching needed for backpropagation. The derivative has no closed form in
/// the output, so the layer caches the input
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::activation::gelu::GELU;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with GELU activation
/// let mut model = SequentialBuilder::new()
///     .add(GELU::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-0.15865526, 1.9544997, -0.004049696], [3.9998734, -1.4332579e-6, 6.0]]
/// ```
///
/// Select the tanh approximation with the builder:
///
/// ```rust
/// use rustyml::neural_network::layers::activation::gelu::GELU;
/// use rustyml::neural_network::traits::UnaryLayer;
/// use rustyml::neural_network::Ctx;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((1, 3), vec![-1.0, 0.0, 1.0])
///     .unwrap()
///     .into_dyn();
/// let mut layer = GELU::new().with_approximate(true);
/// let output = layer.forward_mut(&x, &mut Ctx::inference()).unwrap();
///
/// // Output will be: [[-0.158808, 0.0, 0.841192]]
/// ```
#[derive(Debug)]
pub struct GELU {
    /// When `true`, the layer uses the tanh approximation. When `false`, it uses the exact
    /// `Phi`
    pub(super) approximate: bool,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl GELU {
    /// Creates a new GELU activation layer that uses the exact `Phi`
    ///
    /// # Returns
    ///
    /// - `Self` - A new `GELU` layer
    pub fn new() -> Self {
        GELU {
            approximate: false,
            built: None,
        }
    }

    /// Selects the form of the activation
    ///
    /// # Parameters
    ///
    /// - `approximate` - `true` for the tanh approximation, and `false` for the exact `Phi`
    ///
    /// # Returns
    ///
    /// - `Self` - The layer with the selected form
    pub fn with_approximate(mut self, approximate: bool) -> Self {
        self.approximate = approximate;
        self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for GELU {
    fn layer_type(&self) -> &str {
        "GELU"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for GELU {
    /// Records the shape the layer serves. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "GELU", input)? else {
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

        let activation = Activation::GELU {
            approximate: self.approximate,
        };
        if !ctx.is_training() {
            return activation.forward(input);
        }

        // The backward pass reads the input, so the cache holds a copy of it
        let (output, cache) = activation.forward_train(input.clone())?;
        ctx.push_cache("GELU", cache);

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: ActivationCache = ctx.pop_cache("GELU")?;

        // The derivative of the exact form is Phi(x) + x * phi(x), where phi is the standard
        // normal density
        Activation::GELU {
            approximate: self.approximate,
        }
        .backward(&cache, grad_output)
    }
}
