//! ELU activation layer that applies `x` above 0 and `alpha * (e^x - 1)` at 0 and below, and
//! caches the output for backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

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
/// use rustyml::neural_network::sequential::Sequential;
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
/// let mut model = Sequential::new();
/// model
///     .add(ELU::new(1.0).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Cached activated output from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
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
        Ok(ELU {
            alpha,
            output_cache: None,
        })
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
            output_cache: None,
        }
    }
}

impl Layer for ELU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let output = Activation::ELU { alpha: self.alpha }.forward(input)?;

        // Cache activated output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        Activation::ELU { alpha: self.alpha }.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // ELU preserves shape, so gradient must match the cached output
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // ELU derivative is 1 for x > 0, and alpha * e^x, which is `a + alpha`, below it
            Activation::ELU { alpha: self.alpha }.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("ELU"))
        }
    }

    fn layer_type(&self) -> &str {
        "ELU"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
