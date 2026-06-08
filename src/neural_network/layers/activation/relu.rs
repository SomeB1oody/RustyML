use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::{Activation, format_output_shape};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::traits::Layer;

/// ReLU (Rectified Linear Unit) activation layer.
///
/// Applies `max(0, x)` element-wise to the input tensor, keeping the original shape.
/// Common inputs include 2D tensors for dense layers and 4D tensors for convolutional layers.
///
/// The activation math is provided by [`Activation::ReLU`]; this layer only adds
/// boundary validation and the caching required for backpropagation.
///
/// # Fields
///
/// - `output_cache` - Cached activated output from the forward pass, used during backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::activation::relu::ReLU;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with ReLU activation
/// let mut model = Sequential::new();
/// model
///     .add(ReLU::new())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]]
/// ```
pub struct ReLU {
    output_cache: Option<Tensor>,
}

impl ReLU {
    /// Creates a new ReLU activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `ReLU` layer instance
    pub fn new() -> Self {
        ReLU { output_cache: None }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Check if tensor is empty
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Check for NaN or infinite values
        if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(Error::non_finite("input tensor"));
        }

        // Apply ReLU: max(0, x)
        let output = Activation::ReLU.forward(input)?;

        // Save activated output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Check if tensor is empty
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Check for NaN or infinite values
        if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(Error::non_finite("input tensor"));
        }

        // Apply ReLU: max(0, x)
        Activation::ReLU.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // Validate gradient output shape (ReLU preserves shape)
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Check for NaN or infinite values in gradient output
            if grad_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Err(Error::non_finite("gradient output"));
            }

            // ReLU derivative is 1 for x > 0, and 0 for x <= 0
            Activation::ReLU.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("ReLU"))
        }
    }

    fn layer_type(&self) -> &str {
        "ReLU"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}
