use crate::neural_network::layer::no_trainable_parameters_layer_functions;
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::activation_layer::{Activation, format_output_shape};
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::neural_network_trait::{ActivationLayer, Layer};

/// Tanh (Hyperbolic Tangent) activation layer.
///
/// Applies `tanh(x)` element-wise to the input tensor, mapping values to (-1, 1) while
/// preserving the input shape.
///
/// The activation math is provided by [`Activation::Tanh`]; this layer only adds
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
/// use rustyml::neural_network::layer::activation_layer::tanh::Tanh;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Tanh activation
/// let mut model = Sequential::new();
/// model
///     .add(Tanh::new())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be approximately: [[-0.76, 0.96, -0.99], [0.99, -0.99, 1.0]]
/// ```
pub struct Tanh {
    output_cache: Option<Tensor>,
}

impl Tanh {
    /// Creates a new Tanh activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Tanh` layer instance
    pub fn new() -> Self {
        Tanh { output_cache: None }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Check if tensor is empty
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Check for NaN or infinite values
        if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(Error::non_finite("input tensor"));
        }

        // Apply Tanh: tanh(x) with input clipping for numerical stability
        let output = Activation::Tanh.forward(input)?;

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

        // Apply Tanh: tanh(x) with input clipping for numerical stability
        Activation::Tanh.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(output) = &self.output_cache {
            // Validate gradient output shape (tanh preserves shape)
            if grad_output.shape() != output.shape() {
                return Err(Error::shape_mismatch(output.shape(), grad_output.shape()));
            }

            // Check for NaN or infinite values in gradient output
            if grad_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Err(Error::non_finite("gradient output"));
            }

            // Tanh derivative is: d/dx tanh(x) = 1 - tanh^2(x)
            Activation::Tanh.backward(output, grad_output)
        } else {
            Err(Error::forward_pass_not_run("Tanh"))
        }
    }

    fn layer_type(&self) -> &str {
        "Tanh"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}

impl ActivationLayer for Tanh {}
