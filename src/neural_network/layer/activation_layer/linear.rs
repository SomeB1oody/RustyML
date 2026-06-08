use crate::neural_network::layer::no_trainable_parameters_layer_functions;
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::activation_layer::format_shape;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::neural_network_trait::Layer;

/// Linear (Identity) activation layer.
///
/// Applies the identity function `f(x) = x` element-wise, preserving the input shape.
///
/// # Fields
///
/// - `input_shape` - Shape of the input from the forward pass, used to validate the gradient
///   during backpropagation (Linear's derivative is 1, so the input values themselves are not needed)
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::activation_layer::linear::Linear;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Linear activation
/// let mut model = Sequential::new();
/// model
///     .add(Linear::new())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]
/// ```
pub struct Linear {
    input_shape: Option<Vec<usize>>,
}

impl Linear {
    /// Creates a new Linear activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Linear` layer instance
    pub fn new() -> Self {
        Linear { input_shape: None }
    }
}

impl Default for Linear {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Check if tensor is empty
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Check for NaN or infinite values
        if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(Error::non_finite("input tensor"));
        }

        // Save the input shape for backpropagation (validation only)
        self.input_shape = Some(input.shape().to_vec());

        // Linear activation: f(x) = x (identity function)
        Ok(input.clone())
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

        // Linear activation: f(x) = x (identity function)
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(input_shape) = &self.input_shape {
            // Validate gradient output shape
            if grad_output.shape() != input_shape.as_slice() {
                return Err(Error::shape_mismatch(
                    input_shape.clone(),
                    grad_output.shape(),
                ));
            }

            // Check for NaN or infinite values in gradient output
            if grad_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Err(Error::non_finite("gradient output"));
            }

            // Linear derivative is 1, so gradient passes through unchanged
            Ok(grad_output.clone())
        } else {
            Err(Error::forward_pass_not_run("Linear"))
        }
    }

    fn layer_type(&self) -> &str {
        "Linear"
    }

    fn output_shape(&self) -> String {
        match &self.input_shape {
            Some(shape) => format_shape(shape),
            None => "Unknown".to_string(),
        }
    }

    no_trainable_parameters_layer_functions!();
}
