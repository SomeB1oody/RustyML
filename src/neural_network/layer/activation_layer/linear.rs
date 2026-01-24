use super::*;

/// Linear (Identity) activation layer.
///
/// Applies the identity function `f(x) = x` element-wise, preserving the input shape.
///
/// # Fields
///
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::prelude::*;
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
    input_cache: Option<Tensor>,
}

impl Linear {
    /// Creates a new Linear activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Linear` layer instance
    pub fn new() -> Self {
        Linear { input_cache: None }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Check if tensor is empty
        if input.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input tensor is empty".to_string(),
            ));
        }

        // Check for NaN or infinite values
        if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(ModelError::InputValidationError(
                "Input tensor contains NaN or infinite values".to_string(),
            ));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Linear activation: f(x) = x (identity function)
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            // Validate gradient output shape
            if grad_output.shape() != input.shape() {
                return Err(ModelError::ProcessingError(format!(
                    "Gradient output shape {:?} doesn't match input shape {:?}",
                    grad_output.shape(),
                    input.shape()
                )));
            }

            // Check for NaN or infinite values in gradient output
            if grad_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Err(ModelError::InputValidationError(
                    "Gradient output contains NaN or infinite values".to_string(),
                ));
            }

            // Linear derivative is 1, so gradient passes through unchanged
            Ok(grad_output.clone())
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Linear"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.input_cache)
    }

    no_trainable_parameters_layer_functions!();
}

impl ActivationLayer for Linear {}
