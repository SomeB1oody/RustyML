use super::*;

/// Flattens a 3D, 4D, or 5D tensor into a 2D tensor.
///
/// This layer reshapes inputs from feature extraction layers into a format suitable for dense layers.
/// Input shapes are [batch_size, features, length], [batch_size, channels, height, width], or
/// [batch_size, channels, depth, height, width]. Output shape is always [batch_size, flattened_features],
/// where flattened_features is the product of all dimensions except batch_size.
///
/// # Fields
///
/// - `flattened_features` - Number of features after flattening (product of all dimensions except batch)
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
///
/// // Create a 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 channels, each 4x4 pixels
/// let x = Array4::ones((2, 3, 4, 4)).into_dyn();
///
/// // Build a model containing a Flatten layer
/// let mut model = Sequential::new();
/// model
///     .add(Flatten::new(vec![2, 3, 4, 4]).unwrap())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let flattened = model.predict(&x).unwrap();
///
/// // Check output shape - should be [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
pub struct Flatten {
    flattened_features: usize,
    input_cache: Option<Tensor>,
}

impl Flatten {
    /// Creates a new Flatten layer.
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Input tensor shape, such as [batch_size, features, length],
    ///   [batch_size, channels, height, width], or [batch_size, channels, depth, height, width]
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New `Flatten` layer instance
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_shape` has fewer than 2 dimensions or contains a zero
    pub fn new(input_shape: Vec<usize>) -> Result<Self, ModelError> {
        // Validate input shape dimensions
        if input_shape.len() < 2 {
            return Err(ModelError::InputValidationError(format!(
                "Input shape must have at least 2 dimensions [batch_size, features...], got {}D",
                input_shape.len()
            )));
        }

        // Ensure all dimensions are greater than 0
        for (i, &dim) in input_shape.iter().enumerate() {
            if dim == 0 {
                return Err(ModelError::InputValidationError(format!(
                    "Dimension {} must be greater than 0, got {}",
                    i, dim
                )));
            }
        }

        let flattened_features = input_shape[1..].iter().product();

        Ok(Flatten {
            flattened_features,
            input_cache: None,
        })
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input dimensions
        let input_shape = input.shape();
        if input_shape.len() < 3 || input_shape.len() > 5 {
            return Err(ModelError::InputValidationError(format!(
                "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
                input_shape.len()
            )));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        // Reshape to flatten the tensor
        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape().to_vec();

            // Validate gradient output shape
            let expected_grad_shape = [input_shape[0], input_shape[1..].iter().product()];
            if grad_output.shape() != expected_grad_shape {
                return Err(ModelError::ProcessingError(format!(
                    "Gradient output shape {:?} doesn't match expected shape {:?}",
                    grad_output.shape(),
                    expected_grad_shape
                )));
            }

            // Reshape gradient back to input shape
            let reshaped_grad = grad_output
                .to_shape(IxDyn(&input_shape))
                .map_err(|e| {
                    ModelError::ProcessingError(format!("Failed to reshape gradient: {}", e))
                })?
                .to_owned();

            Ok(reshaped_grad)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Flatten"
    }

    fn output_shape(&self) -> String {
        format!("(batch_size, {})", self.flattened_features)
    }

    no_trainable_parameters_layer_functions!();
}
