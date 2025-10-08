use super::*;

/// A layer that flattens a multi-dimensional tensor (3D, 4D, or 5D) into a 2D tensor.
///
/// This layer is typically used in neural networks to transform the output of feature extraction layers
/// (such as convolutional or pooling layers) into a format that can be processed by dense (fully connected) layers.
///
/// # Input Shape
///
/// - 3D tensor: \[batch_size, features, length\]
/// - 4D tensor: \[batch_size, channels, height, width\]
/// - 5D tensor: \[batch_size, channels, depth, height, width\]
///
/// # Output Shape
///
/// Output is always a 2D tensor with shape \[batch_size, flattened_features\] where:
/// - For 3D: flattened_features = features * length
/// - For 4D: flattened_features = channels * height * width
/// - For 5D: flattened_features = channels * depth * height * width
///
/// # Fields
///
/// - `output_shape` - the output shape of the layer, in the format \[batch_size, flattened_features\]
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
///
/// # Example
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
///     .add(Flatten::new(vec![2, 3, 4, 4]))
///     .compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let flattened = model.predict(&x);
///
/// // Check output shape - should be [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
pub struct Flatten {
    output_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl Flatten {
    /// Creates a new Flatten layer.
    ///
    /// # Parameters
    ///
    /// * `input_shape` - The shape of the input tensor. Supported formats:
    ///   - 3D: \[batch_size, features, length\]
    ///   - 4D: \[batch_size, channels, height, width\]
    ///   - 5D: \[batch_size, channels, depth, height, width\]
    ///
    /// # Returns
    ///
    /// * `Flatten` - A new `Flatten` layer instance
    pub fn new(input_shape: Vec<usize>) -> Self {
        assert!(
            input_shape.len() >= 3 && input_shape.len() <= 5,
            "Input shape must be 3D, 4D, or 5D, got {}D",
            input_shape.len()
        );

        let batch_size = input_shape[0];
        let flattened_features = input_shape[1..].iter().product();
        let output_shape = vec![batch_size, flattened_features];

        Flatten {
            output_shape,
            input_cache: None,
        }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Validate input dimensions
        let input_shape = input.shape();
        assert!(
            input_shape.len() >= 3 && input_shape.len() <= 5,
            "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
            input_shape.len()
        );

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        // Create new shape and flatten
        let output = input.clone();
        output
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned()
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
        format!("({}, {})", self.output_shape[0], self.output_shape[1])
    }

    no_trainable_parameters_layer_functions!();
}
