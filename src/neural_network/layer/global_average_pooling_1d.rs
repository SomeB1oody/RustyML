use super::*;

/// Global Average Pooling 1D Layer
///
/// Performs global average pooling operation on the sequence dimension of the input tensor.
/// Input tensor shape should be `[batch_size, channels, length]`,
/// output tensor shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// - `input_shape` - Stores the shape of the input tensor during forward propagation.
/// - `input_cache` - Caches the input tensor for backward propagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model with multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalAveragePooling1D layer
/// model.add(GlobalAveragePooling1D::new());
///
/// // Create test input tensor: [batch_size, channels, length]
/// let input_data = Array::from_elem(IxDyn(&[2, 3, 4]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data);
///
/// // Check output shape - should be [2, 3]
/// assert_eq!(output.shape(), &[2, 3]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..2 {
///     for c in 0..3 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
pub struct GlobalAveragePooling1D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl GlobalAveragePooling1D {
    /// Creates a new `GlobalAveragePooling1D` layer.
    ///
    /// # Returns
    ///
    /// * `GlobalAveragePooling1D` - A new instance of `GlobalAveragePooling1D`
    pub fn new() -> Self {
        GlobalAveragePooling1D {
            input_shape: Vec::new(),
            input_cache: None,
        }
    }
}

impl Layer for GlobalAveragePooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Verify input is 3D: [batch_size, channels, length]
        let shape = input.shape();
        assert_eq!(
            shape.len(),
            3,
            "Input shape must be 3-dimensional: [batch_size, channels, length]"
        );

        // Extract dimensions
        let (batch_size, channels, length) = (shape[0], shape[1], shape[2]);

        // Store input shape and cache input for backpropagation
        self.input_shape = vec![batch_size, channels, length];
        self.input_cache = Some(input.clone());

        // Create output tensor
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Compute global average pooling
        // Use parallel iteration over batches and channels, compute sequential sum over length
        let length_f32 = length as f32;

        output
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut output_batch)| {
                for c in 0..channels {
                    let mut sum = 0.0;
                    for l in 0..length {
                        sum += input[[b, c, l]];
                    }
                    output_batch[c] = sum / length_f32;
                }
            });

        output
    }
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Check if we have cached the input
        if self.input_cache.is_none() {
            return Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ));
        }

        // Extract dimensions
        let (channels, length) = (self.input_shape[1], self.input_shape[2]);

        // Create gradient tensor with the same shape as the input tensor
        let mut grad_input = Tensor::zeros(IxDyn(&self.input_shape));

        // Distribute gradients uniformly across the sequence dimension
        // Each position contributes equally to the average, so gradient is divided by length
        let length_f32 = length as f32;

        grad_input
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut grad_batch)| {
                for c in 0..channels {
                    let grad_val = grad_output[[b, c]] / length_f32;
                    for l in 0..length {
                        grad_batch[[c, l]] = grad_val;
                    }
                }
            });

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling1D"
    }

    layer_functions_global_pooling!();
}
