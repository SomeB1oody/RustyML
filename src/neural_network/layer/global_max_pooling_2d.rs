use super::*;

/// Global Max Pooling Layer
///
/// Performs global max pooling operation on the input tensor across spatial dimensions (height and width).
/// The input tensor shape should be `[batch_size, channels, height, width]`,
/// and the output tensor shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// * `input_shape` - Stores the shape of the input tensor during the forward pass.
/// * `max_positions` - Stores the positions of maximum values found during the forward pass, used for gradient propagation during backpropagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model containing multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalMaxPooling2D layer
/// model.add(GlobalMaxPooling2D::new());
///
/// // Create a test input tensor: [batch_size, channels, height, width]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 5, 5]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data);
///
/// // Check output shape - should be [3, 4]
/// assert_eq!(output.shape(), &[3, 4]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..3 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
pub struct GlobalMaxPooling2D {
    input_shape: Vec<usize>,
    max_positions: Option<Vec<(usize, usize)>>,
}

impl GlobalMaxPooling2D {
    /// Creates a new instance of the global max pooling layer
    ///
    /// # Returns
    ///
    /// * `GlobalMaxPooling2D` - A new `GlobalMaxPooling2D` instance
    pub fn new() -> Self {
        GlobalMaxPooling2D {
            input_shape: Vec::new(),
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Store the input shape for backpropagation
        self.input_shape = input.shape().to_vec();

        // Verify input is 4D: [batch_size, channels, height, width]
        assert_eq!(
            self.input_shape.len(),
            4,
            "Input shape must be 4-dimensional: [batch_size, channels, height, width]"
        );

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let height = self.input_shape[2];
        let width = self.input_shape[3];

        // Initialize output tensor with shape [batch_size, channels]
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Pre-allocate max_positions vector with correct capacity
        let mut max_positions = vec![(0, 0); batch_size * channels];

        // Process each batch-channel pair in parallel
        let results: Vec<(usize, f32, (usize, usize))> = (0..batch_size * channels)
            .into_par_iter()
            .map(|idx| {
                let b = idx / channels;
                let c = idx % channels;

                let mut max_val = f32::NEG_INFINITY;
                let mut max_pos = (0, 0);

                // Find maximum value and its position across spatial dimensions
                for h in 0..height {
                    for w in 0..width {
                        let val = input[[b, c, h, w]];
                        if val > max_val {
                            max_val = val;
                            max_pos = (h, w);
                        }
                    }
                }

                (idx, max_val, max_pos)
            })
            .collect();

        // Fill the output tensor and max_positions in the correct order
        for (idx, max_val, max_pos) in results {
            let b = idx / channels;
            let c = idx % channels;
            output[[b, c]] = max_val;
            max_positions[idx] = max_pos;
        }

        // Cache the positions of maximum values for backpropagation
        self.max_positions = Some(max_positions);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(max_positions) = &self.max_positions {
            let batch_size = self.input_shape[0];
            let channels = self.input_shape[1];

            // Initialize input gradient tensor
            let mut grad_input = Tensor::zeros(IxDyn(&self.input_shape));

            // Collect positions and values to update
            let updates: Vec<(usize, usize, usize, usize, f32)> = (0..batch_size * channels)
                .into_par_iter()
                .map(|idx| {
                    let b = idx / channels;
                    let c = idx % channels;
                    let (max_h, max_w) = max_positions[idx];
                    let gradient_value = grad_output[[b, c]];

                    (b, c, max_h, max_w, gradient_value)
                })
                .collect();

            // Apply updates in the main thread
            for (b, c, max_h, max_w, gradient_value) in updates {
                grad_input[[b, c, max_h, max_w]] = gradient_value;
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling2D"
    }

    layer_functions_global_pooling!();
}
