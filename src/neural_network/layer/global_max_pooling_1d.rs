use super::*;

/// Threshold for deciding between parallel and sequential execution.
/// When batch_size * channels >= this threshold, use parallel execution.
const GLOBAL_MAX_POOLING_1D_PARALLEL_THRESHOLD: usize = 32;

/// Global Max Pooling 1D Layer
///
/// Performs global max pooling operation on the input tensor across the sequence dimension.
/// The input tensor shape should be `[batch_size, channels, length]`,
/// and the output tensor shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// - `input_shape` - Stores the shape of the input tensor during the forward pass.
/// - `max_positions` - Stores the positions of maximum values found during the forward pass, used for gradient propagation during backpropagation.
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
/// // Add a GlobalMaxPooling1D layer
/// model.add(GlobalMaxPooling1D::new());
///
/// // Create a test input tensor: [batch_size, channels, length]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 8]), 1.0);
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
pub struct GlobalMaxPooling1D {
    input_shape: Vec<usize>,
    max_positions: Option<Vec<usize>>,
}

impl GlobalMaxPooling1D {
    /// Creates a new GlobalMaxPooling1D layer.
    ///
    /// # Returns
    ///
    /// * `GlobalMaxPooling1D` - A new `GlobalMaxPooling1D` layer instance
    pub fn new() -> Self {
        GlobalMaxPooling1D {
            input_shape: Vec::new(),
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Store the input shape for backpropagation
        self.input_shape = input.shape().to_vec();

        // Verify input is 3D: [batch_size, channels, length]
        assert_eq!(
            self.input_shape.len(),
            3,
            "Input shape must be 3-dimensional: [batch_size, channels, length]"
        );

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let length = self.input_shape[2];

        // Initialize output tensor with shape [batch_size, channels]
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Pre-allocate max_positions vector with correct capacity
        let mut max_positions = vec![0; batch_size * channels];

        // Helper closure to compute global max pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_pos = 0;

            // Find maximum value and its position along the length dimension
            for l in 0..length {
                let val = input[[b, c, l]];
                if val > max_val {
                    max_val = val;
                    max_pos = l;
                }
            }

            ((b, c), (max_val, max_pos))
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_MAX_POOLING_1D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Fill the output tensor and max_positions
        for ((b, c), (max_val, max_pos)) in results {
            output[[b, c]] = max_val;
            let idx = b * channels + c;
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
            let length = self.input_shape[2];

            // Initialize input gradient tensor
            let mut grad_input = Tensor::zeros(IxDyn(&self.input_shape));

            // Helper closure to compute gradient for a single (batch, channel) pair
            let compute_gradient = |b: usize, c: usize| {
                let idx = b * channels + c;
                let max_pos = max_positions[idx];
                let gradient_value = grad_output[[b, c]];

                // Create a gradient vector for this channel (only one position gets the gradient)
                let mut seq_grad = vec![0.0; length];
                seq_grad[max_pos] = gradient_value;

                ((b, c), seq_grad)
            };

            // Choose parallel or sequential execution based on workload size
            let results: Vec<_> = execute_parallel_or_sequential!(
                batch_size,
                channels,
                GLOBAL_MAX_POOLING_1D_PARALLEL_THRESHOLD,
                compute_gradient
            );

            // Apply updates to gradient tensor
            for ((b, c), seq_grad) in results {
                for (l, grad_val) in seq_grad.iter().enumerate() {
                    grad_input[[b, c, l]] = *grad_val;
                }
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling1D"
    }

    layer_functions_global_pooling!();
}
