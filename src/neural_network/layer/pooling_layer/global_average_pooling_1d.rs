use super::*;

/// Threshold for determining when to use parallel vs sequential execution.
/// When batch_size * channels >= this threshold, parallel execution is used.
/// Otherwise, sequential execution is used to avoid parallel overhead.
const GLOBAL_AVERAGE_POOLING_1D_PARALLEL_THRESHOLD: usize = 32;

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
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 3D".to_string(),
            ));
        }

        // Extract dimensions
        let shape = input.shape();
        let (batch_size, channels, length) = (shape[0], shape[1], shape[2]);

        // Store input shape and cache input for backpropagation
        self.input_shape = vec![batch_size, channels, length];
        self.input_cache = Some(input.clone());

        // Create output tensor
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Compute global average pooling
        let length_f32 = length as f32;

        // Helper closure to compute pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut sum = 0.0;
            for l in 0..length {
                sum += input[[b, c, l]];
            }
            ((b, c), sum / length_f32)
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_AVERAGE_POOLING_1D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Merge results into the output tensor
        for ((b, c), val) in results {
            output[[b, c]] = val;
        }

        Ok(output)
    }
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Check if we have cached the input
        if self.input_cache.is_none() {
            return Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ));
        }

        // Extract dimensions
        let (batch_size, channels, length) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        );

        // Distribute gradients uniformly across the sequence dimension
        // Each position contributes equally to the average, so gradient is divided by length
        let length_f32 = length as f32;

        // Helper closure to compute gradient for a single (batch, channel) pair
        let compute_gradient = |b: usize, c: usize| {
            let grad_val = grad_output[[b, c]] / length_f32;
            let seq_grad = vec![grad_val; length];
            ((b, c), seq_grad)
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_AVERAGE_POOLING_1D_PARALLEL_THRESHOLD,
            compute_gradient
        );

        // Create gradient tensor with the same shape as the input tensor
        let mut grad_input = Tensor::zeros(IxDyn(&self.input_shape));

        // Merge gradients from all batches and channels
        merge_gradients_1d!(grad_input, results, length);

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling1D"
    }

    layer_functions_global_pooling!();
}
