use super::*;

/// Threshold for determining when to use parallel vs sequential execution.
/// When batch_size * channels >= this threshold, parallel execution is used.
/// Otherwise, sequential execution is used to avoid parallel overhead.
const GLOBAL_AVERAGE_POOLING_2D_PARALLEL_THRESHOLD: usize = 32;

/// Global average pooling layer for 2D inputs.
///
/// Computes the mean value across the height and width dimensions.
/// Input tensor shape: `[batch_size, channels, height, width]`. Output tensor shape:
/// `[batch_size, channels]`.
///
/// # Fields
///
/// - `input_shape` - Shape of the input tensor cached during the forward pass
/// - `input_cache` - Cached input tensor from the forward pass
///
/// # Examples
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model with multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalAveragePooling2D layer
/// model.add(GlobalAveragePooling2D::new());
///
/// // Create test input tensor: [batch_size, channels, height, width]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 5, 5]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
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
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= GLOBAL_AVERAGE_POOLING_2D_PARALLEL_THRESHOLD` (32).
pub struct GlobalAveragePooling2D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl GlobalAveragePooling2D {
    /// Creates a new global average pooling 2D layer.
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling2D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling2D {
            input_shape: Vec::new(),
            input_cache: None,
        }
    }
}

impl Layer for GlobalAveragePooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 4D".to_string(),
            ));
        }

        // Extract dimensions
        let shape = input.shape();
        let (batch_size, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

        // Store input shape and cache input for backpropagation
        self.input_shape = vec![batch_size, channels, height, width];
        self.input_cache = Some(input.clone());

        // Create output tensor
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Compute global average pooling
        let spatial_size = (height * width) as f32;

        // Helper closure to compute pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut sum = 0.0;
            for h in 0..height {
                for w in 0..width {
                    sum += input[[b, c, h, w]];
                }
            }
            ((b, c), sum / spatial_size)
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_AVERAGE_POOLING_2D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Merge results into the output tensor
        for ((b, c), val) in results {
            output[[b, c]] = val;
        }

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Check if there is a valid input cache
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run yet".to_string(),
                ));
            }
        };

        // Get input dimensions
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let spatial_size = height * width;
        let scale_factor = 1.0 / (spatial_size as f32);

        // Helper closure to compute gradient for a single (batch, channel) pair
        let compute_gradient = |b: usize, c: usize| {
            let grad_val = grad_output[[b, c]] * scale_factor;
            let spatial_grad = vec![grad_val; spatial_size];
            ((b, c), spatial_grad)
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_AVERAGE_POOLING_2D_PARALLEL_THRESHOLD,
            compute_gradient
        );

        // Create and fill gradient tensor
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, channels, height, width]));

        // Merge gradients from all batches and channels
        merge_gradients_2d!(grad_input, results, height, width);

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling2D"
    }

    layer_functions_global_pooling!();
}
