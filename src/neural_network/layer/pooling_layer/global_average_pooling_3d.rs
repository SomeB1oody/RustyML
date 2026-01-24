use super::*;

/// Threshold for determining when to use parallel vs sequential execution.
/// When batch_size * channels >= this threshold, parallel execution is used.
/// Otherwise, sequential execution is used to avoid parallel overhead.
const GLOBAL_AVERAGE_POOLING_3D_PARALLEL_THRESHOLD: usize = 32;

/// Global average pooling layer for 3D inputs.
///
/// Computes the mean value across the depth, height, and width dimensions.
/// Input tensor shape: `[batch_size, channels, depth, height, width]`. Output tensor shape:
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
/// // Add a GlobalAveragePooling3D layer
/// model.add(GlobalAveragePooling3D::new());
///
/// // Create test input tensor: [batch_size, channels, depth, height, width]
/// let input_data = Array::from_elem(IxDyn(&[2, 4, 8, 8, 8]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Check output shape - should be [2, 4]
/// assert_eq!(output.shape(), &[2, 4]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..2 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= GLOBAL_AVERAGE_POOLING_3D_PARALLEL_THRESHOLD` (32).
pub struct GlobalAveragePooling3D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl GlobalAveragePooling3D {
    /// Creates a new global average pooling 3D layer.
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling3D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling3D {
            input_shape: Vec::new(),
            input_cache: None,
        }
    }
}

impl Layer for GlobalAveragePooling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 5D".to_string(),
            ));
        }

        // Extract dimensions
        let shape = input.shape();
        let (batch_size, channels, depth, height, width) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);

        // Store input shape and cache input for backpropagation
        self.input_shape = vec![batch_size, channels, depth, height, width];
        self.input_cache = Some(input.clone());

        // Create output tensor
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Compute global average pooling
        let spatial_size = (depth * height * width) as f32;

        // Helper closure to compute pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut sum = 0.0;
            for d in 0..depth {
                for h in 0..height {
                    for w in 0..width {
                        sum += input[[b, c, d, h, w]];
                    }
                }
            }
            ((b, c), sum / spatial_size)
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_AVERAGE_POOLING_3D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Merge results into the output tensor
        for ((b, c), val) in results {
            output[[b, c]] = val;
        }

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let depth = input_shape[2];
            let height = input_shape[3];
            let width = input_shape[4];

            // Calculate the total number of spatial elements
            let spatial_size = depth * height * width;
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
                GLOBAL_AVERAGE_POOLING_3D_PARALLEL_THRESHOLD,
                compute_gradient
            );

            // Create gradient tensor with the same shape as input
            let mut grad_input = Array::zeros(IxDyn(&input_shape));

            // Merge gradients from all batches and channels
            merge_gradients_3d!(grad_input, results, depth, height, width);

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling3D"
    }

    layer_functions_global_pooling!();
}
