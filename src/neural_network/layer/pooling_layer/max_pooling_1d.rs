use super::*;

/// Threshold for deciding between parallel and sequential execution.
/// When batch_size * channels >= this threshold, use parallel execution.
const MAX_POOLING_1D_PARALLEL_THRESHOLD: usize = 32;

/// 1D Max Pooling layer for neural networks.
///
/// This layer performs max pooling operation on a 3D tensor.
/// Max pooling selects the maximum value from each patch as defined by the pool size.
///
/// # Input Shape
///
/// Input is a 3D tensor with shape \[batch_size, channels, length\]
///
/// # Output Shape
///
/// Output is a 3D tensor with shape \[batch_size, channels, pooled_length\]
/// Where:
/// - pooled_length = (length - pool_size) / stride + 1
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window
/// - `stride` - Stride of the pooling operation
/// - `input_shape` - Shape of the input tensor
/// - `input_cache` - Input tensor cached during forward pass, used for backward pass
/// - `max_positions` - 3D array storing the input indices of maximum values for each output position, used for backpropagation
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array3;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, channels, length]
/// // batch_size=2, 3 input channels, 8 elements per channel
/// let mut input_data = Array3::zeros((2, 3, 8));
///
/// // Set test data to make max pooling results predictable
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..8 {
///              input_data[[b, c, i]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test MaxPooling1D using Sequential model
/// let mut model = Sequential::new();
/// model
///     .add(MaxPooling1D::new(
///         2,              // pool window size
///         2,              // stride
///         vec![2, 3, 8],  // input shape
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Output shape should be [2, 3, 4]
/// let output = model.predict(&x);
/// assert_eq!(output.shape(), &[2, 3, 4]);
///
/// // Verify correctness of pooling results
/// // For window size 2 and stride 2, we expect the maximum element in each window
/// for b in 0..2 {
///     for c in 0..3 {
///         // First window (0,1) -> max value should be 1.0
///        assert_relative_eq!(output[[b, c, 0]], 1.0);
///         // Second window (2,3) -> max value should be 3.0
///         assert_relative_eq!(output[[b, c, 1]], 3.0);
///         // Third window (4,5) -> max value should be 5.0
///         assert_relative_eq!(output[[b, c, 2]], 5.0);
///         // Fourth window (6,7) -> max value should be 7.0
///         assert_relative_eq!(output[[b, c, 3]], 7.0);
///     }
/// }
/// ```
pub struct MaxPooling1D {
    pool_size: usize,
    stride: usize,
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Array3<usize>>,
}

impl MaxPooling1D {
    /// Create a new 1D Max Pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    /// - `stride` - Stride of the pooling operation
    /// - `input_shape` - Shape of the input tensor \[batch_size, channels, length\]
    ///
    /// # Returns
    ///
    /// * `Result<MaxPooling1D, ModelError>` - a new instance of `MaxPooling1D` or an error
    pub fn new(
        pool_size: usize,
        stride: usize,
        input_shape: Vec<usize>,
    ) -> Result<Self, ModelError> {
        // input validation
        validate_input_shape_dims(&input_shape, 3, "MaxPooling1D")?;
        validate_pool_size_1d(pool_size, input_shape[2])?;
        validate_stride_1d(stride)?;

        Ok(MaxPooling1D {
            pool_size,
            stride,
            input_shape,
            input_cache: None,
            max_positions: None,
        })
    }
}

impl Layer for MaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 3D".to_string(),
            ));
        }

        // Cache input for backward pass
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let output_length = (length - self.pool_size) / self.stride + 1;
        let mut output = Array3::<f32>::zeros((batch_size, channels, output_length));
        let mut max_positions = Array3::<usize>::zeros((batch_size, channels, output_length));

        // Copy needed values from self to avoid capturing self in closure
        let pool_size = self.pool_size;
        let stride = self.stride;

        // Helper closure to compute max pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut channel_output = Vec::new();
            let mut channel_max_pos = Vec::new();

            for i in 0..output_length {
                let start_idx = i * stride;
                let end_idx = start_idx + pool_size;

                // Find maximum value and its position in the window
                let mut max_val = input[[b, c, start_idx]];
                let mut max_idx = start_idx;
                for j in (start_idx + 1)..end_idx {
                    if input[[b, c, j]] > max_val {
                        max_val = input[[b, c, j]];
                        max_idx = j;
                    }
                }

                channel_output.push((i, max_val));
                channel_max_pos.push((i, max_idx));
            }

            ((b, c), (channel_output, channel_max_pos))
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            MAX_POOLING_1D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Write results back to output arrays
        for ((b, c), (channel_output, channel_max_pos)) in results {
            for (i, val) in channel_output {
                output[[b, c, i]] = val;
            }
            for (i, pos) in channel_max_pos {
                max_positions[[b, c, i]] = pos;
            }
        }

        self.max_positions = Some(max_positions);
        Ok(output.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Ensure we have cached input
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "No cached input for MaxPooling1D".to_string(),
                ));
            }
        };

        // Ensure we have stored max positions
        let max_positions = match &self.max_positions {
            Some(positions) => positions,
            None => {
                return Err(ModelError::ProcessingError(
                    "No cached max positions for MaxPooling1D".to_string(),
                ));
            }
        };

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];
        let output_length = grad_output.shape()[2];

        let mut grad_input = Array3::<f32>::zeros((batch_size, channels, length));

        // Helper closure to compute gradient for a single (batch, channel) pair
        let compute_gradient = |b: usize, c: usize| {
            let mut channel_grad = vec![0.0; length];

            // For max pooling, gradients flow only through the maximum value in each pooling window
            for i in 0..output_length {
                let max_idx = max_positions[[b, c, i]];
                channel_grad[max_idx] += grad_output[[b, c, i]];
            }

            ((b, c), channel_grad)
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            MAX_POOLING_1D_PARALLEL_THRESHOLD,
            compute_gradient
        );

        // Write results back to gradient array
        merge_gradients_1d!(grad_input, results, length);

        Ok(grad_input.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "MaxPooling1D"
    }

    layer_functions_1d_pooling!();
}
