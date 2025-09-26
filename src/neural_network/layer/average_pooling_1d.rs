use super::*;

/// 1D Average Pooling layer for neural networks.
///
/// This layer performs average pooling operation on a 3D tensor.
/// Average pooling computes the mean value of each patch as defined by the pool size.
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
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array3;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, channels, length]
/// // Batch size=2, 3 input channels, each channel has 8 elements
/// let mut input_data = Array3::zeros((2, 3, 8));
///
/// // Set test data to make average pooling results predictable
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..8 {
///             input_data[[b, c, i]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test AveragePooling1D with a Sequential model
/// let mut model = Sequential::new();
/// model
///     .add(AveragePooling1D::new(
///         2,              // Pool window size
///         2,              // Stride
///         vec![2, 3, 8],  // Input shape
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Output shape should be [2, 3, 4]
/// let output = model.predict(&x);
/// assert_eq!(output.shape(), &[2, 3, 4]);
///
/// // Verify correctness of pooling results
/// // For window size of 2 and stride of 2, we expect the average of elements in each window
/// for b in 0..2 {
///     for c in 0..3 {
///         // First window (0,1) -> average should be (0+1)/2 = 0.5
///         assert_relative_eq!(output[[b, c, 0]], 0.5);
///         // Second window (2,3) -> average should be (2+3)/2 = 2.5
///         assert_relative_eq!(output[[b, c, 1]], 2.5);
///         // Third window (4,5) -> average should be (4+5)/2 = 4.5
///         assert_relative_eq!(output[[b, c, 2]], 4.5);
///         // Fourth window (6,7) -> average should be (6+7)/2 = 6.5
///         assert_relative_eq!(output[[b, c, 3]], 6.5);
///     }
/// }
/// ```
pub struct AveragePooling1D {
    pool_size: usize,
    stride: usize,
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl AveragePooling1D {
    /// Create a new 1D Average Pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    /// - `stride` - Stride of the pooling operation
    /// - `input_shape` - Shape of the input tensor \[batch_size, channels, length\]
    ///
    /// # Returns
    ///
    /// * `AveragePooling1D` - A new `AveragePooling1D` layer instance
    pub fn new(pool_size: usize, stride: usize, input_shape: Vec<usize>) -> Self {
        // verify input is 3D: [batch_size, channels, length]
        assert_eq!(
            input_shape.len(),
            3,
            "Input shape must be 3-dimensional: [batch_size, channels, length]"
        );

        AveragePooling1D {
            pool_size,
            stride,
            input_shape,
            input_cache: None,
        }
    }
}

impl Layer for AveragePooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Cache input for backward pass
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let output_length = (length - self.pool_size) / self.stride + 1;
        let mut output = Array3::<f32>::zeros((batch_size, channels, output_length)).into_dyn();

        // Copy needed values from self to avoid capturing self in closure
        let pool_size = self.pool_size;
        let stride = self.stride;

        // Use rayon to process batches and channels in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();

                    // Perform pooling for each output position
                    for i in 0..output_length {
                        let start_idx = i * stride;
                        let end_idx = start_idx + pool_size;

                        // Calculate average of elements in the window
                        let mut sum = 0.0;
                        for j in start_idx..end_idx {
                            sum += input[[b, c, j]];
                        }
                        batch_channel_output.push((i, sum / (pool_size as f32)));
                    }

                    ((b, c), batch_channel_output)
                })
            })
            .collect();

        // Merge results into output tensor
        for ((b, c), outputs) in results {
            for (i, val) in outputs {
                output[[b, c, i]] = val;
            }
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Ensure we have cached input
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "No cached input for AveragePooling1D".to_string(),
                ));
            }
        };

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let mut grad_input = Array3::<f32>::zeros((batch_size, channels, length)).into_dyn();

        // Calculate gradient with respect to input
        let scale_factor = 1.0 / (self.pool_size as f32);

        // Copy member variables needed in closure
        let pool_size = self.pool_size;
        let stride = self.stride;

        // Use rayon to process batches and channels in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_grad =
                        Array3::<f32>::zeros((batch_size, channels, length)).into_dyn();

                    for i in 0..grad_output.shape()[2] {
                        let start_idx = i * stride;
                        let end_idx = start_idx + pool_size;

                        // Distribute gradient evenly to each element in the input window
                        for j in start_idx..end_idx {
                            if j < length {
                                batch_channel_grad[[b, c, j]] +=
                                    grad_output[[b, c, i]] * scale_factor;
                            }
                        }
                    }

                    ((b, c), batch_channel_grad)
                })
            })
            .collect();

        // Merge gradients from all batches and channels
        for ((b, c), grad) in results {
            for j in 0..length {
                grad_input[[b, c, j]] += grad[[b, c, j]];
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "AveragePooling1D"
    }

    layer_functions_1d_pooling!();
}
