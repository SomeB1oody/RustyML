use super::super::Tensor;
use super::calculate_output_shape_1d_pooling;
use crate::ModelError;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::Array3;
use rayon::prelude::*;

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
/// - `max_positions` - Positions of the maximum values in each pooling window, used for backpropagation
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
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
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
    max_positions: Option<Vec<(usize, usize, usize)>>,
}

impl MaxPooling1D {
    /// Create a new 1D Max Pooling layer
    ///
    /// # Parameters
    ///
    /// * `pool_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling operation
    /// * `input_shape` - Shape of the input tensor \[batch_size, channels, length\]
    pub fn new(pool_size: usize, stride: usize, input_shape: Vec<usize>) -> Self {
        MaxPooling1D {
            pool_size,
            stride,
            input_shape,
            input_cache: None,
            max_positions: None,
        }
    }

    /// Calculate output shape
    ///
    /// Computes the output shape based on input shape, pool size, and stride
    fn compute_output_shape_max_pooling_1d(&self) -> Vec<usize> {
        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let length = self.input_shape[2];

        calculate_output_shape_1d_pooling(batch_size, channels, length, self.pool_size, self.stride)
    }
}

impl Layer for MaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Cache input for backward pass
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let output_length = (length - self.pool_size) / self.stride + 1;
        let mut output = Array3::<f32>::zeros((batch_size, channels, output_length)).into_dyn();

        // Store the positions of maximum values for backpropagation
        let mut max_positions = Vec::new();

        // Copy needed values from self to avoid capturing self in closure
        let pool_size = self.pool_size;
        let stride = self.stride;

        // Use rayon to process batches and channels in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();
                    let mut batch_channel_max_positions = Vec::new();

                    // Perform pooling for each output position
                    for i in 0..output_length {
                        let start_idx = i * stride;
                        let end_idx = start_idx + pool_size;

                        // Find maximum value and its position in the window
                        let mut max_val = f32::MIN;
                        let mut max_idx = start_idx;
                        for j in start_idx..end_idx {
                            if input[[b, c, j]] > max_val {
                                max_val = input[[b, c, j]];
                                max_idx = j;
                            }
                        }

                        batch_channel_output.push((i, max_val));
                        batch_channel_max_positions.push((b, c, max_idx));
                    }

                    ((b, c), (batch_channel_output, batch_channel_max_positions))
                })
            })
            .collect();

        // Merge results into output tensor and collect max positions
        for ((b, c), (outputs, positions)) in results {
            for (i, val) in outputs {
                output[[b, c, i]] = val;
            }
            max_positions.extend(positions);
        }

        self.max_positions = Some(max_positions);
        output
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

        let mut grad_input = Array3::<f32>::zeros((batch_size, channels, length)).into_dyn();

        // For max pooling, gradients flow only through the maximum value in each pooling window
        for (b, c, idx) in max_positions {
            // Calculate which output position this max value contributed to
            let output_idx = (*idx - (*idx % self.stride)) / self.stride;

            // Ensure we're within bounds of grad_output
            if output_idx < grad_output.shape()[2] {
                // Pass the gradient to the input position that had the maximum value
                grad_input[[*b, *c, *idx]] += grad_output[[*b, *c, output_idx]];
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "MaxPooling1D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.compute_output_shape_max_pooling_1d();
        format!("{:?}", output_shape)
    }

    no_trainable_parameters_layer_functions!();
}
