use crate::neural_network::layer::LayerWeight;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::ArrayD;
use rayon::prelude::*;

/// A 2D average pooling layer for neural networks.
///
/// This layer performs average pooling operations on 4D tensors.
/// Average pooling computes the average value of each patch defined by the pool size.
///
/// # Input Shape
///
/// Input is a 4D tensor with shape \[batch_size, channels, height, width\]
///
/// # Output Shape
///
/// Output is a 4D tensor with shape \[batch_size, channels, pooled_height, pooled_width\]
/// where:
/// - pooled_height = (height - pool_size_h) / stride_h + 1
/// - pooled_width = (width - pool_size_w) / stride_w + 1
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window as (height, width)
/// - `strides` - Stride of the pooling operation as (height, width)
/// - `input_shape` - Shape of the input tensor
/// - `input_cache` - Cached input tensor from forward pass, used in backpropagation
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 input channels, each channel is 4x4 pixels
/// let mut input_data = Array4::zeros((2, 3, 4, 4));
///
///  // Set test data to make average pooling results predictable
///  for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..4 {
///             for j in 0..4 {
///                 input_data[[b, c, i, j]] = (i + j) as f32;
///             }
///         }
///     }
///  }
///
///  let x = input_data.clone().into_dyn();
///
///  // Test AveragePooling with Sequential model
///  let mut model = Sequential::new();
///  model
///  .add(AveragePooling::new(
///  (2, 2),           // Pooling window size
///  (2, 2),           // Stride
///  vec![2, 3, 4, 4], // Input shape
///  ))
///  .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
///  // Output shape should be [2, 3, 2, 2]
///  let output = model.predict(&x);
///  assert_eq!(output.shape(), &[2, 3, 2, 2]);
///
///  // Verify correctness of pooling results
///  // For a 2x2 window with stride 2, we expect the result to be the average of the elements in the window
///  for b in 0..2 {
///     for c in 0..3 {
///         // First window (0,0), (0,1), (1,0), (1,1) -> average should be (0+1+1+2)/4 = 1.0
///         assert_relative_eq!(output[[b, c, 0, 0]], 1.0);
///         // Second window (0,2), (0,3), (1,2), (1,3) -> average should be (2+3+3+4)/4 = 3.0
///         assert_relative_eq!(output[[b, c, 0, 1]], 3.0);
///         // Third window (2,0), (2,1), (3,0), (3,1) -> average should be (2+3+3+4)/4 = 3.0
///         assert_relative_eq!(output[[b, c, 1, 0]], 3.0);
///         // Fourth window (2,2), (2,3), (3,2), (3,3) -> average should be (4+5+5+6)/4 = 5.0
///         assert_relative_eq!(output[[b, c, 1, 1]], 5.0);
///     }
///  }
/// ```
pub struct AveragePooling {
    pool_size: (usize, usize),
    strides: (usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl AveragePooling {
    /// Creates a new AveragePooling layer.
    ///
    /// # Parameters
    ///
    /// * `pool_size` - Size of the pooling window as (height, width)
    /// * `strides` - Stride of the pooling operation as (height, width)
    /// * `input_shape` - Shape of the input tensor \[batch_size, channels, height, width\]
    ///
    /// # Returns
    ///
    /// * `Self` - A new `AveragePooling` layer instance
    pub fn new(
        pool_size: (usize, usize),
        strides: (usize, usize),
        input_shape: Vec<usize>,
    ) -> Self {
        assert_eq!(
            input_shape.len(),
            4,
            "Input shape must be 4-dimensional for AveragePooling"
        );

        AveragePooling {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
        }
    }

    /// Calculates the output shape for the pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - Shape of the output tensor after pooling
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let output_height = (input_shape[2] - self.pool_size.0) / self.strides.0 + 1;
        let output_width = (input_shape[3] - self.pool_size.1) / self.strides.1 + 1;

        vec![input_shape[0], input_shape[1], output_height, output_width]
    }

    /// Performs average pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input` - Input tensor with shape \[batch_size, channels, height, width\]
    ///
    /// # Returns
    ///
    /// * `Tensor` - Result of the pooling operation
    fn avg_pool(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // Pre-allocate output array
        let mut output = ArrayD::zeros(output_shape.clone());

        // Process each batch and channel in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                // Clone output_shape to avoid ownership movement issues
                let output_shape_clone = output_shape.clone();
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();

                    // Perform pooling for each output position
                    for i in 0..output_shape_clone[2] {
                        let i_start = i * self.strides.0;

                        for j in 0..output_shape_clone[3] {
                            let j_start = j * self.strides.1;

                            // Calculate average value within the pooling window
                            let mut sum = 0.0;
                            let mut count = 0;

                            for di in 0..self.pool_size.0 {
                                let i_pos = i_start + di;
                                if i_pos >= input_shape[2] {
                                    continue;
                                }

                                for dj in 0..self.pool_size.1 {
                                    let j_pos = j_start + dj;
                                    if j_pos >= input_shape[3] {
                                        continue;
                                    }

                                    sum += input[[b, c, i_pos, j_pos]];
                                    count += 1;
                                }
                            }

                            // Calculate average, avoiding division by zero
                            let avg_val = if count > 0 { sum / count as f32 } else { 0.0 };
                            batch_channel_output.push((i, j, avg_val));
                        }
                    }

                    ((b, c), batch_channel_output)
                })
            })
            .collect();

        // Merge results into the output tensor
        for ((b, c), outputs) in results {
            for (i, j, val) in outputs {
                output[[b, c, i, j]] = val;
            }
        }

        output
    }
}

impl Layer for AveragePooling {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform average pooling
        self.avg_pool(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];

            // Initialize input gradient to zero
            let mut input_grad = ArrayD::zeros(input_shape.to_vec());

            // Get output dimensions from the output gradient shape
            let output_shape = grad_output.shape();

            // Copy member variables needed in closures
            let pool_size = self.pool_size;
            let strides = self.strides;

            // Process each batch and channel in parallel
            let results: Vec<_> = (0..batch_size)
                .into_par_iter()
                .flat_map(|b| {
                    (0..channels).into_par_iter().map(move |c| {
                        let mut batch_channel_grad: ArrayD<f32> =
                            ArrayD::zeros(input_shape.to_vec());

                        // For each output position
                        for i in 0..output_shape[2] {
                            let i_start = i * strides.0; // Use copied variable instead of self.strides

                            for j in 0..output_shape[3] {
                                let j_start = j * strides.1; // Use copied variable instead of self.strides

                                // Get current output gradient
                                let grad = grad_output[[b, c, i, j]];

                                // Calculate the actual number of elements in the pooling window (considering boundaries)
                                let mut count = 0;
                                for di in 0..pool_size.0 {
                                    // Use copied variable instead of self.pool_size
                                    let i_pos = i_start + di;
                                    if i_pos >= input_shape[2] {
                                        continue;
                                    }

                                    for dj in 0..pool_size.1 {
                                        // Use copied variable instead of self.pool_size
                                        let j_pos = j_start + dj;
                                        if j_pos >= input_shape[3] {
                                            continue;
                                        }

                                        count += 1;
                                    }
                                }

                                // Distribute gradient evenly to all input elements that participated in the calculation
                                let grad_per_element =
                                    if count > 0 { grad / count as f32 } else { 0.0 };

                                for di in 0..pool_size.0 {
                                    // Use copied variable instead of self.pool_size
                                    let i_pos = i_start + di;
                                    if i_pos >= input_shape[2] {
                                        continue;
                                    }

                                    for dj in 0..pool_size.1 {
                                        // Use copied variable instead of self.pool_size
                                        let j_pos = j_start + dj;
                                        if j_pos >= input_shape[3] {
                                            continue;
                                        }

                                        batch_channel_grad[[b, c, i_pos, j_pos]] +=
                                            grad_per_element;
                                    }
                                }
                            }
                        }

                        ((b, c), batch_channel_grad)
                    })
                })
                .collect();

            // Merge gradients from all batches and channels
            for ((b, c), grad) in results {
                for i in 0..input_shape[2] {
                    for j in 0..input_shape[3] {
                        input_grad[[b, c, i, j]] += grad[[b, c, i, j]];
                    }
                }
            }

            Ok(input_grad)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "AveragePooling"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> usize {
        // Average pooling layer has no trainable parameters
        0
    }

    fn update_parameters_sgd(&mut self, _lr: f32) {
        // No parameters to update
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // No parameters to update
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // No parameters to update
    }

    fn get_weights(&self) -> LayerWeight {
        // Average pooling layer has no weights
        LayerWeight::Empty
    }
}
