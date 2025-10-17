use super::*;

/// Threshold for deciding between parallel and sequential execution.
/// When batch_size * channels >= this threshold, use parallel execution.
const MAX_POOLING_2D_PARALLEL_THRESHOLD: usize = 32;

/// Defines a structure for max pooling operation, used to perform max pooling on 2D data.
///
/// Max pooling is a common downsampling technique in CNNs that reduces
/// the spatial dimensions of feature maps by selecting the maximum value in each pooling window,
/// thereby reducing computation and controlling overfitting.
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window, expressed as (height, width).
/// - `strides` - Stride of the pooling operation, expressed as (vertical stride, horizontal stride).
/// - `input_shape` - Shape of the input tensor.
/// - `input_cache` - Cached input data, used for backpropagation.
/// - `max_positions` - Cache of maximum value positions, used for backpropagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
///
/// // Create a simple 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 input channels, 6x6 pixels
/// let mut input_data = Array4::zeros((2, 3, 6, 6));
///
/// // Set some specific values so we can predict the max pooling result
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..6 {
///             for j in 0..6 {
///                 // Create input data with an easily observable pattern
///                 input_data[[b, c, i, j]] = (i * j) as f32 + b as f32 * 0.1 + c as f32 * 0.01;
///             }
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test using MaxPooling2D in a model
/// let mut model = Sequential::new();
/// model
///     .add(MaxPooling2D::new(
///         (2, 2),           // Pool window size
///         vec![2, 3, 6, 6], // Input shape
///         None,             // Use default stride (2,2)
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Create target tensor - corresponding to the pooled shape
/// let y = Array4::ones((2, 3, 3, 3)).into_dyn();
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("MaxPooling2D prediction results: {:?}", prediction);
///
/// // Check if output shape is correct
/// assert_eq!(prediction.shape(), &[2, 3, 3, 3]);
/// ```
pub struct MaxPooling2D {
    pool_size: (usize, usize),
    strides: (usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize, usize, usize, usize)>>,
}

impl MaxPooling2D {
    /// Creates a new 2D max pooling layer.
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window, expressed as (height, width).
    /// - `input_shape` - Shape of the input tensor, in format \[batch_size, channels, height, width\].
    /// - `strides` - Stride of the pooling operation, expressed as (vertical stride, horizontal stride). If None, uses the same value as pool_size.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the MaxPooling2D layer.
    pub fn new(
        pool_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize)>,
    ) -> Self {
        // Verify input is 4D: [batch_size, channels, height, width]
        assert_eq!(
            input_shape.len(),
            4,
            "Input shape must be 4-dimensional: [batch_size, channels, height, width]"
        );

        // If stride is not specified, use the same stride as pool size
        let strides = strides.unwrap_or(pool_size);

        MaxPooling2D {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
            max_positions: None,
        }
    }

    /// Calculates the output shape of the max pooling layer.
    ///
    /// # Parameters
    ///
    /// * `input_shape` - Shape of the input tensor, in format \[batch_size, channels, height, width\].
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - A vector containing the calculated output shape, in format \[batch_size, channels, output_height, output_width\].
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        calculate_output_shape_2d_pooling(input_shape, self.pool_size, self.strides)
    }

    /// Performs max pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input` - Input tensor with shape \[batch_size, channels, height, width\].
    ///
    /// # Returns
    ///
    /// * `(Tensor, Vec<(usize, usize, usize, usize, usize, usize)>)` - Result of the pooling operation
    ///   and positions mapping: (batch, channel, out_i, out_j, in_i, in_j)
    fn max_pool(
        &self,
        input: &Tensor,
    ) -> (Tensor, Vec<(usize, usize, usize, usize, usize, usize)>) {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // Pre-allocate output array
        let mut output = ArrayD::zeros(output_shape.clone());
        // Vector to store mapping from output positions to input positions with maximum values
        let mut max_positions = Vec::new();

        // Helper closure to compute max pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut batch_channel_output = Vec::new();
            let mut batch_channel_positions = Vec::new();

            // Perform pooling for each output position
            for out_i in 0..output_shape[2] {
                let i_start = out_i * self.strides.0;

                for out_j in 0..output_shape[3] {
                    let j_start = out_j * self.strides.1;

                    // Find maximum value in pooling window
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_pos = (0, 0);

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

                            let val = input[[b, c, i_pos, j_pos]];
                            if val > max_val {
                                max_val = val;
                                max_pos = (i_pos, j_pos);
                            }
                        }
                    }

                    batch_channel_output.push((out_i, out_j, max_val));
                    // Store complete mapping: (batch, channel, output_i, output_j, input_i, input_j)
                    batch_channel_positions.push((b, c, out_i, out_j, max_pos.0, max_pos.1));
                }
            }

            ((b, c), (batch_channel_output, batch_channel_positions))
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            MAX_POOLING_2D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Merge results into output tensor
        for ((b, c), (outputs, positions)) in results {
            for (i, j, val) in outputs {
                output[[b, c, i, j]] = val;
            }
            max_positions.extend(positions);
        }

        (output, max_positions)
    }
}

impl Layer for MaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 4D".to_string(),
            ));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform max pooling operation
        let (output, max_positions) = self.max_pool(input);

        // Store maximum value positions for backpropagation
        self.max_positions = Some(max_positions);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let (Some(input), Some(max_positions)) = (&self.input_cache, &self.max_positions) {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let height = input_shape[2];
            let width = input_shape[3];

            // Initialize input gradients with same shape as input
            let mut input_gradients = ArrayD::zeros(input_shape.to_vec());

            // Group max_positions by (batch, channel) for parallel processing
            let mut positions_by_bc: std::collections::HashMap<
                (usize, usize),
                Vec<(usize, usize, usize, usize)>,
            > = std::collections::HashMap::new();

            for &(b, c, out_i, out_j, in_i, in_j) in max_positions.iter() {
                positions_by_bc
                    .entry((b, c))
                    .or_insert_with(Vec::new)
                    .push((out_i, out_j, in_i, in_j));
            }

            // Helper closure to compute gradient for a single (batch, channel) pair
            let compute_gradient = |b: usize, c: usize| {
                let mut spatial_grad = vec![0.0; height * width];

                if let Some(positions) = positions_by_bc.get(&(b, c)) {
                    for &(out_i, out_j, in_i, in_j) in positions {
                        let flat_idx = in_i * width + in_j;
                        spatial_grad[flat_idx] += grad_output[[b, c, out_i, out_j]];
                    }
                }

                ((b, c), spatial_grad)
            };

            // Choose parallel or sequential execution based on workload size
            let results: Vec<_> = execute_parallel_or_sequential!(
                batch_size,
                channels,
                MAX_POOLING_2D_PARALLEL_THRESHOLD,
                compute_gradient
            );

            // Write results back to gradient array
            merge_gradients_2d!(input_gradients, results, height, width);

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "MaxPooling2D"
    }

    layer_functions_2d_pooling!();
}
