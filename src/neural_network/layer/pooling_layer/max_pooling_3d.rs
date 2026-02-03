use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::helper_function::calculate_output_shape_3d_pooling;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::pooling_layer::input_validation_function::{
    validate_input_shape_dims, validate_pool_size_3d, validate_strides_3d,
};
use crate::neural_network::neural_network_trait::Layer;
use ndarray::ArrayD;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Threshold for deciding between parallel and sequential execution.
/// When batch_size * channels >= this threshold, use parallel execution.
const MAX_POOLING_3D_PARALLEL_THRESHOLD: usize = 32;

/// 3D max pooling layer.
///
/// Selects the maximum value within each pooling window across depth, height, and width.
/// Input tensor shape: `[batch_size, channels, depth, height, width]`. Output tensor shape:
/// `[batch_size, channels, pooled_depth, pooled_height, pooled_width]` where
/// `pooled_depth = (depth - pool_size_d) / stride_d + 1`,
/// `pooled_height = (height - pool_size_h) / stride_h + 1`, and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`.
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window as (depth, height, width)
/// - `strides` - Step size of the pooling operation as (depth stride, height stride, width stride)
/// - `input_shape` - Shape of the input tensor
/// - `input_cache` - Cached input tensor from the forward pass
/// - `max_positions` - Cached positions of maximum values for backpropagation
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::{Array5, ArrayD};
///
/// // Create a Sequential model for 3D data processing
/// let mut model = Sequential::new();
///
/// // Add MaxPooling3D layer to the model
/// model.add(MaxPooling3D::new(
///     (2, 2, 2),                    // pooling window size: 2×2×2
///     vec![1, 16, 32, 32, 32],      // input shape: [batch, channels, depth, height, width]
///     Some((2, 2, 2)),              // stride: move window by 2 in each dimension
/// ).unwrap());
///
/// // Compile the model with optimizer and loss function
/// model.compile(
///     RMSprop::new(0.001, 0.9, 1e-8).unwrap(),    // RMSprop optimizer
///     MeanSquaredError::new()              // Mean Squared Error loss
/// );
///
/// // Create sample 3D input data (e.g., 3D medical images or volumetric data)
/// // Input: [1 batch, 16 channels, 32×32×32 3D volume]
/// let input_data = Array5::from_shape_fn((1, 16, 32, 32, 32), |(b, c, d, h, w)| {
///     // Generate sample data with spatial patterns
///     ((d + h + w) as f32 * 0.1) + (c as f32 * 0.01)
/// }).into_dyn();
///
/// // Create target data for training (output shape: [1, 16, 16, 16, 16])
/// let target_data = Array5::ones((1, 16, 16, 16, 16)).into_dyn();
///
/// // Display model architecture
/// model.summary();
///
/// // Train the model
/// model.fit(&input_data, &target_data, 5).unwrap();
///
/// // Make predictions on new data
/// let predictions = model.predict(&input_data).unwrap();
/// println!("Output shape after max pooling: {:?}", predictions.shape());
/// // Expected output: [1, 16, 16, 16, 16] (spatial dimensions reduced by factor of 2)
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= MAX_POOLING_3D_PARALLEL_THRESHOLD` (32).
pub struct MaxPooling3D {
    pool_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>>,
}

impl MaxPooling3D {
    /// Creates a new 3D max pooling layer.
    ///
    /// If `strides` is None, it defaults to `pool_size`.
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (depth, height, width)
    /// - `input_shape` - Input tensor shape `[batch_size, channels, depth, height, width]`
    /// - `strides` - Optional strides of the pooling operation
    ///
    /// # Returns
    ///
    /// - `Result<MaxPooling3D, ModelError>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_shape` is not 5D, `pool_size` has a zero
    ///   dimension, or any stride is zero
    pub fn new(
        pool_size: (usize, usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize, usize)>,
    ) -> Result<Self, ModelError> {
        let strides = strides.unwrap_or(pool_size);

        // input validation
        validate_input_shape_dims(&input_shape, 5, "MaxPooling3D")?;
        validate_pool_size_3d(pool_size)?;
        validate_strides_3d(strides)?;

        Ok(MaxPooling3D {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
            max_positions: None,
        })
    }

    /// Calculates the output shape of the max pooling layer.
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        calculate_output_shape_3d_pooling(input_shape, self.pool_size, self.strides)
    }

    /// Performs 3D max pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input` - The input tensor, with shape \[batch_size, channels, depth, height, width\].
    ///
    /// # Returns
    ///
    /// * `(Tensor, Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>)` - Result of the pooling operation
    ///   and positions mapping: (batch, channel, out_d, out_i, out_j, in_d, in_i, in_j)
    fn max_pool(
        &self,
        input: &Tensor,
    ) -> (
        Tensor,
        Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>,
    ) {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // Pre-allocate the output array
        let mut output = ArrayD::zeros(output_shape.clone());
        // Vector to store maximum value positions
        let mut max_positions = Vec::new();

        // Helper closure to compute max pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut batch_channel_output = Vec::new();
            let mut batch_channel_positions = Vec::new();

            // Perform pooling for each output position
            for out_d in 0..output_shape[2] {
                let d_start = out_d * self.strides.0;

                for out_i in 0..output_shape[3] {
                    let i_start = out_i * self.strides.1;

                    for out_j in 0..output_shape[4] {
                        let j_start = out_j * self.strides.2;

                        // Find maximum value in pooling window
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_pos = (0, 0, 0);

                        for dd in 0..self.pool_size.0 {
                            let d_pos = d_start + dd;
                            if d_pos >= input_shape[2] {
                                continue;
                            }

                            for di in 0..self.pool_size.1 {
                                let i_pos = i_start + di;
                                if i_pos >= input_shape[3] {
                                    continue;
                                }

                                for dj in 0..self.pool_size.2 {
                                    let j_pos = j_start + dj;
                                    if j_pos >= input_shape[4] {
                                        continue;
                                    }

                                    let val = input[[b, c, d_pos, i_pos, j_pos]];
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = (d_pos, i_pos, j_pos);
                                    }
                                }
                            }
                        }

                        batch_channel_output.push((out_d, out_i, out_j, max_val));
                        // Store complete mapping: (batch, channel, output_d, output_i, output_j, input_d, input_i, input_j)
                        batch_channel_positions
                            .push((b, c, out_d, out_i, out_j, max_pos.0, max_pos.1, max_pos.2));
                    }
                }
            }

            ((b, c), (batch_channel_output, batch_channel_positions))
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            MAX_POOLING_3D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Merge results into the output tensor
        for ((b, c), (outputs, positions)) in results {
            for (d, i, j, val) in outputs {
                output[[b, c, d, i, j]] = val;
            }
            max_positions.extend(positions);
        }

        (output, max_positions)
    }
}

impl Layer for MaxPooling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 5D".to_string(),
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
            let depth = input_shape[2];
            let height = input_shape[3];
            let width = input_shape[4];

            // Initialize input gradients with same shape as input
            let mut input_gradients = ArrayD::zeros(input_shape.to_vec());

            // Group max_positions by (batch, channel) for parallel processing
            let mut positions_by_bc: std::collections::HashMap<
                (usize, usize),
                Vec<(usize, usize, usize, usize, usize, usize)>,
            > = std::collections::HashMap::new();

            for &(b, c, out_d, out_i, out_j, in_d, in_i, in_j) in max_positions.iter() {
                positions_by_bc
                    .entry((b, c))
                    .or_insert_with(Vec::new)
                    .push((out_d, out_i, out_j, in_d, in_i, in_j));
            }

            // Helper closure to compute gradient for a single (batch, channel) pair
            let compute_gradient = |b: usize, c: usize| {
                let mut spatial_grad = vec![0.0; depth * height * width];

                if let Some(positions) = positions_by_bc.get(&(b, c)) {
                    for &(out_d, out_i, out_j, in_d, in_i, in_j) in positions {
                        let flat_idx = in_d * (height * width) + in_i * width + in_j;
                        spatial_grad[flat_idx] += grad_output[[b, c, out_d, out_i, out_j]];
                    }
                }

                ((b, c), spatial_grad)
            };

            // Choose parallel or sequential execution based on workload size
            let results: Vec<_> = execute_parallel_or_sequential!(
                batch_size,
                channels,
                MAX_POOLING_3D_PARALLEL_THRESHOLD,
                compute_gradient
            );

            // Write results back to gradient array
            merge_gradients_3d!(input_gradients, results, depth, height, width);

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "MaxPooling3D"
    }

    layer_functions_3d_pooling!();
}
