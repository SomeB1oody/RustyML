use super::helper_functions::*;
use super::*;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::ArrayD;
use rayon::prelude::*;

/// 3D data max pooling Layer.
///
/// Max pooling is a common downsampling technique in convolutional neural networks that
/// reduces the spatial dimensions of feature maps by selecting the maximum value from each
/// pooling window, thereby reducing computational load and controlling overfitting.
///
/// # Data Structure
///
/// - Input tensor format: 5D tensor with shape `[batch_size, channels, depth, height, width]`
/// - Output tensor format: 5D tensor with reduced spatial dimensions
///
/// # Fields
///
/// - `pool_size` - The size of the pooling window, represented as (depth, height, width).
/// - `strides` - The stride of the pooling operation, represented as (depth stride, vertical stride, horizontal stride).
/// - `input_shape` - The shape of the input tensor.
/// - `input_cache` - Cached input data, used for backpropagation.
/// - `max_positions` - Cache of maximum value positions, used for backpropagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
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
/// ));
///
/// // Compile the model with optimizer and loss function
/// model.compile(
///     RMSprop::new(0.001, 0.9, 1e-8),    // RMSprop optimizer
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
/// let predictions = model.predict(&input_data);
/// println!("Output shape after max pooling: {:?}", predictions.shape());
/// // Expected output: [1, 16, 16, 16, 16] (spatial dimensions reduced by factor of 2)
/// ```
pub struct MaxPooling3D {
    pool_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize, usize, usize)>>,
}

impl MaxPooling3D {
    /// Creates a new 3D max pooling layer.
    ///
    /// # Parameters
    ///
    /// - `pool_size` - The size of the pooling window, represented as (depth, height, width).
    /// - `input_shape` - The shape of the input tensor, formatted as \[batch_size, channels, depth, height, width\].
    /// - `strides` - The stride of the pooling operation, represented as (depth stride, vertical stride, horizontal stride).
    ///   If None, the same value as pool_size is used.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the MaxPooling3D layer.
    pub fn new(
        pool_size: (usize, usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize, usize)>,
    ) -> Self {
        // Verify input is 5D: [batch_size, channels, depth, height, width]
        assert_eq!(
            input_shape.len(),
            5,
            "Input tensor must be 5-dimensional: [batch_size, channels, depth, height, width]"
        );

        // If no stride is specified, use the same stride as the pooling window
        let strides = strides.unwrap_or(pool_size);

        MaxPooling3D {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
            max_positions: None,
        }
    }

    /// Performs 3D max pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input` - The input tensor, with shape \[batch_size, channels, depth, height, width\].
    ///
    /// # Returns
    ///
    /// * `(Tensor, Vec<(usize, usize, usize, usize, usize)>)` - The result of the pooling operation and the positions of the maximum values.
    fn max_pool(&self, input: &Tensor) -> (Tensor, Vec<(usize, usize, usize, usize, usize)>) {
        let input_shape = input.shape();

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape =
            calculate_output_shape_3d_pooling(input_shape, self.pool_size, self.strides);

        // Pre-allocate the output array
        let mut output = ArrayD::zeros(output_shape.clone());
        // Vector to store maximum value positions
        let mut max_positions = Vec::new();

        // Process each batch and channel in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                // Clone output_shape here to avoid ownership movement issues
                let output_shape_clone = output_shape.clone();
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();
                    let mut batch_channel_positions = Vec::new();

                    // Perform pooling for each output position
                    for d in 0..output_shape_clone[2] {
                        let d_start = d * self.strides.0;

                        for i in 0..output_shape_clone[3] {
                            let i_start = i * self.strides.1;

                            for j in 0..output_shape_clone[4] {
                                let j_start = j * self.strides.2;

                                // Find the maximum value in the pooling window
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

                                batch_channel_output.push((d, i, j, max_val));
                                batch_channel_positions
                                    .push((b, c, max_pos.0, max_pos.1, max_pos.2));
                            }
                        }
                    }

                    ((b, c), (batch_channel_output, batch_channel_positions))
                })
            })
            .collect();

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
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform max pooling operation
        let (output, max_positions) = self.max_pool(input);

        // Store maximum value positions for backpropagation
        self.max_positions = Some(max_positions);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let (Some(input), Some(max_positions)) = (&self.input_cache, &self.max_positions) {
            let grad_shape = grad_output.shape();

            // Initialize input gradients with the same shape as the input
            let mut input_gradients = ArrayD::zeros(input.dim());

            // Create a vector containing update positions and values
            let gradient_updates: Vec<_> = max_positions
                .par_iter()
                .filter_map(|&(b, c, d, i, j)| {
                    // Calculate the corresponding output gradient index
                    let out_d = d / self.strides.0;
                    let out_i = i / self.strides.1;
                    let out_j = j / self.strides.2;

                    // Ensure the index is within valid range
                    if out_d < grad_shape[2] && out_i < grad_shape[3] && out_j < grad_shape[4] {
                        // Return index and gradient value
                        Some(((b, c, d, i, j), grad_output[[b, c, out_d, out_i, out_j]]))
                    } else {
                        None
                    }
                })
                .collect();

            // Apply gradient updates sequentially
            for ((b, c, d, i, j), grad_val) in gradient_updates {
                input_gradients[[b, c, d, i, j]] = grad_val;
            }

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
