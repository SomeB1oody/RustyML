use super::helper_functions::*;
use crate::neural_network::{Layer, LayerWeight, ModelError, Tensor};
use ndarray::{Array, IxDyn};
use rayon::prelude::*;

/// 3D Average Pooling Layer
///
/// Average pooling is a commonly used down-sampling technique in convolutional neural networks
/// that reduces the spatial dimensions of feature maps by calculating the average value
/// of all values in each pooling window. This reduces computational load and helps control overfitting.
///
/// # Data Structure
///
/// - Input tensor format: 5D tensor with shape `[batch_size, channels, depth, height, width]`
/// - Output tensor format: 5D tensor with reduced spatial dimensions
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window, represented as (depth, height, width)
/// - `strides` - Stride of the pooling operation, represented as (depth_stride, height_stride, width_stride)
/// - `input_shape` - Shape of the input tensor
/// - `input_cache` - Cached input data used for backpropagation
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array5, ArrayD};
///
/// // Create a Sequential model for 3D data processing
/// let mut model = Sequential::new();
///
/// // Add an AveragePooling3D layer to the model
/// model.add(AveragePooling3D::new(
///     (2, 2, 2),                    // Pooling window size: 2×2×2
///     vec![1, 16, 32, 32, 32],      // Input shape: [batch, channels, depth, height, width]
///     Some((2, 2, 2)),              // Strides: move by 2 in each dimension
/// ));
///
/// // Compile the model with optimizer and loss function
/// model.compile(
///     RMSprop::new(0.001, 0.9, 1e-8),    // RMSprop optimizer
///     MeanSquaredError::new()            // Mean squared error loss
/// );
///
/// // Create example 3D input data (e.g., 3D medical imaging or volume data)
/// // Input: [1 batch, 16 channels, 32×32×32 3D volume]
/// let input_data = Array5::from_shape_fn((1, 16, 32, 32, 32), |(b, c, d, h, w)| {
///     // Generate example data with spatial patterns
///     ((d + h + w) as f32 * 0.1) + (c as f32 * 0.01)
/// }).into_dyn();
///
/// // Create target data for training (output shape: [1, 16, 16, 16, 16])
/// let target_data = Array5::ones((1, 16, 16, 16, 16)).into_dyn();
///
/// // Display the model architecture
/// model.summary();
///
/// // Train the model
/// model.fit(&input_data, &target_data, 5).unwrap();
///
/// // Make predictions on new data
/// let predictions = model.predict(&input_data);
/// println!("Output shape after average pooling: {:?}", predictions.shape());
/// // Expected output: [1, 16, 16, 16, 16] (spatial dimensions are halved)
/// ```
pub struct AveragePooling3D {
    pool_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl AveragePooling3D {
    /// Create a new AveragePooling3D layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window (depth, height, width)
    /// - `input_shape` - Shape of the input tensor \[batch_size, channels, depth, height, width\]
    /// - `strides` - Optional strides, if None, the values will match `pool_size`
    ///
    /// # Returns
    ///
    /// * `AveragePooling3D` - A new `AveragePooling3D` instance
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

        let strides = strides.unwrap_or(pool_size);

        Self {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
        }
    }
}

impl Layer for AveragePooling3D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();

        // Cache input for backpropagation
        self.input_cache = Some(input.clone());
        self.input_shape = input_shape.to_vec();

        let output_shape =
            calculate_output_shape_3d_pooling(input_shape, self.pool_size, self.strides);
        let mut output = Array::zeros(IxDyn(&output_shape));

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];

        let output_depth = output_shape[2];
        let output_height = output_shape[3];
        let output_width = output_shape[4];

        let pool_volume = (self.pool_size.0 * self.pool_size.1 * self.pool_size.2) as f32;

        // Create index pairs for all (batch, channel) combinations
        let batch_channel_pairs: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c)))
            .collect();

        // Parallel computation for each (batch, channel) pair
        let results: Vec<((usize, usize), Vec<((usize, usize, usize), f32)>)> = batch_channel_pairs
            .into_par_iter()
            .map(|(b, c)| {
                let mut local_results = Vec::new();

                for od in 0..output_depth {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let start_d = od * self.strides.0;
                            let start_h = oh * self.strides.1;
                            let start_w = ow * self.strides.2;

                            let end_d = (start_d + self.pool_size.0).min(input_depth);
                            let end_h = (start_h + self.pool_size.1).min(input_height);
                            let end_w = (start_w + self.pool_size.2).min(input_width);

                            // Calculate average value within the pooling window
                            let mut sum = 0.0;
                            for d in start_d..end_d {
                                for h in start_h..end_h {
                                    for w in start_w..end_w {
                                        sum += input[[b, c, d, h, w]];
                                    }
                                }
                            }

                            let pooled_value = sum / pool_volume;
                            local_results.push(((od, oh, ow), pooled_value));
                        }
                    }
                }

                ((b, c), local_results)
            })
            .collect();

        // Merge results into output tensor
        for ((b, c), local_results) in results {
            for ((od, oh, ow), value) in local_results {
                output[[b, c, od, oh, ow]] = value;
            }
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let input = self.input_cache.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run yet".to_string())
        })?;

        let input_shape = input.shape();
        let mut grad_input = Array::zeros(IxDyn(input_shape));

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];

        let output_depth = grad_output.shape()[2];
        let output_height = grad_output.shape()[3];
        let output_width = grad_output.shape()[4];

        let pool_volume = (self.pool_size.0 * self.pool_size.1 * self.pool_size.2) as f32;

        // Create index pairs for all (batch, channel) combinations
        let batch_channel_pairs: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c)))
            .collect();

        // Parallel computation of gradient contributions for each (batch, channel) pair
        let results: Vec<((usize, usize), Vec<((usize, usize, usize), f32)>)> = batch_channel_pairs
            .into_par_iter()
            .map(|(b, c)| {
                let mut local_gradients = Vec::new();

                for od in 0..output_depth {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let start_d = od * self.strides.0;
                            let start_h = oh * self.strides.1;
                            let start_w = ow * self.strides.2;

                            let end_d = (start_d + self.pool_size.0).min(input_depth);
                            let end_h = (start_h + self.pool_size.1).min(input_height);
                            let end_w = (start_w + self.pool_size.2).min(input_width);

                            // Distribute gradient evenly to all elements in the pooling window
                            let grad_value = grad_output[[b, c, od, oh, ow]] / pool_volume;

                            for d in start_d..end_d {
                                for h in start_h..end_h {
                                    for w in start_w..end_w {
                                        local_gradients.push(((d, h, w), grad_value));
                                    }
                                }
                            }
                        }
                    }
                }

                ((b, c), local_gradients)
            })
            .collect();

        // Merge parallel computation results into output gradient tensor
        for ((b, c), local_gradients) in results {
            for ((d, h, w), grad_value) in local_gradients {
                grad_input[[b, c, d, h, w]] += grad_value;
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "AveragePooling3D"
    }

    layer_functions_3d_pooling!();
}
