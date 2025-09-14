use super::*;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::{Array, IxDyn};
use rayon::prelude::*;

/// Global Max Pooling 3D Layer
///
/// Performs global max pooling operation on the spatial dimensions (depth, height, and width) of the input tensor.
/// The input tensor's shape should be `[batch_size, channels, depth, height, width]`,
/// and the output tensor's shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// - `input_shape` - Stores the shape of the input tensor during forward propagation.
/// - `input_cache` - Caches the input tensor for backpropagation.
/// - `max_positions` - Stores the positions of maximum values found during forward propagation, used for gradient propagation in backpropagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model with multiple layers
/// let mut model = Sequential::new();
///
/// // Add GlobalMaxPooling3D layer
/// model.add(GlobalMaxPooling3D::new());
///
/// // Create test input tensor: [batch_size, channels, depth, height, width]
/// let input_data = Array::from_elem(IxDyn(&[2, 3, 4, 5, 5]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data);
///
/// // Check output shape - should be [2, 3]
/// assert_eq!(output.shape(), &[2, 3]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..2 {
///     for c in 0..3 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
pub struct GlobalMaxPooling3D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize)>>,
}

impl GlobalMaxPooling3D {
    /// Creates a new GlobalMaxPooling3D layer.
    ///
    /// # Returns
    ///
    /// * `GlobalMaxPooling3D` - A new `GlobalMaxPooling3D` layer instance
    pub fn new() -> Self {
        GlobalMaxPooling3D {
            input_shape: Vec::new(),
            input_cache: None,
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling3D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();

        // Verify input is 5D: [batch_size, channels, depth, height, width]
        assert_eq!(
            input_shape.len(),
            5,
            "Input tensor must be 5-dimensional: [batch_size, channels, depth, height, width]"
        );

        // Save input shape and data for backpropagation
        self.input_shape = input_shape.to_vec();
        self.input_cache = Some(input.clone());

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let depth = input_shape[2];
        let height = input_shape[3];
        let width = input_shape[4];

        // Create output tensor
        let mut output = Array::zeros(IxDyn(&[batch_size, channels]));

        // Create a vector to store results from parallel computation
        let batch_channel_pairs: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c)))
            .collect();

        // Perform global max pooling in parallel across batch and channel dimensions
        let results: Vec<(usize, usize, f32, (usize, usize, usize))> = batch_channel_pairs
            .par_iter()
            .map(|&(b, c)| {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_pos = (0, 0, 0);

                // Find maximum value across spatial dimensions
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            let val = input[[b, c, d, h, w]];
                            if val > max_val {
                                max_val = val;
                                max_pos = (d, h, w);
                            }
                        }
                    }
                }

                (b, c, max_val, max_pos)
            })
            .collect();

        // Collect results into output tensor and max_positions vector
        let mut max_positions = Vec::with_capacity(batch_size * channels);

        // Sort results by batch and channel to maintain order
        let mut sorted_results = results;
        sorted_results.sort_by_key(|&(b, c, _, _)| (b, c));

        for (b, c, max_val, max_pos) in sorted_results {
            output[[b, c]] = max_val;
            max_positions.push(max_pos);
        }

        // Save maximum value positions for backpropagation
        self.max_positions = Some(max_positions);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let (Some(input), Some(max_positions)) = (&self.input_cache, &self.max_positions) {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];

            // Create gradient tensor with the same shape as input
            let mut grad_input = Array::zeros(IxDyn(&input_shape));

            // Collect all update operations into a vector
            let updates: Vec<(Vec<usize>, f32)> = (0..batch_size)
                .into_par_iter()
                .flat_map(|b| {
                    (0..channels).into_par_iter().map(move |c| {
                        let pos_idx = b * channels + c;
                        let (max_d, max_h, max_w) = max_positions[pos_idx];
                        let grad_value = grad_output[[b, c]];

                        // Return indices and gradient value
                        (vec![b, c, max_d, max_h, max_w], grad_value)
                    })
                })
                .collect();

            // Apply all updates serially to avoid data races
            for (indices, grad_value) in updates {
                grad_input[IxDyn(&indices)] = grad_value;
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling3D"
    }

    layer_functions_global_pooling!();
}
