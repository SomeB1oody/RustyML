use super::*;

/// Global Average Pooling 3D Layer
///
/// Performs global average pooling operation on the spatial dimensions (depth, height, and width) of the input tensor.
/// Input tensor shape should be `[batch_size, channels, depth, height, width]`,
/// output tensor shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// - `input_shape` - Stores the shape of the input tensor during forward propagation.
/// - `input_cache` - Caches the input tensor for backward propagation.
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
/// // Add a GlobalAveragePooling3D layer
/// model.add(GlobalAveragePooling3D::new());
///
/// // Create test input tensor: [batch_size, channels, depth, height, width]
/// let input_data = Array::from_elem(IxDyn(&[2, 4, 8, 8, 8]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data);
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
pub struct GlobalAveragePooling3D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl GlobalAveragePooling3D {
    /// Creates a new GlobalAveragePooling3D layer.
    ///
    /// # Returns
    ///
    /// * `GlobalAveragePooling3D` - A new `GlobalAveragePooling3D` layer instance
    pub fn new() -> Self {
        GlobalAveragePooling3D {
            input_shape: Vec::new(),
            input_cache: None,
        }
    }
}

impl Layer for GlobalAveragePooling3D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Verify input is 5D: [batch_size, channels, depth, height, width]
        let shape = input.shape();
        assert_eq!(
            shape.len(),
            5,
            "Input tensor must be 5-dimensional: [batch_size, channels, depth, height, width]"
        );

        // Extract dimensions
        let (batch_size, channels, depth, height, width) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);

        // Store input shape and cache input for backpropagation
        self.input_shape = vec![batch_size, channels, depth, height, width];
        self.input_cache = Some(input.clone());

        // Create output tensor
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Compute global average pooling
        // Use parallel iteration over batches, compute sequential sum over channels and spatial dimensions
        let spatial_size = (depth * height * width) as f32;

        output
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut output_batch)| {
                for c in 0..channels {
                    let mut sum = 0.0;
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                sum += input[[b, c, d, h, w]];
                            }
                        }
                    }
                    output_batch[c] = sum / spatial_size;
                }
            });

        output
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
            let spatial_elements = (depth * height * width) as f32;

            // Create gradient tensor with the same shape as input
            let mut grad_input = Array::zeros(IxDyn(&input_shape));

            // Create a flat vector of (batch, channel) indices for parallel processing
            let batch_channel_indices: Vec<(usize, usize)> = (0..batch_size)
                .flat_map(|b| (0..channels).map(move |c| (b, c)))
                .collect();

            // Process gradient distribution in parallel
            let grad_updates: Vec<((usize, usize), f32)> = batch_channel_indices
                .par_iter()
                .map(|&(b, c)| {
                    let grad_value = grad_output[[b, c]] / spatial_elements;
                    ((b, c), grad_value)
                })
                .collect();

            // Distribute gradients uniformly across all spatial positions
            for ((b, c), grad_value) in grad_updates {
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            grad_input[[b, c, d, h, w]] = grad_value;
                        }
                    }
                }
            }

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
