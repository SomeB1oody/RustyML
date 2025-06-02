use crate::neural_network::layer::LayerWeight;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::IxDyn;
use rayon::prelude::*;

/// Global Max Pooling 1D Layer
///
/// Performs global max pooling operation on the input tensor across the sequence dimension.
/// The input tensor shape should be `[batch_size, channels, length]`,
/// and the output tensor shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// - `input_shape` - Stores the shape of the input tensor during the forward pass.
/// - `input_cache` - Caches the input tensor for backpropagation.
/// - `max_positions` - Stores the positions of maximum values found during the forward pass, used for gradient propagation during backpropagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model containing multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalMaxPooling1D layer
/// model.add(GlobalMaxPooling1D::new());
///
/// // Create a test input tensor: [batch_size, channels, length]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 8]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data);
///
/// // Check output shape - should be [3, 4]
/// assert_eq!(output.shape(), &[3, 4]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..3 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
pub struct GlobalMaxPooling1D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<usize>>,
}

impl GlobalMaxPooling1D {
    /// Creates a new GlobalMaxPooling1D layer.
    ///
    /// # Returns
    ///
    /// * `Self` - A new `GlobalMaxPooling1D` layer instance
    pub fn new() -> Self {
        GlobalMaxPooling1D {
            input_shape: Vec::new(),
            input_cache: None,
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Store the input shape and cache the input for backpropagation
        self.input_shape = input.shape().to_vec();
        self.input_cache = Some(input.clone());

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let length = self.input_shape[2];

        // Initialize output tensor with shape [batch_size, channels]
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Use rayon to process batches and channels in parallel, return (batch_idx, channel_idx, max_value, max_position)
        let results: Vec<(usize, usize, f32, usize)> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels)
                    .into_par_iter()
                    .map(move |c| {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_pos = 0;

                        // Find maximum value and its position
                        for l in 0..length {
                            let val = input[[b, c, l]];
                            if val > max_val {
                                max_val = val;
                                max_pos = l;
                            }
                        }

                        // Return a tuple of indices, max value and max position
                        (b, c, max_val, max_pos)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Fill the output tensor and collect max value positions
        let mut max_positions = Vec::with_capacity(batch_size * channels);

        for (b, c, max_val, max_pos) in results {
            output[[b, c]] = max_val;
            max_positions.push(max_pos);
        }

        // Cache the positions of maximum values for backpropagation
        self.max_positions = Some(max_positions);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(max_positions) = &self.max_positions {
            let batch_size = self.input_shape[0];
            let channels = self.input_shape[1];

            // Initialize input gradient tensor
            let mut grad_input = Tensor::zeros(IxDyn(&self.input_shape));

            // Collect positions and values to update
            let updates: Vec<(usize, usize, usize, f32)> = (0..batch_size * channels)
                .into_par_iter()
                .map(|idx| {
                    let b = idx / channels;
                    let c = idx % channels;
                    let max_pos = max_positions[idx];
                    let gradient_value = grad_output[[b, c]];

                    (b, c, max_pos, gradient_value)
                })
                .collect();

            // Apply updates in the main thread
            for (b, c, max_pos, gradient_value) in updates {
                grad_input[[b, c, max_pos]] = gradient_value;
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling1D"
    }

    layer_functions_avg_pooling!();
}
