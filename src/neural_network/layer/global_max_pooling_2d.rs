use crate::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::{Array, IxDyn};
use rayon::prelude::*;

/// Global Max Pooling Layer
///
/// Performs global max pooling operation on the input tensor across spatial dimensions (height and width).
/// The input tensor shape should be `[batch_size, channels, height, width]`,
/// and the output tensor shape will be `[batch_size, channels]`.
///
/// This layer has no trainable parameters.
///
/// # Fields
///
/// * `input_shape` - Stores the shape of the input tensor during the forward pass.
/// * `input_cache` - Caches the input tensor for backpropagation.
/// * `max_positions` - Stores the positions of maximum values found during the forward pass, used for gradient propagation during backpropagation.
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
/// // Add a GlobalMaxPooling2D layer
/// model.add(GlobalMaxPooling2D::new());
///
/// // Create a test input tensor: [batch_size, channels, height, width]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 5, 5]), 1.0);
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
pub struct GlobalMaxPooling2D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize)>>,
}

impl GlobalMaxPooling2D {
    /// Creates a new instance of the global max pooling layer
    ///
    /// # Returns
    ///
    /// * `GlobalMaxPooling2D` - A new `GlobalMaxPooling2D` instance
    pub fn new() -> Self {
        GlobalMaxPooling2D {
            input_shape: Vec::new(),
            input_cache: None,
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input shape and cache input for backpropagation
        self.input_shape = input.shape().to_vec();

        // Verify input is 4D: [batch_size, channels, height, width]
        assert_eq!(
            self.input_shape.len(),
            4,
            "Input shape must be 4-dimensional: [batch_size, channels, height, width]"
        );

        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];

        // Create result tensor and storage for maximum value positions
        let mut result = Array::zeros(IxDyn(&[batch_size, channels]));

        // Create index collection representing all (batch, channel) combinations
        let indices: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c)))
            .collect();

        // Process each (batch, channel) combination in parallel
        let max_results: Vec<((usize, usize), f32, (usize, usize))> = indices
            .par_iter()
            .map(|&(b, c)| {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_h = 0;
                let mut max_w = 0;

                // Find the maximum value and its position in each channel
                for h in 0..height {
                    for w in 0..width {
                        let val = input[[b, c, h, w]];
                        if val > max_val {
                            max_val = val;
                            max_h = h;
                            max_w = w;
                        }
                    }
                }

                ((b, c), max_val, (max_h, max_w))
            })
            .collect();

        // Fill output tensor and position records using parallel computation results
        let mut max_positions = Vec::with_capacity(batch_size * channels);

        for ((b, c), max_val, (max_h, max_w)) in max_results {
            result[[b, c]] = max_val;
            max_positions.push((max_h, max_w));
        }

        self.max_positions = Some(max_positions);
        result
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Check if input cache is valid
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run yet".to_string(),
                ));
            }
        };

        // Check if maximum value position records are valid
        let max_positions = match &self.max_positions {
            Some(positions) => positions,
            None => {
                return Err(ModelError::ProcessingError(
                    "Forward pass has not been run yet".to_string(),
                ));
            }
        };

        // Get input dimensions
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];

        // Create a gradient tensor with the same shape as input, initialized to zero
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, channels, height, width]));

        // Create index collection representing all (batch, channel) combinations
        let indices: Vec<(usize, usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c, b * channels + c)))
            .collect();

        // Process gradients for each (batch, channel) combination in parallel
        let grad_results: Vec<(usize, usize, usize, usize, f32)> = indices
            .par_iter()
            .map(|&(b, c, idx)| {
                let (max_h, max_w) = max_positions[idx];
                let grad_value = grad_output[[b, c]];
                (b, c, max_h, max_w, grad_value)
            })
            .collect();

        // Fill gradient tensor
        for (b, c, max_h, max_w, grad_value) in grad_results {
            grad_input[[b, c, max_h, max_w]] = grad_value;
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling2D"
    }

    layer_functions_global_pooling!();
}
