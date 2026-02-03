use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::neural_network_trait::Layer;
use ndarray::IxDyn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Threshold for deciding between parallel and sequential execution.
/// When batch_size * channels >= this threshold, use parallel execution.
const GLOBAL_MAX_POOLING_3D_PARALLEL_THRESHOLD: usize = 32;

/// Global max pooling layer for 3D inputs.
///
/// Selects the maximum value across the depth, height, and width dimensions.
/// Input tensor shape: `[batch_size, channels, depth, height, width]`. Output tensor shape:
/// `[batch_size, channels]`.
///
/// # Fields
///
/// - `input_shape` - Shape of the input tensor cached during the forward pass
/// - `max_positions` - Cached positions of maximum values for backpropagation
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
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
/// let output = model.predict(&input_data).unwrap();
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
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= GLOBAL_MAX_POOLING_3D_PARALLEL_THRESHOLD` (32).
pub struct GlobalMaxPooling3D {
    input_shape: Vec<usize>,
    max_positions: Option<Vec<(usize, usize, usize)>>,
}

impl GlobalMaxPooling3D {
    /// Creates a new global max pooling 3D layer.
    ///
    /// # Returns
    ///
    /// - `GlobalMaxPooling3D` - New layer instance
    pub fn new() -> Self {
        GlobalMaxPooling3D {
            input_shape: Vec::new(),
            max_positions: None,
        }
    }
}

impl Layer for GlobalMaxPooling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 5D".to_string(),
            ));
        }

        // Store the input shape for backpropagation
        self.input_shape = input.shape().to_vec();

        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let depth = self.input_shape[2];
        let height = self.input_shape[3];
        let width = self.input_shape[4];

        // Initialize output tensor with shape [batch_size, channels]
        let mut output = Tensor::zeros(IxDyn(&[batch_size, channels]));

        // Pre-allocate max_positions vector with correct capacity
        let mut max_positions = vec![(0, 0, 0); batch_size * channels];

        // Helper closure to compute global max pooling for a single (batch, channel) pair
        let compute_pooling = |b: usize, c: usize| {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_pos = (0, 0, 0);

            // Find maximum value and its position across spatial dimensions
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

            ((b, c), (max_val, max_pos))
        };

        // Choose parallel or sequential execution based on workload size
        let results: Vec<_> = execute_parallel_or_sequential!(
            batch_size,
            channels,
            GLOBAL_MAX_POOLING_3D_PARALLEL_THRESHOLD,
            compute_pooling
        );

        // Fill the output tensor and max_positions
        for ((b, c), (max_val, max_pos)) in results {
            output[[b, c]] = max_val;
            let idx = b * channels + c;
            max_positions[idx] = max_pos;
        }

        // Cache the positions of maximum values for backpropagation
        self.max_positions = Some(max_positions);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(max_positions) = &self.max_positions {
            let batch_size = self.input_shape[0];
            let channels = self.input_shape[1];
            let depth = self.input_shape[2];
            let height = self.input_shape[3];
            let width = self.input_shape[4];

            // Initialize input gradient tensor
            let mut grad_input = Tensor::zeros(IxDyn(&self.input_shape));

            // Helper closure to compute gradient for a single (batch, channel) pair
            let compute_gradient = |b: usize, c: usize| {
                let idx = b * channels + c;
                let (max_d, max_h, max_w) = max_positions[idx];
                let gradient_value = grad_output[[b, c]];

                // Create a gradient volume for this channel (only one position gets the gradient)
                let mut spatial_grad = vec![0.0; depth * height * width];
                let flat_idx = max_d * (height * width) + max_h * width + max_w;
                spatial_grad[flat_idx] = gradient_value;

                ((b, c), spatial_grad)
            };

            // Choose parallel or sequential execution based on workload size
            let results: Vec<_> = execute_parallel_or_sequential!(
                batch_size,
                channels,
                GLOBAL_MAX_POOLING_3D_PARALLEL_THRESHOLD,
                compute_gradient
            );

            // Apply updates to gradient tensor
            merge_gradients_3d!(grad_input, results, depth, height, width);

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
