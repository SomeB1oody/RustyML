use crate::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::{Array, IxDyn};
use rayon::prelude::*;

/// Global Average Pooling 2D Layer
///
/// Performs global average pooling operation on the spatial dimensions (height and width) of the input tensor.
/// Input tensor shape should be `[batch_size, channels, height, width]`,
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
/// // Add a GlobalAveragePooling2D layer
/// model.add(GlobalAveragePooling2D::new());
///
/// // Create test input tensor: [batch_size, channels, height, width]
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
pub struct GlobalAveragePooling2D {
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl GlobalAveragePooling2D {
    /// Creates a new instance of global average pooling layer
    ///
    /// # Returns
    ///
    /// A new `GlobalAveragePooling2D` instance
    pub fn new() -> Self {
        GlobalAveragePooling2D {
            input_shape: Vec::new(),
            input_cache: None,
        }
    }
}

impl Layer for GlobalAveragePooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input shape and cache input for backpropagation
        self.input_shape = input.shape().to_vec();
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let spatial_size = (height * width) as f32;

        // Create result array
        let mut result = Array::zeros(IxDyn(&[batch_size, channels]));

        let indices: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c)))
            .collect();

        let avgs: Vec<f32> = indices
            .par_iter()
            .map(|&(b, c)| {
                let mut sum = 0.0;
                for h in 0..height {
                    for w in 0..width {
                        sum += input[[b, c, h, w]];
                    }
                }
                sum / spatial_size
            })
            .collect();

        // Fill the calculated results into the result tensor
        for (i, &(b, c)) in indices.iter().enumerate() {
            result[[b, c]] = avgs[i];
        }

        result
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Check if there is a valid input cache
        let input = match &self.input_cache {
            Some(input) => input,
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
        let spatial_size = height * width;
        let scale_factor = 1.0 / (spatial_size as f32);

        // Create a collection containing all indices
        let indices: Vec<(usize, usize, usize, usize)> = (0..batch_size)
            .flat_map(|b| {
                (0..channels).flat_map(move |c| {
                    (0..height).flat_map(move |h| (0..width).map(move |w| (b, c, h, w)))
                })
            })
            .collect();

        // Create gradient tensor using into_par_iter().map().collect() pattern
        let grad_values: Vec<(usize, usize, usize, usize, f32)> = indices
            .into_par_iter()
            .map(|(b, c, h, w)| {
                let grad = grad_output[[b, c]] * scale_factor;
                (b, c, h, w, grad)
            })
            .collect();

        // Create and fill gradient tensor
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, channels, height, width]));

        // Fill the calculated results into the gradient tensor
        for (b, c, h, w, grad) in grad_values {
            grad_input[[b, c, h, w]] = grad;
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling2D"
    }

    fn output_shape(&self) -> String {
        if self.input_shape.is_empty() {
            return "unknown".to_string();
        }
        format!("[{}, {}]", self.input_shape[0], self.input_shape[1])
    }

    fn param_count(&self) -> usize {
        0 // GlobalAveragePooling2D layer has no trainable parameters
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
        // Return empty LayerWeight as there are no weights
        LayerWeight::Empty
    }
}
