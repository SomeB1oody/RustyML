use super::*;

/// Epsilon value for floating point precision handling
const EPSILON: f32 = 1e-8;

/// Gradient clipping value to prevent exploding gradients
const GRAD_CLIP_VALUE: f32 = 1e6;

/// Threshold for parallel computation (in number of elements)
/// For tensors with fewer elements, sequential computation is faster due to overhead
const RELU_PARALLEL_THRESHOLD: usize = 10_000;

/// ReLU (Rectified Linear Unit) activation layer.
///
/// Applies `max(0, x)` element-wise to the input tensor, keeping the original shape.
/// Common inputs include 2D tensors for dense layers and 4D tensors for convolutional layers.
///
/// # Fields
///
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with ReLU activation
/// let mut model = Sequential::new();
/// model
///     .add(ReLU::new())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be: [[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]]
/// ```
pub struct ReLU {
    input_cache: Option<Tensor>,
}

impl ReLU {
    /// Creates a new ReLU activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `ReLU` layer instance
    pub fn new() -> Self {
        ReLU { input_cache: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Check if tensor is empty
        if input.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input tensor is empty".to_string(),
            ));
        }

        // Check for NaN or infinite values
        if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(ModelError::InputValidationError(
                "Input tensor contains NaN or infinite values".to_string(),
            ));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Apply ReLU: max(0, x) with epsilon for precision handling
        // Use parallel computation for large tensors, sequential for small ones
        let mut output = input.clone();

        if input.len() >= RELU_PARALLEL_THRESHOLD {
            output.par_mapv_inplace(|x| if x > EPSILON { x } else { 0.0 })
        } else {
            output.mapv_inplace(|x| if x > EPSILON { x } else { 0.0 })
        };

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            // Validate gradient output shape
            if grad_output.shape() != input.shape() {
                return Err(ModelError::ProcessingError(format!(
                    "Gradient output shape {:?} doesn't match input shape {:?}",
                    grad_output.shape(),
                    input.shape()
                )));
            }

            // Check for NaN or infinite values in gradient output
            if grad_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Err(ModelError::InputValidationError(
                    "Gradient output contains NaN or infinite values".to_string(),
                ));
            }

            // Compute gradient: grad_input = grad_output * (input > 0)
            // ReLU derivative is 1 for x > 0, and 0 for x <= 0
            // Use parallel computation for large tensors, sequential for small ones
            let mut grad_input = grad_output.clone();
            if input.len() >= RELU_PARALLEL_THRESHOLD {
                Zip::from(&mut grad_input)
                    .and(input)
                    .par_for_each(|grad, &inp| {
                        if inp <= EPSILON {
                            *grad = 0.0;
                        } else {
                            // Apply gradient clipping to prevent exploding gradients
                            *grad = grad.clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
                        }
                    });
            } else {
                Zip::from(&mut grad_input)
                    .and(input)
                    .for_each(|grad, &inp| {
                        if inp <= EPSILON {
                            *grad = 0.0;
                        } else {
                            // Apply gradient clipping to prevent exploding gradients
                            *grad = grad.clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
                        }
                    });
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "ReLU"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.input_cache)
    }

    no_trainable_parameters_layer_functions!();
}

impl ActivationLayer for ReLU {}
