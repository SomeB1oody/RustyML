use super::*;

/// Epsilon value for floating point precision handling
const EPSILON: f32 = 1e-8;

/// Gradient clipping value to prevent exploding gradients
const GRAD_CLIP_VALUE: f32 = 1e6;

/// Max input clipping values to prevent overflow in exp function
const INPUT_CLIP_MIN: f32 = -500.0;

/// Min input clipping values to prevent overflow in exp function
const INPUT_CLIP_MAX: f32 = 500.0;

/// Threshold for determining whether to use parallel or sequential processing (number of rows)
const SOFTMAX_PARALLEL_THRESHOLD: usize = 8;

/// Softmax activation layer.
///
/// This layer applies the Softmax activation function to the input tensor along
/// the last axis, converting raw scores (logits) into a probability distribution.
///
/// The Softmax function is defined as:
/// softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
///
/// where max(x) is subtracted for numerical stability to prevent overflow.
///
/// Softmax is commonly used in multi-class classification tasks as the final
/// activation layer, transforming the network's output into class probabilities
/// that sum to 1.
///
/// # Input Shape
///
/// Typically accepts 2D tensors of shape \[batch_size, num_classes\], though it can
/// handle higher dimensional inputs by applying softmax along the last axis for each
/// batch element.
///
/// # Output Shape
///
/// Same as input shape, with values normalized to form a probability distribution
/// along the last axis.
///
/// # Fields
///
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
/// - `output_cache` - Cached output tensor from the forward pass, used during backpropagation
///
/// # Example
///
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor with logits
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Softmax activation
/// let mut model = Sequential::new();
/// model
///     .add(Softmax::new())
///     .compile(SGD::new(0.01).unwrap(), CategoricalCrossEntropy::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be probability distributions that sum to 1.0 for each batch
/// ```
pub struct Softmax {
    input_cache: Option<Tensor>,
    output_cache: Option<Tensor>,
}

impl Softmax {
    /// Creates a new Softmax activation layer.
    ///
    /// # Returns
    ///
    /// * `Softmax` - A new `Softmax` layer instance
    pub fn new() -> Self {
        Softmax {
            input_cache: None,
            output_cache: None,
        }
    }
}

impl Layer for Softmax {
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

        // Ensure input is at least 2D for softmax computation
        let shape = input.shape();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(ModelError::InputValidationError(format!(
                "Softmax requires input with at least 2 dimensions, got shape: {:?}",
                shape
            )));
        }

        let output = input.clone();

        // Flatten to 2D: [batch, features]
        let batch_size: usize = shape[..ndim - 1].iter().product();
        let num_features = shape[ndim - 1];

        let mut output_2d = output
            .into_shape_with_order((batch_size, num_features))
            .map_err(|e| {
                ModelError::ProcessingError(format!(
                    "Failed to reshape for softmax computation: {}",
                    e
                ))
            })?;

        // Apply softmax row-wise with numerical stability
        // For each row: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        let apply_softmax = |mut row: ArrayViewMut1<f32>| {
            // Find max for numerical stability
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let clipped_max = max_val.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);

            // Compute exp(x - max) with input clipping
            row.map_inplace(|x| {
                let clipped_x = x.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);
                *x = (clipped_x - clipped_max).exp()
            });

            // Normalize by sum with epsilon for numerical stability
            let sum = row.sum().max(EPSILON);
            row.map_inplace(|x| *x /= sum);
        };

        if batch_size > SOFTMAX_PARALLEL_THRESHOLD {
            // Parallel processing for larger batches
            output_2d
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(apply_softmax);
        } else {
            // Sequential processing for smaller batches
            output_2d.axis_iter_mut(Axis(0)).for_each(apply_softmax);
        }

        // Reshape back to original shape
        let output = output_2d
            .into_shape_with_order(shape)
            .map_err(|e| {
                ModelError::ProcessingError(format!(
                    "Failed to reshape back after softmax computation: {}",
                    e
                ))
            })?
            .into_dyn();

        // Cache output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        match (&self.input_cache, &self.output_cache) {
            (Some(input), Some(output)) => {
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

                let shape = input.shape();
                let ndim = shape.len();
                let batch_size: usize = shape[..ndim - 1].iter().product();
                let num_features = shape[ndim - 1];

                // Flatten to 2D for computation
                let output_2d = output
                    .clone()
                    .into_shape_with_order((batch_size, num_features))
                    .map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to reshape output for backward: {}",
                            e
                        ))
                    })?;

                let grad_output_2d = grad_output
                    .clone()
                    .into_shape_with_order((batch_size, num_features))
                    .map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to reshape grad_output for backward: {}",
                            e
                        ))
                    })?;

                // Compute gradient using the softmax derivative
                // For softmax: grad_input[i] = output[i] * (grad_output[i] - sum_j(output[j] * grad_output[j]))
                let mut grad_input_2d = Array2::<f32>::zeros((batch_size, num_features));

                let compute_gradient =
                    |mut grad_row: ArrayViewMut1<f32>,
                     out_row: ArrayView1<f32>,
                     grad_out_row: ArrayView1<f32>| {
                        // Compute dot product: sum(output * grad_output)
                        let dot: f32 = out_row
                            .iter()
                            .zip(grad_out_row.iter())
                            .map(|(&o, &g)| o * g)
                            .sum();

                        // Apply softmax gradient formula with gradient clipping
                        for j in 0..num_features {
                            grad_row[j] = out_row[j] * (grad_out_row[j] - dot);

                            // Apply gradient clipping to prevent exploding gradients
                            if grad_row[j].is_nan() || grad_row[j].is_infinite() {
                                grad_row[j] = 0.0;
                            } else {
                                grad_row[j] = grad_row[j].clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
                            }
                        }
                    };

                if batch_size > SOFTMAX_PARALLEL_THRESHOLD {
                    // Parallel processing for larger batches
                    Zip::from(grad_input_2d.axis_iter_mut(Axis(0)))
                        .and(output_2d.axis_iter(Axis(0)))
                        .and(grad_output_2d.axis_iter(Axis(0)))
                        .par_for_each(compute_gradient);
                } else {
                    // Sequential processing for smaller batches
                    Zip::from(grad_input_2d.axis_iter_mut(Axis(0)))
                        .and(output_2d.axis_iter(Axis(0)))
                        .and(grad_output_2d.axis_iter(Axis(0)))
                        .for_each(compute_gradient);
                }

                // Reshape back to original shape
                let grad_input = grad_input_2d
                    .into_shape_with_order(shape)
                    .map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to reshape grad_input back: {}",
                            e
                        ))
                    })?
                    .into_dyn();

                Ok(grad_input)
            }
            _ => Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            )),
        }
    }

    fn layer_type(&self) -> &str {
        "Softmax"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.input_cache)
    }

    no_trainable_parameters_layer_functions!();
}

impl ActivationLayer for Softmax {}
