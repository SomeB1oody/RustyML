use super::*;

/// Gradient clipping value to prevent exploding gradients
const GRAD_CLIP_VALUE: f32 = 1e6;

/// Max input clipping values to prevent overflow in tanh function
const INPUT_CLIP_MIN: f32 = -500.0;

/// Min input clipping values to prevent overflow in tanh function
const INPUT_CLIP_MAX: f32 = 500.0;

/// Threshold for using parallel computation instead of sequential
const TANH_PARALLEL_THRESHOLD: usize = 2048;

/// Tanh (Hyperbolic Tangent) activation layer.
///
/// Applies `tanh(x)` element-wise to the input tensor, mapping values to (-1, 1) while
/// preserving the input shape.
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
/// // Build a model with Tanh activation
/// let mut model = Sequential::new();
/// model
///     .add(Tanh::new())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be approximately: [[-0.76, 0.96, -0.99], [0.99, -0.99, 1.0]]
/// ```
pub struct Tanh {
    input_cache: Option<Tensor>,
}

impl Tanh {
    /// Creates a new Tanh activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Tanh` layer instance
    pub fn new() -> Self {
        Tanh { input_cache: None }
    }
}

impl Layer for Tanh {
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

        // Apply Tanh: tanh(x) with input clipping for numerical stability
        let output = if input.len() >= TANH_PARALLEL_THRESHOLD {
            let mut output = input.clone();
            Zip::from(&mut output).par_for_each(|x| {
                let clipped_x = x.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);
                *x = clipped_x.tanh();
            });
            output
        } else {
            input.mapv(|x| {
                let clipped_x = x.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);
                clipped_x.tanh()
            })
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

            // Compute gradient: grad_input = grad_output * (1 - tanh^2(input))
            // Tanh derivative is: d/dx tanh(x) = 1 - tanh^2(x) = sech^2(x)
            let mut grad_input = grad_output.clone();

            let compute_gradient = |grad: &mut f32, &inp: &f32| {
                let clipped_inp = inp.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);
                let tanh_val = clipped_inp.tanh();
                let derivative = 1.0 - tanh_val * tanh_val;
                *grad *= derivative;

                // Apply gradient clipping to prevent exploding gradients
                if grad.is_nan() || grad.is_infinite() {
                    *grad = 0.0;
                } else {
                    *grad = grad.clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
                }
            };

            if input.len() >= TANH_PARALLEL_THRESHOLD {
                Zip::from(&mut grad_input)
                    .and(input)
                    .par_for_each(compute_gradient);
            } else {
                Zip::from(&mut grad_input)
                    .and(input)
                    .for_each(compute_gradient);
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Tanh"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.input_cache)
    }

    no_trainable_parameters_layer_functions!();
}

impl ActivationLayer for Tanh {}
