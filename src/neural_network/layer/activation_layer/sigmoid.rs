use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::activation_layer::format_output_shape;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::neural_network_trait::{ActivationLayer, Layer};
use ndarray::Zip;

/// Gradient clipping value to prevent exploding gradients
const GRAD_CLIP_VALUE: f32 = 1e6;

/// Max input clipping values to prevent overflow in exp function
const INPUT_CLIP_MIN: f32 = -500.0;

/// Min input clipping values to prevent overflow in exp function
const INPUT_CLIP_MAX: f32 = 500.0;

/// Threshold for using parallel computation (number of elements)
const SIGMOID_PARALLEL_THRESHOLD: usize = 1000;

/// Sigmoid activation layer.
///
/// Applies `1 / (1 + e^(-x))` element-wise to the input tensor, squashing values to (0, 1)
/// while preserving the input shape.
///
/// # Fields
///
/// - `output_cache` - Cached output tensor from the forward pass, used during backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::activation_layer::sigmoid::Sigmoid;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::MeanSquaredError;
/// use ndarray::Array2;
///
/// // Create a 2D input tensor
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// // Build a model with Sigmoid activation
/// let mut model = Sequential::new();
/// model
///     .add(Sigmoid::new())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&x);
///
/// // Output will be approximately: [[0.27, 0.88, 0.05], [0.98, 0.007, 0.998]]
/// ```
pub struct Sigmoid {
    output_cache: Option<Tensor>,
}

impl Sigmoid {
    /// Creates a new Sigmoid activation layer.
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Sigmoid` layer instance
    pub fn new() -> Self {
        Sigmoid { output_cache: None }
    }
}

impl Layer for Sigmoid {
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

        // Apply Sigmoid: 1 / (1 + e^(-x)) with input clipping for numerical stability
        let mut output = input.to_owned();

        let sigmoid_fn = |x: f32| {
            let clipped_x = x.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);
            1.0 / (1.0 + (-clipped_x).exp())
        };

        if input.len() >= SIGMOID_PARALLEL_THRESHOLD {
            output.par_mapv_inplace(sigmoid_fn);
        } else {
            output.mapv_inplace(sigmoid_fn);
        }

        // Save output for backpropagation
        self.output_cache = Some(output.clone());

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(output) = &self.output_cache {
            // Validate gradient output shape
            if grad_output.shape() != output.shape() {
                return Err(ModelError::ProcessingError(format!(
                    "Gradient output shape {:?} doesn't match output shape {:?}",
                    grad_output.shape(),
                    output.shape()
                )));
            }

            // Check for NaN or infinite values in gradient output
            if grad_output.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Err(ModelError::InputValidationError(
                    "Gradient output contains NaN or infinite values".to_string(),
                ));
            }

            // Compute gradient: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
            // Sigmoid derivative is: f'(x) = f(x) * (1 - f(x))
            let mut grad_input = grad_output.clone();

            let gradient_fn = |grad: &mut f32, &out: &f32| {
                // Compute derivative with numerical stability check
                let derivative = out * (1.0 - out);
                *grad *= derivative;

                // Apply gradient clipping to prevent exploding gradients
                if grad.is_nan() || grad.is_infinite() {
                    *grad = 0.0;
                } else {
                    *grad = grad.clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
                }
            };

            if grad_output.len() >= SIGMOID_PARALLEL_THRESHOLD {
                Zip::from(&mut grad_input)
                    .and(output)
                    .par_for_each(gradient_fn);
            } else {
                Zip::from(&mut grad_input).and(output).for_each(gradient_fn);
            }

            Ok(grad_input)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Sigmoid"
    }

    fn output_shape(&self) -> String {
        format_output_shape(&self.output_cache)
    }

    no_trainable_parameters_layer_functions!();
}

impl ActivationLayer for Sigmoid {}
