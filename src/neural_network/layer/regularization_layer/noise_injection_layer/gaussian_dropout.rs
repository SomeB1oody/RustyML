use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::regularization_layer::input_validation_function::{
    validate_input_shape, validate_rate_exclusive,
};
use crate::neural_network::neural_network_trait::Layer;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

/// Gaussian Dropout layer for neural networks.
///
/// Multiplies inputs with random samples drawn from a Gaussian distribution during
/// training. Each input value x is transformed as: x' = x * N(1, stddev^2).
/// The standard deviation is computed as sqrt(rate / (1 - rate)). During inference,
/// inputs pass through unchanged.
///
/// # Fields
///
/// - `rate` - Dropout rate used to compute standard deviation
/// - `input_shape` - Expected shape of the input tensor
/// - `training` - Whether the layer is in training mode or inference mode
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::neural_network_trait::Layer;
/// use ndarray::Array2;
///
/// // Create a GaussianDropout layer with dropout rate of 0.3
/// let mut gaussian_dropout = GaussianDropout::new(0.3, vec![32, 128]).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, values will be multiplied by Gaussian noise N(1, sqrt(rate/(1-rate)))
/// let output = gaussian_dropout.forward(&input).unwrap();
/// ```
pub struct GaussianDropout {
    rate: f32,
    input_shape: Vec<usize>,
    training: bool,
}

impl GaussianDropout {
    /// Creates a new GaussianDropout layer.
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate, must be between 0 and 1 (exclusive)
    /// - `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New GaussianDropout layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `rate` is not in range [0, 1)
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, ModelError> {
        validate_rate_exclusive(rate, "Dropout rate")?;

        Ok(GaussianDropout {
            rate,
            input_shape,
            training: true,
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for GaussianDropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        validate_rate_exclusive(self.rate, "Dropout rate")?;
        validate_input_shape(input.shape(), &self.input_shape)?;

        // During inference or when rate is 0, pass input through unchanged
        if !self.training || self.rate == 0.0 {
            return Ok(input.clone());
        }

        // Compute standard deviation for Gaussian noise
        // The formula stddev = sqrt(rate / (1 - rate)) ensures that:
        // - When rate = 0, stddev = 0 (no noise)
        // - As rate approaches 1, stddev increases (more noise)
        let stddev = (self.rate / (1.0 - self.rate)).sqrt();

        // Generate multiplicative Gaussian noise with mean=1 and computed stddev
        let noise = Tensor::random(input.raw_dim(), Normal::new(1.0, stddev).unwrap());

        // Multiply input by noise
        let output = input * &noise;

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Gradient passes through unchanged
        Ok(grad_output.clone())
    }

    fn layer_type(&self) -> &str {
        "GaussianDropout"
    }

    fn output_shape(&self) -> String {
        format!("{:?}", self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
