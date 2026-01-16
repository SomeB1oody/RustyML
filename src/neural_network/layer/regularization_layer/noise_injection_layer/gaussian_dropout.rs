use super::*;
use rand_distr::Normal;

/// Gaussian Dropout layer for neural networks, which multiplies inputs with random samples
/// drawn from a Gaussian (normal) distribution during training time.
///
/// Unlike standard dropout which randomly sets units to zero, Gaussian dropout multiplies
/// each input by a random value sampled from a Gaussian distribution with mean 1 and
/// configurable standard deviation. This provides a continuous form of dropout regularization
/// that can lead to better generalization performance.
///
/// During training, each input value x is transformed as: x' = x * N(1, stddev�)
/// During inference, inputs pass through unchanged.
///
/// # Fields
///
/// - `rate` - Dropout rate used to compute standard deviation. Higher rate means more noise.
/// - `input_shape` - Shape of the input tensor.
/// - `training` - Whether the layer is in training mode or inference mode.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a GaussianDropout layer with dropout rate of 0.3
/// let mut gaussian_dropout = GaussianDropout::new(0.3, vec![32, 128]);
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
    /// - `rate` - Dropout rate, must be between 0 and 1. The standard deviation of the
    ///   multiplicative Gaussian noise will be computed as sqrt(rate / (1 - rate)).
    /// - `input_shape` - Shape of the input tensor.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the GaussianDropout layer.
    ///
    /// # Panics
    ///
    /// Panics if rate is not between 0 and 1.
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Self {
        assert!(
            rate >= 0.0 && rate < 1.0,
            "Dropout rate must be in [0, 1), got {}",
            rate
        );
        GaussianDropout {
            rate,
            input_shape,
            training: true,
        }
    }

    mode_dependent_layer_set_training!();
}

impl Layer for GaussianDropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate dropout rate
        if self.rate < 0.0 || self.rate >= 1.0 {
            return Err(ModelError::InputValidationError(
                "Dropout rate must be in range [0, 1)".to_string(),
            ));
        }

        // Validate input shape matches expected shape
        if !self.input_shape.is_empty() && input.shape() != self.input_shape.as_slice() {
            return Err(ModelError::InputValidationError(format!(
                "Input shape mismatch: expected {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )));
        }

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
