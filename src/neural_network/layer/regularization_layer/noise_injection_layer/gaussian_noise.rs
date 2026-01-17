use super::*;

/// Gaussian Noise layer for neural networks, which adds random Gaussian (normal) noise
/// to the input during training time, helping prevent overfitting and improving robustness.
///
/// Gaussian noise injection is a regularization technique that adds random noise sampled
/// from a normal distribution with mean 0 and a specified standard deviation. This forces
/// the network to be more robust to small variations in the input.
///
/// # Fields
///
/// - `stddev` - Standard deviation of the Gaussian noise to be added.
/// - `input_shape` - Shape of the input tensor.
/// - `training` - Whether the layer is in training mode or inference mode.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a GaussianNoise layer with standard deviation of 0.1
/// let mut noise_layer = GaussianNoise::new(0.1, vec![32, 128]);
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, Gaussian noise with stddev=0.1 will be added
/// let output = noise_layer.forward(&input).unwrap();
/// ```
pub struct GaussianNoise {
    stddev: f32,
    input_shape: Vec<usize>,
    training: bool,
}

impl GaussianNoise {
    /// Creates a new GaussianNoise layer.
    ///
    /// # Parameters
    ///
    /// - `stddev` - Standard deviation of the Gaussian noise. Must be non-negative.
    /// - `input_shape` - Shape of the input tensor.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the GaussianNoise layer.
    ///
    /// # Panics
    ///
    /// Panics if stddev is negative.
    pub fn new(stddev: f32, input_shape: Vec<usize>) -> Self {
        assert!(
            stddev >= 0.0,
            "Standard deviation must be non-negative, got {}",
            stddev
        );
        GaussianNoise {
            stddev,
            input_shape,
            training: true,
        }
    }

    mode_dependent_layer_set_training!();
}

impl Layer for GaussianNoise {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate standard deviation
        if self.stddev < 0.0 {
            return Err(ModelError::InputValidationError(
                "Standard deviation cannot be negative".to_string(),
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

        // During inference or when stddev is 0, pass input through unchanged
        if !self.training || self.stddev == 0.0 {
            return Ok(input.clone());
        }

        // Generate random Gaussian noise with mean=0 and stddev=self.stddev
        let noise = Tensor::random(input.raw_dim(), Normal::new(0.0, self.stddev).unwrap());

        // Add noise to input
        let output = input + &noise;

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Gradient passes through unchanged since d/dx(x + noise) = 1
        // The noise is not a function of the input during backpropagation
        Ok(grad_output.clone())
    }

    fn layer_type(&self) -> &str {
        "GaussianNoise"
    }

    fn output_shape(&self) -> String {
        format!("{:?}", self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
