use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::regularization_layer::input_validation_function::{
    validate_input_shape, validate_stddev,
};
use crate::neural_network::neural_network_trait::Layer;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

/// Gaussian Noise layer for neural networks.
///
/// Adds random noise sampled from a normal distribution with mean 0 during training
/// to improve robustness and reduce overfitting.
///
/// # Fields
///
/// - `stddev` - Standard deviation of the Gaussian noise to add
/// - `input_shape` - Expected shape of the input tensor
/// - `training` - Whether the layer is in training mode or inference mode
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::neural_network_trait::Layer;
/// use ndarray::Array2;
///
/// // Create a GaussianNoise layer with standard deviation of 0.1
/// let mut noise_layer = GaussianNoise::new(0.1, vec![32, 128]).unwrap();
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
    /// - `stddev` - Standard deviation of the Gaussian noise, must be non-negative
    /// - `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New GaussianNoise layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `stddev` is negative
    pub fn new(stddev: f32, input_shape: Vec<usize>) -> Result<Self, ModelError> {
        validate_stddev(stddev)?;

        Ok(GaussianNoise {
            stddev,
            input_shape,
            training: true,
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for GaussianNoise {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        validate_stddev(self.stddev)?;
        validate_input_shape(input.shape(), &self.input_shape)?;

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
