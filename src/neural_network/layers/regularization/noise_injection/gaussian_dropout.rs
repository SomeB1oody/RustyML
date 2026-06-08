use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::regularization::validation::{
    validate_input_shape, validate_rate_exclusive,
};
use crate::neural_network::traits::Layer;
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
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
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
    /// - `Result<Self, Error>` - New GaussianDropout layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not in range [0, 1)
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, Error> {
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
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is immutable and already validated in `new()`; only validate the runtime input.
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

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is immutable and already validated in `new()`; only validate the runtime input.
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Inference is identity: pass input through unchanged (no noise sampling).
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Gradient passes through unchanged
        Ok(grad_output.clone())
    }

    fn layer_type(&self) -> &str {
        "GaussianDropout"
    }

    fn output_shape(&self) -> String {
        if self.input_shape.is_empty() {
            String::from("Unknown")
        } else {
            format!(
                "({})",
                self.input_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
