use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    validate_input_shape, validate_stddev,
};
use crate::neural_network::traits::Layer;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
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
/// - `rng` - Random number generator used to sample the Gaussian noise
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a GaussianNoise layer with standard deviation of 0.1
/// let mut noise_layer = GaussianNoise::new(0.1, vec![32, 128], None).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, Gaussian noise with stddev=0.1 will be added
/// let output = noise_layer.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct GaussianNoise {
    stddev: f32,
    input_shape: Vec<usize>,
    training: bool,
    rng: StdRng,
}

impl GaussianNoise {
    /// Creates a new GaussianNoise layer.
    ///
    /// # Parameters
    ///
    /// - `stddev` - Standard deviation of the Gaussian noise, must be non-negative
    /// - `input_shape` - Shape of the input tensor
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random.
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New GaussianNoise layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `stddev` is negative
    pub fn new(
        stddev: f32,
        input_shape: Vec<usize>,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        validate_stddev(stddev)?;

        let rng = crate::random::make_rng(random_state);

        Ok(GaussianNoise {
            stddev,
            input_shape,
            training: true,
            rng,
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for GaussianNoise {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `stddev` is immutable and already validated in `new()`; only validate the runtime input.
        validate_input_shape(input.shape(), &self.input_shape)?;

        // During inference or when stddev is 0, pass input through unchanged
        if !self.training || self.stddev == 0.0 {
            return Ok(input.clone());
        }

        // Generate random Gaussian noise with mean=0 and stddev=self.stddev
        let noise = Tensor::random_using(
            input.raw_dim(),
            Normal::new(0.0, self.stddev).unwrap(),
            &mut self.rng,
        );

        // Add noise to input
        let output = input + &noise;

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `stddev` is immutable and already validated in `new()`; only validate the runtime input.
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Inference is identity: pass input through unchanged without sampling noise.
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Gradient passes through unchanged since d/dx(x + noise) = 1
        // The noise is not a function of the input during backpropagation
        Ok(grad_output.clone())
    }

    fn layer_type(&self) -> &str {
        "GaussianNoise"
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
