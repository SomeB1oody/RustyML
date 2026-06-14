//! Gaussian noise regularization layer that injects zero-mean normal noise during training

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

/// Gaussian noise layer for neural networks
///
/// Adds random noise sampled from a normal distribution with mean 0 during training
/// to improve robustness and reduce overfitting
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
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
#[derive(Debug)]
pub struct GaussianNoise {
    /// Standard deviation of the Gaussian noise to add
    stddev: f32,
    /// Expected shape of the input tensor
    input_shape: Vec<usize>,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Random number generator used to sample the Gaussian noise
    rng: StdRng,
}

impl GaussianNoise {
    /// Creates a new GaussianNoise layer
    ///
    /// # Parameters
    ///
    /// - `stddev` - Standard deviation of the Gaussian noise, must be non-negative
    /// - `input_shape` - Shape of the input tensor
    ///
    /// # Notes
    ///
    /// The noise RNG is seeded from the global seed or entropy by default. For reproducible noise,
    /// set a seed with [`GaussianNoise::with_random_state`].
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New GaussianNoise layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `stddev` is negative
    pub fn new(stddev: f32, input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_stddev(stddev)?;

        let rng = crate::random::make_rng(None);

        Ok(GaussianNoise {
            stddev,
            input_shape,
            training: true,
            rng,
        })
    }

    /// Sets the seed for reproducible noise sampling
    ///
    /// By default the RNG is seeded from the global seed or entropy (see [`crate::random`]). This
    /// re-seeds it deterministically from `random_state`.
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for the layer's random number generator
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.rng = crate::random::make_rng(Some(random_state));
        self
    }

    mode_dependent_layer_set_training!();
}

impl Layer for GaussianNoise {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `stddev` is immutable and already validated in `new()`; only validate the runtime input
        validate_input_shape(input.shape(), &self.input_shape)?;

        // During inference or when stddev is 0, pass input through unchanged
        if !self.training || self.stddev == 0.0 {
            return Ok(input.clone());
        }

        // Sample mean-0, stddev=self.stddev Gaussian noise and add it to the input
        let noise = Tensor::random_using(
            input.raw_dim(),
            Normal::new(0.0, self.stddev).unwrap(),
            &mut self.rng,
        );
        let output = input + &noise;

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches), see [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `stddev` is immutable and already validated in `new()`; only validate the runtime input
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Inference is identity: pass input through unchanged without sampling noise
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Gradient passes through unchanged since d/dx(x + noise) = 1 (noise is independent of the input)
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
