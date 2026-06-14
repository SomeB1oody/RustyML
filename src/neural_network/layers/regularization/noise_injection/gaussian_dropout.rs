//! Gaussian Dropout layer: multiplicative Gaussian noise applied during training

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    validate_input_shape, validate_rate_exclusive,
};
use crate::neural_network::traits::Layer;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Normal;

/// Gaussian Dropout layer for neural networks
///
/// Multiplies inputs with random samples drawn from a Gaussian distribution during
/// training. Each input value x is transformed as x' = x * N(1, stddev^2), where the
/// standard deviation is sqrt(rate / (1 - rate)). During inference, inputs pass through
/// unchanged
///
/// # Examples
///
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
/// // During training, values are multiplied by Gaussian noise N(1, sqrt(rate/(1-rate)))
/// let output = gaussian_dropout.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct GaussianDropout {
    /// Dropout rate used to compute the noise standard deviation
    rate: f32,
    /// Expected shape of the input tensor
    input_shape: Vec<usize>,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Random number generator used to sample the multiplicative Gaussian noise
    rng: StdRng,
    /// Multiplicative noise sampled during the forward pass, cached so backward can reuse the
    /// exact same draw (y = x * noise => dx = grad * noise)
    noise_cache: Option<Tensor>,
}

impl GaussianDropout {
    /// Creates a new GaussianDropout layer
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate, must be in range [0, 1)
    /// - `input_shape` - Shape of the input tensor
    ///
    /// # Notes
    ///
    /// The noise RNG is seeded from the global seed or entropy by default. For reproducible noise,
    /// set a seed with [`GaussianDropout::with_random_state`]
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

        let rng = crate::random::make_rng(None);

        Ok(GaussianDropout {
            rate,
            input_shape,
            training: true,
            rng,
            noise_cache: None,
        })
    }

    /// Sets the seed for reproducible noise sampling
    ///
    /// By default the RNG is seeded from the global seed or entropy (see [`crate::random`]). This
    /// re-seeds it deterministically from `random_state`
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

impl Layer for GaussianDropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is immutable and already validated in `new()`; only validate the runtime input
        validate_input_shape(input.shape(), &self.input_shape)?;

        // During inference or when rate is 0, pass input through unchanged
        if !self.training || self.rate == 0.0 {
            return Ok(input.clone());
        }

        // Noise stddev grows from 0 (rate = 0) toward inf as rate approaches 1
        let stddev = (self.rate / (1.0 - self.rate)).sqrt();

        // Multiplicative Gaussian noise with mean 1 and the computed stddev
        let noise = Tensor::random_using(
            input.raw_dim(),
            Normal::new(1.0, stddev).unwrap(),
            &mut self.rng,
        );

        let output = input * &noise;

        // Cache the exact draw so backward can apply the same multiplier
        self.noise_cache = Some(noise);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is immutable and already validated in `new()`; only validate the runtime input
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Inference is identity: pass input through unchanged (no noise sampling)
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // During inference or zero rate, forward was the identity, so pass the gradient through
        if !self.training || self.rate == 0.0 {
            return Ok(grad_output.clone());
        }

        // y = x * noise => dL/dx = dL/dy * noise, reusing the exact draw from the forward pass
        match self.noise_cache.as_ref() {
            Some(noise) => Ok(grad_output * noise),
            None => Err(Error::forward_pass_not_run("GaussianDropout")),
        }
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
