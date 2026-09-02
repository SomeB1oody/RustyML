//! Gaussian Dropout layer that applies multiplicative Gaussian noise during training

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::validate_rate_exclusive;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    build_on_forward, built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::Layer;
use crate::neural_network::{Shape, Tensor};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Normal;

/// Gaussian Dropout layer for neural networks
///
/// Multiplies inputs with random samples drawn from a Gaussian distribution during
/// training. The layer transforms each input value x as x' = x * N(1, stddev^2), where the
/// standard deviation is sqrt(rate / (1 - rate)). During inference, inputs pass through
/// unchanged
///
/// # Shape freedom
///
/// The layer owns no array and reads no extent of its input. It therefore accepts a tensor of
/// any shape and of any rank. [`Layer::build`] records the shape it is given, and
/// [`Layer::output_shape`] reports it, but no later input is checked against it. See the
/// "Shape freedom" section of
/// [`Dropout`](crate::neural_network::layers::regularization::dropout::dropout::Dropout)
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a GaussianDropout layer with dropout rate of 0.3
/// let mut gaussian_dropout = GaussianDropout::new(0.3).unwrap();
///
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // The forward pass multiplies values by Gaussian noise (mean 1, variance rate/(1 - rate))
/// let output = gaussian_dropout.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct GaussianDropout {
    /// Dropout rate used to compute the noise standard deviation
    rate: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
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
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New GaussianDropout layer instance or a validation error
    ///
    /// # Notes
    ///
    /// By default, `new` seeds the noise RNG from the global seed or entropy. For reproducible
    /// noise, set a seed with [`GaussianDropout::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not in range [0, 1)
    pub fn new(rate: f32) -> Result<Self, Error> {
        validate_rate_exclusive(rate, "Dropout rate")?;

        let rng = crate::random::make_rng(None);

        Ok(GaussianDropout {
            rate,
            built: None,
            training: true,
            rng,
            noise_cache: None,
        })
    }

    /// Sets the seed for reproducible noise sampling
    ///
    /// By default, `new` seeds the RNG from the global seed or entropy (see [`crate::random`]).
    /// This method re-seeds it deterministically from `random_state`
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
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    ///
    /// The recorded shape is what [`Layer::output_shape`] reports, and no more. The layer owns
    /// no array and reads no extent, so it checks no later input against it. See the
    /// "Shape freedom" section of the type
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "GaussianDropout", input)? else {
            return Ok(());
        };
        input.check_min_rank("GaussianDropout", 1)?;
        self.built = Some(built);
        Ok(())
    }

    /// Scales a tensor of any shape by the drawn noise. See the "Shape freedom" section of the type
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` was already validated in `new()`, and the input needs no check
        build_on_forward!(self, input);

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
    ///
    /// The input needs no check, and the layer takes a tensor of any shape. `predict` cannot
    /// build, so it still refuses a layer that holds no build
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` was already validated in `new()`, and the input needs no check
        if self.built.is_none() {
            return Err(Error::not_built("GaussianDropout"));
        }

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

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
