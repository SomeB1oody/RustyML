//! Gaussian noise regularization layer that injects zero-mean normal noise during training

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::validate_stddev;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    build_on_forward, built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::Layer;
use crate::neural_network::{Shape, Tensor};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Normal;

/// Gaussian noise layer for neural networks
///
/// Adds random noise sampled from a normal distribution with mean 0 during training
/// to improve robustness and reduce overfitting
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
/// // GaussianNoise layer with standard deviation 0.1
/// let mut noise_layer = GaussianNoise::new(0.1).unwrap();
///
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, the layer adds Gaussian noise with stddev=0.1
/// let output = noise_layer.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct GaussianNoise {
    /// Standard deviation of the Gaussian noise to add
    stddev: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
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
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New GaussianNoise layer instance or a validation error
    ///
    /// # Notes
    ///
    /// By default, `new` seeds the noise RNG from the global seed or entropy. For reproducible
    /// noise, set a seed with [`GaussianNoise::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `stddev` is negative
    pub fn new(stddev: f32) -> Result<Self, Error> {
        validate_stddev(stddev)?;

        let rng = crate::random::make_rng(None);

        Ok(GaussianNoise {
            stddev,
            built: None,
            training: true,
            rng,
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

impl Layer for GaussianNoise {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    ///
    /// The recorded shape is what [`Layer::output_shape`] reports, and no more. The layer owns
    /// no array and reads no extent, so it checks no later input against it. See the
    /// "Shape freedom" section of the type
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "GaussianNoise", input)? else {
            return Ok(());
        };
        input.check_min_rank("GaussianNoise", 1)?;
        self.built = Some(built);
        Ok(())
    }

    /// Adds noise to a tensor of any shape. See the "Shape freedom" section of the type
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `stddev` was validated in `new()`, and the input needs no check
        build_on_forward!(self, input);

        // During inference or when stddev is 0, pass input through unchanged
        if !self.training || self.stddev == 0.0 {
            return Ok(input.clone());
        }

        // Sample mean-0 Gaussian noise and add it to the input
        let noise = Tensor::random_using(
            input.raw_dim(),
            Normal::new(0.0, self.stddev).unwrap(),
            &mut self.rng,
        );
        let output = input + &noise;

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    ///
    /// The input needs no check, and the layer takes a tensor of any shape. `predict` cannot
    /// build, so it still refuses a layer that holds no build
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `stddev` was validated in `new()`, and the input needs no check
        if self.built.is_none() {
            return Err(Error::not_built("GaussianNoise"));
        }

        // Inference is identity: pass input through without sampling noise
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // The gradient passes through unchanged, since noise does not depend on x and
        // d/dx(x + noise) = 1
        Ok(grad_output.clone())
    }

    fn layer_type(&self) -> &str {
        "GaussianNoise"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
