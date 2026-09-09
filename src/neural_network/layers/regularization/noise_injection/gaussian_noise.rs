//! Gaussian noise regularization layer that injects zero-mean normal noise during training

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::regularization::validation::validate_stddev;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, StateSlot, Tensor};
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
/// The layer owns no array and reads no extent of its input. It therefore accepts a tensor of any
/// shape at rank 1 or higher. [`UnaryLayer::build`] records the shape it is given, and
/// [`Layer::output_shape`](crate::neural_network::traits::Layer::output_shape) reports it, but no
/// later input is checked against it. See the "Shape freedom" section of
/// [`Dropout`](crate::neural_network::layers::regularization::dropout::dropout::Dropout)
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::UnaryLayer;
/// use rustyml::neural_network::Ctx;
/// use ndarray::Array2;
///
/// // GaussianNoise layer with standard deviation 0.1
/// let mut noise_layer = GaussianNoise::new(0.1).unwrap();
///
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, the layer adds Gaussian noise with stddev=0.1
/// let mut ctx = Ctx::training();
/// let output = noise_layer.forward_mut(&input, &mut ctx).unwrap();
/// ```
#[derive(Debug)]
pub struct GaussianNoise {
    /// Standard deviation of the Gaussian noise to add
    stddev: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Random number generator used to sample the Gaussian noise
    rng: StdRng,
}

impl GaussianNoise {
    /// Creates a new GaussianNoise layer
    ///
    /// # Parameters
    ///
    /// - `stddev` - Standard deviation of the Gaussian noise, must be finite and non-negative
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
    /// - `Error::InvalidParameter` - If `stddev` is negative or not finite
    pub fn new(stddev: f32) -> Result<Self, Error> {
        validate_stddev(stddev)?;

        let rng = crate::random::make_rng(None);

        Ok(GaussianNoise {
            stddev,
            built: None,
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
}

impl LayerBase for GaussianNoise {
    fn layer_type(&self) -> &str {
        "GaussianNoise"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();

    /// Takes back the random stream that the forward pass advanced
    ///
    /// The forward pass draws from a copy that lives in the context, because it takes `&self`.
    /// This method moves that copy into the layer, so the next pass starts where the last one
    /// stopped
    fn apply_state(&mut self, state: &mut StateSlot<'_>) {
        if let Some(rng) = state.take::<StdRng>("rng") {
            self.rng = rng;
        }
    }
}

impl UnaryLayer for GaussianNoise {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    ///
    /// The recorded shape is what
    /// [`Layer::output_shape`](crate::neural_network::traits::Layer::output_shape) reports, and no
    /// more. The layer owns no array and reads no extent, so it checks no later input against it.
    /// See the "Shape freedom" section of the type
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "GaussianNoise", input)? else {
            return Ok(());
        };
        input.check_min_rank("GaussianNoise", 1)?;
        self.built = Some(built);
        Ok(())
    }

    /// Adds noise to a tensor of any shape. See the "Shape freedom" section of the type
    ///
    /// An inference pass is the identity, and it draws nothing at all
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // `stddev` was validated in `new()`, and the input needs no check
        if !self.is_built() {
            return Err(Error::not_built("GaussianNoise"));
        }

        if !ctx.is_training() || self.stddev == 0.0 {
            return Ok(input.clone());
        }

        // The stream lives in the context for the length of the pass, so the layer stays
        // read-only and the draws of 2 calls follow each other
        let mut rng = ctx
            .take_state::<StdRng>("rng")
            .unwrap_or_else(|| self.rng.clone());

        let noise = Tensor::random_using(
            input.raw_dim(),
            Normal::new(0.0, self.stddev).unwrap(),
            &mut rng,
        );
        ctx.set_state("rng", rng);

        let output = input + &noise;

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, _ctx: &mut Ctx) -> Result<Tensor, Error> {
        // The gradient passes through unchanged, since noise does not depend on x and
        // d/dx(x + noise) = 1
        Ok(grad_output.clone())
    }
}
