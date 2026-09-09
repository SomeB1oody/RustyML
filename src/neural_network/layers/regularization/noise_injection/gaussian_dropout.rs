//! Gaussian Dropout layer that applies multiplicative Gaussian noise during training

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::regularization::validation::validate_rate_exclusive;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, StateSlot, Tensor};
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
/// // Create a GaussianDropout layer with dropout rate of 0.3
/// let mut gaussian_dropout = GaussianDropout::new(0.3).unwrap();
///
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // The forward pass multiplies values by Gaussian noise (mean 1, variance rate/(1 - rate))
/// let mut ctx = Ctx::training();
/// let output = gaussian_dropout.forward_mut(&input, &mut ctx).unwrap();
/// ```
#[derive(Debug)]
pub struct GaussianDropout {
    /// Dropout rate used to compute the noise standard deviation
    rate: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Random number generator used to sample the multiplicative Gaussian noise
    rng: StdRng,
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

impl LayerBase for GaussianDropout {
    fn layer_type(&self) -> &str {
        "GaussianDropout"
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

impl UnaryLayer for GaussianDropout {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    ///
    /// The recorded shape is what
    /// [`Layer::output_shape`](crate::neural_network::traits::Layer::output_shape) reports, and no
    /// more. The layer owns no array and reads no extent, so it checks no later input against it.
    /// See the "Shape freedom" section of the type
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "GaussianDropout", input)? else {
            return Ok(());
        };
        input.check_min_rank("GaussianDropout", 1)?;
        self.built = Some(built);
        Ok(())
    }

    /// Scales a tensor of any shape by the drawn noise. See the "Shape freedom" section of the type
    ///
    /// An inference pass is the identity, and it draws nothing at all
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // `rate` was validated in `new()`, and the input needs no check
        if !self.is_built() {
            return Err(Error::not_built("GaussianDropout"));
        }

        if !ctx.is_training() || self.rate == 0.0 {
            return Ok(input.clone());
        }

        // Noise stddev grows from 0 at rate 0 toward infinity as rate approaches 1
        let stddev = (self.rate / (1.0 - self.rate)).sqrt();

        // The stream lives in the context for the length of the pass, so the layer stays
        // read-only and the draws of 2 calls follow each other
        let mut rng = ctx
            .take_state::<StdRng>("rng")
            .unwrap_or_else(|| self.rng.clone());

        let noise =
            Tensor::random_using(input.raw_dim(), Normal::new(1.0, stddev).unwrap(), &mut rng);
        ctx.set_state("rng", rng);

        let output = input * &noise;

        // Park the exact draw so backward can apply the same multiplier
        ctx.push_cache("GaussianDropout", noise);

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // During inference or zero rate, forward was the identity, so pass the gradient through
        if !ctx.is_training() || self.rate == 0.0 {
            return Ok(grad_output.clone());
        }

        // y = x * noise => dL/dx = dL/dy * noise, reusing the exact draw from the forward pass
        let noise: Tensor = ctx.pop_cache("GaussianDropout")?;
        Ok(grad_output * &noise)
    }
}
