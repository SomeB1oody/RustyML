//! Dropout layer that randomly zeroes input units during training to reduce overfitting

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::regularization::dropout::{
    broadcast_dropout_scale, dropout_backward,
};
use crate::neural_network::layers::regularization::validation::validate_rate;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, StateSlot, Tensor};
use crate::parallel_gates::cheap_map_parallel_threshold;
use ndarray::IxDyn;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Dropout layer for neural networks
///
/// Randomly drops a fraction of input units during training to reduce overfitting and
/// improve generalization
///
/// By default the layer draws 1 independent value per input element. [`Dropout::with_noise_shape`]
/// changes the shape of the drawn mask, so an axis can carry 1 shared draw instead of 1 draw per
/// position
///
/// # Shape freedom
///
/// The layer owns no array and reads no extent of its input. It therefore accepts a tensor of
/// any shape and of any rank, and 1 layer serves a rank-2 batch of feature vectors and a
/// rank-4 batch of images alike. [`UnaryLayer::build`] records the shape it is given, and
/// [`Layer::output_shape`](crate::neural_network::traits::Layer::output_shape) reports it, but no later input is checked against it. The 2 noise
/// layers,
/// [`GaussianNoise`](crate::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise)
/// and
/// [`GaussianDropout`](crate::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout),
/// take the same freedom. The 3 spatial dropout layers do not, because each of them reads a
/// channel axis at a fixed position and needs the rank that puts it there
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::UnaryLayer;
/// use rustyml::neural_network::Ctx;
/// use ndarray::Array2;
///
/// // Create a Dropout layer with 50% dropout rate
/// let mut dropout = Dropout::new(0.5).unwrap();
///
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, about 50% of values are set to 0
/// let mut ctx = Ctx::training();
/// let output = dropout.forward_mut(&input, &mut ctx).unwrap();
/// ```
#[derive(Debug)]
pub struct Dropout {
    /// Fraction of input units to drop (between 0 and 1)
    rate: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Shape of the random mask, or `None` for 1 independent draw per input element
    ///
    /// An entry of `None` takes the extent of the input on that axis, and an entry of 1 makes
    /// the axis share 1 draw. A shorter vector lines up against the last axes of the input
    noise_shape: Option<Vec<Option<usize>>>,
    /// Random number generator used to sample the dropout mask
    rng: StdRng,
}

impl Dropout {
    /// Creates a new Dropout layer
    ///
    /// # Parameters
    ///
    /// - `rate` - Fraction of the input units to drop (between 0 and 1)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New Dropout layer instance, or a validation error
    ///
    /// # Notes
    ///
    /// By default, `new` seeds the mask RNG from the global seed or entropy. For reproducible
    /// masks, set a seed with [`Dropout::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not between 0 and 1
    pub fn new(rate: f32) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(Dropout {
            rate,
            built: None,
            noise_shape: None,
            rng: crate::random::make_rng(None),
        })
    }

    /// Sets the shape of the random mask, which then broadcasts up to the input
    ///
    /// By default the layer draws 1 independent value per input element. A `noise_shape` makes
    /// the draw coarser. An entry of 1 gives the whole axis 1 shared draw, so the same units
    /// drop at every position of that axis. An entry of `None` takes the extent of the input on
    /// that axis, which keeps the draws on that axis independent. Any other entry must equal the
    /// extent of the input on that axis
    ///
    /// A vector that is shorter than the rank of the input lines up against the last axes,
    /// which is the usual right-aligned broadcast rule. The axes that the vector leaves out
    /// therefore share 1 draw. A caller that omits the batch axis shares 1 mask across the whole
    /// batch
    ///
    /// The layer samples the mask at this shape, keeps it at this shape for the backward pass,
    /// and broadcasts it in both directions. It never builds a full-size copy
    ///
    /// # Parameters
    ///
    /// - `noise_shape` - Shape of the random mask, 1 entry per axis
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `noise_shape` is empty
    /// - `Error::InvalidParameter` - If any entry is `Some(0)`
    pub fn with_noise_shape(mut self, noise_shape: Vec<Option<usize>>) -> Result<Self, Error> {
        if noise_shape.is_empty() {
            return Err(Error::empty_input("noise_shape"));
        }
        if let Some(axis) = noise_shape.iter().position(|entry| *entry == Some(0)) {
            return Err(Error::invalid_parameter(
                "noise_shape",
                format!(
                    "every entry must be positive or None, got 0 at axis {}",
                    axis
                ),
            ));
        }

        self.noise_shape = Some(noise_shape);
        Ok(self)
    }

    /// Sets the seed for reproducible mask sampling
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

    /// Resolves the configured noise shape against the shape of an actual input
    ///
    /// Without a `noise_shape` the answer is the input shape itself, which gives 1 draw per
    /// input element. With one, the vector lines up against the last axes of the input, and
    /// each entry becomes the input extent (`None`), 1, or the input extent again
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If the noise shape has a higher rank than the input, or if
    ///   an entry is neither 1 nor the extent of the input on its axis
    fn resolve_noise_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, Error> {
        let Some(noise_shape) = &self.noise_shape else {
            return Ok(input_shape.to_vec());
        };

        if noise_shape.len() > input_shape.len() {
            return Err(Error::invalid_parameter(
                "noise_shape",
                format!(
                    "has rank {}, which is more than the input rank {}",
                    noise_shape.len(),
                    input_shape.len()
                ),
            ));
        }

        // Right-aligned against the input, so the leading axes that the caller left out keep an
        // extent of 1 and share 1 draw
        let leading = input_shape.len() - noise_shape.len();
        let mut resolved = Vec::with_capacity(noise_shape.len());
        for (index, entry) in noise_shape.iter().enumerate() {
            let extent = input_shape[leading + index];
            match *entry {
                None => resolved.push(extent),
                Some(1) => resolved.push(1),
                Some(size) if size == extent => resolved.push(size),
                Some(size) => {
                    return Err(Error::invalid_parameter(
                        "noise_shape",
                        format!(
                            "entry {} at axis {} of the input must be 1, {}, or None",
                            size,
                            leading + index,
                            extent
                        ),
                    ));
                }
            }
        }
        Ok(resolved)
    }
}

impl LayerBase for Dropout {
    fn layer_type(&self) -> &str {
        "Dropout"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();

    /// Takes back the random stream that the forward pass advanced
    ///
    /// The stream is the only state of the layer that a forward pass changes. A training pass
    /// leaves the advanced stream in the context, and this moves it into the layer, so the next
    /// pass draws the values that follow
    fn apply_state(&mut self, state: &mut StateSlot<'_>) {
        if let Some(rng) = state.take::<StdRng>("rng") {
            self.rng = rng;
        }
    }
}

impl UnaryLayer for Dropout {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    ///
    /// The recorded shape is what [`Layer::output_shape`](crate::neural_network::traits::Layer::output_shape) reports, and no more. The layer owns
    /// no array and reads no extent, so it checks no later input against it. See the
    /// "Shape freedom" section of the type
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Dropout", input)? else {
            return Ok(());
        };
        input.check_min_rank("Dropout", 1)?;
        self.built = Some(built);
        Ok(())
    }

    /// Drops units of a tensor of any shape. See the "Shape freedom" section of the type
    ///
    /// An inference pass draws nothing and passes the input through unchanged, which is what
    /// inverted dropout needs. The input needs no check, and the layer takes a tensor of any
    /// shape, but the pass still refuses a layer that holds no build
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // `rate` was validated in `new()`
        if !self.is_built() {
            return Err(Error::not_built("Dropout"));
        }

        if !ctx.is_training() {
            // Inference passes the input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            return Ok(input.clone());
        }

        // The mask shape, which is the input shape unless a noise shape makes it coarser
        let noise_shape = self.resolve_noise_shape(input.shape())?;

        if self.rate == 1.0 {
            // Dropping every unit yields all zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        // The stream of the pass, which starts from the stream of the layer. The draw below
        // advances the copy in the context, and `apply_state` moves it back into the layer
        let mut rng = ctx
            .take_state::<StdRng>("rng")
            .unwrap_or_else(|| self.rng.clone());

        // Sample a uniform value per draw. With no noise shape this is 1 value per input
        // element. With one, the sampler runs at the smaller shape, so an axis of extent 1
        // takes 1 draw that every position of that axis then shares
        let mut mask = Tensor::random_using(
            IxDyn(&noise_shape),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut rng,
        );
        ctx.set_state("rng", rng);

        // Threshold into a binary mask, in parallel for large masks
        if mask.len() >= cheap_map_parallel_threshold() {
            mask.par_mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        } else {
            mask.mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        }

        // Inverted dropout: scale kept units by 1 / (1 - rate) to preserve the expected value
        let output = broadcast_dropout_scale(input, &mask, self.rate)?;

        // Park the mask, at its own shape, for backpropagation
        ctx.push_cache(mask);

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // A pass that drew no mask parked none, and the helper reads the mode and the rate
        // before it reads the mask. It reports the missing mask with the same error that
        // `pop_cache` gives, so a backward pass with no forward pass behind it still refuses
        let mask = ctx.pop_cache::<Tensor>("Dropout").ok();
        dropout_backward(grad_output, &mask, ctx.is_training(), self.rate, "Dropout")
    }
}
