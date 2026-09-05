//! 1D spatial dropout layer that drops whole channels of `(batch_size, length, channels)` inputs

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::regularization::dropout::{
    spatial_dropout_backward, spatial_dropout_scale,
};
use crate::neural_network::layers::regularization::validation::{
    validate_input_ndim, validate_rate,
};
use crate::neural_network::layers::validation::{start_build, validate_built_input};
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, StateSlot, Tensor};
use crate::parallel_gates::spatial_dropout_scale_parallel_min_elems;
use ndarray::IxDyn;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Spatial dropout layer for 1D data
///
/// Drops entire channels instead of individual elements, which suits convolutional
/// layers where adjacent positions are correlated. Input shape is
/// `(batch_size, length, channels)`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::UnaryLayer;
/// use rustyml::neural_network::Ctx;
/// use ndarray::Array3;
///
/// // Create a SpatialDropout1D layer with 20% dropout rate
/// let mut spatial_dropout = SpatialDropout1D::new(0.2).unwrap();
///
/// // Create input tensor (batch_size=32, length=128, channels=64)
/// let input = Array3::ones((32, 128, 64)).into_dyn();
///
/// // During training, ~20% of channels are set to 0
/// let mut ctx = Ctx::training();
/// let output = spatial_dropout.forward_mut(&input, &mut ctx).unwrap();
/// ```
#[derive(Debug)]
pub struct SpatialDropout1D {
    /// Dropout rate, fraction of channels to drop (between 0 and 1)
    rate: f32,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Random number generator backing mask sampling
    rng: StdRng,
}

impl SpatialDropout1D {
    /// Creates a new SpatialDropout1D layer
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate, fraction of channels to drop (between 0 and 1)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New SpatialDropout1D layer instance or a validation error
    ///
    /// # Notes
    ///
    /// By default, `new` seeds the mask RNG from the global seed or entropy. For reproducible
    /// masks, set a seed with [`SpatialDropout1D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not between 0 and 1
    pub fn new(rate: f32) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(SpatialDropout1D {
            rate,
            built: None,
            rng: crate::random::make_rng(None),
        })
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
}

impl LayerBase for SpatialDropout1D {
    fn layer_type(&self) -> &str {
        "SpatialDropout1D"
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

impl UnaryLayer for SpatialDropout1D {
    /// Records the shape the layer serves. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "SpatialDropout1D", input)? else {
            return Ok(());
        };
        input.check_rank("SpatialDropout1D", 3)?;
        self.built = Some(built);
        Ok(())
    }

    /// Drops whole channels of a `(batch_size, length, channels)` tensor
    ///
    /// An inference pass draws nothing and passes the input through unchanged, which is what
    /// inverted dropout needs
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // `rate` is validated in `new()`
        validate_built_input(&self.built, "SpatialDropout1D", input.shape())?;
        validate_input_ndim(
            input.ndim(),
            3,
            "SpatialDropout1D (batch_size, length, channels)",
        )?;

        if !ctx.is_training() {
            // During inference, pass input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            return Ok(input.clone());
        }

        if self.rate == 1.0 {
            // Rate of 1.0 drops every channel
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[shape.len() - 1];

        // The stream of the pass, which starts from the stream of the layer. The draw below
        // advances the copy in the context, and `apply_state` moves it back into the layer
        let mut rng = ctx
            .take_state::<StdRng>("rng")
            .unwrap_or_else(|| self.rng.clone());

        // Per-channel mask of shape (batch_size, channels): 1 keep/drop value per channel
        let mut mask_2d = Tensor::random_using(
            IxDyn(&[batch_size, channels]),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut rng,
        );
        ctx.set_state("rng", rng);

        // Threshold the samples into a binary keep/drop mask. The mask holds 1 value per
        // (batch, channel), so it stays far too small for rayon to pay
        let rate = self.rate;
        mask_2d.mapv_inplace(|x| if x >= rate { 1.0 } else { 0.0 });

        let channel_mask = mask_2d.as_slice().expect("per-channel mask is contiguous");
        let output = spatial_dropout_scale(
            input,
            channel_mask,
            self.rate,
            spatial_dropout_scale_parallel_min_elems(),
        );

        // Park the small per-channel mask for backpropagation
        ctx.push_cache(mask_2d);

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // A pass that drew no mask parked none, and the helper reads the mode and the rate
        // before it reads the mask. It reports the missing mask with the same error that
        // `pop_cache` gives, so a backward pass with no forward pass behind it still refuses
        let mask = ctx.pop_cache::<Tensor>("SpatialDropout1D").ok();
        spatial_dropout_backward(
            grad_output,
            &mask,
            ctx.is_training(),
            self.rate,
            "SpatialDropout1D",
            spatial_dropout_scale_parallel_min_elems(),
        )
    }

    /// The layer changes values and not extents
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("SpatialDropout1D", 3)?;
        Ok(input.clone())
    }
}
