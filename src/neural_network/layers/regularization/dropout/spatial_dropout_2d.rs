//! Spatial dropout layer for 2D feature maps, dropping whole channels at once

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::dropout::{
    apply_spatial_dropout_threshold, dropout_output_shape, spatial_dropout_backward,
    spatial_dropout_scale,
};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    validate_input_ndim, validate_input_shape, validate_rate,
};
use crate::neural_network::traits::Layer;
use crate::parallel_gates::{
    CHEAP_MAP_PARALLEL_THRESHOLD, SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS,
};
use ndarray::IxDyn;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Spatial Dropout layer for 2D data
///
/// Drops entire channels instead of individual elements, which is effective for
/// convolutional layers where adjacent pixels are correlated. Input shape is
/// `(batch_size, channels, height, width)`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array4;
///
/// // Create a SpatialDropout2D layer with 20% dropout rate
/// let mut spatial_dropout = SpatialDropout2D::new(0.2, vec![32, 64, 28, 28]).unwrap();
///
/// // Create input tensor (batch_size=32, channels=64, height=28, width=28)
/// let input = Array4::ones((32, 64, 28, 28)).into_dyn();
///
/// // During training, approximately 20% of channels will be set to 0
/// let output = spatial_dropout.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct SpatialDropout2D {
    /// Dropout rate, fraction of channels to drop (between 0 and 1)
    rate: f32,
    /// Expected shape of the input tensor
    input_shape: Vec<usize>,
    /// Binary mask used during training to determine which channels to drop
    mask: Option<Tensor>,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Random number generator backing mask sampling
    rng: StdRng,
}

impl SpatialDropout2D {
    /// Creates a new SpatialDropout2D layer
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate, fraction of channels to drop (between 0 and 1)
    /// - `input_shape` - Shape of the input tensor `(batch_size, channels, height, width)`
    ///
    /// # Notes
    ///
    /// The mask RNG is seeded from the global seed or entropy by default. For reproducible masks,
    /// set a seed with [`SpatialDropout2D::with_random_state`].
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New SpatialDropout2D layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not between 0 and 1
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(SpatialDropout2D {
            rate,
            input_shape,
            mask: None,
            training: true,
            rng: crate::random::make_rng(None),
        })
    }

    /// Sets the seed for reproducible mask sampling
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

impl Layer for SpatialDropout2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is already validated in `new()`
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_input_ndim(
            input.ndim(),
            4,
            "SpatialDropout2D (batch_size, channels, height, width)",
        )?;

        if !self.training {
            // During inference, pass input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            return Ok(input.clone());
        }

        if self.rate == 1.0 {
            // Dropping every channel yields all zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];

        // Per-channel mask of shape (batch_size, channels): one keep/drop value per channel
        let mut mask_2d = Tensor::random_using(
            IxDyn(&[batch_size, channels]),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut self.rng,
        );

        // Threshold the samples into a binary keep/drop mask
        apply_spatial_dropout_threshold(&mut mask_2d, self.rate, CHEAP_MAP_PARALLEL_THRESHOLD);

        let channel_mask = mask_2d.as_slice().expect("per-channel mask is contiguous");
        let output = spatial_dropout_scale(
            input,
            channel_mask,
            self.rate,
            SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS,
        );

        // Store the small per-channel mask for backpropagation
        self.mask = Some(mask_2d);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is already validated in `new()`
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_input_ndim(
            input.ndim(),
            4,
            "SpatialDropout2D (batch_size, channels, height, width)",
        )?;

        // During inference, pass input through unchanged (no mask, no scaling)
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        spatial_dropout_backward(
            grad_output,
            &self.mask,
            self.training,
            self.rate,
            "SpatialDropout2D",
            SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS,
        )
    }

    fn layer_type(&self) -> &str {
        "SpatialDropout2D"
    }

    fn output_shape(&self) -> String {
        dropout_output_shape(&self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
