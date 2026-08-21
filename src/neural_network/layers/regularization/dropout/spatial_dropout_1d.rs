//! 1D spatial dropout layer that drops whole channels of `(batch_size, length, channels)` inputs

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::dropout::{
    dropout_output_shape, spatial_dropout_backward, spatial_dropout_scale,
};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    validate_input_ndim, validate_input_shape, validate_rate,
};
use crate::neural_network::traits::Layer;
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
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array3;
///
/// // Create a SpatialDropout1D layer with 20% dropout rate
/// let mut spatial_dropout = SpatialDropout1D::new(0.2, vec![32, 128, 64]).unwrap();
///
/// // Create input tensor (batch_size=32, length=128, channels=64)
/// let input = Array3::ones((32, 128, 64)).into_dyn();
///
/// // During training, ~20% of channels are set to 0
/// let output = spatial_dropout.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct SpatialDropout1D {
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

impl SpatialDropout1D {
    /// Creates a new SpatialDropout1D layer
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate, fraction of channels to drop (between 0 and 1)
    /// - `input_shape` - Shape of the input tensor `(batch_size, length, channels)`
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
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(SpatialDropout1D {
            rate,
            input_shape,
            mask: None,
            training: true,
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

    mode_dependent_layer_set_training!();
}

impl Layer for SpatialDropout1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is validated in `new()`
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_input_ndim(
            input.ndim(),
            3,
            "SpatialDropout1D (batch_size, length, channels)",
        )?;

        if !self.training {
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

        // Per-channel mask of shape (batch_size, channels): 1 keep/drop value per channel
        let mut mask_2d = Tensor::random_using(
            IxDyn(&[batch_size, channels]),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut self.rng,
        );

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

        // Store the small per-channel mask for backpropagation
        self.mask = Some(mask_2d);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is validated in `new()`
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_input_ndim(
            input.ndim(),
            3,
            "SpatialDropout1D (batch_size, length, channels)",
        )?;

        // During inference, pass input through unchanged
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        spatial_dropout_backward(
            grad_output,
            &self.mask,
            self.training,
            self.rate,
            "SpatialDropout1D",
            spatial_dropout_scale_parallel_min_elems(),
        )
    }

    fn layer_type(&self) -> &str {
        "SpatialDropout1D"
    }

    fn output_shape(&self) -> String {
        dropout_output_shape(&self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
