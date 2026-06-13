//! 1D spatial dropout layer that drops whole channels of `(batch_size, channels, length)` inputs

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::dropout::{
    apply_spatial_dropout_threshold, dropout_backward, dropout_output_shape,
};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    validate_input_ndim, validate_input_shape, validate_rate,
};
use crate::neural_network::traits::Layer;
use crate::parallel_gates::CHEAP_MAP_PARALLEL_THRESHOLD;
use ndarray::IxDyn;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Spatial Dropout layer for 1D data
///
/// Drops entire channels instead of individual elements, which is effective for
/// convolutional layers where adjacent positions are correlated. Input shape is
/// `(batch_size, channels, length)`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array3;
///
/// // Create a SpatialDropout1D layer with 20% dropout rate
/// let mut spatial_dropout = SpatialDropout1D::new(0.2, vec![32, 64, 128], None).unwrap();
///
/// // Create input tensor (batch_size=32, channels=64, length=128)
/// let input = Array3::ones((32, 64, 128)).into_dyn();
///
/// // During training, approximately 20% of channels will be set to 0
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
    /// - `input_shape` - Shape of the input tensor `(batch_size, channels, length)`
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New SpatialDropout1D layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not between 0 and 1
    pub fn new(
        rate: f32,
        input_shape: Vec<usize>,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(SpatialDropout1D {
            rate,
            input_shape,
            mask: None,
            training: true,
            rng: crate::random::make_rng(random_state),
        })
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
            "SpatialDropout1D (batch_size, channels, length)",
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
        let channels = shape[1];
        let length = shape[2];

        // Per-channel mask of shape (batch_size, channels)
        let mut mask_2d = Tensor::random_using(
            IxDyn(&[batch_size, channels]),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut self.rng,
        );

        // Threshold the mask into binary values (parallel above the threshold)
        apply_spatial_dropout_threshold(&mut mask_2d, self.rate, CHEAP_MAP_PARALLEL_THRESHOLD);

        // Broadcast the 2D mask across the length dimension to (batch_size, channels, length)
        let mut mask = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

        for b in 0..batch_size {
            for c in 0..channels {
                let mask_value = mask_2d[[b, c]];
                for l in 0..length {
                    mask[[b, c, l]] = mask_value;
                }
            }
        }

        // Inverted dropout: scale by 1 / (1 - rate) to preserve the expected value
        let scale = 1.0 / (1.0 - self.rate);
        let output = input * &mask * scale;

        // Store mask for backpropagation
        self.mask = Some(mask);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is validated in `new()`; only validate the runtime input here
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_input_ndim(
            input.ndim(),
            3,
            "SpatialDropout1D (batch_size, channels, length)",
        )?;

        // During inference, pass input through unchanged
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        dropout_backward(
            grad_output,
            &self.mask,
            self.training,
            self.rate,
            "SpatialDropout1D",
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
