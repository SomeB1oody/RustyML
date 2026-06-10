//! Spatial dropout layer for 2D feature maps, dropping whole channels at once

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
use ndarray::IxDyn;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Mask expansion runs in parallel once batch_size * channels reaches this threshold
const SPATIAL_DROPOUT_2D_PARALLEL_THRESHOLD: usize = 64;

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
/// let mut spatial_dropout = SpatialDropout2D::new(0.2, vec![32, 64, 28, 28], None).unwrap();
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
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New SpatialDropout2D layer instance or a validation error
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

        Ok(SpatialDropout2D {
            rate,
            input_shape,
            mask: None,
            training: true,
            rng: crate::random::make_rng(random_state),
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for SpatialDropout2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is already validated in `new()`; only validate the runtime input
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
        let height = shape[2];
        let width = shape[3];

        // Per-channel mask of shape (batch_size, channels); each channel is fully kept or fully dropped
        let mut mask_2d = Tensor::random_using(
            IxDyn(&[batch_size, channels]),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut self.rng,
        );

        // Threshold the samples into a binary mask
        apply_spatial_dropout_threshold(
            &mut mask_2d,
            self.rate,
            SPATIAL_DROPOUT_2D_PARALLEL_THRESHOLD,
        );

        // Broadcast the 2D mask across spatial dimensions to (batch_size, channels, height, width)
        let mut mask = Tensor::zeros(IxDyn(&[batch_size, channels, height, width]));

        for b in 0..batch_size {
            for c in 0..channels {
                let mask_value = mask_2d[[b, c]];
                for h in 0..height {
                    for w in 0..width {
                        mask[[b, c, h, w]] = mask_value;
                    }
                }
            }
        }

        // Inverted dropout: scale by 1 / (1 - rate) so the expected value is preserved
        let scale = 1.0 / (1.0 - self.rate);
        let output = input * &mask * scale;

        // Store mask for backpropagation
        self.mask = Some(mask);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` is already validated in `new()`; only validate the runtime input
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
        dropout_backward(
            grad_output,
            &self.mask,
            self.training,
            self.rate,
            "SpatialDropout2D",
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
