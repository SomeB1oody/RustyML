use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::regularization_layer::dropout_layer::{
    apply_spatial_dropout_threshold, dropout_backward, dropout_output_shape,
};
use crate::neural_network::layer::regularization_layer::input_validation_function::{
    validate_input_ndim, validate_input_shape, validate_rate,
};
use crate::neural_network::neural_network_trait::Layer;
use ndarray::IxDyn;
use ndarray_rand::RandomExt;
use rand::distr::Uniform;

/// Threshold for using parallel computation in SpatialDropout3D layer.
/// When batch_size * channels >= this threshold, parallel computation is used for mask expansion.
const SPATIAL_DROPOUT_3D_PARALLEL_THRESHOLD: usize = 64;

/// Spatial Dropout layer for 3D data.
///
/// Drops entire channels instead of individual elements, which is effective for
/// 3D convolutional layers where adjacent voxels are correlated. Input shape is
/// `(batch_size, channels, depth, height, width)`.
///
/// # Fields
///
/// - `rate` - Dropout rate, fraction of channels to drop (between 0 and 1)
/// - `input_shape` - Expected shape of the input tensor
/// - `mask` - Binary mask used during training to determine which channels to drop
/// - `training` - Whether the layer is in training mode or inference mode
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::layer::*;
/// use rustyml::prelude::neural_network_trait::Layer;
/// use ndarray::Array5;
///
/// // Create a SpatialDropout3D layer with 20% dropout rate
/// let mut spatial_dropout = SpatialDropout3D::new(0.2, vec![32, 64, 16, 28, 28]).unwrap();
///
/// // Create input tensor (batch_size=32, channels=64, depth=16, height=28, width=28)
/// let input = Array5::ones((32, 64, 16, 28, 28)).into_dyn();
///
/// // During training, approximately 20% of channels will be set to 0
/// let output = spatial_dropout.forward(&input).unwrap();
/// ```
pub struct SpatialDropout3D {
    rate: f32,
    input_shape: Vec<usize>,
    mask: Option<Tensor>,
    training: bool,
}

impl SpatialDropout3D {
    /// Creates a new SpatialDropout3D layer.
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate, fraction of channels to drop (between 0 and 1)
    /// - `input_shape` - Shape of the input tensor `(batch_size, channels, depth, height, width)`
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New SpatialDropout3D layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `rate` is not between 0 and 1
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, ModelError> {
        validate_rate(rate, "Dropout rate")?;

        Ok(SpatialDropout3D {
            rate,
            input_shape,
            mask: None,
            training: true,
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for SpatialDropout3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        validate_rate(self.rate, "Dropout rate")?;
        validate_input_shape(input.shape(), &self.input_shape)?;
        validate_input_ndim(
            input.ndim(),
            5,
            "SpatialDropout3D (batch_size, channels, depth, height, width)",
        )?;

        if !self.training {
            // During inference, pass input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            eprintln!("Dropout rate is 0.0, so this layer has no effect on the output.");
            return Ok(input.clone());
        }

        if self.rate == 1.0 {
            eprintln!("SpatialDropout3D rate is 1.0, so this layer will return all zeros.");
            // If dropout rate is 1.0, return zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let depth = shape[2];
        let height = shape[3];
        let width = shape[4];

        // Generate mask for channels: shape (batch_size, channels)
        // Each channel is either fully kept or fully dropped
        let mut mask_2d = Tensor::random(
            IxDyn(&[batch_size, channels]),
            Uniform::new(0.0, 1.0).unwrap(),
        );

        // Apply threshold to create binary mask with parallel or sequential computation
        apply_spatial_dropout_threshold(
            &mut mask_2d,
            self.rate,
            SPATIAL_DROPOUT_3D_PARALLEL_THRESHOLD,
        );

        // Expand mask to match input shape (batch_size, channels, depth, height, width)
        // by broadcasting the mask across the spatial dimensions
        let mut mask = Tensor::zeros(IxDyn(&[batch_size, channels, depth, height, width]));

        // Broadcast the 2D mask to 5D mask efficiently
        for b in 0..batch_size {
            for c in 0..channels {
                let mask_value = mask_2d[[b, c]];
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            mask[[b, c, d, h, w]] = mask_value;
                        }
                    }
                }
            }
        }

        // Apply mask and scale by (1 - rate) to maintain expected value
        // This is "inverted dropout" technique
        let scale = 1.0 / (1.0 - self.rate);
        let output = input * &mask * scale;

        // Store mask for backpropagation
        self.mask = Some(mask);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        dropout_backward(grad_output, &self.mask, self.training, self.rate)
    }

    fn layer_type(&self) -> &str {
        "SpatialDropout3D"
    }

    fn output_shape(&self) -> String {
        dropout_output_shape(&self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
