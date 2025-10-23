use super::*;

/// Threshold for using parallel computation in SpatialDropout1D layer.
/// When batch_size * channels >= this threshold, parallel computation is used for mask expansion.
const SPATIAL_DROPOUT_1D_PARALLEL_THRESHOLD: usize = 64;

/// Spatial Dropout layer for 1D data, which randomly sets entire feature maps to 0
/// at each update during training time.
///
/// Unlike regular Dropout which drops individual elements, Spatial Dropout drops entire
/// channels (feature maps) along the spatial dimension. This is particularly effective
/// for convolutional layers where adjacent pixels are highly correlated.
///
/// Input shape: (batch_size, channels, length)
///
/// # Fields
///
/// - `rate` - Dropout rate: fraction of the channels to drop (between 0 and 1).
/// - `input_shape` - Shape of the input tensor.
/// - `mask` - Binary mask used during training to determine which channels to drop.
/// - `training` - Whether the layer is in training mode or inference mode.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array3;
///
/// // Create a SpatialDropout1D layer with 20% dropout rate
/// let mut spatial_dropout = SpatialDropout1D::new(0.2, vec![32, 64, 128]);
///
/// // Create input tensor (batch_size=32, channels=64, length=128)
/// let input = Array3::ones((32, 64, 128)).into_dyn();
///
/// // During training, approximately 20% of channels will be set to 0
/// let output = spatial_dropout.forward(&input).unwrap();
/// ```
pub struct SpatialDropout1D {
    rate: f32,
    input_shape: Vec<usize>,
    mask: Option<Tensor>,
    training: bool,
}

impl SpatialDropout1D {
    /// Creates a new SpatialDropout1D layer.
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate: fraction of the channels to drop. Must be between 0 and 1.
    /// - `input_shape` - Shape of the input tensor (batch_size, channels, length).
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the SpatialDropout1D layer.
    ///
    /// # Panics
    ///
    /// Panics if rate is not between 0 and 1.
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Self {
        SpatialDropout1D {
            rate,
            input_shape,
            mask: None,
            training: true,
        }
    }

    mode_dependent_layer_set_training!();
}

impl Layer for SpatialDropout1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        if self.rate < 0.0 || self.rate > 1.0 {
            return Err(ModelError::InputValidationError(
                "Dropout rate cannot be less than 0 or greater than 1".to_string(),
            ));
        }

        // Validate input shape matches expected shape
        if !self.input_shape.is_empty() && input.shape() != self.input_shape.as_slice() {
            return Err(ModelError::InputValidationError(format!(
                "Input shape mismatch: expected {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )));
        }

        // Validate input is 3D (batch_size, channels, length)
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(format!(
                "SpatialDropout1D expects 3D input (batch_size, channels, length), got {}D",
                input.ndim()
            )));
        }

        if !self.training {
            // During inference, pass input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            eprintln!("Dropout rate is 0.0, so this layer has no effect on the output.");
            return Ok(input.clone());
        }

        if self.rate == 1.0 {
            eprintln!("SpatialDropout1D rate is 1.0, so this layer will return all zeros.");
            // If dropout rate is 1.0, return zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let length = shape[2];

        // Generate mask for channels: shape (batch_size, channels)
        // Each channel is either fully kept or fully dropped
        let mut mask_2d = Tensor::random(IxDyn(&[batch_size, channels]), Uniform::new(0.0, 1.0));

        // Apply threshold to create binary mask with parallel or sequential computation
        apply_spatial_dropout_threshold(
            &mut mask_2d,
            self.rate,
            SPATIAL_DROPOUT_1D_PARALLEL_THRESHOLD,
        );

        // Expand mask to match input shape (batch_size, channels, length)
        // by broadcasting the mask across the spatial dimension
        let mut mask = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

        // Broadcast the 2D mask to 3D mask efficiently
        for b in 0..batch_size {
            for c in 0..channels {
                let mask_value = mask_2d[[b, c]];
                for l in 0..length {
                    mask[[b, c, l]] = mask_value;
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
        "SpatialDropout1D"
    }

    fn output_shape(&self) -> String {
        dropout_output_shape(&self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
