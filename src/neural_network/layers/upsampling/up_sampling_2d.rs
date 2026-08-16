//! 2D upsampling layer that enlarges an image by a whole-number factor per axis

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::upsampling::resize_engine::{
    upsample_backward, upsample_forward, upsample_summary, validate_factors,
};
use crate::neural_network::layers::upsampling::{Factor2D, Interpolation};
use crate::neural_network::traits::Layer;

/// Enlarges the height and the width of a rank-4 tensor
///
/// The input shape is `[batch_size, height, width, channels]`. The output shape multiplies the
/// height and the width by their own factor from `size`. The batch axis and the channel axis
/// pass through unchanged
///
/// The layer holds no parameter. [`Interpolation::Nearest`] repeats each pixel into a block of
/// the factor size. Every other mode resamples with a separable kernel, so a new pixel takes a
/// weighted sum of its neighbors along each axis
///
/// The layer is the decoder counterpart of
/// [`MaxPooling2D`](crate::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D) and
/// [`AveragePooling2D`](crate::neural_network::layers::pooling::average_pooling_2d::AveragePooling2D).
/// An upsampling stage followed by a
/// [`Conv2D`](crate::neural_network::layers::convolution::conv_2d::Conv2D) is the usual way to
/// build a decoder without a transposed convolution
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
///
/// // A rank-4 input: 1 sample, 2x2 pixels, 1 channel
/// let x = Array4::from_shape_vec((1, 2, 2, 1), vec![1.0, 2.0, 3.0, 4.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(UpSampling2D::new(2, Interpolation::Nearest).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let larger = model.predict(&x).unwrap();
///
/// // 2x2 pixels become 4x4, and each input pixel covers a 2x2 block
/// assert_eq!(larger.shape(), &[1, 4, 4, 1]);
/// assert_eq!(larger[[0, 0, 0, 0]], 1.0);
/// assert_eq!(larger[[0, 1, 1, 0]], 1.0);
/// assert_eq!(larger[[0, 1, 2, 0]], 2.0);
/// assert_eq!(larger[[0, 3, 3, 0]], 4.0);
/// ```
///
/// # Performance
///
/// [`Interpolation::Nearest`] copies whole runs of channels, so it runs at copy speed. Every
/// other mode costs 1 multiply-add per tap per output element, and the taps grow with the
/// kernel radius. The count is 3 for bilinear, 5 for bicubic, 7 for `Lanczos3`, and 11 for
/// `Lanczos5`. The layer pays that cost once per spatial axis, not once per pixel pair
#[derive(Debug)]
pub struct UpSampling2D {
    /// Factor each spatial axis grows by
    size: Factor2D,
    /// How the layer fills the new pixels
    interpolation: Interpolation,
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
}

impl UpSampling2D {
    /// Creates a new UpSampling2D layer
    ///
    /// # Parameters
    ///
    /// - `size` - Factor each spatial axis grows by. An integer gives the same factor to both
    ///   axes. A `(height, width)` pair names the factor of each axis. See [`Factor2D`]
    /// - `interpolation` - How the layer fills the new pixels. See [`Interpolation`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `UpSampling2D` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If either factor is 0
    pub fn new(size: impl Into<Factor2D>, interpolation: Interpolation) -> Result<Self, Error> {
        let size = size.into();
        validate_factors(&size.0)?;
        Ok(UpSampling2D {
            size,
            interpolation,
            input_shape: None,
        })
    }
}

impl Layer for UpSampling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = upsample_forward(input, &self.size.0, self.interpolation, 4, "UpSampling2D")?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        upsample_forward(input, &self.size.0, self.interpolation, 4, "UpSampling2D")
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        upsample_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.size.0,
            self.interpolation,
            "UpSampling2D",
        )
    }

    fn layer_type(&self) -> &str {
        "UpSampling2D"
    }

    fn output_shape(&self) -> String {
        upsample_summary(self.input_shape.as_deref(), &self.size.0)
    }

    no_trainable_parameters_layer_functions!();
}
