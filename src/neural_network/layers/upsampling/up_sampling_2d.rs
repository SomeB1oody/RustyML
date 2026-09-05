//! 2D upsampling layer that enlarges an image by a whole-number factor per axis

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::upsampling::resize_engine::{
    upsample_backward, upsample_forward, upsample_output_shape, validate_factors,
};
use crate::neural_network::layers::upsampling::{Factor2D, Interpolation};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

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
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(UpSampling2D::new(2, Interpolation::Nearest).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
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
            built: None,
        })
    }
}

impl LayerBase for UpSampling2D {
    fn layer_type(&self) -> &str {
        "UpSampling2D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for UpSampling2D {
    /// Records the shape the layer enlarges. The layer holds no array, so nothing is allocated.
    /// The shape algebra checks the rank
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "UpSampling2D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output = upsample_forward(input, &self.size.0, self.interpolation, 4, "UpSampling2D")?;

        if ctx.is_training() {
            ctx.push_cache(input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("UpSampling2D")?;

        upsample_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.size.0,
            self.interpolation,
            "UpSampling2D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        upsample_output_shape(input, &self.size.0, "UpSampling2D")
    }
}
