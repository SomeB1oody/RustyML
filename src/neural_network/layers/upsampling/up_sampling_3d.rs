//! 3D upsampling layer that enlarges a volume by a whole-number factor per axis

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::upsampling::resize_engine::{
    upsample_backward, upsample_forward, upsample_output_shape, validate_factors,
};
use crate::neural_network::layers::upsampling::{Factor3D, Interpolation};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Enlarges the 3 spatial axes of a rank-5 tensor
///
/// The input shape is `[batch_size, dim1, dim2, dim3, channels]`. The output shape multiplies
/// each spatial extent by its factor. The batch axis and the channel axis pass through
/// unchanged
///
/// The layer holds no parameter. Each output position copies the input position it came from,
/// so the layer takes no interpolation argument
///
/// The layer is the decoder counterpart of
/// [`MaxPooling3D`](crate::neural_network::layers::pooling::max_pooling_3d::MaxPooling3D) and
/// [`AveragePooling3D`](crate::neural_network::layers::pooling::average_pooling_3d::AveragePooling3D).
/// A volume model pairs each pooling stage with 1 upsampling stage of the same factor. Examples
/// include a model over a medical scan or a video clip
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array5;
///
/// // A rank-5 input: 1 sample, a 2x1x1 volume, 2 channels
/// let x = Array5::from_shape_vec((1, 2, 1, 1, 2), vec![1.0, 2.0, 3.0, 4.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(UpSampling3D::new((2, 3, 1)).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let larger = model.predict(&x).unwrap();
///
/// // Each spatial axis grows by its own factor, and the channel axis stays put
/// assert_eq!(larger.shape(), &[1, 4, 3, 1, 2]);
/// assert_eq!(larger[[0, 0, 2, 0, 1]], 2.0);
/// assert_eq!(larger[[0, 3, 0, 0, 0]], 3.0);
/// ```
#[derive(Debug)]
pub struct UpSampling3D {
    /// Factor each spatial axis grows by
    size: Factor3D,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl UpSampling3D {
    /// Creates a new UpSampling3D layer
    ///
    /// # Parameters
    ///
    /// - `size` - Factor each spatial axis grows by. An integer gives the same factor to all 3
    ///   axes. A triple names the factor of each axis. See [`Factor3D`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `UpSampling3D` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any factor is 0
    pub fn new(size: impl Into<Factor3D>) -> Result<Self, Error> {
        let size = size.into();
        validate_factors(&size.0)?;
        Ok(UpSampling3D { size, built: None })
    }
}

impl LayerBase for UpSampling3D {
    fn layer_type(&self) -> &str {
        "UpSampling3D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for UpSampling3D {
    /// Records the shape the layer enlarges. The layer holds no array, so nothing is allocated.
    /// The shape algebra checks the rank
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "UpSampling3D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output = upsample_forward(
            input,
            &self.size.0,
            Interpolation::Nearest,
            5,
            "UpSampling3D",
        )?;

        if ctx.is_training() {
            ctx.push_cache(input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("UpSampling3D")?;

        upsample_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.size.0,
            Interpolation::Nearest,
            "UpSampling3D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        upsample_output_shape(input, &self.size.0, "UpSampling3D")
    }
}
