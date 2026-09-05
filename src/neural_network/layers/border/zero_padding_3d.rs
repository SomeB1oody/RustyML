//! 3D zero-padding layer that adds zero planes at the 6 faces of a volume

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::border::Border3D;
use crate::neural_network::layers::border::pad_crop_engine::{
    pad_backward, pad_forward, pad_output_shape,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Adds zero planes at the 6 faces of a rank-5 tensor
///
/// The input shape is `[batch_size, depth, height, width, channels]`. Each of the 3 spatial
/// axes grows by the amount at its 2 ends. The batch axis and the channel axis pass through
/// unchanged
///
/// The layer holds no parameter. A volumetric convolution with
/// [`PaddingType::Valid`](crate::neural_network::layers::convolution::PaddingType) after this
/// layer keeps the border under the caller's control
///
/// [`Cropping3D`](crate::neural_network::layers::border::Cropping3D) is the inverse layer, and
/// it is also this layer's backward pass
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
/// // A rank-5 input: 2 samples, a 3x4x4 volume, 1 channel
/// let x = Array5::ones((2, 3, 4, 4, 1)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(ZeroPadding3D::new((1, 2, 0)))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let padded = model.predict(&x).unwrap();
///
/// // 1 plane at each end of the first axis, 2 at each end of the second, none on the third
/// assert_eq!(padded.shape(), &[2, 5, 8, 4, 1]);
/// ```
#[derive(Debug)]
pub struct ZeroPadding3D {
    /// Zero planes to add at each of the 6 faces
    padding: Border3D,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl ZeroPadding3D {
    /// Creates a new ZeroPadding3D layer
    ///
    /// # Parameters
    ///
    /// - `padding` - Zero planes to add. An integer gives the same amount at all 6 faces. A
    ///   triple gives 1 amount per axis. A triple of pairs names all 6 faces. See [`Border3D`]
    ///
    /// # Returns
    ///
    /// - `Self` - New `ZeroPadding3D` layer instance
    pub fn new(padding: impl Into<Border3D>) -> Self {
        ZeroPadding3D {
            padding: padding.into(),
            built: None,
        }
    }
}

impl LayerBase for ZeroPadding3D {
    fn layer_type(&self) -> &str {
        "ZeroPadding3D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for ZeroPadding3D {
    /// Records the shape the pad runs over. The layer holds no array, so nothing is
    /// allocated. The shape algebra checks the rank
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "ZeroPadding3D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("ZeroPadding3D"));
        }
        let output = pad_forward(input, &self.padding.0, 5, "ZeroPadding3D")?;

        if ctx.is_training() {
            ctx.push_cache(input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("ZeroPadding3D")?;
        pad_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.padding.0,
            "ZeroPadding3D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        pad_output_shape(input, &self.padding.0, "ZeroPadding3D")
    }
}
