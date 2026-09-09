//! 2D zero-padding layer that adds zero rows and columns at the edges of an image

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::border::Border2D;
use crate::neural_network::layers::border::pad_crop_engine::{
    pad_backward, pad_forward, pad_output_shape,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Adds zero rows and columns at the edges of a rank-4 tensor
///
/// The input shape is `[batch_size, height, width, channels]`. The output shape is
/// `[batch_size, height + top + bottom, width + left + right, channels]`. The batch axis and
/// the channel axis pass through unchanged.
///
/// The layer holds no parameter. Put it before a convolution with
/// [`PaddingType::Valid`](crate::neural_network::layers::convolution::PaddingType) to control
/// the border yourself. A convolution with `Same` padding splits an odd padding amount by its
/// own rule. This layer instead takes the amount at each of the 4 edges.
///
/// [`Cropping2D`](crate::neural_network::layers::border::Cropping2D) is the inverse layer, and
/// it is also this layer's backward pass.
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
/// // A rank-4 input: 2 samples, 4x4 pixels, 3 channels
/// let x = Array4::ones((2, 4, 4, 3)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(ZeroPadding2D::new(1))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let padded = model.predict(&x).unwrap();
///
/// // 1 zero row and 1 zero column at every edge, so 4x4 becomes 6x6
/// assert_eq!(padded.shape(), &[2, 6, 6, 3]);
/// ```
#[derive(Debug)]
pub struct ZeroPadding2D {
    /// Zero rows and columns to add at each of the 4 edges
    padding: Border2D,
    /// Shape the layer was built for, batch axis first. `None` before the build.
    built: Option<Shape>,
}

impl ZeroPadding2D {
    /// Creates a new ZeroPadding2D layer
    ///
    /// # Parameters
    ///
    /// - `padding` - Zero rows and columns to add. An integer gives the same amount at all 4
    ///   edges. A `(height, width)` pair gives 1 amount per axis. A pair of pairs
    ///   `((top, bottom), (left, right))` names all 4 edges. See [`Border2D`]
    ///
    /// # Returns
    ///
    /// - `Self` - New `ZeroPadding2D` layer instance
    pub fn new(padding: impl Into<Border2D>) -> Self {
        ZeroPadding2D {
            padding: padding.into(),
            built: None,
        }
    }
}

impl LayerBase for ZeroPadding2D {
    fn layer_type(&self) -> &str {
        "ZeroPadding2D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for ZeroPadding2D {
    /// Records the shape the pad runs over. The layer holds no array, so nothing is
    /// allocated. The shape algebra checks the rank.
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "ZeroPadding2D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("ZeroPadding2D"));
        }
        let output = pad_forward(input, &self.padding.0, 4, "ZeroPadding2D")?;

        if ctx.is_training() {
            ctx.push_cache("ZeroPadding2D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("ZeroPadding2D")?;
        pad_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.padding.0,
            "ZeroPadding2D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        pad_output_shape(input, &self.padding.0, "ZeroPadding2D")
    }
}
