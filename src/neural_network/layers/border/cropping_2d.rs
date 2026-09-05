//! 2D cropping layer that removes rows and columns at the edges of an image

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::border::Border2D;
use crate::neural_network::layers::border::pad_crop_engine::{
    crop_backward, crop_forward, crop_output_shape,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Removes rows and columns at the edges of a rank-4 tensor
///
/// The input shape is `[batch_size, height, width, channels]`. The output shape is
/// `[batch_size, height - top - bottom, width - left - right, channels]`. The batch axis and
/// the channel axis pass through unchanged
///
/// The layer holds no parameter. Each spatial axis must keep at least 1 position, so the
/// forward pass fails when the 2 amounts on an axis together reach its extent
///
/// A common use is a decoder that upsamples past the wanted size. The layer then trims the
/// output to the size of the matching encoder feature map
///
/// [`ZeroPadding2D`](crate::neural_network::layers::border::ZeroPadding2D) is the inverse
/// layer, and it is also this layer's backward pass
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
/// // A rank-4 input: 2 samples, 8x8 pixels, 3 channels
/// let x = Array4::ones((2, 8, 8, 3)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(Cropping2D::new(((1, 1), (2, 0))))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let cropped = model.predict(&x).unwrap();
///
/// // 1 row off the top and bottom, and 2 columns off the left only
/// assert_eq!(cropped.shape(), &[2, 6, 6, 3]);
/// ```
#[derive(Debug)]
pub struct Cropping2D {
    /// Rows and columns to remove at each of the 4 edges
    cropping: Border2D,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl Cropping2D {
    /// Creates a new Cropping2D layer
    ///
    /// # Parameters
    ///
    /// - `cropping` - Rows and columns to remove. An integer gives the same amount at all 4
    ///   edges. A `(height, width)` pair gives 1 amount per axis. A pair of pairs
    ///   `((top, bottom), (left, right))` names all 4 edges. See [`Border2D`]
    ///
    /// # Returns
    ///
    /// - `Self` - New `Cropping2D` layer instance
    pub fn new(cropping: impl Into<Border2D>) -> Self {
        Cropping2D {
            cropping: cropping.into(),
            built: None,
        }
    }
}

impl LayerBase for Cropping2D {
    fn layer_type(&self) -> &str {
        "Cropping2D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Cropping2D {
    /// Records the shape the crop runs over. The layer holds no array, so nothing is
    /// allocated. The shape algebra checks the rank and the crop fit
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Cropping2D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Cropping2D"));
        }
        let output = crop_forward(input, &self.cropping.0, 4, "Cropping2D")?;

        if ctx.is_training() {
            ctx.push_cache("Cropping2D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("Cropping2D")?;
        crop_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.cropping.0,
            "Cropping2D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        crop_output_shape(input, &self.cropping.0, "Cropping2D")
    }
}
