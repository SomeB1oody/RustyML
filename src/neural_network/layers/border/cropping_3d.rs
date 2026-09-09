//! 3D cropping layer that removes planes at the 6 faces of a volume

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::border::Border3D;
use crate::neural_network::layers::border::pad_crop_engine::{
    crop_backward, crop_forward, crop_output_shape,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Removes planes at the 6 faces of a rank-5 tensor
///
/// The input shape is `[batch_size, depth, height, width, channels]`. Each of the 3 spatial
/// axes shrinks by the amount at its 2 ends. The batch axis and the channel axis pass through
/// unchanged.
///
/// The layer holds no parameter. Each spatial axis must keep at least 1 position, so the
/// forward pass fails when the 2 amounts on an axis together reach its extent.
///
/// [`ZeroPadding3D`](crate::neural_network::layers::border::ZeroPadding3D) is the inverse
/// layer, and it is also this layer's backward pass.
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
/// // A rank-5 input: 2 samples, a 4x6x6 volume, 1 channel
/// let x = Array5::ones((2, 4, 6, 6, 1)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(Cropping3D::new(1))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let cropped = model.predict(&x).unwrap();
///
/// // 1 plane off each of the 6 faces
/// assert_eq!(cropped.shape(), &[2, 2, 4, 4, 1]);
/// ```
#[derive(Debug)]
pub struct Cropping3D {
    /// Planes to remove at each of the 6 faces
    cropping: Border3D,
    /// Shape the layer was built for, batch axis first. `None` before the build.
    built: Option<Shape>,
}

impl Cropping3D {
    /// Creates a new Cropping3D layer
    ///
    /// # Parameters
    ///
    /// - `cropping` - Planes to remove. An integer gives the same amount at all 6 faces. A
    ///   triple gives 1 amount per axis. A triple of pairs names all 6 faces. See [`Border3D`]
    ///
    /// # Returns
    ///
    /// - `Self` - New `Cropping3D` layer instance
    pub fn new(cropping: impl Into<Border3D>) -> Self {
        Cropping3D {
            cropping: cropping.into(),
            built: None,
        }
    }
}

impl LayerBase for Cropping3D {
    fn layer_type(&self) -> &str {
        "Cropping3D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Cropping3D {
    /// Records the shape the crop runs over. The layer holds no array, so nothing is
    /// allocated. The shape algebra checks the rank and the crop fit.
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Cropping3D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Cropping3D"));
        }
        let output = crop_forward(input, &self.cropping.0, 5, "Cropping3D")?;

        if ctx.is_training() {
            ctx.push_cache("Cropping3D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("Cropping3D")?;
        crop_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.cropping.0,
            "Cropping3D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        crop_output_shape(input, &self.cropping.0, "Cropping3D")
    }
}
