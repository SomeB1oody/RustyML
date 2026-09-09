//! 1D cropping layer that removes steps at each end of the step axis

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::border::Border1D;
use crate::neural_network::layers::border::pad_crop_engine::{
    crop_backward, crop_forward, crop_output_shape,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Removes steps at each end of the step axis of a rank-3 tensor
///
/// The input shape is `[batch_size, steps, features]`. The output shape is
/// `[batch_size, steps - before - after, features]`. The batch axis and the feature axis pass
/// through unchanged.
///
/// The layer holds no parameter. At least 1 step must remain, so the forward pass fails when
/// the 2 amounts together reach the extent of the step axis.
///
/// [`ZeroPadding1D`](crate::neural_network::layers::border::ZeroPadding1D) is the inverse
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
/// use ndarray::Array3;
///
/// // A rank-3 input: 2 samples, 6 steps, 3 features
/// let x = Array3::ones((2, 6, 3)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(Cropping1D::new((1, 2)))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let cropped = model.predict(&x).unwrap();
///
/// // 1 step off the front and 2 off the back, so 6 steps become 3
/// assert_eq!(cropped.shape(), &[2, 3, 3]);
/// ```
#[derive(Debug)]
pub struct Cropping1D {
    /// Steps to remove at each end of the step axis
    cropping: Border1D,
    /// Shape the layer was built for, batch axis first. `None` before the build.
    built: Option<Shape>,
}

impl Cropping1D {
    /// Creates a new Cropping1D layer
    ///
    /// # Parameters
    ///
    /// - `cropping` - Steps to remove at each end of the step axis. An integer gives an equal
    ///   amount at both ends. A `(before, after)` pair names each end. See [`Border1D`]
    ///
    /// # Returns
    ///
    /// - `Self` - New `Cropping1D` layer instance
    pub fn new(cropping: impl Into<Border1D>) -> Self {
        Cropping1D {
            cropping: cropping.into(),
            built: None,
        }
    }
}

impl LayerBase for Cropping1D {
    fn layer_type(&self) -> &str {
        "Cropping1D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Cropping1D {
    /// Records the shape the crop runs over. The layer holds no array, so nothing is
    /// allocated. The shape algebra checks the rank and the crop fit.
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Cropping1D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Cropping1D"));
        }
        let output = crop_forward(input, &self.cropping.0, 3, "Cropping1D")?;

        if ctx.is_training() {
            ctx.push_cache("Cropping1D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("Cropping1D")?;
        crop_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.cropping.0,
            "Cropping1D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        crop_output_shape(input, &self.cropping.0, "Cropping1D")
    }
}
