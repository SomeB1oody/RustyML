//! 1D zero-padding layer that adds zero steps at each end of the step axis

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::border::Border1D;
use crate::neural_network::layers::border::pad_crop_engine::{
    pad_backward, pad_forward, pad_output_shape,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Adds zero steps at each end of the step axis of a rank-3 tensor
///
/// The input shape is `[batch_size, steps, features]`. The output shape is
/// `[batch_size, steps + before + after, features]`. The batch axis and the feature axis pass
/// through unchanged.
///
/// The layer holds no parameter. It writes zeros in the new steps, so a later layer sees a
/// longer sequence whose ends carry no signal. A convolution over the padded sequence then
/// keeps its output length, without the layer itself choosing a padding mode.
///
/// [`Cropping1D`](crate::neural_network::layers::border::Cropping1D) is the inverse layer, and
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
/// use ndarray::Array3;
///
/// // A rank-3 input: 2 samples, 4 steps, 3 features
/// let x = Array3::ones((2, 4, 3)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(ZeroPadding1D::new((1, 2)))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let padded = model.predict(&x).unwrap();
///
/// // 1 zero step before the first step and 2 after the last, so 4 steps become 7
/// assert_eq!(padded.shape(), &[2, 7, 3]);
/// ```
#[derive(Debug)]
pub struct ZeroPadding1D {
    /// Zero steps to add at each end of the step axis
    padding: Border1D,
    /// Shape the layer was built for, batch axis first. `None` before the build.
    built: Option<Shape>,
}

impl ZeroPadding1D {
    /// Creates a new ZeroPadding1D layer
    ///
    /// # Parameters
    ///
    /// - `padding` - Zero steps to add at each end of the step axis. An integer gives an equal
    ///   amount at both ends. A `(before, after)` pair names each end. See [`Border1D`]
    ///
    /// # Returns
    ///
    /// - `Self` - New `ZeroPadding1D` layer instance
    pub fn new(padding: impl Into<Border1D>) -> Self {
        ZeroPadding1D {
            padding: padding.into(),
            built: None,
        }
    }
}

impl LayerBase for ZeroPadding1D {
    fn layer_type(&self) -> &str {
        "ZeroPadding1D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for ZeroPadding1D {
    /// Records the shape the pad runs over. The layer holds no array, so nothing is
    /// allocated. The shape algebra checks the rank.
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "ZeroPadding1D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("ZeroPadding1D"));
        }
        let output = pad_forward(input, &self.padding.0, 3, "ZeroPadding1D")?;

        if ctx.is_training() {
            ctx.push_cache("ZeroPadding1D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("ZeroPadding1D")?;
        pad_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &self.padding.0,
            "ZeroPadding1D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        pad_output_shape(input, &self.padding.0, "ZeroPadding1D")
    }
}
