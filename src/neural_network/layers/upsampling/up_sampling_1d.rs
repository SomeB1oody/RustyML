//! 1D upsampling layer that repeats each step of the step axis

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::upsampling::Interpolation;
use crate::neural_network::layers::upsampling::resize_engine::{
    upsample_backward, upsample_forward, upsample_output_shape, validate_factors,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Repeats each step of the step axis of a rank-3 tensor
///
/// The input shape is `[batch_size, steps, features]`. The output shape is
/// `[batch_size, steps * size, features]`. The batch axis and the feature axis pass through
/// unchanged
///
/// The layer holds no parameter. Each output step copies the input step it came from, so the
/// layer takes no interpolation argument
///
/// The layer is the decoder counterpart of
/// [`MaxPooling1D`](crate::neural_network::layers::pooling::max_pooling_1d::MaxPooling1D) and
/// [`AveragePooling1D`](crate::neural_network::layers::pooling::average_pooling_1d::AveragePooling1D).
/// A pooling stage of `pool_size` and an upsampling stage of the same `size` restore the
/// original length
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
/// // A rank-3 input: 1 sample, 2 steps, 2 features
/// let x = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(UpSampling1D::new(3).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let longer = model.predict(&x).unwrap();
///
/// // 2 steps become 6, and each input step covers 3 of them
/// assert_eq!(longer.shape(), &[1, 6, 2]);
/// assert_eq!(longer[[0, 0, 0]], 1.0);
/// assert_eq!(longer[[0, 2, 0]], 1.0);
/// assert_eq!(longer[[0, 3, 0]], 3.0);
/// ```
#[derive(Debug)]
pub struct UpSampling1D {
    /// Times the layer repeats each step
    size: usize,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl UpSampling1D {
    /// Creates a new UpSampling1D layer
    ///
    /// # Parameters
    ///
    /// - `size` - Times the layer repeats each step. A size of 1 leaves the length alone
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `UpSampling1D` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `size` is 0
    pub fn new(size: usize) -> Result<Self, Error> {
        validate_factors(&[size])?;
        Ok(UpSampling1D { size, built: None })
    }
}

impl LayerBase for UpSampling1D {
    fn layer_type(&self) -> &str {
        "UpSampling1D"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for UpSampling1D {
    /// Records the shape the layer enlarges. The layer holds no array, so nothing is allocated.
    /// The shape algebra checks the rank
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "UpSampling1D", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let output = upsample_forward(
            input,
            &[self.size],
            Interpolation::Nearest,
            3,
            "UpSampling1D",
        )?;

        if ctx.is_training() {
            ctx.push_cache("UpSampling1D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("UpSampling1D")?;

        upsample_backward(
            grad_output,
            Some(input_shape.as_slice()),
            &[self.size],
            Interpolation::Nearest,
            "UpSampling1D",
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        upsample_output_shape(input, &[self.size], "UpSampling1D")
    }
}
