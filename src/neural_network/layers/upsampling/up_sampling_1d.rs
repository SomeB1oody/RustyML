//! 1D upsampling layer that repeats each step of the step axis

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::upsampling::Interpolation;
use crate::neural_network::layers::upsampling::resize_engine::{
    upsample_backward, upsample_forward, upsample_summary, validate_factors,
};
use crate::neural_network::traits::Layer;

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
/// use rustyml::neural_network::sequential::Sequential;
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
/// let mut model = Sequential::new();
/// model
///     .add(UpSampling1D::new(3).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
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
        Ok(UpSampling1D {
            size,
            input_shape: None,
        })
    }
}

impl Layer for UpSampling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = upsample_forward(
            input,
            &[self.size],
            Interpolation::Nearest,
            3,
            "UpSampling1D",
        )?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        upsample_forward(
            input,
            &[self.size],
            Interpolation::Nearest,
            3,
            "UpSampling1D",
        )
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        upsample_backward(
            grad_output,
            self.input_shape.as_deref(),
            &[self.size],
            Interpolation::Nearest,
            "UpSampling1D",
        )
    }

    fn layer_type(&self) -> &str {
        "UpSampling1D"
    }

    fn output_shape(&self) -> String {
        upsample_summary(self.input_shape.as_deref(), &[self.size])
    }

    no_trainable_parameters_layer_functions!();
}
