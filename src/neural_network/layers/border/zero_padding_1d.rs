//! 1D zero-padding layer that adds zero steps at each end of the step axis

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::border::Border1D;
use crate::neural_network::layers::border::pad_crop_engine::{
    pad_backward, pad_forward, pad_summary,
};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Adds zero steps at each end of the step axis of a rank-3 tensor
///
/// The input shape is `[batch_size, steps, features]`. The output shape is
/// `[batch_size, steps + before + after, features]`. The batch axis and the feature axis pass
/// through unchanged
///
/// The layer holds no parameter. It writes zeros in the new steps, so a later layer sees a
/// longer sequence whose ends carry no signal. A convolution over the padded sequence then
/// keeps its output length, without the layer itself choosing a padding mode
///
/// [`Cropping1D`](crate::neural_network::layers::border::Cropping1D) is the inverse layer, and
/// it is also this layer's backward pass
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
/// // A rank-3 input: 2 samples, 4 steps, 3 features
/// let x = Array3::ones((2, 4, 3)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(ZeroPadding1D::new((1, 2)))
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
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
            input_shape: None,
        }
    }
}

impl Layer for ZeroPadding1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = pad_forward(input, &self.padding.0, 3, "ZeroPadding1D")?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        pad_forward(input, &self.padding.0, 3, "ZeroPadding1D")
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        pad_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.padding.0,
            "ZeroPadding1D",
        )
    }

    fn layer_type(&self) -> &str {
        "ZeroPadding1D"
    }

    fn output_shape(&self) -> String {
        pad_summary(self.input_shape.as_deref(), &self.padding.0)
    }

    no_trainable_parameters_layer_functions!();
}
