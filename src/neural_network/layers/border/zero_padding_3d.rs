//! 3D zero-padding layer that adds zero planes at the 6 faces of a volume

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::border::Border3D;
use crate::neural_network::layers::border::pad_crop_engine::{
    pad_backward, pad_forward, pad_summary,
};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

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
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array5;
///
/// // A rank-5 input: 2 samples, a 3x4x4 volume, 1 channel
/// let x = Array5::ones((2, 3, 4, 4, 1)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(ZeroPadding3D::new((1, 2, 0)))
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
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
            input_shape: None,
        }
    }
}

impl Layer for ZeroPadding3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = pad_forward(input, &self.padding.0, 5, "ZeroPadding3D")?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        pad_forward(input, &self.padding.0, 5, "ZeroPadding3D")
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        pad_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.padding.0,
            "ZeroPadding3D",
        )
    }

    fn layer_type(&self) -> &str {
        "ZeroPadding3D"
    }

    fn output_shape(&self) -> String {
        pad_summary(self.input_shape.as_deref(), &self.padding.0)
    }

    no_trainable_parameters_layer_functions!();
}
