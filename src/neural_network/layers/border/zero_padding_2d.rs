//! 2D zero-padding layer that adds zero rows and columns at the edges of an image

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::border::Border2D;
use crate::neural_network::layers::border::pad_crop_engine::{
    pad_backward, pad_forward, pad_summary,
};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Adds zero rows and columns at the edges of a rank-4 tensor
///
/// The input shape is `[batch_size, height, width, channels]`. The output shape is
/// `[batch_size, height + top + bottom, width + left + right, channels]`. The batch axis and
/// the channel axis pass through unchanged
///
/// The layer holds no parameter. Put it before a convolution with
/// [`PaddingType::Valid`](crate::neural_network::layers::convolution::PaddingType) to control
/// the border yourself. A convolution with `Same` padding splits an odd padding amount by its
/// own rule. This layer instead takes the amount at each of the 4 edges
///
/// [`Cropping2D`](crate::neural_network::layers::border::Cropping2D) is the inverse layer, and
/// it is also this layer's backward pass
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
///
/// // A rank-4 input: 2 samples, 4x4 pixels, 3 channels
/// let x = Array4::ones((2, 4, 4, 3)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(ZeroPadding2D::new(1))
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
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
            input_shape: None,
        }
    }
}

impl Layer for ZeroPadding2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = pad_forward(input, &self.padding.0, 4, "ZeroPadding2D")?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        pad_forward(input, &self.padding.0, 4, "ZeroPadding2D")
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        pad_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.padding.0,
            "ZeroPadding2D",
        )
    }

    fn layer_type(&self) -> &str {
        "ZeroPadding2D"
    }

    fn output_shape(&self) -> String {
        pad_summary(self.input_shape.as_deref(), &self.padding.0)
    }

    no_trainable_parameters_layer_functions!();
}
