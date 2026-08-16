//! 3D upsampling layer that enlarges a volume by a whole-number factor per axis

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::upsampling::resize_engine::{
    upsample_backward, upsample_forward, upsample_summary, validate_factors,
};
use crate::neural_network::layers::upsampling::{Factor3D, Interpolation};
use crate::neural_network::traits::Layer;

/// Enlarges the 3 spatial axes of a rank-5 tensor
///
/// The input shape is `[batch_size, dim1, dim2, dim3, channels]`. The output shape multiplies
/// each spatial extent by its factor. The batch axis and the channel axis pass through
/// unchanged
///
/// The layer holds no parameter. Each output position copies the input position it came from,
/// so the layer takes no interpolation argument
///
/// The layer is the decoder counterpart of
/// [`MaxPooling3D`](crate::neural_network::layers::pooling::max_pooling_3d::MaxPooling3D) and
/// [`AveragePooling3D`](crate::neural_network::layers::pooling::average_pooling_3d::AveragePooling3D).
/// A volume model pairs each pooling stage with 1 upsampling stage of the same factor. Examples
/// include a model over a medical scan or a video clip
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
/// // A rank-5 input: 1 sample, a 2x1x1 volume, 2 channels
/// let x = Array5::from_shape_vec((1, 2, 1, 1, 2), vec![1.0, 2.0, 3.0, 4.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(UpSampling3D::new((2, 3, 1)).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let larger = model.predict(&x).unwrap();
///
/// // Each spatial axis grows by its own factor, and the channel axis stays put
/// assert_eq!(larger.shape(), &[1, 4, 3, 1, 2]);
/// assert_eq!(larger[[0, 0, 2, 0, 1]], 2.0);
/// assert_eq!(larger[[0, 3, 0, 0, 0]], 3.0);
/// ```
#[derive(Debug)]
pub struct UpSampling3D {
    /// Factor each spatial axis grows by
    size: Factor3D,
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
}

impl UpSampling3D {
    /// Creates a new UpSampling3D layer
    ///
    /// # Parameters
    ///
    /// - `size` - Factor each spatial axis grows by. An integer gives the same factor to all 3
    ///   axes. A triple names the factor of each axis. See [`Factor3D`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `UpSampling3D` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any factor is 0
    pub fn new(size: impl Into<Factor3D>) -> Result<Self, Error> {
        let size = size.into();
        validate_factors(&size.0)?;
        Ok(UpSampling3D {
            size,
            input_shape: None,
        })
    }
}

impl Layer for UpSampling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = upsample_forward(
            input,
            &self.size.0,
            Interpolation::Nearest,
            5,
            "UpSampling3D",
        )?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        upsample_forward(
            input,
            &self.size.0,
            Interpolation::Nearest,
            5,
            "UpSampling3D",
        )
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        upsample_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.size.0,
            Interpolation::Nearest,
            "UpSampling3D",
        )
    }

    fn layer_type(&self) -> &str {
        "UpSampling3D"
    }

    fn output_shape(&self) -> String {
        upsample_summary(self.input_shape.as_deref(), &self.size.0)
    }

    no_trainable_parameters_layer_functions!();
}
