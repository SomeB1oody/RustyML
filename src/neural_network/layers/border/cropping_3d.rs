//! 3D cropping layer that removes planes at the 6 faces of a volume

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::border::Border3D;
use crate::neural_network::layers::border::pad_crop_engine::{
    crop_backward, crop_forward, crop_summary,
};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Removes planes at the 6 faces of a rank-5 tensor
///
/// The input shape is `[batch_size, depth, height, width, channels]`. Each of the 3 spatial
/// axes shrinks by the amount at its 2 ends. The batch axis and the channel axis pass through
/// unchanged
///
/// The layer holds no parameter. Each spatial axis must keep at least 1 position, so the
/// forward pass fails when the 2 amounts on an axis together reach its extent
///
/// [`ZeroPadding3D`](crate::neural_network::layers::border::ZeroPadding3D) is the inverse
/// layer, and it is also this layer's backward pass
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
/// // A rank-5 input: 2 samples, a 4x6x6 volume, 1 channel
/// let x = Array5::ones((2, 4, 6, 6, 1)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(Cropping3D::new(1))
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
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
            input_shape: None,
        }
    }
}

impl Layer for Cropping3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = crop_forward(input, &self.cropping.0, 5, "Cropping3D")?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        crop_forward(input, &self.cropping.0, 5, "Cropping3D")
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        crop_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.cropping.0,
            "Cropping3D",
        )
    }

    fn layer_type(&self) -> &str {
        "Cropping3D"
    }

    fn output_shape(&self) -> String {
        crop_summary(self.input_shape.as_deref(), &self.cropping.0)
    }

    no_trainable_parameters_layer_functions!();
}
