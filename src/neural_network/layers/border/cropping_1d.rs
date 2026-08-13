//! 1D cropping layer that removes steps at each end of the step axis

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::border::Border1D;
use crate::neural_network::layers::border::pad_crop_engine::{
    crop_backward, crop_forward, crop_summary,
};
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Removes steps at each end of the step axis of a rank-3 tensor
///
/// The input shape is `[batch_size, steps, features]`. The output shape is
/// `[batch_size, steps - before - after, features]`. The batch axis and the feature axis pass
/// through unchanged
///
/// The layer holds no parameter. At least 1 step must remain, so the forward pass fails when
/// the 2 amounts together reach the extent of the step axis
///
/// [`ZeroPadding1D`](crate::neural_network::layers::border::ZeroPadding1D) is the inverse
/// layer, and it is also this layer's backward pass
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
/// // A rank-3 input: 2 samples, 6 steps, 3 features
/// let x = Array3::ones((2, 6, 3)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(Cropping1D::new((1, 2)))
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Shape of the most recent forward input. The backward pass needs it to size the gradient
    input_shape: Option<Vec<usize>>,
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
            input_shape: None,
        }
    }
}

impl Layer for Cropping1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let output = crop_forward(input, &self.cropping.0, 3, "Cropping1D")?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        crop_forward(input, &self.cropping.0, 3, "Cropping1D")
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        crop_backward(
            grad_output,
            self.input_shape.as_deref(),
            &self.cropping.0,
            "Cropping1D",
        )
    }

    fn layer_type(&self) -> &str {
        "Cropping1D"
    }

    fn output_shape(&self) -> String {
        crop_summary(self.input_shape.as_deref(), &self.cropping.0)
    }

    no_trainable_parameters_layer_functions!();
}
