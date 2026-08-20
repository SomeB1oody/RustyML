//! Flatten layer that reshapes a 3D, 4D, or 5D tensor into a 2D tensor for dense layers

use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;
use ndarray::IxDyn;

/// Flattens a 3D, 4D, or 5D tensor into a 2D tensor
///
/// Reshapes inputs from feature extraction layers into a format suitable for dense layers
///
/// The reshape itself is layout-agnostic. It collapses every axis after the batch axis in C
/// order, without regard to which feature lands at which output index. Under the crate's
/// channels-last layout, the channel axis is innermost. The flattened vector then runs position
/// by position, with all channels of one position adjacent, rather than plane by plane. A
/// `Dense` layer trained against the other ordering then reads its inputs permuted, even though
/// its weight shape stays the same. This is why saved models carry a format version (see
/// [`MODEL_FORMAT_VERSION`](crate::neural_network::layers::serialize_model::MODEL_FORMAT_VERSION))
/// instead of relying on a shape check to catch the mismatch
///
/// Input shapes are `[batch_size, length, features]`, `[batch_size, height, width, channels]`,
/// or `[batch_size, depth, height, width, channels]`. The output shape is always
/// `[batch_size, flattened_features]`, where `flattened_features` is the product of all
/// dimensions except the batch size
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
/// // Create a 4D input tensor: [batch_size, height, width, channels]
/// // Batch size=2, 4x4 pixels, 3 channels
/// let x = Array4::ones((2, 4, 4, 3)).into_dyn();
///
/// // Build a model containing a Flatten layer
/// let mut model = Sequential::new();
/// model
///     .add(Flatten::new(vec![2, 4, 4, 3]).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let flattened = model.predict(&x).unwrap();
///
/// // The output shape should be [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
#[derive(Debug)]
pub struct Flatten {
    /// Number of features after flattening (product of all dimensions except batch)
    flattened_features: usize,
    /// Shape of the most recent forward input. The backward pass restores it
    ///
    /// A flatten moves no data, so the backward pass needs the shape alone
    input_shape: Option<Vec<usize>>,
}

impl Flatten {
    /// Creates a new Flatten layer
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Input tensor shape, such as `[batch_size, length, features]`,
    ///   `[batch_size, height, width, channels]`, or
    ///   `[batch_size, depth, height, width, channels]`
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Flatten` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If `input_shape` has fewer than 2 dimensions or contains a zero
    pub fn new(input_shape: Vec<usize>) -> Result<Self, Error> {
        if input_shape.len() < 2 {
            return Err(Error::invalid_input(format!(
                "Input shape must have at least 2 dimensions [batch_size, features...], got {}D",
                input_shape.len()
            )));
        }

        for (i, &dim) in input_shape.iter().enumerate() {
            if dim == 0 {
                return Err(Error::invalid_input(format!(
                    "Dimension {} must be greater than 0, got {}",
                    i, dim
                )));
            }
        }

        let flattened_features = input_shape[1..].iter().product();

        Ok(Flatten {
            flattened_features,
            input_shape: None,
        })
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let input_shape = input.shape();
        if input_shape.len() < 3 || input_shape.len() > 5 {
            return Err(Error::invalid_input(format!(
                "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
                input_shape.len()
            )));
        }

        self.input_shape = Some(input_shape.to_vec());

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_shape = input.shape();
        if input_shape.len() < 3 || input_shape.len() > 5 {
            return Err(Error::invalid_input(format!(
                "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
                input_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(input_shape) = &self.input_shape {
            let expected_grad_shape = [input_shape[0], input_shape[1..].iter().product()];
            if grad_output.shape() != expected_grad_shape {
                return Err(Error::shape_mismatch(
                    expected_grad_shape,
                    grad_output.shape(),
                ));
            }

            // Reshape gradient back to input shape
            let reshaped_grad = grad_output
                .to_shape(IxDyn(input_shape))
                .context("reshape gradient")?
                .to_owned();

            Ok(reshaped_grad)
        } else {
            Err(Error::forward_pass_not_run("Flatten"))
        }
    }

    fn layer_type(&self) -> &str {
        "Flatten"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.flattened_features)
    }

    no_trainable_parameters_layer_functions!();
}
