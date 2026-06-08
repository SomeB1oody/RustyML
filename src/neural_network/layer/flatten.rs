use crate::error::{Error, Context};
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::no_trainable_parameters_layer_functions;
use crate::neural_network::neural_network_trait::Layer;
use ndarray::IxDyn;

/// Flattens a 3D, 4D, or 5D tensor into a 2D tensor.
///
/// This layer reshapes inputs from feature extraction layers into a format suitable for dense layers.
/// Input shapes are \[batch_size, features, length\], \[batch_size, channels, height, width\], or
/// \[batch_size, channels, depth, height, width\]. Output shape is always \[batch_size, flattened_features\],
/// where flattened_features is the product of all dimensions except batch_size.
///
/// # Fields
///
/// - `flattened_features` - Number of features after flattening (product of all dimensions except batch)
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::Array4;
///
/// // Create a 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 channels, each 4x4 pixels
/// let x = Array4::ones((2, 3, 4, 4)).into_dyn();
///
/// // Build a model containing a Flatten layer
/// let mut model = Sequential::new();
/// model
///     .add(Flatten::new(vec![2, 3, 4, 4]).unwrap())
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let flattened = model.predict(&x).unwrap();
///
/// // Check output shape - should be [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
pub struct Flatten {
    flattened_features: usize,
    input_cache: Option<Tensor>,
}

impl Flatten {
    /// Creates a new Flatten layer.
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Input tensor shape, such as [batch_size, features, length],
    ///   [batch_size, channels, height, width], or [batch_size, channels, depth, height, width]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Flatten` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If `input_shape` has fewer than 2 dimensions or contains a zero
    pub fn new(input_shape: Vec<usize>) -> Result<Self, Error> {
        // Validate input shape dimensions
        if input_shape.len() < 2 {
            return Err(Error::invalid_input(format!(
                "Input shape must have at least 2 dimensions [batch_size, features...], got {}D",
                input_shape.len()
            )));
        }

        // Ensure all dimensions are greater than 0
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
            input_cache: None,
        })
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input dimensions
        let input_shape = input.shape();
        if input_shape.len() < 3 || input_shape.len() > 5 {
            return Err(Error::invalid_input(format!(
                "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
                input_shape.len()
            )));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        // Reshape to flatten the tensor
        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`](crate::neural_network::neural_network_trait::Layer::predict).
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input dimensions
        let input_shape = input.shape();
        if input_shape.len() < 3 || input_shape.len() > 5 {
            return Err(Error::invalid_input(format!(
                "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
                input_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        // Reshape to flatten the tensor
        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape().to_vec();

            // Validate gradient output shape
            let expected_grad_shape = [input_shape[0], input_shape[1..].iter().product()];
            if grad_output.shape() != expected_grad_shape {
                return Err(Error::shape_mismatch(
                    expected_grad_shape,
                    grad_output.shape(),
                ));
            }

            // Reshape gradient back to input shape
            let reshaped_grad = grad_output
                .to_shape(IxDyn(&input_shape))
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
        // `None` is the batch placeholder used by the other definition-time-shaped layers
        // (Dense, SimpleRNN, GRU, LSTM), whose batch size is not fixed until forward.
        format!("(None, {})", self.flattened_features)
    }

    no_trainable_parameters_layer_functions!();
}
