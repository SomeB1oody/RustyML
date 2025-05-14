use crate::neural_network::layer::LayerWeight;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::IxDyn;

/// A layer that flattens a 4D tensor into a 2D tensor.
///
/// This layer is typically used in convolutional networks to transform the output of feature extraction layers
/// (such as convolutional or pooling layers) into a format that can be processed by dense (fully connected) layers.
///
/// # Input Shape
///
/// Input is a 4D tensor with shape \[batch_size, channels, height, width\]
///
/// # Output Shape
///
/// Output is a 2D tensor with shape \[batch_size, channels * height * width\]
///
/// # Fields
///
/// - `output_shape` - the output shape of the layer, in the format \[batch_size, channels * height * width\]
/// - `input_cache` - Cached input tensor from the forward pass, used during backpropagation
///
/// # Example
///
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array4;
///
/// // Create a 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 channels, each 4x4 pixels
/// let x = Array4::ones((2, 3, 4, 4)).into_dyn();
///
/// // Build a model containing a Flatten layer
/// let mut model = Sequential::new();
/// model
///     .add(Flatten::new(vec![2, 3, 4, 4]))
///     .compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let flattened = model.predict(&x);
///
/// // Check output shape - should be [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
pub struct Flatten {
    output_shape: Vec<usize>,
    input_cache: Option<Tensor>,
}

impl Flatten {
    /// Creates a new Flatten layer.
    ///
    /// # Parameters
    ///
    /// * `input_shape` - The shape of the input tensor, format is \[batch_size, channels, height, width\]
    ///
    /// # Returns
    ///
    /// * `Self` - A new `Flatten` layer instance
    pub fn new(input_shape: Vec<usize>) -> Self {
        assert_eq!(input_shape.len(), 4, "Input shape must be 4-dimensional");

        let batch_size = input_shape[0];
        let flattened_features = input_shape[1..].iter().product();
        let output_shape = vec![batch_size, flattened_features];

        Flatten {
            output_shape,
            input_cache: None,
        }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        // Create new shape
        let output = input.clone();
        output
            .into_shape_with_order(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let Some(input) = &self.input_cache {
            let input_shape = input.shape().to_vec();

            // Reshape gradient back to input shape
            let reshaped_grad = grad_output
                .clone()
                .into_shape_with_order(IxDyn(&input_shape))
                .unwrap();

            Ok(reshaped_grad)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run yet".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "Flatten"
    }

    fn output_shape(&self) -> String {
        format!("({}, {})", self.output_shape[0], self.output_shape[1])
    }

    fn param_count(&self) -> usize {
        // Flatten layer has no trainable parameters
        0
    }

    fn update_parameters_sgd(&mut self, _lr: f32) {
        // No parameters to update
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // No parameters to update
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // No parameters to update
    }

    fn get_weights(&self) -> LayerWeight {
        // Flatten layer has no weights
        LayerWeight::Empty
    }
}
