//! Global average pooling layer for 2D inputs.

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_global_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::traits::Layer;

/// Global average pooling layer for 2D inputs
///
/// Computes the mean value across the height and width dimensions.
/// Input tensor shape: `[batch_size, height, width, channels]`. Output tensor shape:
/// `[batch_size, channels]`.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model with multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalAveragePooling2D layer
/// model.add(GlobalAveragePooling2D::new());
///
/// // Create test input tensor: [batch_size, height, width, channels]
/// let input_data = Array::from_elem(IxDyn(&[3, 5, 5, 4]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Check output shape - should be [3, 4]
/// assert_eq!(output.shape(), &[3, 4]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..3 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// The pass runs in parallel when its estimated element count (`batch * positions * channels`)
/// clears the gate in [`tuning::pool`](crate::tuning::pool). The gate does not depend on any
/// fixed shape.
#[derive(Debug)]
pub struct GlobalAveragePooling2D {
    /// Shape of the input tensor cached during the forward pass
    input_shape: Vec<usize>,
}

impl GlobalAveragePooling2D {
    /// Creates a new global average pooling 2D layer
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling2D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling2D {
            input_shape: Vec::new(),
        }
    }
}

impl Default for GlobalAveragePooling2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalAveragePooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Cache the input shape for the backward pass
        self.input_shape = input.shape().to_vec();

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    /// Runs the forward pass for inference. Writes no cache. See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // An empty shape means forward has not run
        if self.input_shape.is_empty() {
            return Err(Error::forward_pass_not_run("GlobalAveragePooling2D"));
        }

        Ok(global_pool_backward(
            grad_output,
            &self.input_shape,
            PoolKind::Average,
            None,
        ))
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling2D"
    }

    layer_functions_global_pooling!();
}
