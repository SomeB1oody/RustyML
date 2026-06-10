use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_global_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::traits::Layer;

/// Global average pooling layer for 1D inputs.
///
/// Computes the mean value across the length dimension.
/// Input tensor shape: `[batch_size, channels, length]`. Output tensor shape:
/// `[batch_size, channels]`.
///
/// # Fields
///
/// - `input_shape` - Shape of the input tensor cached during the forward pass
///   (global average pooling only needs the shape, not the input values, in backward)
///
/// # Examples
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
/// // Add a GlobalAveragePooling1D layer
/// model.add(GlobalAveragePooling1D::new());
///
/// // Create test input tensor: [batch_size, channels, length]
/// let input_data = Array::from_elem(IxDyn(&[2, 3, 4]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Check output shape - should be [2, 3]
/// assert_eq!(output.shape(), &[2, 3]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..2 {
///     for c in 0..3 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`.
#[derive(Debug)]
pub struct GlobalAveragePooling1D {
    input_shape: Vec<usize>,
}

impl GlobalAveragePooling1D {
    /// Creates a new global average pooling 1D layer.
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling1D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling1D {
            input_shape: Vec::new(),
        }
    }
}

impl Default for GlobalAveragePooling1D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalAveragePooling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        // Store input shape for backpropagation (only the shape is needed)
        self.input_shape = input.shape().to_vec();

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Check that the forward pass has populated the input shape
        if self.input_shape.is_empty() {
            return Err(Error::forward_pass_not_run("GlobalAveragePooling1D"));
        }

        Ok(global_pool_backward(
            grad_output,
            &self.input_shape,
            PoolKind::Average,
            None,
        ))
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling1D"
    }

    layer_functions_global_pooling!();
}
