use crate::neural_network::layer::pooling_layer::layer_functions_global_pooling;
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::pooling_layer::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::neural_network_trait::Layer;

/// Global average pooling layer for 3D inputs.
///
/// Computes the mean value across the depth, height, and width dimensions.
/// Input tensor shape: `[batch_size, channels, depth, height, width]`. Output tensor shape:
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
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a Sequential model with multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalAveragePooling3D layer
/// model.add(GlobalAveragePooling3D::new());
///
/// // Create test input tensor: [batch_size, channels, depth, height, width]
/// let input_data = Array::from_elem(IxDyn(&[2, 4, 8, 8, 8]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Check output shape - should be [2, 4]
/// assert_eq!(output.shape(), &[2, 4]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..2 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`.
pub struct GlobalAveragePooling3D {
    input_shape: Vec<usize>,
}

impl GlobalAveragePooling3D {
    /// Creates a new global average pooling 3D layer.
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling3D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling3D {
            input_shape: Vec::new(),
        }
    }
}

impl Default for GlobalAveragePooling3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalAveragePooling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        // Store input shape for backpropagation (only the shape is needed)
        self.input_shape = input.shape().to_vec();

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`](crate::neural_network::neural_network_trait::Layer::predict).
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Check that the forward pass has populated the input shape
        if self.input_shape.is_empty() {
            return Err(Error::forward_pass_not_run("GlobalAveragePooling3D"));
        }

        Ok(global_pool_backward(
            grad_output,
            &self.input_shape,
            PoolKind::Average,
            None,
        ))
    }

    fn layer_type(&self) -> &str {
        "GlobalAveragePooling3D"
    }

    layer_functions_global_pooling!();
}
