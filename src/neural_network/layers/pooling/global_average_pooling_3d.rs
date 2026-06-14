//! Global average pooling layer for 3D inputs

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_global_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::traits::Layer;

/// Global average pooling layer for 3D inputs
///
/// Computes the mean value across the depth, height, and width dimensions. Input tensor
/// shape: `[batch_size, channels, depth, height, width]`, output tensor shape:
/// `[batch_size, channels]`
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
/// // Create a Sequential model
/// let mut model = Sequential::new();
///
/// // Add a GlobalAveragePooling3D layer
/// model.add(GlobalAveragePooling3D::new());
///
/// // Test input tensor: [batch_size, channels, depth, height, width]
/// let input_data = Array::from_elem(IxDyn(&[2, 4, 8, 8, 8]), 1.0);
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Output shape should be [2, 4]
/// assert_eq!(output.shape(), &[2, 4]);
///
/// // All input values are 1.0, so all output values are 1.0
/// for b in 0..2 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`
#[derive(Debug)]
pub struct GlobalAveragePooling3D {
    /// Shape of the input tensor cached during the forward pass; backward only needs the
    /// shape, not the input values
    input_shape: Vec<usize>,
}

impl GlobalAveragePooling3D {
    /// Creates a new global average pooling 3D layer
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
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        // Backward only needs the shape, not the input values
        self.input_shape = input.shape().to_vec();

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        let (output, _) = global_pool_forward(input, PoolKind::Average);
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Forward must have run to populate the input shape
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
