//! Global max pooling layer for 3D inputs, reducing each channel to its maximum value

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_global_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::traits::Layer;

/// Global max pooling layer for 3D inputs
///
/// Selects the maximum value across the depth, height, and width dimensions. Input
/// tensor shape: `[batch_size, channels, depth, height, width]`. Output tensor shape:
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
/// // Create a Sequential model with multiple layers
/// let mut model = Sequential::new();
///
/// // Add GlobalMaxPooling3D layer
/// model.add(GlobalMaxPooling3D::new());
///
/// // Create test input tensor: [batch_size, channels, depth, height, width]
/// let input_data = Array::from_elem(IxDyn(&[2, 3, 4, 5, 5]), 1.0);
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
/// Parallel execution is used when `batch_size * channels >= 32`
#[derive(Debug)]
pub struct GlobalMaxPooling3D {
    /// Shape of the input tensor cached during the forward pass
    input_shape: Vec<usize>,
    /// Cached flat per-channel arg-max indices for backpropagation
    argmax: Option<Vec<usize>>,
}

impl GlobalMaxPooling3D {
    /// Creates a new global max pooling 3D layer
    ///
    /// # Returns
    ///
    /// - `GlobalMaxPooling3D` - New layer instance
    pub fn new() -> Self {
        GlobalMaxPooling3D {
            input_shape: Vec::new(),
            argmax: None,
        }
    }
}

impl Default for GlobalMaxPooling3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalMaxPooling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        // Cache the input shape and arg-max positions for backpropagation
        self.input_shape = input.shape().to_vec();

        let (output, argmax) = global_pool_forward(input, PoolKind::Max);
        self.argmax = argmax;
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        Ok(global_pool_forward(input, PoolKind::Max).0)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(argmax) = &self.argmax {
            Ok(global_pool_backward(
                grad_output,
                &self.input_shape,
                PoolKind::Max,
                Some(argmax),
            ))
        } else {
            Err(Error::forward_pass_not_run("GlobalMaxPooling3D"))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling3D"
    }

    layer_functions_global_pooling!();
}
