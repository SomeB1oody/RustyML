use crate::neural_network::layer::pooling_layer::layer_functions_global_pooling;
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::pooling_layer::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::neural_network_trait::Layer;

/// Global max pooling layer for 2D inputs.
///
/// Selects the maximum value across the height and width dimensions.
/// Input tensor shape: `[batch_size, channels, height, width]`. Output tensor shape:
/// `[batch_size, channels]`.
///
/// # Fields
///
/// - `input_shape` - Shape of the input tensor cached during the forward pass
/// - `argmax` - Cached flat per-channel arg-max indices for backpropagation
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
/// // Create a Sequential model containing multiple layers
/// let mut model = Sequential::new();
///
/// // Add a GlobalMaxPooling2D layer
/// model.add(GlobalMaxPooling2D::new());
///
/// // Create a test input tensor: [batch_size, channels, height, width]
/// let input_data = Array::from_elem(IxDyn(&[3, 4, 5, 5]), 1.0);
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
/// Parallel execution is used when `batch_size * channels >= 32`.
pub struct GlobalMaxPooling2D {
    input_shape: Vec<usize>,
    argmax: Option<Vec<usize>>,
}

impl GlobalMaxPooling2D {
    /// Creates a new global max pooling 2D layer.
    ///
    /// # Returns
    ///
    /// - `GlobalMaxPooling2D` - New layer instance
    pub fn new() -> Self {
        GlobalMaxPooling2D {
            input_shape: Vec::new(),
            argmax: None,
        }
    }
}

impl Default for GlobalMaxPooling2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalMaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Store the input shape and arg-max positions for backpropagation
        self.input_shape = input.shape().to_vec();

        let (output, argmax) = global_pool_forward(input, PoolKind::Max);
        self.argmax = argmax;
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
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
            Err(Error::forward_pass_not_run("GlobalMaxPooling2D"))
        }
    }

    fn layer_type(&self) -> &str {
        "GlobalMaxPooling2D"
    }

    layer_functions_global_pooling!();
}
