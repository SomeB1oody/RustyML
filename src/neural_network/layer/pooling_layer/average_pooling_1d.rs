use crate::neural_network::layer::pooling_layer::layer_functions_1d_pooling;
use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::shape_helpers::calculate_output_shape_1d_pooling;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::pooling_layer::validation::{
    validate_all_dims_positive, validate_input_shape_dims, validate_pool_size_1d,
    validate_stride_1d,
};
use crate::neural_network::layer::pooling_layer::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::neural_network_trait::Layer;

/// 1D average pooling layer.
///
/// Computes the mean value over each pooling window along the length dimension.
/// Input tensor shape: `[batch_size, channels, length]`. Output tensor shape:
/// `[batch_size, channels, pooled_length]` where
/// `pooled_length = (length - pool_size) / stride + 1`.
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window
/// - `stride` - Step size of the pooling operation
/// - `input_shape` - Shape of the input tensor declared at construction time
/// - `forward_input_shape` - Shape of the most recent forward input, cached for backpropagation
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::Array3;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, channels, length]
/// // Batch size=2, 3 input channels, each channel has 8 elements
/// let mut input_data = Array3::zeros((2, 3, 8));
///
/// // Set test data to make average pooling results predictable
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..8 {
///             input_data[[b, c, i]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test AveragePooling1D with a Sequential model
/// let mut model = Sequential::new();
/// model
///     .add(AveragePooling1D::new(
///         2,              // Pool window size
///         vec![2, 3, 8],  // Input shape
///         Some(2),        // Stride (optional, defaults to pool_size if None)
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Output shape should be [2, 3, 4]
/// let output = model.predict(&x).unwrap();
/// assert_eq!(output.shape(), &[2, 3, 4]);
///
/// // Verify correctness of pooling results
/// // For window size of 2 and stride of 2, we expect the average of elements in each window
/// for b in 0..2 {
///     for c in 0..3 {
///         // First window (0,1) -> average should be (0+1)/2 = 0.5
///         assert_relative_eq!(output[[b, c, 0]], 0.5);
///         // Second window (2,3) -> average should be (2+3)/2 = 2.5
///         assert_relative_eq!(output[[b, c, 1]], 2.5);
///         // Third window (4,5) -> average should be (4+5)/2 = 4.5
///         assert_relative_eq!(output[[b, c, 2]], 4.5);
///         // Fourth window (6,7) -> average should be (6+7)/2 = 6.5
///         assert_relative_eq!(output[[b, c, 3]], 6.5);
///     }
/// }
/// ```
pub struct AveragePooling1D {
    pool_size: usize,
    stride: usize,
    input_shape: Vec<usize>,
    forward_input_shape: Option<Vec<usize>>,
}

impl AveragePooling1D {
    /// Creates a new 1D average pooling layer.
    ///
    /// If `stride` is None, it defaults to `pool_size`.
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    /// - `input_shape` - Input tensor shape `[batch_size, channels, length]`
    /// - `stride` - Optional stride of the pooling operation
    ///
    /// # Returns
    ///
    /// - `Result<AveragePooling1D, ModelError>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_shape` is not 3D, contains non-positive
    ///   dimensions, `pool_size` is zero or larger than the input length, or `stride` is zero
    pub fn new(
        pool_size: usize,
        input_shape: Vec<usize>,
        stride: Option<usize>,
    ) -> Result<Self, ModelError> {
        let stride = stride.unwrap_or(pool_size);

        // input validation
        validate_input_shape_dims(&input_shape, 3, "AveragePooling1D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_1d(pool_size, input_shape[2])?;
        validate_stride_1d(stride)?;

        Ok(AveragePooling1D {
            pool_size,
            stride,
            input_shape,
            forward_input_shape: None,
        })
    }
}

impl Layer for AveragePooling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 3D".to_string(),
            ));
        }

        // Cache the actual input shape for backward (only the shape is needed for averaging)
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
        );
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`](crate::neural_network::neural_network_trait::Layer::predict).
    fn predict(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(ModelError::InputValidationError(
                "input tensor is not 3D".to_string(),
            ));
        }

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
        );
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let input_shape = self.forward_input_shape.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run yet".to_string())
        })?;

        Ok(windowed_pool_backward(
            grad_output,
            input_shape,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
            None,
        ))
    }

    fn layer_type(&self) -> &str {
        "AveragePooling1D"
    }

    layer_functions_1d_pooling!();
}
