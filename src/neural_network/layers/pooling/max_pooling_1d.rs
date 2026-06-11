//! 1D max pooling layer that selects the maximum value within each window along the length dimension

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_1d_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::layers::pooling::validation::{
    validate_all_dims_positive, validate_input_shape_dims, validate_pool_size_1d,
    validate_stride_1d,
};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_1d_pooling;
use crate::neural_network::traits::Layer;

/// 1D max pooling layer
///
/// Selects the maximum value within each pooling window along the length dimension. Input
/// tensor shape `[batch_size, channels, length]`, output tensor shape
/// `[batch_size, channels, pooled_length]` where
/// `pooled_length = (length - pool_size) / stride + 1`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array3;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, channels, length]
/// // batch_size=2, 3 input channels, 8 elements per channel
/// let mut input_data = Array3::zeros((2, 3, 8));
///
/// // Set test data to make max pooling results predictable
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..8 {
///              input_data[[b, c, i]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test MaxPooling1D using Sequential model
/// let mut model = Sequential::new();
/// model
///     .add(MaxPooling1D::new(
///         2,              // pool window size
///         vec![2, 3, 8],  // input shape
///         Some(2),        // stride
///         PaddingType::Valid,
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, None, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Output shape should be [2, 3, 4]
/// let output = model.predict(&x).unwrap();
/// assert_eq!(output.shape(), &[2, 3, 4]);
///
/// // Verify correctness of pooling results
/// // For window size 2 and stride 2, the maximum element in each window is selected
/// for b in 0..2 {
///     for c in 0..3 {
///         // First window (0,1) -> max value should be 1.0
///        assert_relative_eq!(output[[b, c, 0]], 1.0);
///         // Second window (2,3) -> max value should be 3.0
///         assert_relative_eq!(output[[b, c, 1]], 3.0);
///         // Third window (4,5) -> max value should be 5.0
///         assert_relative_eq!(output[[b, c, 2]], 5.0);
///         // Fourth window (6,7) -> max value should be 7.0
///         assert_relative_eq!(output[[b, c, 3]], 7.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`
#[derive(Debug)]
pub struct MaxPooling1D {
    /// Size of the pooling window
    pool_size: usize,
    /// Step size of the pooling operation
    stride: usize,
    /// Shape of the input tensor declared at construction time
    input_shape: Vec<usize>,
    /// Padding mode applied around the input before pooling
    padding: PaddingType,
    /// Shape of the most recent forward input, cached for backpropagation
    forward_input_shape: Option<Vec<usize>>,
    /// Cached flat per-output arg-max indices used for backpropagation
    argmax: Option<Vec<usize>>,
}

impl MaxPooling1D {
    /// Creates a new 1D max pooling layer
    ///
    /// If `stride` is None, it defaults to `pool_size`
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    /// - `input_shape` - Input tensor shape `[batch_size, channels, length]`
    /// - `stride` - Optional stride of the pooling operation
    ///
    /// # Returns
    ///
    /// - `Result<MaxPooling1D, Error>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - `Error::DimensionMismatch` - If `input_shape` is not 3D
    /// - `Error::InvalidInput` - If `input_shape` contains a zero dimension
    /// - `Error::InvalidParameter` - If `pool_size` is zero or larger than the input length, or
    ///   `stride` is zero
    pub fn new(
        pool_size: usize,
        input_shape: Vec<usize>,
        stride: Option<usize>,
        padding: PaddingType,
    ) -> Result<Self, Error> {
        let stride = stride.unwrap_or(pool_size);

        // input validation
        validate_input_shape_dims(&input_shape, 3, "MaxPooling1D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_1d(pool_size, input_shape[2])?;
        validate_stride_1d(stride)?;

        Ok(MaxPooling1D {
            pool_size,
            stride,
            input_shape,
            padding,
            forward_input_shape: None,
            argmax: None,
        })
    }
}

impl Layer for MaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        // Cache the actual input shape and arg-max positions for the backward pass
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, argmax) =
            windowed_pool_forward(input, &[self.pool_size], &[self.stride], PoolKind::Max, self.padding);
        self.argmax = argmax;
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        let (output, _) =
            windowed_pool_forward(input, &[self.pool_size], &[self.stride], PoolKind::Max, self.padding);
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let input_shape = self
            .forward_input_shape
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("MaxPooling1D"))?;
        let argmax = self
            .argmax
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("MaxPooling1D"))?;

        Ok(windowed_pool_backward(
            grad_output,
            input_shape,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Max,
            Some(argmax),
            self.padding,
        ))
    }

    fn layer_type(&self) -> &str {
        "MaxPooling1D"
    }

    layer_functions_1d_pooling!();
}
