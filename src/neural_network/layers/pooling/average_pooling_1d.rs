//! 1D average pooling layer that computes the mean over each window along the length dimension

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

/// 1D average pooling layer
///
/// Computes the mean value over each pooling window along the length dimension
/// Input tensor shape: `[batch_size, channels, length]`. Output tensor shape:
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
///     // stride defaults to pool_size (2) and padding defaults to Valid
///     .add(AveragePooling1D::new(2, vec![2, 3, 8]).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
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
#[derive(Debug)]
pub struct AveragePooling1D {
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
}

impl AveragePooling1D {
    /// Creates a new 1D average pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    /// - `input_shape` - Input tensor shape `[batch_size, channels, length]`
    ///
    /// # Notes
    ///
    /// The stride defaults to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override
    /// them with [`AveragePooling1D::with_stride`] and [`AveragePooling1D::with_padding`].
    ///
    /// # Returns
    ///
    /// - `Result<AveragePooling1D, Error>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `input_shape` is not 3D
    /// - [`Error::InvalidInput`] if `input_shape` contains non-positive dimensions
    /// - [`Error::InvalidParameter`] if `pool_size` is zero or larger than the input length
    pub fn new(pool_size: usize, input_shape: Vec<usize>) -> Result<Self, Error> {
        // input validation
        validate_input_shape_dims(&input_shape, 3, "AveragePooling1D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_1d(pool_size, input_shape[2])?;

        Ok(AveragePooling1D {
            pool_size,
            stride: pool_size,
            input_shape,
            padding: PaddingType::Valid,
            forward_input_shape: None,
        })
    }

    /// Sets the pooling stride (defaults to `pool_size`)
    ///
    /// # Parameters
    ///
    /// - `stride` - Stride of the pooling operation
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error if `stride` is zero
    pub fn with_stride(mut self, stride: usize) -> Result<Self, Error> {
        validate_stride_1d(stride)?;
        self.stride = stride;
        Ok(self)
    }

    /// Sets the padding mode (defaults to [`PaddingType::Valid`])
    ///
    /// # Parameters
    ///
    /// - `padding` - `Valid` (no padding) or `Same` (pad so the output covers the input)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }
}

impl Layer for AveragePooling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        // Cache the actual input shape for backward (only the shape is needed for averaging)
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
            self.padding,
        );
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
            self.padding,
        );
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let input_shape = self
            .forward_input_shape
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("AveragePooling1D"))?;

        Ok(windowed_pool_backward(
            grad_output,
            input_shape,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
            None,
            self.padding,
        ))
    }

    fn layer_type(&self) -> &str {
        "AveragePooling1D"
    }

    layer_functions_1d_pooling!();
}
