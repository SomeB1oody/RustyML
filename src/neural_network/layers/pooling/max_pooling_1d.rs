//! 1D max pooling layer that selects the maximum value within each pooling window along the
//! length dimension.

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
/// Selects the maximum value within each pooling window along the length dimension.
/// Input tensor shape: `[batch_size, length, channels]`. Output tensor shape:
/// `[batch_size, pooled_length, channels]`. With `Valid` padding,
/// `pooled_length = (length - pool_size) / stride + 1`.
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
/// // Create an input tensor: [batch_size, length, channels]
/// // batch size 2, 8 elements per channel, 3 input channels
/// let mut input_data = Array3::zeros((2, 8, 3));
///
/// // Set test data to make the max pooling results predictable
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..8 {
///              input_data[[b, i, c]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test MaxPooling1D with a Sequential model
/// let mut model = Sequential::new();
/// model
///     // stride defaults to pool_size (2) and padding defaults to Valid
///     .add(MaxPooling1D::new(2, vec![2, 8, 3]).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Output shape should be [2, 4, 3]
/// let output = model.predict(&x).unwrap();
/// assert_eq!(output.shape(), &[2, 4, 3]);
///
/// // Verify the pooling results
/// // Window size 2 and stride 2: each window keeps its maximum element
/// for b in 0..2 {
///     for c in 0..3 {
///         // First window (0, 1): max value 1.0
///        assert_relative_eq!(output[[b, 0, c]], 1.0);
///         // Second window (2, 3): max value 3.0
///         assert_relative_eq!(output[[b, 1, c]], 3.0);
///         // Third window (4, 5): max value 5.0
///         assert_relative_eq!(output[[b, 2, c]], 5.0);
///         // Fourth window (6, 7): max value 7.0
///         assert_relative_eq!(output[[b, 3, c]], 7.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// The pass runs in parallel when its estimated element count
/// (`batch * out_positions * channels * window taps`) clears the gate in
/// [`tuning::pool`](crate::tuning::pool). The gate does not depend on any fixed shape.
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
    /// Shape of the most recent forward input, cached for the backward pass
    forward_input_shape: Option<Vec<usize>>,
    /// Cached flat per-output arg-max indices used for the backward pass
    argmax: Option<Vec<usize>>,
}

impl MaxPooling1D {
    /// Creates a new 1D max pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    /// - `input_shape` - Input tensor shape `[batch_size, length, channels]`
    ///
    /// # Returns
    ///
    /// - `Result<MaxPooling1D, Error>` - New layer instance on success
    ///
    /// # Notes
    ///
    /// The stride defaults to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override
    /// them with [`MaxPooling1D::with_stride`] and [`MaxPooling1D::with_padding`].
    ///
    /// # Errors
    ///
    /// - `Error::DimensionMismatch` - If `input_shape` is not 3D
    /// - `Error::InvalidInput` - If any `input_shape` dimension is zero
    /// - `Error::InvalidParameter` - If `pool_size` is zero or larger than the input length
    pub fn new(pool_size: usize, input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_input_shape_dims(&input_shape, 3, "MaxPooling1D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_1d(pool_size, input_shape[1])?;

        Ok(MaxPooling1D {
            pool_size,
            stride: pool_size,
            input_shape,
            padding: PaddingType::Valid,
            forward_input_shape: None,
            argmax: None,
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
    /// - `padding` - `Valid` (no padding) or `Same` (pad so the output covers the input, with
    ///   padded cells excluded from each window)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }
}

impl Layer for MaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        // Cache the input shape and arg-max positions for the backward pass
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, argmax) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Max,
            self.padding,
        );
        self.argmax = argmax;
        Ok(output)
    }

    /// Runs the forward pass for inference. Writes no cache. See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Max,
            self.padding,
        );
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
