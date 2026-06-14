//! 3D max pooling layer that downsamples a 5D tensor by taking the maximum over each pooling window

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_3d_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::layers::pooling::validation::{
    validate_all_dims_positive, validate_input_shape_dims, validate_pool_size_3d,
    validate_strides_3d,
};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_3d_pooling;
use crate::neural_network::traits::Layer;

/// 3D max pooling layer
///
/// Selects the maximum value within each pooling window across depth, height, and width
/// Input tensor shape: `[batch_size, channels, depth, height, width]`. Output tensor shape:
/// `[batch_size, channels, pooled_depth, pooled_height, pooled_width]` where
/// `pooled_depth = (depth - pool_size_d) / stride_d + 1`,
/// `pooled_height = (height - pool_size_h) / stride_h + 1`, and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::{Array5, ArrayD};
///
/// // Create a Sequential model for 3D data processing
/// let mut model = Sequential::new();
///
/// // Add MaxPooling3D layer to the model
/// // stride defaults to pool_size (2, 2, 2) and padding defaults to Valid
/// model.add(MaxPooling3D::new((2, 2, 2), vec![1, 16, 32, 32, 32]).unwrap());
///
/// // Compile the model with optimizer and loss function
/// model.compile(
///     RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(),    // RMSprop optimizer
///     MeanSquaredError::new()              // Mean Squared Error loss
/// );
///
/// // Create sample 3D input data (e.g., 3D medical images or volumetric data)
/// // Input: [1 batch, 16 channels, 32x32x32 3D volume]
/// let input_data = Array5::from_shape_fn((1, 16, 32, 32, 32), |(b, c, d, h, w)| {
///     // Generate sample data with spatial patterns
///     ((d + h + w) as f32 * 0.1) + (c as f32 * 0.01)
/// }).into_dyn();
///
/// // Create target data for training (output shape: [1, 16, 16, 16, 16])
/// let target_data = Array5::ones((1, 16, 16, 16, 16)).into_dyn();
///
/// // Display model architecture
/// model.summary();
///
/// // Train the model
/// model.fit(&input_data, &target_data, 5).unwrap();
///
/// // Make predictions on new data
/// let predictions = model.predict(&input_data).unwrap();
/// println!("Output shape after max pooling: {:?}", predictions.shape());
/// // Expected output: [1, 16, 16, 16, 16] (spatial dimensions reduced by factor of 2)
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`
#[derive(Debug)]
pub struct MaxPooling3D {
    /// Size of the pooling window as (depth, height, width)
    pool_size: (usize, usize, usize),
    /// Step size of the pooling operation as (depth stride, height stride, width stride)
    strides: (usize, usize, usize),
    /// Shape of the input tensor declared at construction time
    input_shape: Vec<usize>,
    /// Padding mode applied around the input before pooling
    padding: PaddingType,
    /// Shape of the most recent forward input, cached for backpropagation
    forward_input_shape: Option<Vec<usize>>,
    /// Cached flat per-output arg-max indices used for backpropagation
    argmax: Option<Vec<usize>>,
}

impl MaxPooling3D {
    /// Creates a new 3D max pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (depth, height, width)
    /// - `input_shape` - Input tensor shape `[batch_size, channels, depth, height, width]`
    ///
    /// # Notes
    ///
    /// Strides default to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override them
    /// with [`MaxPooling3D::with_strides`] and [`MaxPooling3D::with_padding`].
    ///
    /// # Returns
    ///
    /// - `Result<MaxPooling3D, Error>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - `Error::DimensionMismatch` - If `input_shape` is not 5D
    /// - `Error::InvalidInput` - If any dimension of `input_shape` is zero
    /// - `Error::InvalidParameter` - If `pool_size` has a zero dimension or a pool dimension exceeds
    ///   the corresponding input dimension
    pub fn new(pool_size: (usize, usize, usize), input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_input_shape_dims(&input_shape, 5, "MaxPooling3D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_3d(pool_size, input_shape[2], input_shape[3], input_shape[4])?;

        Ok(MaxPooling3D {
            pool_size,
            strides: pool_size,
            input_shape,
            padding: PaddingType::Valid,
            forward_input_shape: None,
            argmax: None,
        })
    }

    /// Sets the pooling strides (defaults to `pool_size`)
    ///
    /// # Parameters
    ///
    /// - `strides` - Strides of the pooling operation as (depth, height, width)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error if any stride is zero
    pub fn with_strides(mut self, strides: (usize, usize, usize)) -> Result<Self, Error> {
        validate_strides_3d(strides)?;
        self.strides = strides;
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

impl Layer for MaxPooling3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        // Cache input shape and arg-max positions for the backward pass
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, argmax) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1, self.pool_size.2],
            &[self.strides.0, self.strides.1, self.strides.2],
            PoolKind::Max,
            self.padding,
        );
        self.argmax = argmax;
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        let (output, _argmax) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1, self.pool_size.2],
            &[self.strides.0, self.strides.1, self.strides.2],
            PoolKind::Max,
            self.padding,
        );
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let input_shape = self
            .forward_input_shape
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("MaxPooling3D"))?;
        let argmax = self
            .argmax
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("MaxPooling3D"))?;

        Ok(windowed_pool_backward(
            grad_output,
            input_shape,
            &[self.pool_size.0, self.pool_size.1, self.pool_size.2],
            &[self.strides.0, self.strides.1, self.strides.2],
            PoolKind::Max,
            Some(argmax),
            self.padding,
        ))
    }

    fn layer_type(&self) -> &str {
        "MaxPooling3D"
    }

    layer_functions_3d_pooling!();
}
