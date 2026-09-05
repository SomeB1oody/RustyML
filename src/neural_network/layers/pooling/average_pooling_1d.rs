//! 1D average pooling layer that computes the mean value within each pooling window along the
//! length dimension.

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::layers::pooling::validation::{
    validate_pool_size_1d, validate_stride_1d,
};
use crate::neural_network::layers::pooling::{
    layer_base_functions_pooling, layer_functions_1d_pooling,
};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_1d_pooling;
use crate::neural_network::layers::validation::validate_built_input;
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// 1D average pooling layer
///
/// Computes the mean value within each pooling window along the length dimension.
/// Input tensor shape: `[batch_size, length, channels]`. Output tensor shape:
/// `[batch_size, pooled_length, channels]`. With `Valid` padding,
/// `pooled_length = (length - pool_size) / stride + 1`.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array3;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, length, channels]
/// // batch size 2, 8 elements along the length axis, 3 input channels
/// let mut input_data = Array3::zeros((2, 8, 3));
///
/// // Set test data to make the average pooling results predictable
/// for b in 0..2 {
///     for i in 0..8 {
///         for c in 0..3 {
///             input_data[[b, i, c]] = i as f32;
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test AveragePooling1D with a Sequential model
/// let mut model = SequentialBuilder::new()
///     // stride defaults to pool_size (2) and padding defaults to Valid
///     .add(AveragePooling1D::new(2))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Output shape should be [2, 4, 3]
/// let output = model.predict(&x).unwrap();
/// assert_eq!(output.shape(), &[2, 4, 3]);
///
/// // Verify the pooling results
/// // Window size 2 and stride 2: each window holds the average of its elements
/// for b in 0..2 {
///     for c in 0..3 {
///         // First window (0, 1): average (0 + 1) / 2 = 0.5
///         assert_relative_eq!(output[[b, 0, c]], 0.5);
///         // Second window (2, 3): average (2 + 3) / 2 = 2.5
///         assert_relative_eq!(output[[b, 1, c]], 2.5);
///         // Third window (4, 5): average (4 + 5) / 2 = 4.5
///         assert_relative_eq!(output[[b, 2, c]], 4.5);
///         // Fourth window (6, 7): average (6 + 7) / 2 = 6.5
///         assert_relative_eq!(output[[b, 3, c]], 6.5);
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
pub struct AveragePooling1D {
    /// Size of the pooling window
    pool_size: usize,
    /// Step size of the pooling operation
    stride: usize,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Padding mode applied around the input before pooling
    padding: PaddingType,
}

impl AveragePooling1D {
    /// Creates a new 1D average pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window
    ///
    /// # Returns
    ///
    /// - `AveragePooling1D` - New layer instance
    ///
    /// # Notes
    ///
    /// The stride defaults to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override
    /// them with [`AveragePooling1D::with_stride`] and [`AveragePooling1D::with_padding`].
    ///
    pub fn new(pool_size: usize) -> Self {
        AveragePooling1D {
            pool_size,
            stride: pool_size,
            built: None,
            padding: PaddingType::Valid,
        }
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

impl LayerBase for AveragePooling1D {
    fn layer_type(&self) -> &str {
        "AveragePooling1D"
    }

    layer_base_functions_pooling!();
}

impl UnaryLayer for AveragePooling1D {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "AveragePooling1D", input.shape())?;

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
            self.padding,
        );

        if ctx.is_training() {
            // The only value the backward pass needs is the shape it must restore
            ctx.push_cache("AveragePooling1D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("AveragePooling1D")?;

        Ok(windowed_pool_backward(
            grad_output,
            &input_shape,
            &[self.pool_size],
            &[self.stride],
            PoolKind::Average,
            None,
            self.padding,
        ))
    }

    layer_functions_1d_pooling!("AveragePooling1D");
}
