//! 3D max pooling layer that selects the maximum value within each pooling window across depth,
//! height, and width.

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::layers::pooling::validation::{
    validate_pool_size_3d, validate_strides_3d,
};
use crate::neural_network::layers::pooling::{
    layer_base_functions_pooling, layer_functions_3d_pooling,
};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_3d_pooling;
use crate::neural_network::layers::validation::validate_built_input;
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// 3D max pooling layer
///
/// Selects the maximum value within each pooling window across depth, height, and width.
/// Input tensor shape: `[batch_size, depth, height, width, channels]`. Output tensor shape:
/// `[batch_size, pooled_depth, pooled_height, pooled_width, channels]`. With `Valid` padding,
/// `pooled_depth = (depth - pool_size_d) / stride_d + 1`. The same rule gives
/// `pooled_height = (height - pool_size_h) / stride_h + 1` and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::{Array5, ArrayD};
///
/// // Create a Sequential model for 3D data processing
/// // Create sample 3D input data, for example 3D medical images or volumetric data
/// // Input: [1 batch, 32x32x32 3D volume, 16 channels]
/// let input_data = Array5::from_shape_fn((1, 32, 32, 32, 16), |(b, d, h, w, c)| {
///     // Generate sample data with spatial patterns
///     ((d + h + w) as f32 * 0.1) + (c as f32 * 0.01)
/// }).into_dyn();
///
/// // Create target data for training (output shape: [1, 16, 16, 16, 16])
/// let target_data = Array5::ones((1, 16, 16, 16, 16)).into_dyn();
///
/// // Build a model that holds 1 MaxPooling3D layer
/// // stride defaults to pool_size (2, 2, 2) and padding defaults to Valid
/// let mut model = SequentialBuilder::new()
///     .add(MaxPooling3D::new((2, 2, 2)))
///     .build(&Shape::known(input_data.shape()))
///     .unwrap();
///
/// // Compile the model with optimizer and loss function
/// model.compile(
///     RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(),    // RMSprop optimizer
///     MeanSquaredError::new()              // Mean Squared Error loss
/// );
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
/// // Expected output: [1, 16, 16, 16, 16], spatial dimensions reduced by a factor of 2
/// ```
///
/// # Performance
///
/// The pass runs in parallel when its estimated element count
/// (`batch * out_positions * channels * window taps`) clears the gate in
/// [`tuning::pool`](crate::tuning::pool). The gate does not depend on any fixed shape.
#[derive(Debug)]
pub struct MaxPooling3D {
    /// Size of the pooling window as (depth, height, width)
    pool_size: (usize, usize, usize),
    /// Step size of the pooling operation as (depth, height, width)
    strides: (usize, usize, usize),
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Padding mode applied around the input before pooling
    padding: PaddingType,
}

impl MaxPooling3D {
    /// Creates a new 3D max pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (depth, height, width)
    ///
    /// # Returns
    ///
    /// - `MaxPooling3D` - New layer instance
    ///
    /// # Notes
    ///
    /// Strides default to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override them
    /// with [`MaxPooling3D::with_strides`] and [`MaxPooling3D::with_padding`].
    ///
    pub fn new(pool_size: (usize, usize, usize)) -> Self {
        MaxPooling3D {
            pool_size,
            strides: pool_size,
            built: None,
            padding: PaddingType::Valid,
        }
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

/// What the forward pass of [`MaxPooling3D`] parks for its backward pass
struct MaxPooling3DCache {
    /// Shape of the tensor that entered the layer, to restore the rank of the gradient
    input_shape: Vec<usize>,
    /// Flat per-output arg-max index of each pooling window, to route the gradient back
    argmax: Vec<usize>,
}

impl LayerBase for MaxPooling3D {
    fn layer_type(&self) -> &str {
        "MaxPooling3D"
    }

    layer_base_functions_pooling!();
}

impl UnaryLayer for MaxPooling3D {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "MaxPooling3D", input.shape())?;

        let (output, argmax) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1, self.pool_size.2],
            &[self.strides.0, self.strides.1, self.strides.2],
            PoolKind::Max,
            self.padding,
        );

        if ctx.is_training() {
            ctx.push_cache(MaxPooling3DCache {
                input_shape: input.shape().to_vec(),
                argmax: argmax.expect("PoolKind::Max always tracks the arg-max"),
            });
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: MaxPooling3DCache = ctx.pop_cache("MaxPooling3D")?;

        Ok(windowed_pool_backward(
            grad_output,
            &cache.input_shape,
            &[self.pool_size.0, self.pool_size.1, self.pool_size.2],
            &[self.strides.0, self.strides.1, self.strides.2],
            PoolKind::Max,
            Some(&cache.argmax),
            self.padding,
        ))
    }

    layer_functions_3d_pooling!("MaxPooling3D");
}
