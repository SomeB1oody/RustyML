//! 2D max pooling layer that selects the maximum value within each pooling window across
//! height and width.

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::layers::pooling::validation::{
    validate_pool_size_2d, validate_strides_2d,
};
use crate::neural_network::layers::pooling::{
    layer_base_functions_pooling, layer_functions_2d_pooling,
};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_2d_pooling;
use crate::neural_network::layers::validation::validate_built_input;
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// 2D max pooling layer
///
/// Selects the maximum value within each pooling window across height and width.
/// Input tensor shape: `[batch_size, height, width, channels]`. Output tensor shape:
/// `[batch_size, pooled_height, pooled_width, channels]`. With `Valid` padding,
/// `pooled_height = (height - pool_size_h) / stride_h + 1` and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`. With `Same` padding,
/// `pooled_height = ceil(height / stride_h)` and `pooled_width = ceil(width / stride_w)`.
///
/// # Notes
///
/// When 2 or more elements in a window tie for the maximum, the layer keeps the earliest one.
/// The scan order runs the last spatial axis fastest.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
///
/// // 4D input tensor: [batch_size, height, width, channels]
/// // batch size 2, 6x6 pixels, 3 input channels
/// let mut input_data = Array4::zeros((2, 6, 6, 3));
///
/// // Set values that make the max pooling result predictable
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..6 {
///             for j in 0..6 {
///                 // Easily observable pattern
///                 input_data[[b, i, j, c]] = (i * j) as f32 + b as f32 * 0.1 + c as f32 * 0.01;
///             }
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     // strides default to pool_size (2, 2) and padding defaults to Valid
///     .add(MaxPooling2D::new((2, 2)))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Target tensor matching the pooled shape
/// let y = Array4::ones((2, 3, 3, 3)).into_dyn();
///
/// model.summary();
///
/// model.fit(&x, &y, 3).unwrap();
///
/// let prediction = model.predict(&x).unwrap();
/// println!("MaxPooling2D prediction results: {:?}", prediction);
///
/// assert_eq!(prediction.shape(), &[2, 3, 3, 3]);
/// ```
///
/// # Performance
///
/// The pass runs in parallel when its estimated element count
/// (`batch * out_positions * channels * window taps`) clears the gate in
/// [`tuning::pool`](crate::tuning::pool). The gate does not depend on any fixed shape.
#[derive(Debug)]
pub struct MaxPooling2D {
    /// Size of the pooling window as (height, width)
    pool_size: (usize, usize),
    /// Step size of the pooling operation as (height, width)
    strides: (usize, usize),
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Padding mode applied around the input before pooling
    padding: PaddingType,
}

impl MaxPooling2D {
    /// Creates a new 2D max pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (height, width)
    ///
    /// # Returns
    ///
    /// - `MaxPooling2D` - New layer instance
    ///
    /// # Notes
    ///
    /// Strides default to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override them
    /// with [`MaxPooling2D::with_strides`] and [`MaxPooling2D::with_padding`].
    ///
    pub fn new(pool_size: (usize, usize)) -> Self {
        MaxPooling2D {
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
    /// - `strides` - Strides of the pooling operation as (height, width)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error if any stride is zero
    pub fn with_strides(mut self, strides: (usize, usize)) -> Result<Self, Error> {
        validate_strides_2d(strides)?;
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

/// What the forward pass of [`MaxPooling2D`] parks for its backward pass
struct MaxPooling2DCache {
    /// Shape of the tensor that entered the layer, to restore the shape of the gradient
    input_shape: Vec<usize>,
    /// Flat per-output arg-max index of each pooling window, to route the gradient back
    argmax: Vec<usize>,
}

impl LayerBase for MaxPooling2D {
    fn layer_type(&self) -> &str {
        "MaxPooling2D"
    }

    layer_base_functions_pooling!();
}

impl UnaryLayer for MaxPooling2D {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "MaxPooling2D", input.shape())?;

        let (output, argmax) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Max,
            self.padding,
        );

        if ctx.is_training() {
            ctx.push_cache(
                "MaxPooling2D",
                MaxPooling2DCache {
                    input_shape: input.shape().to_vec(),
                    argmax: argmax.expect("PoolKind::Max always tracks the arg-max"),
                },
            );
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: MaxPooling2DCache = ctx.pop_cache("MaxPooling2D")?;

        Ok(windowed_pool_backward(
            grad_output,
            &cache.input_shape,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Max,
            Some(&cache.argmax),
            self.padding,
        ))
    }

    layer_functions_2d_pooling!("MaxPooling2D");
}
