//! Global max pooling layer for 2D inputs.

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, global_pool_backward, global_pool_forward,
};
use crate::neural_network::layers::pooling::{
    layer_base_functions_pooling, layer_functions_global_pooling,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Global max pooling layer for 2D inputs
///
/// Selects the maximum value across the height and width dimensions.
/// Input tensor shape: `[batch_size, height, width, channels]`. Output tensor shape:
/// `[batch_size, channels]`.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::{Array, IxDyn};
/// use approx::assert_relative_eq;
///
/// // Create a test input tensor: [batch_size, height, width, channels]
/// let input_data = Array::from_elem(IxDyn(&[3, 5, 5, 4]), 1.0);
///
/// // Build a model that holds 1 GlobalMaxPooling2D layer
/// let model = SequentialBuilder::new()
///     .add(GlobalMaxPooling2D::new())
///     .build(&Shape::known(input_data.shape()))
///     .unwrap();
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Check output shape - should be [3, 4]
/// assert_eq!(output.shape(), &[3, 4]);
///
/// // Since all input values are 1.0, all output values should also be 1.0 too
/// for b in 0..3 {
///     for c in 0..4 {
///         assert_relative_eq!(output[[b, c]], 1.0);
///     }
/// }
/// ```
///
/// # Performance
///
/// The pass runs in parallel when its estimated element count (`batch * positions * channels`)
/// clears the gate in [`tuning::pool`](crate::tuning::pool). The gate does not depend on any
/// fixed shape.
#[derive(Debug)]
pub struct GlobalMaxPooling2D {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl GlobalMaxPooling2D {
    /// Creates a new global max pooling 2D layer
    ///
    /// # Returns
    ///
    /// - `GlobalMaxPooling2D` - New layer instance
    pub fn new() -> Self {
        GlobalMaxPooling2D { built: None }
    }
}

impl Default for GlobalMaxPooling2D {
    fn default() -> Self {
        Self::new()
    }
}

/// What the forward pass of [`GlobalMaxPooling2D`] parks for its backward pass
struct GlobalMaxPooling2DCache {
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Flat per-channel arg-max offsets that the forward pass recorded, one per output element
    argmax: Option<Vec<usize>>,
}

impl LayerBase for GlobalMaxPooling2D {
    fn layer_type(&self) -> &str {
        "GlobalMaxPooling2D"
    }

    layer_base_functions_pooling!();
}

impl UnaryLayer for GlobalMaxPooling2D {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        let (output, argmax) = global_pool_forward(input, PoolKind::Max);

        if ctx.is_training() {
            ctx.push_cache(
                "GlobalMaxPooling2D",
                GlobalMaxPooling2DCache {
                    input_shape: input.shape().to_vec(),
                    argmax,
                },
            );
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: GlobalMaxPooling2DCache = ctx.pop_cache("GlobalMaxPooling2D")?;

        Ok(global_pool_backward(
            grad_output,
            &cache.input_shape,
            PoolKind::Max,
            cache.argmax.as_deref(),
        ))
    }

    layer_functions_global_pooling!("GlobalMaxPooling2D", 4);
}
