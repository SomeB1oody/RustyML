//! Global average pooling layer for 1D inputs

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

/// Global average pooling layer for 1D inputs
///
/// Computes the mean value across the length dimension. Input tensor shape:
/// `[batch_size, length, channels]`. Output tensor shape: `[batch_size, channels]`.
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
/// // Create test input tensor: [batch_size, length, channels]
/// let input_data = Array::from_elem(IxDyn(&[2, 4, 3]), 1.0);
///
/// // Build a model that holds 1 GlobalAveragePooling1D layer
/// let model = SequentialBuilder::new()
///     .add(GlobalAveragePooling1D::new())
///     .build(&Shape::known(input_data.shape()))
///     .unwrap();
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Check output shape - should be [2, 3]
/// assert_eq!(output.shape(), &[2, 3]);
///
/// // Since all input values are 1.0, all output values should also be 1.0
/// for b in 0..2 {
///     for c in 0..3 {
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
pub struct GlobalAveragePooling1D {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl GlobalAveragePooling1D {
    /// Creates a new global average pooling 1D layer
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling1D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling1D { built: None }
    }
}

impl Default for GlobalAveragePooling1D {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for GlobalAveragePooling1D {
    fn layer_type(&self) -> &str {
        "GlobalAveragePooling1D"
    }

    layer_base_functions_pooling!();
}

impl UnaryLayer for GlobalAveragePooling1D {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        let (output, _) = global_pool_forward(input, PoolKind::Average);

        if ctx.is_training() {
            ctx.push_cache("GlobalAveragePooling1D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("GlobalAveragePooling1D")?;

        Ok(global_pool_backward(
            grad_output,
            &input_shape,
            PoolKind::Average,
            None,
        ))
    }

    layer_functions_global_pooling!("GlobalAveragePooling1D", 3);
}
