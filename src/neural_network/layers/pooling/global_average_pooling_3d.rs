//! Global average pooling layer for 3D inputs

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

/// Global average pooling layer for 3D inputs
///
/// Computes the mean value across the depth, height, and width dimensions.
/// Input tensor shape: `[batch_size, depth, height, width, channels]`. Output tensor shape:
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
/// // Test input tensor: [batch_size, depth, height, width, channels]
/// let input_data = Array::from_elem(IxDyn(&[2, 8, 8, 8, 4]), 1.0);
///
/// // Build a model that holds 1 GlobalAveragePooling3D layer
/// let model = SequentialBuilder::new()
///     .add(GlobalAveragePooling3D::new())
///     .build(&Shape::known(input_data.shape()))
///     .unwrap();
///
/// // Forward propagation
/// let output = model.predict(&input_data).unwrap();
///
/// // Output shape should be [2, 4]
/// assert_eq!(output.shape(), &[2, 4]);
///
/// // All input values are 1.0, so all output values are 1.0
/// for b in 0..2 {
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
pub struct GlobalAveragePooling3D {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl GlobalAveragePooling3D {
    /// Creates a new global average pooling 3D layer
    ///
    /// # Returns
    ///
    /// - `GlobalAveragePooling3D` - New layer instance
    pub fn new() -> Self {
        GlobalAveragePooling3D { built: None }
    }
}

impl Default for GlobalAveragePooling3D {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerBase for GlobalAveragePooling3D {
    fn layer_type(&self) -> &str {
        "GlobalAveragePooling3D"
    }

    layer_base_functions_pooling!();
}

impl UnaryLayer for GlobalAveragePooling3D {
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        let (output, _) = global_pool_forward(input, PoolKind::Average);

        if ctx.is_training() {
            ctx.push_cache("GlobalAveragePooling3D", input.shape().to_vec());
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("GlobalAveragePooling3D")?;

        Ok(global_pool_backward(
            grad_output,
            &input_shape,
            PoolKind::Average,
            None,
        ))
    }

    layer_functions_global_pooling!("GlobalAveragePooling3D", 5);
}
