//! Serializable container for `SeparableConv2D` layer weights and the logic to apply them back

use crate::error::Error;
use crate::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

/// Serializable representation of SeparableConv2D layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSeparableConv2DWeight {
    /// 4D depthwise kernel with shape (depth_multiplier, channels, kernel_height, kernel_width)
    pub depthwise_weight: Array4<f32>,
    /// 4D pointwise kernel with shape (filters, channels * depth_multiplier, 1, 1)
    pub pointwise_weight: Array4<f32>,
    /// Bias matrix with shape (1, filters)
    pub bias: Array2<f32>,
}

impl ApplyWeights<SeparableConv2D> for SerializableSeparableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut SeparableConv2D) -> Result<(), Error> {
        layer.set_weights(
            self.depthwise_weight.clone(),
            self.pointwise_weight.clone(),
            self.bias.clone(),
        )?;
        Ok(())
    }
}
