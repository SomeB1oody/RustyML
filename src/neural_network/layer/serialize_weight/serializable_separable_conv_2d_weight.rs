use crate::error::IoError;
use crate::neural_network::layer::convolution_layer::separable_conv_2d::SeparableConv2D;
use crate::neural_network::neural_network_trait::ApplyWeights;
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

/// Serializable representation of SeparableConv2D layer weights.
///
/// # Fields
///
/// - `depthwise_weight` - 4D depthwise convolution kernel
/// - `pointwise_weight` - 4D pointwise convolution kernel
/// - `bias` - Bias matrix (1, filters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSeparableConv2DWeight {
    pub depthwise_weight: Array4<f32>,
    pub pointwise_weight: Array4<f32>,
    pub bias: Array2<f32>,
}

impl ApplyWeights<SeparableConv2D> for SerializableSeparableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut SeparableConv2D) -> Result<(), IoError> {
        layer.set_weights(
            self.depthwise_weight.clone(),
            self.pointwise_weight.clone(),
            self.bias.clone(),
        )?;
        Ok(())
    }
}
