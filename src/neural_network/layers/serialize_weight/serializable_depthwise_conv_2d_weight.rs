use crate::error::Error;
use crate::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array4};
use serde::{Deserialize, Serialize};

/// Serializable representation of DepthwiseConv2D layer weights.
///
/// # Fields
///
/// - `weight` - 4D depthwise convolution kernel
/// - `bias` - 1D bias vector (one entry per filter)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDepthwiseConv2DWeight {
    pub weight: Array4<f32>,
    pub bias: Array1<f32>,
}

impl ApplyWeights<DepthwiseConv2D> for SerializableDepthwiseConv2DWeight {
    fn apply_to_layer(&self, layer: &mut DepthwiseConv2D) -> Result<(), Error> {
        layer.set_weights(self.weight.clone(), self.bias.clone())?;
        Ok(())
    }
}
