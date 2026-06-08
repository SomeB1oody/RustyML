use crate::error::Error;
use crate::neural_network::layer::convolution_layer::conv_2d::Conv2D;
use crate::neural_network::neural_network_trait::ApplyWeights;
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

/// Serializable representation of Conv2D layer weights.
///
/// # Fields
///
/// - `weight` - 4D convolution kernel (filters, channels, kernel_height, kernel_width)
/// - `bias` - Bias matrix (1, filters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv2DWeight {
    pub weight: Array4<f32>,
    pub bias: Array2<f32>,
}

impl ApplyWeights<Conv2D> for SerializableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut Conv2D) -> Result<(), Error> {
        layer.set_weights(self.weight.clone(), self.bias.clone())?;
        Ok(())
    }
}
