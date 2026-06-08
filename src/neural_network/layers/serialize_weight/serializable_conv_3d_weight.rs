use crate::error::Error;
use crate::neural_network::layers::convolution::conv_3d::Conv3D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array2, Array5};
use serde::{Deserialize, Serialize};

/// Serializable representation of Conv3D layer weights.
///
/// # Fields
///
/// - `weight` - 5D convolution kernel (filters, channels, kernel_depth, kernel_height, kernel_width)
/// - `bias` - Bias matrix (1, filters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv3DWeight {
    pub weight: Array5<f32>,
    pub bias: Array2<f32>,
}

impl ApplyWeights<Conv3D> for SerializableConv3DWeight {
    fn apply_to_layer(&self, layer: &mut Conv3D) -> Result<(), Error> {
        layer.set_weights(self.weight.clone(), self.bias.clone())?;
        Ok(())
    }
}
