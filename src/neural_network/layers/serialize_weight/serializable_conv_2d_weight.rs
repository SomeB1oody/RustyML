//! Serializable container for `Conv2D` layer weights and the logic to apply them back

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_2d::Conv2D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

/// Serializable representation of Conv2D layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv2DWeight {
    /// 4D convolution kernel with shape (filters, channels, kernel_height, kernel_width)
    pub weight: Array4<f32>,
    /// Bias matrix with shape (1, filters)
    pub bias: Array2<f32>,
}

impl ApplyWeights<Conv2D> for SerializableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut Conv2D) -> Result<(), Error> {
        layer.set_weights(self.weight.clone(), self.bias.clone())?;
        Ok(())
    }
}
