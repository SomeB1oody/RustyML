use crate::error::Error;
use crate::neural_network::layer::convolution_layer::conv_1d::Conv1D;
use crate::neural_network::neural_network_trait::ApplyWeights;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// Serializable representation of Conv1D layer weights.
///
/// # Fields
///
/// - `weight` - 3D convolution kernel (filters, channels, kernel_size)
/// - `bias` - Bias matrix (1, filters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv1DWeight {
    pub weight: Array3<f32>,
    pub bias: Array2<f32>,
}

impl ApplyWeights<Conv1D> for SerializableConv1DWeight {
    fn apply_to_layer(&self, layer: &mut Conv1D) -> Result<(), Error> {
        layer.set_weights(self.weight.clone(), self.bias.clone())?;
        Ok(())
    }
}
