use super::*;
use crate::neural_network::layer::serialize_weight::helper_function::vec4_to_array4;
use crate::neural_network::neural_network_trait::{ActivationLayer, ApplyWeights};
use ndarray::Array1;

/// Serializable representation of DepthwiseConv2D layer weights.
///
/// # Fields
///
/// - `weight` - 4D depthwise convolution weight tensor stored as nested vectors
/// - `bias` - 1D bias vector stored as a flat list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDepthwiseConv2DWeight {
    pub weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub bias: Vec<f32>,
}

impl<T: ActivationLayer> ApplyWeights<DepthwiseConv2D<T>> for SerializableDepthwiseConv2DWeight {
    fn apply_to_layer(&self, layer: &mut DepthwiseConv2D<T>) -> Result<(), IoError> {
        let weight_array = vec4_to_array4(&self.weight)?;
        let bias_array = Array1::from_vec(self.bias.clone());
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}
