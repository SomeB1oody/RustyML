use super::*;
use crate::neural_network::layer::serialize_weight::helper_function::{
    vec2_to_array2, vec4_to_array4,
};
use crate::neural_network::neural_network_trait::{ActivationLayer, ApplyWeights};

/// Serializable representation of Conv2D layer weights.
///
/// # Fields
///
/// - `weight` - 4D convolution weight tensor stored as nested vectors
/// - `bias` - 2D bias matrix stored as nested vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv2DWeight {
    pub weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Conv2D<T>> for SerializableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut Conv2D<T>) -> Result<(), IoError> {
        let weight_array = vec4_to_array4(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}
