use crate::error::IoError;
use crate::neural_network::layer::convolution_layer::conv_1d::Conv1D;
use crate::neural_network::layer::serialize_weight::helper_function::{
    vec2_to_array2, vec3_to_array3,
};
use crate::neural_network::neural_network_trait::{ActivationLayer, ApplyWeights};
use serde::{Deserialize, Serialize};

/// Serializable representation of Conv1D layer weights.
///
/// # Fields
///
/// - `weight` - 3D convolution weight tensor stored as nested vectors
/// - `bias` - 2D bias matrix stored as nested vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv1DWeight {
    pub weight: Vec<Vec<Vec<f32>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Conv1D<T>> for SerializableConv1DWeight {
    fn apply_to_layer(&self, layer: &mut Conv1D<T>) -> Result<(), IoError> {
        let weight_array = vec3_to_array3(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}
