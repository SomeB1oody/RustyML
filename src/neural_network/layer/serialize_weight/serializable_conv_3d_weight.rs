use super::*;

/// Serializable representation of a Conv3D layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv3DWeight {
    pub weight: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Conv3D<T>> for SerializableConv3DWeight {
    fn apply_to_layer(&self, layer: &mut Conv3D<T>) -> Result<(), IoError> {
        let weight_array = vec5_to_array5(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}
