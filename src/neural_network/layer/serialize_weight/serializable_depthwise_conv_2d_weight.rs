use super::*;

/// Serializable representation of a DepthwiseConv2D layer's weights
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
