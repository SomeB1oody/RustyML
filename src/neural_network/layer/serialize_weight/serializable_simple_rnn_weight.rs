use super::*;

/// Serializable representation of a SimpleRNN layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSimpleRNNWeight {
    pub kernel: Vec<Vec<f32>>,
    pub recurrent_kernel: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<SimpleRNN<T>> for SerializableSimpleRNNWeight {
    fn apply_to_layer(&self, layer: &mut SimpleRNN<T>) -> Result<(), IoError> {
        let kernel = vec2_to_array2(&self.kernel)?;
        let recurrent_kernel = vec2_to_array2(&self.recurrent_kernel)?;
        let bias = vec2_to_array2(&self.bias)?;
        layer.set_weights(kernel, recurrent_kernel, bias);
        Ok(())
    }
}
