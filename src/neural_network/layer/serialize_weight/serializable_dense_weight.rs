use super::*;

/// Serializable representation of Dense layer weights.
///
/// # Fields
///
/// - `weight` - 2D weight matrix stored as nested vectors
/// - `bias` - 2D bias matrix stored as nested vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDenseWeight {
    pub weight: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Dense<T>> for SerializableDenseWeight {
    fn apply_to_layer(&self, layer: &mut Dense<T>) -> Result<(), IoError> {
        let weight_array = vec2_to_array2(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}
