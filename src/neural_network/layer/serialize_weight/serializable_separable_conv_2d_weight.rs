use super::*;

/// Serializable representation of SeparableConv2D layer weights.
///
/// # Fields
///
/// - `depthwise_weight` - 4D depthwise convolution weights stored as nested vectors
/// - `pointwise_weight` - 4D pointwise convolution weights stored as nested vectors
/// - `bias` - 2D bias matrix stored as nested vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSeparableConv2DWeight {
    pub depthwise_weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub pointwise_weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<SeparableConv2D<T>> for SerializableSeparableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut SeparableConv2D<T>) -> Result<(), IoError> {
        let depthwise_weight = vec4_to_array4(&self.depthwise_weight)?;
        let pointwise_weight = vec4_to_array4(&self.pointwise_weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(depthwise_weight, pointwise_weight, bias_array);
        Ok(())
    }
}
