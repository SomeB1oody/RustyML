use super::*;

/// Serializable representation of a LayerNormalization layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayerNormalizationWeight {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub shape: Vec<usize>,
}

impl ApplyWeights<LayerNormalization> for SerializableLayerNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut LayerNormalization) -> Result<(), IoError> {
        let gamma =
            ArrayD::from_shape_vec(self.shape.as_slice(), self.gamma.clone()).map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
        let beta =
            ArrayD::from_shape_vec(self.shape.as_slice(), self.beta.clone()).map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
        layer.set_weights(gamma, beta);
        Ok(())
    }
}
