use crate::error::IoError;
use crate::neural_network::layer::regularization_layer::normalization_layer::layer_normalization::LayerNormalization;
use crate::neural_network::neural_network_trait::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of LayerNormalization layer weights.
///
/// # Fields
///
/// - `gamma` - Scale parameter values flattened into a vector
/// - `beta` - Shift parameter values flattened into a vector
/// - `shape` - Original parameter shape used to rebuild arrays
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
