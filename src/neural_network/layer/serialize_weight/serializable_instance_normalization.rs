use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use crate::error::IoError;
use crate::neural_network::layer::regularization_layer::normalization_layer::instance_normalization::InstanceNormalization;
use crate::neural_network::neural_network_trait::ApplyWeights;

/// Serializable representation of InstanceNormalization layer weights.
///
/// # Fields
///
/// - `gamma` - Scale parameter values flattened into a vector
/// - `beta` - Shift parameter values flattened into a vector
/// - `shape` - Original parameter shape used to rebuild arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableInstanceNormalizationWeight {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub shape: Vec<usize>,
}

impl ApplyWeights<InstanceNormalization> for SerializableInstanceNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut InstanceNormalization) -> Result<(), IoError> {
        // Convert Vec to ArrayD
        let gamma =
            ArrayD::from_shape_vec(IxDyn(&self.shape), self.gamma.clone()).map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to reshape gamma: {}", e),
                ))
            })?;

        let beta = ArrayD::from_shape_vec(IxDyn(&self.shape), self.beta.clone()).map_err(|e| {
            IoError::StdIoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to reshape beta: {}", e),
            ))
        })?;

        // Apply weights to the layer
        layer.set_weights(gamma, beta);

        Ok(())
    }
}
