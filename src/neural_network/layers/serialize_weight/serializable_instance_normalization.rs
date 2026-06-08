use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of InstanceNormalization layer weights.
///
/// # Fields
///
/// - `gamma` - Scale parameter
/// - `beta` - Shift parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableInstanceNormalizationWeight {
    pub gamma: ArrayD<f32>,
    pub beta: ArrayD<f32>,
}

impl ApplyWeights<InstanceNormalization> for SerializableInstanceNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut InstanceNormalization) -> Result<(), Error> {
        layer.set_weights(self.gamma.clone(), self.beta.clone())?;
        Ok(())
    }
}
