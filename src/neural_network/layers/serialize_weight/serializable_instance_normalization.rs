//! Serializable form of InstanceNormalization layer weights and the logic to apply them back

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of InstanceNormalization layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableInstanceNormalizationWeight {
    /// Scale parameter
    pub gamma: ArrayD<f32>,
    /// Shift parameter
    pub beta: ArrayD<f32>,
}

impl ApplyWeights<InstanceNormalization> for SerializableInstanceNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut InstanceNormalization) -> Result<(), Error> {
        layer.set_weights(self.gamma.clone(), self.beta.clone())?;
        Ok(())
    }
}
