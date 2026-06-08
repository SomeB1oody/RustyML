use crate::error::IoError;
use crate::neural_network::layer::recurrent_layer::gru::GRU;
use crate::neural_network::layer::serialize_weight::SerializableGateWeight;
use crate::neural_network::neural_network_trait::ApplyWeights;
use serde::{Deserialize, Serialize};

/// Serializable representation of GRU layer weights.
///
/// # Fields
///
/// - `reset` - Weights for the reset gate
/// - `update` - Weights for the update gate
/// - `candidate` - Weights for the candidate state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUWeight {
    pub reset: SerializableGateWeight,
    pub update: SerializableGateWeight,
    pub candidate: SerializableGateWeight,
}

impl ApplyWeights<GRU> for SerializableGRUWeight {
    fn apply_to_layer(&self, layer: &mut GRU) -> Result<(), IoError> {
        layer.set_weights(
            self.reset.kernel.clone(),
            self.reset.recurrent_kernel.clone(),
            self.reset.bias.clone(),
            self.update.kernel.clone(),
            self.update.recurrent_kernel.clone(),
            self.update.bias.clone(),
            self.candidate.kernel.clone(),
            self.candidate.recurrent_kernel.clone(),
            self.candidate.bias.clone(),
        )?;
        Ok(())
    }
}
