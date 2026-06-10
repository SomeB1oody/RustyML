//! Serializable representation of GRU layer weights and its weight-application logic

use crate::error::Error;
use crate::neural_network::layers::recurrent::gru::GRU;
use crate::neural_network::layers::serialize_weight::SerializableGateWeight;
use crate::neural_network::traits::ApplyWeights;
use serde::{Deserialize, Serialize};

/// Serializable representation of GRU layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUWeight {
    /// Weights for the reset gate
    pub reset: SerializableGateWeight,
    /// Weights for the update gate
    pub update: SerializableGateWeight,
    /// Weights for the candidate state
    pub candidate: SerializableGateWeight,
}

impl ApplyWeights<GRU> for SerializableGRUWeight {
    fn apply_to_layer(&self, layer: &mut GRU) -> Result<(), Error> {
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
