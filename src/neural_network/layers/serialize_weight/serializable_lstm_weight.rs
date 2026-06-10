//! Serializable weights for an LSTM layer

use crate::error::Error;
use crate::neural_network::layers::recurrent::lstm::LSTM;
use crate::neural_network::layers::serialize_weight::SerializableGateWeight;
use crate::neural_network::traits::ApplyWeights;
use serde::{Deserialize, Serialize};

/// Serializable representation of LSTM layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMWeight {
    /// Weights for the input gate
    pub input: SerializableGateWeight,
    /// Weights for the forget gate
    pub forget: SerializableGateWeight,
    /// Weights for the cell gate
    pub cell: SerializableGateWeight,
    /// Weights for the output gate
    pub output: SerializableGateWeight,
}

impl ApplyWeights<LSTM> for SerializableLSTMWeight {
    fn apply_to_layer(&self, layer: &mut LSTM) -> Result<(), Error> {
        layer.set_weights(
            self.input.kernel.clone(),
            self.input.recurrent_kernel.clone(),
            self.input.bias.clone(),
            self.forget.kernel.clone(),
            self.forget.recurrent_kernel.clone(),
            self.forget.bias.clone(),
            self.cell.kernel.clone(),
            self.cell.recurrent_kernel.clone(),
            self.cell.bias.clone(),
            self.output.kernel.clone(),
            self.output.recurrent_kernel.clone(),
            self.output.bias.clone(),
        )?;
        Ok(())
    }
}
