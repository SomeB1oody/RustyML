use crate::error::Error;
use crate::neural_network::layer::recurrent_layer::lstm::LSTM;
use crate::neural_network::layer::serialize_weight::SerializableGateWeight;
use crate::neural_network::neural_network_trait::ApplyWeights;
use serde::{Deserialize, Serialize};

/// Serializable representation of LSTM layer weights.
///
/// # Fields
///
/// - `input` - Weights for the input gate
/// - `forget` - Weights for the forget gate
/// - `cell` - Weights for the cell gate
/// - `output` - Weights for the output gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMWeight {
    pub input: SerializableGateWeight,
    pub forget: SerializableGateWeight,
    pub cell: SerializableGateWeight,
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
