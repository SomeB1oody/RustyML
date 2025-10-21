use super::*;

/// Serializable representation of an LSTM layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMWeight {
    pub input: SerializableGateWeight,
    pub forget: SerializableGateWeight,
    pub cell: SerializableGateWeight,
    pub output: SerializableGateWeight,
}
