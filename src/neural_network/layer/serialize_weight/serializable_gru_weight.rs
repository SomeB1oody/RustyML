use super::*;

/// Serializable representation of a GRU layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUWeight {
    pub reset: SerializableGateWeight,
    pub update: SerializableGateWeight,
    pub candidate: SerializableGateWeight,
}
