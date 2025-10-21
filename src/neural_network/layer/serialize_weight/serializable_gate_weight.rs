use super::*;

/// Serializable representation of a single gate's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGateWeight {
    pub kernel: Vec<Vec<f32>>,
    pub recurrent_kernel: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}
