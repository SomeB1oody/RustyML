use serde::{Deserialize, Serialize};

/// Serializable representation of a single gate's weights.
///
/// # Fields
///
/// - `kernel` - 2D input kernel matrix stored as nested vectors
/// - `recurrent_kernel` - 2D recurrent kernel matrix stored as nested vectors
/// - `bias` - 2D bias matrix stored as nested vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGateWeight {
    pub kernel: Vec<Vec<f32>>,
    pub recurrent_kernel: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}
