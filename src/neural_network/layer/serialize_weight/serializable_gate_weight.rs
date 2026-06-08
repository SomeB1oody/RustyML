use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of a single recurrent gate's weights.
///
/// # Fields
///
/// - `kernel` - Input kernel matrix (input_dim, units)
/// - `recurrent_kernel` - Recurrent kernel matrix (units, units)
/// - `bias` - Bias matrix (1, units)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGateWeight {
    pub kernel: Array2<f32>,
    pub recurrent_kernel: Array2<f32>,
    pub bias: Array2<f32>,
}

impl SerializableGateWeight {
    /// Clones gate weight references into an owned serializable gate.
    pub(super) fn from_arrays(
        kernel: &Array2<f32>,
        recurrent_kernel: &Array2<f32>,
        bias: &Array2<f32>,
    ) -> Self {
        SerializableGateWeight {
            kernel: kernel.clone(),
            recurrent_kernel: recurrent_kernel.clone(),
            bias: bias.clone(),
        }
    }
}
