//! Serializable weights for a single recurrent gate

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of a single recurrent gate's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGateWeight {
    /// Input kernel matrix (input_dim, units)
    pub kernel: Array2<f32>,
    /// Recurrent kernel matrix (units, units)
    pub recurrent_kernel: Array2<f32>,
    /// Bias matrix (1, units)
    pub bias: Array2<f32>,
}

impl SerializableGateWeight {
    /// Clones gate weight references into an owned serializable gate
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
