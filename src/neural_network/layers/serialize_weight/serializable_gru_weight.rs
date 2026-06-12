//! Serializable representation of GRU layer weights and its weight-application logic

use crate::error::Error;
use crate::neural_network::layers::recurrent::gru::GRU;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of GRU layer weights
///
/// The three gates are stored fused, gate column blocks in the order `[r | z | h]`
/// (reset, update, candidate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUWeight {
    /// Fused input kernel (input_dim, 3 * units)
    pub kernel: Array2<f32>,
    /// Fused recurrent kernel (units, 3 * units)
    pub recurrent_kernel: Array2<f32>,
    /// Fused bias (1, 3 * units)
    pub bias: Array2<f32>,
}

impl ApplyWeights<GRU> for SerializableGRUWeight {
    fn apply_to_layer(&self, layer: &mut GRU) -> Result<(), Error> {
        layer.set_weights(
            self.kernel.clone(),
            self.recurrent_kernel.clone(),
            self.bias.clone(),
        )?;
        Ok(())
    }
}
