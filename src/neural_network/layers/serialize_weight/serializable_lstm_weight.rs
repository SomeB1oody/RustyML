//! Serializable weights for an LSTM layer

use crate::error::Error;
use crate::neural_network::layers::recurrent::lstm::LSTM;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of LSTM layer weights
///
/// The four gates are stored fused, gate column blocks in the order `[i | f | g | o]`
/// (input, forget, cell, output - the Keras LSTM layout)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMWeight {
    /// Fused input kernel (input_dim, 4 * units)
    pub kernel: Array2<f32>,
    /// Fused recurrent kernel (units, 4 * units)
    pub recurrent_kernel: Array2<f32>,
    /// Fused bias (1, 4 * units)
    pub bias: Array2<f32>,
}

impl ApplyWeights<LSTM> for SerializableLSTMWeight {
    fn apply_to_layer(&self, layer: &mut LSTM) -> Result<(), Error> {
        layer.set_weights(
            self.kernel.clone(),
            self.recurrent_kernel.clone(),
            self.bias.clone(),
        )?;
        Ok(())
    }
}
