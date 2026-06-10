//! Serializable representation of [`SimpleRNN`] layer weights

use crate::error::Error;
use crate::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of SimpleRNN layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSimpleRNNWeight {
    /// Input kernel matrix of shape (input_dim, units)
    pub kernel: Array2<f32>,
    /// Recurrent kernel matrix of shape (units, units)
    pub recurrent_kernel: Array2<f32>,
    /// Bias matrix of shape (1, units)
    pub bias: Array2<f32>,
}

impl ApplyWeights<SimpleRNN> for SerializableSimpleRNNWeight {
    fn apply_to_layer(&self, layer: &mut SimpleRNN) -> Result<(), Error> {
        layer.set_weights(
            self.kernel.clone(),
            self.recurrent_kernel.clone(),
            self.bias.clone(),
        )?;
        Ok(())
    }
}
