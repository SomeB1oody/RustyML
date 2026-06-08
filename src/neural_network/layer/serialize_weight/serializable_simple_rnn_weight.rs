use crate::error::IoError;
use crate::neural_network::layer::recurrent_layer::simple_rnn::SimpleRNN;
use crate::neural_network::neural_network_trait::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Serializable representation of SimpleRNN layer weights.
///
/// # Fields
///
/// - `kernel` - Input kernel matrix (input_dim, units)
/// - `recurrent_kernel` - Recurrent kernel matrix (units, units)
/// - `bias` - Bias matrix (1, units)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSimpleRNNWeight {
    pub kernel: Array2<f32>,
    pub recurrent_kernel: Array2<f32>,
    pub bias: Array2<f32>,
}

impl ApplyWeights<SimpleRNN> for SerializableSimpleRNNWeight {
    fn apply_to_layer(&self, layer: &mut SimpleRNN) -> Result<(), IoError> {
        layer.set_weights(
            self.kernel.clone(),
            self.recurrent_kernel.clone(),
            self.bias.clone(),
        )?;
        Ok(())
    }
}
