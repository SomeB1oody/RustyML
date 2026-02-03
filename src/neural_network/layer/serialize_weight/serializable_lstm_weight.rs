use crate::error::IoError;
use crate::neural_network::layer::recurrent_layer::lstm::LSTM;
use crate::neural_network::layer::serialize_weight::SerializableGateWeight;
use crate::neural_network::layer::serialize_weight::helper_function::vec2_to_array2;
use crate::neural_network::neural_network_trait::{ActivationLayer, ApplyWeights};
use serde::{Deserialize, Serialize};

/// Serializable representation of LSTM layer weights.
///
/// # Fields
///
/// - `input` - Weights for the input gate
/// - `forget` - Weights for the forget gate
/// - `cell` - Weights for the cell gate
/// - `output` - Weights for the output gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMWeight {
    pub input: SerializableGateWeight,
    pub forget: SerializableGateWeight,
    pub cell: SerializableGateWeight,
    pub output: SerializableGateWeight,
}

impl<T: ActivationLayer> ApplyWeights<LSTM<T>> for SerializableLSTMWeight {
    fn apply_to_layer(&self, layer: &mut LSTM<T>) -> Result<(), IoError> {
        let input_kernel = vec2_to_array2(&self.input.kernel)?;
        let input_recurrent = vec2_to_array2(&self.input.recurrent_kernel)?;
        let input_bias = vec2_to_array2(&self.input.bias)?;

        let forget_kernel = vec2_to_array2(&self.forget.kernel)?;
        let forget_recurrent = vec2_to_array2(&self.forget.recurrent_kernel)?;
        let forget_bias = vec2_to_array2(&self.forget.bias)?;

        let cell_kernel = vec2_to_array2(&self.cell.kernel)?;
        let cell_recurrent = vec2_to_array2(&self.cell.recurrent_kernel)?;
        let cell_bias = vec2_to_array2(&self.cell.bias)?;

        let output_kernel = vec2_to_array2(&self.output.kernel)?;
        let output_recurrent = vec2_to_array2(&self.output.recurrent_kernel)?;
        let output_bias = vec2_to_array2(&self.output.bias)?;

        layer.set_weights(
            input_kernel,
            input_recurrent,
            input_bias,
            forget_kernel,
            forget_recurrent,
            forget_bias,
            cell_kernel,
            cell_recurrent,
            cell_bias,
            output_kernel,
            output_recurrent,
            output_bias,
        );
        Ok(())
    }
}
