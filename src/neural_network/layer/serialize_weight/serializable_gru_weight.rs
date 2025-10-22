use super::*;

/// Serializable representation of a GRU layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUWeight {
    pub reset: SerializableGateWeight,
    pub update: SerializableGateWeight,
    pub candidate: SerializableGateWeight,
}

impl<T: ActivationLayer> ApplyWeights<GRU<T>> for SerializableGRUWeight {
    fn apply_to_layer(&self, layer: &mut GRU<T>) -> Result<(), IoError> {
        let reset_kernel = vec2_to_array2(&self.reset.kernel)?;
        let reset_recurrent = vec2_to_array2(&self.reset.recurrent_kernel)?;
        let reset_bias = vec2_to_array2(&self.reset.bias)?;

        let update_kernel = vec2_to_array2(&self.update.kernel)?;
        let update_recurrent = vec2_to_array2(&self.update.recurrent_kernel)?;
        let update_bias = vec2_to_array2(&self.update.bias)?;

        let candidate_kernel = vec2_to_array2(&self.candidate.kernel)?;
        let candidate_recurrent = vec2_to_array2(&self.candidate.recurrent_kernel)?;
        let candidate_bias = vec2_to_array2(&self.candidate.bias)?;

        layer.set_weights(
            reset_kernel,
            reset_recurrent,
            reset_bias,
            update_kernel,
            update_recurrent,
            update_bias,
            candidate_kernel,
            candidate_recurrent,
            candidate_bias,
        );
        Ok(())
    }
}
