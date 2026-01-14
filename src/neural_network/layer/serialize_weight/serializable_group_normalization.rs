use super::*;

/// Serializable representation of GroupNormalization layer weights
///
/// This struct provides a serializable format for the weights of a group normalization layer,
/// allowing the model to be saved to and loaded from disk. Group normalization learns
/// scale (gamma) and shift (beta) parameters per channel.
///
/// # Fields
///
/// - `gamma` - Flattened scale parameter vector
/// - `beta` - Flattened shift parameter vector
/// - `shape` - Original shape of the parameters (typically \[num_channels\])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGroupNormalizationWeight {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub shape: Vec<usize>,
}

impl ApplyWeights<GroupNormalization> for SerializableGroupNormalizationWeight {
    /// Applies the serialized weights to a GroupNormalization layer instance.
    ///
    /// # Parameters
    ///
    /// * `layer` - Mutable reference to the GroupNormalization layer that will receive the weights
    ///
    /// # Returns
    ///
    /// - `Ok(())` - Weights were successfully applied
    /// - `Err(IoError)` - Weight shape mismatch or conversion error
    fn apply_to_layer(&self, layer: &mut GroupNormalization) -> Result<(), IoError> {
        // Convert Vec to ArrayD
        let gamma =
            ArrayD::from_shape_vec(IxDyn(&self.shape), self.gamma.clone()).map_err(|e| {
                IoError::StdIoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to reshape gamma: {}", e),
                ))
            })?;

        let beta = ArrayD::from_shape_vec(IxDyn(&self.shape), self.beta.clone()).map_err(|e| {
            IoError::StdIoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to reshape beta: {}", e),
            ))
        })?;

        // Apply weights to the layer
        layer.set_weights(gamma, beta);

        Ok(())
    }
}
