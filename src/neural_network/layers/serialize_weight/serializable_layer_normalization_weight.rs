use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::layer_normalization::LayerNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of LayerNormalization layer weights.
///
/// # Fields
///
/// - `gamma` - Scale parameter
/// - `beta` - Shift parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayerNormalizationWeight {
    pub gamma: ArrayD<f32>,
    pub beta: ArrayD<f32>,
}

impl ApplyWeights<LayerNormalization> for SerializableLayerNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut LayerNormalization) -> Result<(), Error> {
        layer.set_weights(self.gamma.clone(), self.beta.clone())?;
        Ok(())
    }
}
