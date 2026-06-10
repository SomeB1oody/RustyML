//! Serializable representation of `LayerNormalization` layer weights

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::layer_normalization::LayerNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of `LayerNormalization` layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayerNormalizationWeight {
    /// Scale parameter
    pub gamma: ArrayD<f32>,
    /// Shift parameter
    pub beta: ArrayD<f32>,
}

impl ApplyWeights<LayerNormalization> for SerializableLayerNormalizationWeight {
    /// Applies the stored `gamma` and `beta` weights to the target layer
    ///
    /// # Errors
    ///
    /// - Returns an error if `gamma` or `beta` does not match the layer's expected parameter shape
    fn apply_to_layer(&self, layer: &mut LayerNormalization) -> Result<(), Error> {
        layer.set_weights(self.gamma.clone(), self.beta.clone())?;
        Ok(())
    }
}
