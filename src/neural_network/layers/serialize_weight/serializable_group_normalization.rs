//! Serializable representation of GroupNormalization layer weights

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of GroupNormalization layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGroupNormalizationWeight {
    /// Scale parameter
    pub gamma: ArrayD<f32>,
    /// Shift parameter
    pub beta: ArrayD<f32>,
}

impl ApplyWeights<GroupNormalization> for SerializableGroupNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut GroupNormalization) -> Result<(), Error> {
        layer.set_weights(self.gamma.clone(), self.beta.clone())?;
        Ok(())
    }
}
