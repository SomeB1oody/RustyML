//! Serializable representation of [`BatchNormalization`] layer weights

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Serializable representation of BatchNormalization layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableBatchNormalizationWeight {
    /// Scale parameter
    pub gamma: ArrayD<f32>,
    /// Shift parameter
    pub beta: ArrayD<f32>,
    /// Running mean used at inference
    pub running_mean: ArrayD<f32>,
    /// Running variance used at inference
    pub running_var: ArrayD<f32>,
}

impl ApplyWeights<BatchNormalization> for SerializableBatchNormalizationWeight {
    fn apply_to_layer(&self, layer: &mut BatchNormalization) -> Result<(), Error> {
        layer.set_weights(
            self.gamma.clone(),
            self.beta.clone(),
            self.running_mean.clone(),
            self.running_var.clone(),
        )?;
        Ok(())
    }
}
