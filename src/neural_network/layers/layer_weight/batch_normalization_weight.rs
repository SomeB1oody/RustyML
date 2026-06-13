//! Weight container for the BatchNormalization layer

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a BatchNormalization layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalizationLayerWeight<'a> {
    /// Scale parameter
    pub gamma: Cow<'a, ArrayD<f32>>,
    /// Shift parameter
    pub beta: Cow<'a, ArrayD<f32>>,
    /// Running mean used at inference
    pub running_mean: Cow<'a, ArrayD<f32>>,
    /// Running variance used at inference
    pub running_var: Cow<'a, ArrayD<f32>>,
}

impl ApplyWeights<BatchNormalization> for BatchNormalizationLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut BatchNormalization) -> Result<(), Error> {
        layer.set_weights(
            (*self.gamma).clone(),
            (*self.beta).clone(),
            (*self.running_mean).clone(),
            (*self.running_var).clone(),
        )?;
        Ok(())
    }
}
