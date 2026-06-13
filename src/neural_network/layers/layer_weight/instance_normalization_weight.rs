//! Weight container for the InstanceNormalization layer

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of an InstanceNormalization layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceNormalizationLayerWeight<'a> {
    /// Scale parameter
    pub gamma: Cow<'a, ArrayD<f32>>,
    /// Shift parameter
    pub beta: Cow<'a, ArrayD<f32>>,
}

impl ApplyWeights<InstanceNormalization> for InstanceNormalizationLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut InstanceNormalization) -> Result<(), Error> {
        layer.set_weights((*self.gamma).clone(), (*self.beta).clone())?;
        Ok(())
    }
}
