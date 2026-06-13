//! Weight container for the LayerNormalization layer

use crate::error::Error;
use crate::neural_network::layers::regularization::normalization::layer_normalization::LayerNormalization;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a LayerNormalization layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormalizationLayerWeight<'a> {
    /// Scale parameter
    pub gamma: Cow<'a, ArrayD<f32>>,
    /// Shift parameter
    pub beta: Cow<'a, ArrayD<f32>>,
}

impl ApplyWeights<LayerNormalization> for LayerNormalizationLayerWeight<'_> {
    /// Applies the stored `gamma` and `beta` weights to the target layer
    ///
    /// # Errors
    ///
    /// - Returns an error if `gamma` or `beta` does not match the layer's expected parameter shape
    fn apply_to_layer(&self, layer: &mut LayerNormalization) -> Result<(), Error> {
        layer.set_weights((*self.gamma).clone(), (*self.beta).clone())?;
        Ok(())
    }
}
