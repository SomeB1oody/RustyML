//! Weight container for the Embedding layer

use crate::error::Error;
use crate::neural_network::layers::embedding::Embedding;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of an Embedding layer
///
/// This struct stores the field as [`Cow`]. Saving borrows the live layer array without cloning.
/// Loading deserializes it into an owned array. The layer holds 1 parameter tensor, so this
/// container holds 1 field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayerWeight<'a> {
    /// Lookup table of shape (input_dim, output_dim). Row `i` is the vector for index `i`
    pub embeddings: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<Embedding> for EmbeddingLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Embedding) -> Result<(), Error> {
        layer.set_weights((*self.embeddings).clone())?;
        Ok(())
    }
}
