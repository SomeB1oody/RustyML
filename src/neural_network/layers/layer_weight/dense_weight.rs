//! Weight container for the Dense layer

use crate::error::Error;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Dense layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayerWeight<'a> {
    /// Weight matrix of shape (input_dim, output_dim)
    pub weight: Cow<'a, Array2<f32>>,
    /// Bias matrix of shape (1, output_dim)
    pub bias: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<Dense> for DenseLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Dense) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
