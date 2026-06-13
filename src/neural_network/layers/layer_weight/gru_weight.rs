//! Weight container for the GRU layer

use crate::error::Error;
use crate::neural_network::layers::recurrent::gru::GRU;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a GRU layer
///
/// The three gates are stored fused, gate column blocks in the order `[r | z | h]`
/// (reset, update, candidate)
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRULayerWeight<'a> {
    /// Fused input kernel (input_dim, 3 * units)
    pub kernel: Cow<'a, Array2<f32>>,
    /// Fused recurrent kernel (units, 3 * units)
    pub recurrent_kernel: Cow<'a, Array2<f32>>,
    /// Fused bias (1, 3 * units)
    pub bias: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<GRU> for GRULayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut GRU) -> Result<(), Error> {
        layer.set_weights(
            (*self.kernel).clone(),
            (*self.recurrent_kernel).clone(),
            (*self.bias).clone(),
        )?;
        Ok(())
    }
}
