//! Weight container for the LSTM layer

use crate::error::Error;
use crate::neural_network::layers::recurrent::lstm::LSTM;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of an LSTM layer
///
/// The four gates are stored fused, gate column blocks in the order `[i | f | g | o]`
/// (input, forget, cell, output - the Keras LSTM layout)
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMLayerWeight<'a> {
    /// Fused input kernel (input_dim, 4 * units)
    pub kernel: Cow<'a, Array2<f32>>,
    /// Fused recurrent kernel (units, 4 * units)
    pub recurrent_kernel: Cow<'a, Array2<f32>>,
    /// Fused bias (1, 4 * units)
    pub bias: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<LSTM> for LSTMLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut LSTM) -> Result<(), Error> {
        layer.set_weights(
            (*self.kernel).clone(),
            (*self.recurrent_kernel).clone(),
            (*self.bias).clone(),
        )?;
        Ok(())
    }
}
