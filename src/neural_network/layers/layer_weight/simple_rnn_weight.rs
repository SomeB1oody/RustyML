//! Weight container for the SimpleRNN layer

use crate::error::Error;
use crate::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use crate::neural_network::traits::ApplyWeights;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a SimpleRNN layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleRNNLayerWeight<'a> {
    /// Input kernel matrix of shape (input_dim, units)
    pub kernel: Cow<'a, Array2<f32>>,
    /// Recurrent kernel matrix of shape (units, units)
    pub recurrent_kernel: Cow<'a, Array2<f32>>,
    /// Bias matrix of shape (1, units)
    pub bias: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<SimpleRNN> for SimpleRNNLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut SimpleRNN) -> Result<(), Error> {
        layer.set_weights(
            (*self.kernel).clone(),
            (*self.recurrent_kernel).clone(),
            (*self.bias).clone(),
        )?;
        Ok(())
    }
}
