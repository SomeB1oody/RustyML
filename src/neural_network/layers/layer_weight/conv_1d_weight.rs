//! Weight container for the Conv1D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_1d::Conv1D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Conv1D layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1DLayerWeight<'a> {
    /// 3D convolution kernel with shape (filters, channels, kernel_size)
    pub weight: Cow<'a, Array3<f32>>,
    /// Bias matrix with shape (1, filters)
    pub bias: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<Conv1D> for Conv1DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Conv1D) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
