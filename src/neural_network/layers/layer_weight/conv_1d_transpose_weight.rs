//! Weight container for the Conv1DTranspose layer

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_1d_transpose::Conv1DTranspose;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Conv1DTranspose layer
///
/// This struct stores each field as [`Cow`]. Saving borrows the live layer arrays without
/// cloning. Loading deserializes them into owned arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1DTransposeLayerWeight<'a> {
    /// 3D transposed convolution kernel with shape (kernel_size, filters, channels). The filter
    /// axis comes before the channel axis, which is the reverse of the Conv1D kernel
    pub weight: Cow<'a, Array3<f32>>,
    /// Bias vector with shape (filters,)
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<Conv1DTranspose> for Conv1DTransposeLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Conv1DTranspose) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
