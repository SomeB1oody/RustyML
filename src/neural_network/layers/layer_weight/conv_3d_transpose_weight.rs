//! Weight container for the Conv3DTranspose layer

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_3d_transpose::Conv3DTranspose;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array5};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Conv3DTranspose layer
///
/// This struct stores each field as [`Cow`]. Saving borrows the live layer arrays without
/// cloning. Loading deserializes them into owned arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv3DTransposeLayerWeight<'a> {
    /// 5D transposed convolution kernel with shape (kernel_depth, kernel_height, kernel_width,
    /// filters, channels). The filter axis comes before the channel axis, which is the reverse of
    /// the Conv3D kernel
    pub weight: Cow<'a, Array5<f32>>,
    /// Bias vector with shape (filters,)
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<Conv3DTranspose> for Conv3DTransposeLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Conv3DTranspose) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
