//! Weight container for the Conv2DTranspose layer

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_2d_transpose::Conv2DTranspose;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array4};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Conv2DTranspose layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DTransposeLayerWeight<'a> {
    /// 4D transposed convolution kernel with shape (kernel_height, kernel_width, filters,
    /// channels). The filter axis comes before the channel axis, which is the reverse of the
    /// Conv2D kernel
    pub weight: Cow<'a, Array4<f32>>,
    /// Bias vector with shape (filters,)
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<Conv2DTranspose> for Conv2DTransposeLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Conv2DTranspose) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
