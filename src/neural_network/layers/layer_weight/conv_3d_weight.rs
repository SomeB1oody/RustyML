//! Weight container for the Conv3D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_3d::Conv3D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array5};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Conv3D layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv3DLayerWeight<'a> {
    /// 5D convolution kernel with shape \[kernel_depth, kernel_height, kernel_width, channels, filters\]
    pub weight: Cow<'a, Array5<f32>>,
    /// Bias vector with shape \[filters\]
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<Conv3D> for Conv3DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Conv3D) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
