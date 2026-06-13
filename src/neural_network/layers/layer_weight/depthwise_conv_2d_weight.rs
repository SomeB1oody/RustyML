//! Weight container for the DepthwiseConv2D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array4};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a DepthwiseConv2D layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthwiseConv2DLayerWeight<'a> {
    /// 4D depthwise convolution kernel with shape (filters, 1, kernel_height, kernel_width)
    pub weight: Cow<'a, Array4<f32>>,
    /// 1D bias vector with shape (filters), one entry per filter
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<DepthwiseConv2D> for DepthwiseConv2DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut DepthwiseConv2D) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
