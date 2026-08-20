//! Weight container for the DepthwiseConv1D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a DepthwiseConv1D layer
///
/// This struct stores each field as [`Cow`]. Saving borrows the live layer arrays without
/// cloning. Loading deserializes them into owned arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthwiseConv1DLayerWeight<'a> {
    /// 3D depthwise convolution kernel with shape (kernel_size, channels, depth_multiplier)
    pub weight: Cow<'a, Array3<f32>>,
    /// 1D bias vector with shape (channels * depth_multiplier), 1 entry per output channel
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<DepthwiseConv1D> for DepthwiseConv1DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut DepthwiseConv1D) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
