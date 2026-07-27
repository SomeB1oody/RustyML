//! Weight container for the SeparableConv2D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array4};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a SeparableConv2D layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparableConv2DLayerWeight<'a> {
    /// 4D depthwise kernel with shape (kernel_height, kernel_width, channels, depth_multiplier)
    pub depthwise_weight: Cow<'a, Array4<f32>>,
    /// 4D pointwise kernel with shape (1, 1, channels * depth_multiplier, filters)
    pub pointwise_weight: Cow<'a, Array4<f32>>,
    /// Bias vector with shape (filters,)
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<SeparableConv2D> for SeparableConv2DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut SeparableConv2D) -> Result<(), Error> {
        layer.set_weights(
            (*self.depthwise_weight).clone(),
            (*self.pointwise_weight).clone(),
            (*self.bias).clone(),
        )?;
        Ok(())
    }
}
