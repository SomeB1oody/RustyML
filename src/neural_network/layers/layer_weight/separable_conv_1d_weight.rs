//! Weight container for the SeparableConv1D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a SeparableConv1D layer
///
/// This struct stores each field as [`Cow`]. Saving borrows the live layer arrays without
/// cloning. Loading deserializes them into owned arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparableConv1DLayerWeight<'a> {
    /// 3D depthwise kernel with shape (kernel_size, channels, depth_multiplier)
    pub depthwise_weight: Cow<'a, Array3<f32>>,
    /// 3D pointwise kernel with shape (1, channels * depth_multiplier, filters)
    pub pointwise_weight: Cow<'a, Array3<f32>>,
    /// Bias vector with shape (filters,)
    pub bias: Cow<'a, Array1<f32>>,
}

impl ApplyWeights<SeparableConv1D> for SeparableConv1DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut SeparableConv1D) -> Result<(), Error> {
        layer.set_weights(
            (*self.depthwise_weight).clone(),
            (*self.pointwise_weight).clone(),
            (*self.bias).clone(),
        )?;
        Ok(())
    }
}
