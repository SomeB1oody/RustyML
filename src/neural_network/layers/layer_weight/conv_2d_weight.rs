//! Weight container for the Conv2D layer

use crate::error::Error;
use crate::neural_network::layers::convolution::conv_2d::Conv2D;
use crate::neural_network::traits::ApplyWeights;
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a Conv2D layer
///
/// Stored as [`Cow`] so saving borrows the live layer arrays without cloning, while loading
/// deserializes them into owned arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DLayerWeight<'a> {
    /// 4D convolution kernel with shape (filters, channels, kernel_height, kernel_width)
    pub weight: Cow<'a, Array4<f32>>,
    /// Bias matrix with shape (1, filters)
    pub bias: Cow<'a, Array2<f32>>,
}

impl ApplyWeights<Conv2D> for Conv2DLayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut Conv2D) -> Result<(), Error> {
        layer.set_weights((*self.weight).clone(), (*self.bias).clone())?;
        Ok(())
    }
}
