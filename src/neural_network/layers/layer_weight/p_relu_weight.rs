//! Weight container for the PReLU layer

use crate::error::Error;
use crate::neural_network::layers::activation::p_relu::PReLU;
use crate::neural_network::traits::ApplyWeights;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Weights of a PReLU layer
///
/// This struct stores the field as [`Cow`]. Saving borrows the live layer array without cloning.
/// Loading deserializes it into an owned array. The layer holds 1 parameter tensor, so this
/// container holds 1 field.
///
/// The rank of `alpha` follows the input rank, less 1, because the batch axis carries no slope.
/// A shared axis keeps its place in the array at extent 1, so the rank does not drop with it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PReLULayerWeight<'a> {
    /// Negative-side slopes. A shared axis has extent 1, and the slope broadcasts over it
    pub alpha: Cow<'a, ArrayD<f32>>,
}

impl ApplyWeights<PReLU> for PReLULayerWeight<'_> {
    fn apply_to_layer(&self, layer: &mut PReLU) -> Result<(), Error> {
        layer.set_weights((*self.alpha).clone())?;
        Ok(())
    }
}
