//! Model-level serialization scaffolding for a Sequential model
//!
//! The per-layer weight containers live in
//! [`layer_weight`](crate::neural_network::layers::layer_weight); this module wraps them with
//! layer metadata into a serializable whole-model snapshot and applies loaded weights back onto a
//! live model

use crate::error::{Error, IoError};
use crate::neural_network::layers::convolution::{
    conv_1d::Conv1D, conv_2d::Conv2D, conv_3d::Conv3D, depthwise_conv_2d::DepthwiseConv2D,
    separable_conv_2d::SeparableConv2D,
};
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::recurrent::{gru::GRU, lstm::LSTM, simple_rnn::SimpleRNN};
use crate::neural_network::layers::regularization::normalization::{
    batch_normalization::BatchNormalization, group_normalization::GroupNormalization,
    instance_normalization::InstanceNormalization, layer_normalization::LayerNormalization,
};
use crate::neural_network::traits::ApplyWeights;
use crate::neural_network::traits::Layer;
use crate::{Deserialize, Serialize};

/// Serializable layer metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer type name
    pub layer_type: String,
    /// Layer output shape description
    pub output_shape: String,
}

/// Serializable layer with metadata and weights
///
/// The `'a` lifetime is threaded from [`LayerWeight`]: borrowed when saving, `'static` (owned)
/// when loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayer<'a> {
    /// Layer metadata describing type and output shape
    pub info: LayerInfo,
    /// Layer weights in a serializable format
    pub weights: LayerWeight<'a>,
}

/// Serializable representation of a Sequential model
///
/// The `'a` lifetime is threaded from [`LayerWeight`]: borrowed when saving, `'static` (owned)
/// when loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSequential<'a> {
    /// Ordered list of layers with metadata and weights
    pub layers: Vec<SerializableLayer<'a>>,
}

/// Downcasts a layer to the given type and applies weights, surfacing a structural
/// mismatch error if the downcast fails
///
/// # Parameters
///
/// - `$layer_any` - Mutable reference to the layer as `&mut dyn Any`
/// - `$weight` - Weight structure to apply to the layer
/// - `$layer_type` - Concrete layer type (e.g. BatchNormalization, LayerNormalization)
/// - `$layer_name` - String literal of the layer name, used in error messages
/// - `$expected_type` - Expected layer type label, used in error messages
macro_rules! apply_weights_simple {
    ($layer_any:expr, $weight:expr, $layer_type:ident, $layer_name:expr, $expected_type:expr) => {{
        if let Some(layer) = $layer_any.downcast_mut::<$layer_type>() {
            $weight
                .apply_to_layer(layer)
                .map_err(|e| Error::Io(IoError::ModelStructureMismatch(e.to_string())))?;
        } else {
            return Err(Error::Io(IoError::ModelStructureMismatch(format!(
                "expected a {} layer but the target layer has type `{}`",
                $layer_name, $expected_type
            ))));
        }
    }};
}

/// Applies a layer's deserialized weights back onto a live layer instance
///
/// # Parameters
///
/// - `layer` - Mutable reference to the target layer
/// - `weights` - Weights to apply
/// - `expected_type` - Expected layer type label for error messages
///
/// # Returns
///
/// - `Result<(), Error>` - Ok when weights are applied successfully
///
/// # Errors
///
/// - `Error::Io(IoError::ModelStructureMismatch)` - Layer type mismatch or invalid weight shape during conversion
pub fn apply_weights_to_layer(
    layer: &mut dyn Layer,
    weights: &LayerWeight<'_>,
    expected_type: &str,
) -> Result<(), Error> {
    use std::any::Any;
    let layer_any: &mut dyn Any = layer;

    match weights {
        // No weights to set for empty layers
        LayerWeight::Empty => {}

        LayerWeight::Dense(w) => {
            apply_weights_simple!(layer_any, w, Dense, "Dense", expected_type);
        }
        LayerWeight::SimpleRNN(w) => {
            apply_weights_simple!(layer_any, w, SimpleRNN, "SimpleRNN", expected_type);
        }
        LayerWeight::LSTM(w) => {
            apply_weights_simple!(layer_any, w, LSTM, "LSTM", expected_type);
        }
        LayerWeight::GRU(w) => {
            apply_weights_simple!(layer_any, w, GRU, "GRU", expected_type);
        }
        LayerWeight::Conv1D(w) => {
            apply_weights_simple!(layer_any, w, Conv1D, "Conv1D", expected_type);
        }
        LayerWeight::Conv2D(w) => {
            apply_weights_simple!(layer_any, w, Conv2D, "Conv2D", expected_type);
        }
        LayerWeight::Conv3D(w) => {
            apply_weights_simple!(layer_any, w, Conv3D, "Conv3D", expected_type);
        }
        LayerWeight::SeparableConv2D(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                SeparableConv2D,
                "SeparableConv2D",
                expected_type
            );
        }
        LayerWeight::DepthwiseConv2D(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                DepthwiseConv2D,
                "DepthwiseConv2D",
                expected_type
            );
        }
        LayerWeight::BatchNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                BatchNormalization,
                "BatchNormalization",
                expected_type
            );
        }
        LayerWeight::LayerNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                LayerNormalization,
                "LayerNormalization",
                expected_type
            );
        }
        LayerWeight::InstanceNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                InstanceNormalization,
                "InstanceNormalization",
                expected_type
            );
        }
        LayerWeight::GroupNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                GroupNormalization,
                "GroupNormalization",
                expected_type
            );
        }
    }

    Ok(())
}
