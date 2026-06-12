//! Serializable weight containers and (de)serialization helpers for all supported layer types

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

/// Serializable weight container for all supported layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SerializableLayerWeight {
    /// Weights for a Dense layer
    Dense(SerializableDenseWeight),
    /// Weights for a SimpleRNN layer
    SimpleRNN(SerializableSimpleRNNWeight),
    /// Weights for an LSTM layer
    LSTM(SerializableLSTMWeight),
    /// Weights for a GRU layer
    GRU(SerializableGRUWeight),
    /// Weights for a Conv1D layer
    Conv1D(SerializableConv1DWeight),
    /// Weights for a Conv2D layer
    Conv2D(SerializableConv2DWeight),
    /// Weights for a Conv3D layer
    Conv3D(SerializableConv3DWeight),
    /// Weights for a SeparableConv2D layer
    SeparableConv2D(SerializableSeparableConv2DWeight),
    /// Weights for a DepthwiseConv2D layer
    DepthwiseConv2D(SerializableDepthwiseConv2DWeight),
    /// Weights for a BatchNormalization layer
    BatchNormalization(SerializableBatchNormalizationWeight),
    /// Weights for a LayerNormalization layer
    LayerNormalization(SerializableLayerNormalizationWeight),
    /// Weights for an InstanceNormalization layer
    InstanceNormalization(SerializableInstanceNormalizationWeight),
    /// Weights for a GroupNormalization layer
    GroupNormalization(SerializableGroupNormalizationWeight),
    /// No weights for layers without parameters
    Empty,
}

impl SerializableLayerWeight {
    /// Converts a `LayerWeight` reference into an owned serializable weight
    ///
    /// All weight arrays are cloned directly; serde (via ndarray's `serde` feature)
    /// encodes each array compactly as its shape plus a flat data buffer
    ///
    /// # Parameters
    ///
    /// - `weight` - Layer weights to convert into a serializable form
    ///
    /// # Returns
    ///
    /// - `SerializableLayerWeight` - Serializable representation of the provided weights
    pub fn from_layer_weight(weight: &LayerWeight) -> Self {
        match weight {
            LayerWeight::Empty => SerializableLayerWeight::Empty,
            LayerWeight::Dense(w) => SerializableLayerWeight::Dense(SerializableDenseWeight {
                weight: w.weight.clone(),
                bias: w.bias.clone(),
            }),
            LayerWeight::SimpleRNN(w) => {
                SerializableLayerWeight::SimpleRNN(SerializableSimpleRNNWeight {
                    kernel: w.kernel.clone(),
                    recurrent_kernel: w.recurrent_kernel.clone(),
                    bias: w.bias.clone(),
                })
            }
            LayerWeight::LSTM(w) => SerializableLayerWeight::LSTM(SerializableLSTMWeight {
                kernel: w.kernel.clone(),
                recurrent_kernel: w.recurrent_kernel.clone(),
                bias: w.bias.clone(),
            }),
            LayerWeight::GRU(w) => SerializableLayerWeight::GRU(SerializableGRUWeight {
                kernel: w.kernel.clone(),
                recurrent_kernel: w.recurrent_kernel.clone(),
                bias: w.bias.clone(),
            }),
            LayerWeight::Conv1D(w) => SerializableLayerWeight::Conv1D(SerializableConv1DWeight {
                weight: w.weight.clone(),
                bias: w.bias.clone(),
            }),
            LayerWeight::Conv2D(w) => SerializableLayerWeight::Conv2D(SerializableConv2DWeight {
                weight: w.weight.clone(),
                bias: w.bias.clone(),
            }),
            LayerWeight::Conv3D(w) => SerializableLayerWeight::Conv3D(SerializableConv3DWeight {
                weight: w.weight.clone(),
                bias: w.bias.clone(),
            }),
            LayerWeight::SeparableConv2DLayer(w) => {
                SerializableLayerWeight::SeparableConv2D(SerializableSeparableConv2DWeight {
                    depthwise_weight: w.depthwise_weight.clone(),
                    pointwise_weight: w.pointwise_weight.clone(),
                    bias: w.bias.clone(),
                })
            }
            LayerWeight::DepthwiseConv2DLayer(w) => {
                SerializableLayerWeight::DepthwiseConv2D(SerializableDepthwiseConv2DWeight {
                    weight: w.weight.clone(),
                    bias: w.bias.clone(),
                })
            }
            LayerWeight::BatchNormalization(w) => {
                SerializableLayerWeight::BatchNormalization(SerializableBatchNormalizationWeight {
                    gamma: w.gamma.clone(),
                    beta: w.beta.clone(),
                    running_mean: w.running_mean.clone(),
                    running_var: w.running_var.clone(),
                })
            }
            LayerWeight::LayerNormalizationLayer(w) => {
                SerializableLayerWeight::LayerNormalization(SerializableLayerNormalizationWeight {
                    gamma: w.gamma.clone(),
                    beta: w.beta.clone(),
                })
            }
            LayerWeight::InstanceNormalizationLayer(w) => {
                SerializableLayerWeight::InstanceNormalization(
                    SerializableInstanceNormalizationWeight {
                        gamma: w.gamma.clone(),
                        beta: w.beta.clone(),
                    },
                )
            }
            LayerWeight::GroupNormalizationLayer(w) => {
                SerializableLayerWeight::GroupNormalization(SerializableGroupNormalizationWeight {
                    gamma: w.gamma.clone(),
                    beta: w.beta.clone(),
                })
            }
        }
    }
}

/// Serializable layer metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer type name
    pub layer_type: String,
    /// Layer output shape description
    pub output_shape: String,
}

/// Serializable layer with metadata and weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayer {
    /// Layer metadata describing type and output shape
    pub info: LayerInfo,
    /// Layer weights in a serializable format
    pub weights: SerializableLayerWeight,
}

/// Serializable representation of a Sequential model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSequential {
    /// Ordered list of layers with metadata and weights
    pub layers: Vec<SerializableLayer>,
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

/// Applies serializable weights to a layer instance
///
/// # Parameters
///
/// - `layer` - Mutable reference to the target layer
/// - `weights` - Serializable weights to apply
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
    weights: &SerializableLayerWeight,
    expected_type: &str,
) -> Result<(), Error> {
    use std::any::Any;
    let layer_any: &mut dyn Any = layer;

    match weights {
        // No weights to set for empty layers
        SerializableLayerWeight::Empty => {}

        SerializableLayerWeight::Dense(w) => {
            apply_weights_simple!(layer_any, w, Dense, "Dense", expected_type);
        }
        SerializableLayerWeight::SimpleRNN(w) => {
            apply_weights_simple!(layer_any, w, SimpleRNN, "SimpleRNN", expected_type);
        }
        SerializableLayerWeight::LSTM(w) => {
            apply_weights_simple!(layer_any, w, LSTM, "LSTM", expected_type);
        }
        SerializableLayerWeight::GRU(w) => {
            apply_weights_simple!(layer_any, w, GRU, "GRU", expected_type);
        }
        SerializableLayerWeight::Conv1D(w) => {
            apply_weights_simple!(layer_any, w, Conv1D, "Conv1D", expected_type);
        }
        SerializableLayerWeight::Conv2D(w) => {
            apply_weights_simple!(layer_any, w, Conv2D, "Conv2D", expected_type);
        }
        SerializableLayerWeight::Conv3D(w) => {
            apply_weights_simple!(layer_any, w, Conv3D, "Conv3D", expected_type);
        }
        SerializableLayerWeight::SeparableConv2D(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                SeparableConv2D,
                "SeparableConv2D",
                expected_type
            );
        }
        SerializableLayerWeight::DepthwiseConv2D(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                DepthwiseConv2D,
                "DepthwiseConv2D",
                expected_type
            );
        }
        SerializableLayerWeight::BatchNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                BatchNormalization,
                "BatchNormalization",
                expected_type
            );
        }
        SerializableLayerWeight::LayerNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                LayerNormalization,
                "LayerNormalization",
                expected_type
            );
        }
        SerializableLayerWeight::InstanceNormalization(w) => {
            apply_weights_simple!(
                layer_any,
                w,
                InstanceNormalization,
                "InstanceNormalization",
                expected_type
            );
        }
        SerializableLayerWeight::GroupNormalization(w) => {
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

/// Serializable representation of a BatchNormalization layer's weights
pub mod serializable_batch_normalization_weight;
/// Serializable representation of a Conv1D layer's weights
pub mod serializable_conv_1d_weight;
/// Serializable representation of a Conv2D layer's weights
pub mod serializable_conv_2d_weight;
/// Serializable representation of a Conv3D layer's weights
pub mod serializable_conv_3d_weight;
/// Serializable representation of a Dense layer's weights
pub mod serializable_dense_weight;
/// Serializable representation of a DepthwiseConv2D layer's weights
pub mod serializable_depthwise_conv_2d_weight;
/// Serializable representation of a GroupNormalization layer's weights
pub mod serializable_group_normalization;
/// Serializable representation of a GRU layer's weights
pub mod serializable_gru_weight;
/// Serializable representation of an instance normalization layer's weights
pub mod serializable_instance_normalization;
/// Serializable representation of a LayerNormalization layer's weights
pub mod serializable_layer_normalization_weight;
/// Serializable representation of an LSTM layer's weights
pub mod serializable_lstm_weight;
/// Serializable representation of a SeparableConv2D layer's weights
pub mod serializable_separable_conv_2d_weight;
/// Serializable representation of a SimpleRNN layer's weights
pub mod serializable_simple_rnn_weight;

pub use serializable_batch_normalization_weight::*;
pub use serializable_conv_1d_weight::*;
pub use serializable_conv_2d_weight::*;
pub use serializable_conv_3d_weight::*;
pub use serializable_dense_weight::*;
pub use serializable_depthwise_conv_2d_weight::*;
pub use serializable_group_normalization::*;
pub use serializable_gru_weight::*;
pub use serializable_instance_normalization::*;
pub use serializable_layer_normalization_weight::*;
pub use serializable_lstm_weight::*;
pub use serializable_separable_conv_2d_weight::*;
pub use serializable_simple_rnn_weight::*;
