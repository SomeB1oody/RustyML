use super::*;
use crate::{Deserialize, Serialize};

/// Serializable weight container for all supported layer types.
///
/// # Variants
///
/// - `Dense` - Weights for a Dense layer
/// - `SimpleRNN` - Weights for a SimpleRNN layer
/// - `LSTM` - Weights for an LSTM layer
/// - `GRU` - Weights for a GRU layer
/// - `Conv1D` - Weights for a Conv1D layer
/// - `Conv2D` - Weights for a Conv2D layer
/// - `Conv3D` - Weights for a Conv3D layer
/// - `SeparableConv2D` - Weights for a SeparableConv2D layer
/// - `DepthwiseConv2D` - Weights for a DepthwiseConv2D layer
/// - `BatchNormalization` - Weights for a BatchNormalization layer
/// - `LayerNormalization` - Weights for a LayerNormalization layer
/// - `InstanceNormalization` - Weights for an InstanceNormalization layer
/// - `GroupNormalization` - Weights for a GroupNormalization layer
/// - `Empty` - No weights for layers without parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SerializableLayerWeight {
    Dense(SerializableDenseWeight),
    SimpleRNN(SerializableSimpleRNNWeight),
    LSTM(SerializableLSTMWeight),
    GRU(SerializableGRUWeight),
    Conv1D(SerializableConv1DWeight),
    Conv2D(SerializableConv2DWeight),
    Conv3D(SerializableConv3DWeight),
    SeparableConv2D(SerializableSeparableConv2DWeight),
    DepthwiseConv2D(SerializableDepthwiseConv2DWeight),
    BatchNormalization(SerializableBatchNormalizationWeight),
    LayerNormalization(SerializableLayerNormalizationWeight),
    InstanceNormalization(SerializableInstanceNormalizationWeight),
    GroupNormalization(SerializableGroupNormalizationWeight),
    Empty,
}

impl SerializableLayerWeight {
    /// Converts a `LayerWeight` reference into an owned serializable weight.
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
                weight: w.weight.outer_iter().map(|row| row.to_vec()).collect(),
                bias: w.bias.outer_iter().map(|row| row.to_vec()).collect(),
            }),
            LayerWeight::SimpleRNN(w) => {
                SerializableLayerWeight::SimpleRNN(SerializableSimpleRNNWeight {
                    kernel: w.kernel.outer_iter().map(|row| row.to_vec()).collect(),
                    recurrent_kernel: w
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.bias.outer_iter().map(|row| row.to_vec()).collect(),
                })
            }
            LayerWeight::LSTM(w) => SerializableLayerWeight::LSTM(SerializableLSTMWeight {
                input: SerializableGateWeight {
                    kernel: w
                        .input
                        .kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    recurrent_kernel: w
                        .input
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.input.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
                forget: SerializableGateWeight {
                    kernel: w
                        .forget
                        .kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    recurrent_kernel: w
                        .forget
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.forget.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
                cell: SerializableGateWeight {
                    kernel: w.cell.kernel.outer_iter().map(|row| row.to_vec()).collect(),
                    recurrent_kernel: w
                        .cell
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.cell.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
                output: SerializableGateWeight {
                    kernel: w
                        .output
                        .kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    recurrent_kernel: w
                        .output
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.output.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
            }),
            LayerWeight::GRU(w) => SerializableLayerWeight::GRU(SerializableGRUWeight {
                reset: SerializableGateWeight {
                    kernel: w
                        .reset
                        .kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    recurrent_kernel: w
                        .reset
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.reset.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
                update: SerializableGateWeight {
                    kernel: w
                        .update
                        .kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    recurrent_kernel: w
                        .update
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.update.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
                candidate: SerializableGateWeight {
                    kernel: w
                        .candidate
                        .kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    recurrent_kernel: w
                        .candidate
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w
                        .candidate
                        .bias
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                },
            }),
            LayerWeight::Conv1D(w) => SerializableLayerWeight::Conv1D(SerializableConv1DWeight {
                weight: w
                    .weight
                    .outer_iter()
                    .map(|d1| d1.outer_iter().map(|d2| d2.to_vec()).collect())
                    .collect(),
                bias: w.bias.outer_iter().map(|row| row.to_vec()).collect(),
            }),
            LayerWeight::Conv2D(w) => SerializableLayerWeight::Conv2D(SerializableConv2DWeight {
                weight: w
                    .weight
                    .outer_iter()
                    .map(|d1| {
                        d1.outer_iter()
                            .map(|d2| d2.outer_iter().map(|d3| d3.to_vec()).collect())
                            .collect()
                    })
                    .collect(),
                bias: w.bias.outer_iter().map(|row| row.to_vec()).collect(),
            }),
            LayerWeight::Conv3D(w) => SerializableLayerWeight::Conv3D(SerializableConv3DWeight {
                weight: w
                    .weight
                    .outer_iter()
                    .map(|d1| {
                        d1.outer_iter()
                            .map(|d2| {
                                d2.outer_iter()
                                    .map(|d3| d3.outer_iter().map(|d4| d4.to_vec()).collect())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
                bias: w.bias.outer_iter().map(|row| row.to_vec()).collect(),
            }),
            LayerWeight::SeparableConv2DLayer(w) => {
                SerializableLayerWeight::SeparableConv2D(SerializableSeparableConv2DWeight {
                    depthwise_weight: w
                        .depthwise_weight
                        .outer_iter()
                        .map(|d1| {
                            d1.outer_iter()
                                .map(|d2| d2.outer_iter().map(|d3| d3.to_vec()).collect())
                                .collect()
                        })
                        .collect(),
                    pointwise_weight: w
                        .pointwise_weight
                        .outer_iter()
                        .map(|d1| {
                            d1.outer_iter()
                                .map(|d2| d2.outer_iter().map(|d3| d3.to_vec()).collect())
                                .collect()
                        })
                        .collect(),
                    bias: w.bias.outer_iter().map(|row| row.to_vec()).collect(),
                })
            }
            LayerWeight::DepthwiseConv2DLayer(w) => {
                SerializableLayerWeight::DepthwiseConv2D(SerializableDepthwiseConv2DWeight {
                    weight: w
                        .weight
                        .outer_iter()
                        .map(|d1| {
                            d1.outer_iter()
                                .map(|d2| d2.outer_iter().map(|d3| d3.to_vec()).collect())
                                .collect()
                        })
                        .collect(),
                    bias: w.bias.to_vec(),
                })
            }
            LayerWeight::BatchNormalization(w) => {
                SerializableLayerWeight::BatchNormalization(SerializableBatchNormalizationWeight {
                    gamma: w.gamma.iter().cloned().collect(),
                    beta: w.beta.iter().cloned().collect(),
                    running_mean: w.running_mean.iter().cloned().collect(),
                    running_var: w.running_var.iter().cloned().collect(),
                    shape: w.gamma.shape().to_vec(),
                })
            }
            LayerWeight::LayerNormalizationLayer(w) => {
                SerializableLayerWeight::LayerNormalization(SerializableLayerNormalizationWeight {
                    gamma: w.gamma.iter().cloned().collect(),
                    beta: w.beta.iter().cloned().collect(),
                    shape: w.gamma.shape().to_vec(),
                })
            }
            LayerWeight::InstanceNormalizationLayer(w) => {
                SerializableLayerWeight::InstanceNormalization(
                    SerializableInstanceNormalizationWeight {
                        gamma: w.gamma.iter().cloned().collect(),
                        beta: w.beta.iter().cloned().collect(),
                        shape: w.gamma.shape().to_vec(),
                    },
                )
            }
            LayerWeight::GroupNormalizationLayer(w) => {
                SerializableLayerWeight::GroupNormalization(SerializableGroupNormalizationWeight {
                    gamma: w.gamma.iter().cloned().collect(),
                    beta: w.beta.iter().cloned().collect(),
                    shape: w.gamma.shape().to_vec(),
                })
            }
        }
    }
}

/// Serializable layer metadata.
///
/// # Fields
///
/// - `layer_type` - Layer type name
/// - `output_shape` - Layer output shape description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub layer_type: String,
    pub output_shape: String,
}

/// Serializable layer with metadata and weights.
///
/// # Fields
///
/// - `info` - Layer metadata describing type and output shape
/// - `weights` - Layer weights in a serializable format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayer {
    pub info: LayerInfo,
    pub weights: SerializableLayerWeight,
}

/// Serializable representation of a Sequential model.
///
/// # Fields
///
/// - `layers` - Ordered list of layers with metadata and weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSequential {
    pub layers: Vec<SerializableLayer>,
}

/// A macro that applies weights to a layer with activation functions and handles type mismatch errors.
///
/// This macro uses `try_apply_with_activations!` to attempt weight application and
/// returns an error if the layer type doesn't match the expected type.
///
/// # Parameters
///
/// - `$layer_any` - A mutable reference to the layer as `&mut dyn Any`
/// - `$weight` - The weight structure to apply to the layer
/// - `$layer_type` - The specific layer type (e.g., Dense, Conv2D)
/// - `$layer_name` - String literal of the layer name (for error messages)
/// - `$expected_type` - String describing the expected layer type (for error messages)
macro_rules! apply_weights_with_activations {
    ($layer_any:expr, $weight:expr, $layer_type:ident, $layer_name:expr, $expected_type:expr) => {{
        let applied = try_apply_with_activations!($layer_any, $weight, $layer_type, $layer_name);
        if !applied {
            return Err(IoError::StdIoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected {} layer but got {}", $layer_name, $expected_type),
            )));
        }
    }};
}

/// A macro that applies weights to a simple layer (without activation functions) and handles type mismatch errors.
///
/// This macro attempts to downcast the layer to the specified type and applies weights.
/// Returns an error if the downcast fails.
///
/// # Parameters
///
/// - `$layer_any` - A mutable reference to the layer as `&mut dyn Any`
/// - `$weight` - The weight structure to apply to the layer
/// - `$layer_type` - The specific layer type (e.g., BatchNormalization, LayerNormalization)
/// - `$layer_name` - String literal of the layer name (for error messages)
/// - `$expected_type` - String describing the expected layer type (for error messages)
macro_rules! apply_weights_simple {
    ($layer_any:expr, $weight:expr, $layer_type:ident, $layer_name:expr, $expected_type:expr) => {{
        if let Some(layer) = $layer_any.downcast_mut::<$layer_type>() {
            $weight.apply_to_layer(layer)?;
        } else {
            return Err(IoError::StdIoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected {} layer but got {}", $layer_name, $expected_type),
            )));
        }
    }};
}

/// A macro that attempts to apply weights to different activation layer types.
///
/// This macro tries to downcast a generic layer to specific layer types with different
/// activation functions (ReLU, Sigmoid, Softmax, Tanh) and applies the given weights
/// if the downcast is successful.
///
/// # Parameters
///
/// - `$layer_any` - A mutable reference to the layer as `&mut dyn Any`
/// - `$weight` - The weight structure to apply to the layer
/// - `$layer_type` - The specific layer type (e.g., Dense, Conv2D)
/// - `$layer_name` - String literal of the layer name (used for debugging)
///
/// # Returns
///
/// - `true` - If the layer was successfully downcast and weights were applied
/// - `false` - If none of the activation types matched
macro_rules! try_apply_with_activations {
    ($layer_any:expr, $weight:expr, $layer_type:ident, $layer_name:expr) => {{
        if let Some(layer) = $layer_any.downcast_mut::<$layer_type<ReLU>>() {
            $weight.apply_to_layer(layer)?;
            true
        } else if let Some(layer) = $layer_any.downcast_mut::<$layer_type<Sigmoid>>() {
            $weight.apply_to_layer(layer)?;
            true
        } else if let Some(layer) = $layer_any.downcast_mut::<$layer_type<Softmax>>() {
            $weight.apply_to_layer(layer)?;
            true
        } else if let Some(layer) = $layer_any.downcast_mut::<$layer_type<Tanh>>() {
            $weight.apply_to_layer(layer)?;
            true
        } else {
            false
        }
    }};
}

/// Applies serializable weights to a layer instance.
///
/// # Parameters
///
/// - `layer` - Mutable reference to the target layer
/// - `weights` - Serializable weights to apply
/// - `expected_type` - Expected layer type label for error messages
///
/// # Returns
///
/// - `Result<(), IoError>` - Ok when weights are applied successfully
///
/// # Errors
///
/// - `IoError::StdIoError` - Layer type mismatch or invalid weight shape during conversion
pub fn apply_weights_to_layer(
    layer: &mut dyn Layer,
    weights: &SerializableLayerWeight,
    expected_type: &str,
) -> Result<(), IoError> {
    use std::any::Any;
    let layer_any: &mut dyn Any = layer;

    match weights {
        // No weights to set for empty layers
        SerializableLayerWeight::Empty => {}

        SerializableLayerWeight::Dense(w) => {
            apply_weights_with_activations!(layer_any, w, Dense, "Dense", expected_type);
        }
        SerializableLayerWeight::SimpleRNN(w) => {
            apply_weights_with_activations!(layer_any, w, SimpleRNN, "SimpleRNN", expected_type);
        }
        SerializableLayerWeight::LSTM(w) => {
            apply_weights_with_activations!(layer_any, w, LSTM, "LSTM", expected_type);
        }
        SerializableLayerWeight::GRU(w) => {
            apply_weights_with_activations!(layer_any, w, GRU, "GRU", expected_type);
        }
        SerializableLayerWeight::Conv1D(w) => {
            apply_weights_with_activations!(layer_any, w, Conv1D, "Conv1D", expected_type);
        }
        SerializableLayerWeight::Conv2D(w) => {
            apply_weights_with_activations!(layer_any, w, Conv2D, "Conv2D", expected_type);
        }
        SerializableLayerWeight::Conv3D(w) => {
            apply_weights_with_activations!(layer_any, w, Conv3D, "Conv3D", expected_type);
        }
        SerializableLayerWeight::SeparableConv2D(w) => {
            apply_weights_with_activations!(
                layer_any,
                w,
                SeparableConv2D,
                "SeparableConv2D",
                expected_type
            );
        }
        SerializableLayerWeight::DepthwiseConv2D(w) => {
            apply_weights_with_activations!(
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

/// Helper functions used by multiple weight types
mod helper_function;
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
/// Serializable representation of a single gate's weights
pub mod serializable_gate_weight;
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

use helper_function::*;
pub use serializable_batch_normalization_weight::*;
pub use serializable_conv_1d_weight::*;
pub use serializable_conv_2d_weight::*;
pub use serializable_conv_3d_weight::*;
pub use serializable_dense_weight::*;
pub use serializable_depthwise_conv_2d_weight::*;
pub use serializable_gate_weight::*;
pub use serializable_group_normalization::*;
pub use serializable_gru_weight::*;
pub use serializable_instance_normalization::*;
pub use serializable_layer_normalization_weight::*;
pub use serializable_lstm_weight::*;
pub use serializable_separable_conv_2d_weight::*;
pub use serializable_simple_rnn_weight::*;
