use super::*;
use serde::{Deserialize, Serialize};

/// Enum containing all possible serializable layer weight types
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
    Empty,
}

/// Serializable representation of a Dense layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDenseWeight {
    pub weight: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Dense<T>> for SerializableDenseWeight {
    fn apply_to_layer(&self, layer: &mut Dense<T>) -> Result<(), IoError> {
        let weight_array = vec2_to_array2(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}

/// Serializable representation of a SimpleRNN layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSimpleRNNWeight {
    pub kernel: Vec<Vec<f32>>,
    pub recurrent_kernel: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<SimpleRNN<T>> for SerializableSimpleRNNWeight {
    fn apply_to_layer(&self, layer: &mut SimpleRNN<T>) -> Result<(), IoError> {
        let kernel = vec2_to_array2(&self.kernel)?;
        let recurrent_kernel = vec2_to_array2(&self.recurrent_kernel)?;
        let bias = vec2_to_array2(&self.bias)?;
        layer.set_weights(kernel, recurrent_kernel, bias);
        Ok(())
    }
}

/// Serializable representation of a single LSTM gate's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMGateWeight {
    pub kernel: Vec<Vec<f32>>,
    pub recurrent_kernel: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}

/// Serializable representation of an LSTM layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLSTMWeight {
    pub input: SerializableLSTMGateWeight,
    pub forget: SerializableLSTMGateWeight,
    pub cell: SerializableLSTMGateWeight,
    pub output: SerializableLSTMGateWeight,
}

/// Serializable representation of a single GRU gate's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUGateWeight {
    pub kernel: Vec<Vec<f32>>,
    pub recurrent_kernel: Vec<Vec<f32>>,
    pub bias: Vec<Vec<f32>>,
}

/// Serializable representation of a GRU layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGRUWeight {
    pub reset: SerializableGRUGateWeight,
    pub update: SerializableGRUGateWeight,
    pub candidate: SerializableGRUGateWeight,
}

impl<T: ActivationLayer> ApplyWeights<GRU<T>> for SerializableGRUWeight {
    fn apply_to_layer(&self, layer: &mut GRU<T>) -> Result<(), IoError> {
        let reset_kernel = vec2_to_array2(&self.reset.kernel)?;
        let reset_recurrent = vec2_to_array2(&self.reset.recurrent_kernel)?;
        let reset_bias = vec2_to_array2(&self.reset.bias)?;

        let update_kernel = vec2_to_array2(&self.update.kernel)?;
        let update_recurrent = vec2_to_array2(&self.update.recurrent_kernel)?;
        let update_bias = vec2_to_array2(&self.update.bias)?;

        let candidate_kernel = vec2_to_array2(&self.candidate.kernel)?;
        let candidate_recurrent = vec2_to_array2(&self.candidate.recurrent_kernel)?;
        let candidate_bias = vec2_to_array2(&self.candidate.bias)?;

        layer.set_weights(
            reset_kernel,
            reset_recurrent,
            reset_bias,
            update_kernel,
            update_recurrent,
            update_bias,
            candidate_kernel,
            candidate_recurrent,
            candidate_bias,
        );
        Ok(())
    }
}

impl<T: ActivationLayer> ApplyWeights<LSTM<T>> for SerializableLSTMWeight {
    fn apply_to_layer(&self, layer: &mut LSTM<T>) -> Result<(), IoError> {
        let input_kernel = vec2_to_array2(&self.input.kernel)?;
        let input_recurrent = vec2_to_array2(&self.input.recurrent_kernel)?;
        let input_bias = vec2_to_array2(&self.input.bias)?;

        let forget_kernel = vec2_to_array2(&self.forget.kernel)?;
        let forget_recurrent = vec2_to_array2(&self.forget.recurrent_kernel)?;
        let forget_bias = vec2_to_array2(&self.forget.bias)?;

        let cell_kernel = vec2_to_array2(&self.cell.kernel)?;
        let cell_recurrent = vec2_to_array2(&self.cell.recurrent_kernel)?;
        let cell_bias = vec2_to_array2(&self.cell.bias)?;

        let output_kernel = vec2_to_array2(&self.output.kernel)?;
        let output_recurrent = vec2_to_array2(&self.output.recurrent_kernel)?;
        let output_bias = vec2_to_array2(&self.output.bias)?;

        layer.set_weights(
            input_kernel,
            input_recurrent,
            input_bias,
            forget_kernel,
            forget_recurrent,
            forget_bias,
            cell_kernel,
            cell_recurrent,
            cell_bias,
            output_kernel,
            output_recurrent,
            output_bias,
        );
        Ok(())
    }
}

/// Serializable representation of a Conv1D layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv1DWeight {
    pub weight: Vec<Vec<Vec<f32>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Conv1D<T>> for SerializableConv1DWeight {
    fn apply_to_layer(&self, layer: &mut Conv1D<T>) -> Result<(), IoError> {
        let weight_array = vec3_to_array3(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}

/// Serializable representation of a Conv2D layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv2DWeight {
    pub weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Conv2D<T>> for SerializableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut Conv2D<T>) -> Result<(), IoError> {
        let weight_array = vec4_to_array4(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}

/// Serializable representation of a Conv3D layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConv3DWeight {
    pub weight: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<Conv3D<T>> for SerializableConv3DWeight {
    fn apply_to_layer(&self, layer: &mut Conv3D<T>) -> Result<(), IoError> {
        let weight_array = vec5_to_array5(&self.weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}

/// Serializable representation of a SeparableConv2D layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSeparableConv2DWeight {
    pub depthwise_weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub pointwise_weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub bias: Vec<Vec<f32>>,
}

impl<T: ActivationLayer> ApplyWeights<SeparableConv2D<T>> for SerializableSeparableConv2DWeight {
    fn apply_to_layer(&self, layer: &mut SeparableConv2D<T>) -> Result<(), IoError> {
        let depthwise_weight = vec4_to_array4(&self.depthwise_weight)?;
        let pointwise_weight = vec4_to_array4(&self.pointwise_weight)?;
        let bias_array = vec2_to_array2(&self.bias)?;
        layer.set_weights(depthwise_weight, pointwise_weight, bias_array);
        Ok(())
    }
}

/// Serializable representation of a DepthwiseConv2D layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDepthwiseConv2DWeight {
    pub weight: Vec<Vec<Vec<Vec<f32>>>>,
    pub bias: Vec<f32>,
}

impl<T: ActivationLayer> ApplyWeights<DepthwiseConv2D<T>> for SerializableDepthwiseConv2DWeight {
    fn apply_to_layer(&self, layer: &mut DepthwiseConv2D<T>) -> Result<(), IoError> {
        let weight_array = vec4_to_array4(&self.weight)?;
        let bias_array = Array1::from_vec(self.bias.clone());
        layer.set_weights(weight_array, bias_array);
        Ok(())
    }
}

impl SerializableLayerWeight {
    /// Converts LayerWeight references to owned SerializableLayerWeight
    pub fn from_layer_weight(weight: &LayerWeight) -> Self {
        match weight {
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
                input: SerializableLSTMGateWeight {
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
                forget: SerializableLSTMGateWeight {
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
                cell: SerializableLSTMGateWeight {
                    kernel: w.cell.kernel.outer_iter().map(|row| row.to_vec()).collect(),
                    recurrent_kernel: w
                        .cell
                        .recurrent_kernel
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                    bias: w.cell.bias.outer_iter().map(|row| row.to_vec()).collect(),
                },
                output: SerializableLSTMGateWeight {
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
                reset: SerializableGRUGateWeight {
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
                update: SerializableGRUGateWeight {
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
                candidate: SerializableGRUGateWeight {
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
            LayerWeight::Empty => SerializableLayerWeight::Empty,
        }
    }
}

/// Information about a layer's type and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub layer_type: String,
    pub output_shape: String,
}

/// Complete serializable representation of a layer including its weights and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayer {
    pub info: LayerInfo,
    pub weights: SerializableLayerWeight,
}

/// Serializable representation of a Sequential model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSequential {
    pub layers: Vec<SerializableLayer>,
}

// Helper conversion functions used by multiple weight types
fn vec2_to_array2(vec: &[Vec<f32>]) -> Result<Array2<f32>, IoError> {
    let rows = vec.len();
    let cols = if rows > 0 { vec[0].len() } else { 0 };
    let flat: Vec<f32> = vec.iter().flat_map(|row| row.iter().cloned()).collect();
    Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

fn vec3_to_array3(vec: &[Vec<Vec<f32>>]) -> Result<Array3<f32>, IoError> {
    let d0 = vec.len();
    let d1 = if d0 > 0 { vec[0].len() } else { 0 };
    let d2 = if d1 > 0 && d0 > 0 { vec[0][0].len() } else { 0 };
    let flat: Vec<f32> = vec
        .iter()
        .flat_map(|v1| v1.iter().flat_map(|v2| v2.iter().cloned()))
        .collect();
    Array3::from_shape_vec((d0, d1, d2), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

fn vec4_to_array4(vec: &[Vec<Vec<Vec<f32>>>]) -> Result<Array4<f32>, IoError> {
    let d0 = vec.len();
    let d1 = if d0 > 0 { vec[0].len() } else { 0 };
    let d2 = if d1 > 0 && d0 > 0 { vec[0][0].len() } else { 0 };
    let d3 = if d2 > 0 && d1 > 0 && d0 > 0 {
        vec[0][0][0].len()
    } else {
        0
    };
    let flat: Vec<f32> = vec
        .iter()
        .flat_map(|v1| {
            v1.iter()
                .flat_map(|v2| v2.iter().flat_map(|v3| v3.iter().cloned()))
        })
        .collect();
    Array4::from_shape_vec((d0, d1, d2, d3), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

fn vec5_to_array5(vec: &[Vec<Vec<Vec<Vec<f32>>>>]) -> Result<Array5<f32>, IoError> {
    let d0 = vec.len();
    let d1 = if d0 > 0 { vec[0].len() } else { 0 };
    let d2 = if d1 > 0 && d0 > 0 { vec[0][0].len() } else { 0 };
    let d3 = if d2 > 0 && d1 > 0 && d0 > 0 {
        vec[0][0][0].len()
    } else {
        0
    };
    let d4 = if d3 > 0 && d2 > 0 && d1 > 0 && d0 > 0 {
        vec[0][0][0][0].len()
    } else {
        0
    };
    let flat: Vec<f32> = vec
        .iter()
        .flat_map(|v1| {
            v1.iter().flat_map(|v2| {
                v2.iter()
                    .flat_map(|v3| v3.iter().flat_map(|v4| v4.iter().cloned()))
            })
        })
        .collect();
    Array5::from_shape_vec((d0, d1, d2, d3, d4), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
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

/// A macro that applies weights to a layer and handles type mismatch errors.
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
macro_rules! apply_layer_weights {
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

/// Applies serializable weights to a neural network layer.
///
/// This function takes a mutable reference to a layer and applies the appropriate weights
/// based on the layer type. It handles type checking and downcasting to ensure weights
/// are applied to the correct layer type.
///
/// # Parameters
///
/// - `layer` - Mutable reference to the layer that will receive the weights
/// - `weights` - Reference to the serializable weights to apply
/// - `expected_type` - String describing the expected layer type (used for error messages)
///
/// # Returns
///
/// - `Ok(())` - Weights were successfully applied to the layer
/// - `Err(IoError)` - Layer type mismatch or weight application failed
pub fn apply_weights_to_layer(
    layer: &mut dyn Layer,
    weights: &SerializableLayerWeight,
    expected_type: &str,
) -> Result<(), IoError> {
    use std::any::Any;
    let layer_any: &mut dyn Any = layer;

    match weights {
        SerializableLayerWeight::Dense(w) => {
            apply_layer_weights!(layer_any, w, Dense, "Dense", expected_type);
        }
        SerializableLayerWeight::SimpleRNN(w) => {
            apply_layer_weights!(layer_any, w, SimpleRNN, "SimpleRNN", expected_type);
        }
        SerializableLayerWeight::LSTM(w) => {
            apply_layer_weights!(layer_any, w, LSTM, "LSTM", expected_type);
        }
        SerializableLayerWeight::GRU(w) => {
            apply_layer_weights!(layer_any, w, GRU, "GRU", expected_type);
        }
        SerializableLayerWeight::Conv1D(w) => {
            apply_layer_weights!(layer_any, w, Conv1D, "Conv1D", expected_type);
        }
        SerializableLayerWeight::Conv2D(w) => {
            apply_layer_weights!(layer_any, w, Conv2D, "Conv2D", expected_type);
        }
        SerializableLayerWeight::Conv3D(w) => {
            apply_layer_weights!(layer_any, w, Conv3D, "Conv3D", expected_type);
        }
        SerializableLayerWeight::SeparableConv2D(w) => {
            apply_layer_weights!(
                layer_any,
                w,
                SeparableConv2D,
                "SeparableConv2D",
                expected_type
            );
        }
        SerializableLayerWeight::DepthwiseConv2D(w) => {
            apply_layer_weights!(
                layer_any,
                w,
                DepthwiseConv2D,
                "DepthwiseConv2D",
                expected_type
            );
        }
        SerializableLayerWeight::Empty => {
            // No weights to set for empty layers
        }
    }

    Ok(())
}
