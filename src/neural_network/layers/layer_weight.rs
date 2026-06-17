//! Weight containers for neural network layers
//!
//! Defines the [`LayerWeight`](LayerWeight<'a>) enum and the per-layer weight structs it wraps. Each struct holds
//! its arrays as [`Cow`], so one type serves both directions. [`Layer::get_weights`] borrows
//! the live layer arrays (no clone) for inspection or saving, while loading deserializes into
//! owned arrays. The enum derives `Serialize`/`Deserialize`, so it is the on-disk weight format
//!
//! [`Cow`]: std::borrow::Cow
//! [`Layer::get_weights`]: crate::neural_network::traits::Layer::get_weights

use crate::{Deserialize, Serialize};

/// Per-layer weight container for the BatchNormalization layer
pub mod batch_normalization_weight;
/// Per-layer weight container for the Conv1D layer
pub mod conv_1d_weight;
/// Per-layer weight container for the Conv2D layer
pub mod conv_2d_weight;
/// Per-layer weight container for the Conv3D layer
pub mod conv_3d_weight;
/// Per-layer weight container for the Dense layer
pub mod dense_weight;
/// Per-layer weight container for the DepthwiseConv2D layer
pub mod depthwise_conv_2d_weight;
/// Per-layer weight container for the GroupNormalization layer
pub mod group_normalization_weight;
/// Per-layer weight container for the GRU layer
pub mod gru_weight;
/// Per-layer weight container for the InstanceNormalization layer
pub mod instance_normalization_weight;
/// Per-layer weight container for the LayerNormalization layer
pub mod layer_normalization_weight;
/// Per-layer weight container for the LSTM layer
pub mod lstm_weight;
/// Per-layer weight container for the SeparableConv2D layer
pub mod separable_conv_2d_weight;
/// Per-layer weight container for the SimpleRNN layer
pub mod simple_rnn_weight;

pub use batch_normalization_weight::*;
pub use conv_1d_weight::*;
pub use conv_2d_weight::*;
pub use conv_3d_weight::*;
pub use dense_weight::*;
pub use depthwise_conv_2d_weight::*;
pub use group_normalization_weight::*;
pub use gru_weight::*;
pub use instance_normalization_weight::*;
pub use layer_normalization_weight::*;
pub use lstm_weight::*;
pub use separable_conv_2d_weight::*;
pub use simple_rnn_weight::*;

/// Polymorphic container for the weights of different neural network layer types
///
/// Each variant corresponds to a specific layer type and holds the matching weight struct. The
/// `'a` lifetime lets a variant borrow the live layer arrays when inspecting or saving (via
/// [`Cow`](std::borrow::Cow)). Loading deserializes into owned arrays, so the type is used as
/// `LayerWeight<'static>` on the load path. The enum uses serde's default (externally tagged)
/// representation, which the non-self-describing postcard binary format requires
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerWeight<'a> {
    /// Weights for dense (fully connected) layers
    Dense(DenseLayerWeight<'a>),
    /// Weights for simple recurrent neural network layers
    SimpleRNN(SimpleRNNLayerWeight<'a>),
    /// Weights for long short-term memory layers
    LSTM(LSTMLayerWeight<'a>),
    /// Weights for gated recurrent unit layers
    GRU(GRULayerWeight<'a>),
    /// Weights for 1D convolutional layers
    Conv1D(Conv1DLayerWeight<'a>),
    /// Weights for 2D convolutional layers
    Conv2D(Conv2DLayerWeight<'a>),
    /// Weights for 3D convolutional layers
    Conv3D(Conv3DLayerWeight<'a>),
    /// Weights for 2D separable convolutional layers
    SeparableConv2D(SeparableConv2DLayerWeight<'a>),
    /// Weights for 2D depthwise convolutional layers
    DepthwiseConv2D(DepthwiseConv2DLayerWeight<'a>),
    /// Weights for batch normalization layers
    BatchNormalization(BatchNormalizationLayerWeight<'a>),
    /// Weights for layer normalization layers
    LayerNormalization(LayerNormalizationLayerWeight<'a>),
    /// Weights for instance normalization layers
    InstanceNormalization(InstanceNormalizationLayerWeight<'a>),
    /// Weights for group normalization layers
    GroupNormalization(GroupNormalizationLayerWeight<'a>),
    /// Layer with no trainable parameters
    Empty,
}
