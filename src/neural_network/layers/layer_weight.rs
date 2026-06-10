//! Weight container types for neural network layers
//!
//! Defines the [`LayerWeight`](crate::neural_network::layers::layer_weight::LayerWeight) enum and the per-layer weight structs it wraps,
//! each holding borrowed references to a layer's trainable parameters

use ndarray::{Array1, Array2, Array3, Array4, Array5, ArrayD};

/// Polymorphic container for the weights of different neural network layer types
///
/// Each variant corresponds to a specific layer type and holds the matching
/// weight structure for that layer
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
    /// Weights for 2D separable convolutional layers
    SeparableConv2DLayer(SeparableConv2DLayerWeight<'a>),
    /// Weights for 2D depthwise convolutional layers
    DepthwiseConv2DLayer(DepthwiseConv2DLayerWeight<'a>),
    /// Weights for 3D convolutional layers
    Conv3D(Conv3DLayerWeight<'a>),
    /// Weights for batch normalization layers
    BatchNormalization(BatchNormalizationLayerWeight<'a>),
    /// Weights for layer normalization layers
    LayerNormalizationLayer(LayerNormalizationLayerWeight<'a>),
    /// Weights for instance normalization layers
    InstanceNormalizationLayer(InstanceNormalizationLayerWeight<'a>),
    /// Weights for group normalization layers
    GroupNormalizationLayer(GroupNormalizationLayerWeight<'a>),
    /// Layer with no trainable parameters
    Empty,
}

/// Weights for a dense (fully connected) neural network layer
pub struct DenseLayerWeight<'a> {
    /// Weight matrix with shape (input_features, output_features)
    pub weight: &'a Array2<f32>,
    /// Bias vector with shape (1, output_features)
    pub bias: &'a Array2<f32>,
}

/// Weights for a simple recurrent neural network layer
pub struct SimpleRNNLayerWeight<'a> {
    /// Weight matrix for input features
    pub kernel: &'a Array2<f32>,
    /// Weight matrix for recurrent connections
    pub recurrent_kernel: &'a Array2<f32>,
    /// Bias vector
    pub bias: &'a Array2<f32>,
}

/// Weights for a single gate in an LSTM layer
pub struct LSTMGateWeight<'a> {
    /// Weight matrix for input features
    pub kernel: &'a Array2<f32>,
    /// Weight matrix for recurrent connections
    pub recurrent_kernel: &'a Array2<f32>,
    /// Bias vector for the gate
    pub bias: &'a Array2<f32>,
}

/// Weights for a Long Short-Term Memory (LSTM) layer
///
/// Holds the weights for the four gates that control information flow in an LSTM
/// cell: input gate, forget gate, cell gate, and output gate
pub struct LSTMLayerWeight<'a> {
    /// Input gate weights, which control what new information to store
    pub input: LSTMGateWeight<'a>,
    /// Forget gate weights, which control what information to discard
    pub forget: LSTMGateWeight<'a>,
    /// Cell gate weights, which propose new cell state values
    pub cell: LSTMGateWeight<'a>,
    /// Output gate weights, which control what to output
    pub output: LSTMGateWeight<'a>,
}

/// Weights for a single gate in a GRU layer
pub struct GRUGateWeight<'a> {
    /// Weight matrix for input features
    pub kernel: &'a Array2<f32>,
    /// Weight matrix for recurrent connections
    pub recurrent_kernel: &'a Array2<f32>,
    /// Bias vector for the gate
    pub bias: &'a Array2<f32>,
}

/// Weights for a Gated Recurrent Unit (GRU) layer
///
/// Holds the weights for the three gates that control information flow in a GRU
/// cell: reset gate, update gate, and candidate gate
pub struct GRULayerWeight<'a> {
    /// Reset gate weights, which control what information to forget
    pub reset: GRUGateWeight<'a>,
    /// Update gate weights, which control how much to update the hidden state
    pub update: GRUGateWeight<'a>,
    /// Candidate gate weights, which propose new hidden state values
    pub candidate: GRUGateWeight<'a>,
}

/// Weights for a 1D convolutional layer
pub struct Conv1DLayerWeight<'a> {
    /// 3D convolution kernel with shape (output_channels, input_channels, kernel_size)
    pub weight: &'a Array3<f32>,
    /// Bias vector with shape (1, output_channels)
    pub bias: &'a Array2<f32>,
}

/// Weights for a 2D convolutional layer
pub struct Conv2DLayerWeight<'a> {
    /// 4D convolution kernel with shape (output_channels, input_channels, kernel_height, kernel_width)
    pub weight: &'a Array4<f32>,
    /// Bias vector with shape (1, output_channels)
    pub bias: &'a Array2<f32>,
}

/// Weights for a 3D convolutional layer
pub struct Conv3DLayerWeight<'a> {
    /// 5D convolution kernel with shape (output_channels, input_channels, kernel_depth, kernel_height, kernel_width)
    pub weight: &'a Array5<f32>,
    /// Bias vector with shape (1, output_channels)
    pub bias: &'a Array2<f32>,
}

/// Weights for a 2D separable convolutional layer
pub struct SeparableConv2DLayerWeight<'a> {
    /// 4D weight tensor for depthwise convolution filters with shape (depth_multiplier, input_channels, kernel_height, kernel_width)
    pub depthwise_weight: &'a Array4<f32>,
    /// 4D weight tensor for pointwise (1x1) convolution filters with shape (output_filters, input_channels * depth_multiplier, 1, 1)
    pub pointwise_weight: &'a Array4<f32>,
    /// Bias vector with shape (1, output_filters)
    pub bias: &'a Array2<f32>,
}

/// Weights for a 2D depthwise convolutional layer
pub struct DepthwiseConv2DLayerWeight<'a> {
    /// 4D weight tensor for depthwise filters with shape (depth_multiplier, input_channels, kernel_height, kernel_width)
    pub weight: &'a Array4<f32>,
    /// Bias vector with one bias per input channel
    pub bias: &'a Array1<f32>,
}

/// Weights for a batch normalization layer
pub struct BatchNormalizationLayerWeight<'a> {
    /// Learned scale parameter that controls the variance of normalized values
    pub gamma: &'a ArrayD<f32>,
    /// Learned shift parameter that controls the mean of normalized values
    pub beta: &'a ArrayD<f32>,
    /// Exponentially weighted moving average of batch means, updated during training and used during inference
    pub running_mean: &'a ArrayD<f32>,
    /// Exponentially weighted moving average of batch variances, updated during training and used during inference
    pub running_var: &'a ArrayD<f32>,
}

/// Weights for a layer normalization layer
pub struct LayerNormalizationLayerWeight<'a> {
    /// Learned scale parameter that controls the variance of normalized values
    pub gamma: &'a ArrayD<f32>,
    /// Learned shift parameter that controls the mean of normalized values
    pub beta: &'a ArrayD<f32>,
}

/// Weights for an instance normalization layer
pub struct InstanceNormalizationLayerWeight<'a> {
    /// Learned scale parameter that controls the variance of normalized values
    pub gamma: &'a ArrayD<f32>,
    /// Learned shift parameter that controls the mean of normalized values
    pub beta: &'a ArrayD<f32>,
}

/// Weights for a group normalization layer
pub struct GroupNormalizationLayerWeight<'a> {
    /// Learned scale parameter that controls the variance of normalized values
    pub gamma: &'a ArrayD<f32>,
    /// Learned shift parameter that controls the mean of normalized values
    pub beta: &'a ArrayD<f32>,
}
