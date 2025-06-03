/// Container for different types of neural network layer weights
///
/// This enum serves as a polymorphic container for the weights of various
/// neural network layer types. Each variant corresponds to a specific layer
/// type and contains the appropriate weight structure for that layer.
///
/// # Variants
///
/// - `Dense` - Contains weights for dense (fully connected) layers
/// - `SimpleRNN` - Contains weights for simple recurrent neural network layers
/// - `LSTM` - Contains weights for long short-term memory layers
/// - `Conv2D` - Contains weights for 2D convolutional layers
/// - `Empty` - Represents a layer with no trainable parameters
pub enum LayerWeight<'a> {
    Dense(DenseLayerWeight<'a>),
    SimpleRNN(SimpleRNNLayerWeight<'a>),
    LSTM(LSTMLayerWeight<'a>),
    Conv2D(Conv2DLayerWeight<'a>),
    Conv1D(Conv1DLayerWeight<'a>),
    Empty,
}

/// Weights for a dense (fully connected) neural network layer
///
/// # Fields
///
/// - `weight` - Weight matrix with shape (input_features, output_features)
/// - `bias` - Bias vector with shape (1, output_features)
pub struct DenseLayerWeight<'a> {
    pub weight: &'a ndarray::Array2<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

/// Weights for a simple recurrent neural network layer
///
/// # Fields
///
/// - `kernel` - Weight matrix for input features
/// - `recurrent_kernel` - Weight matrix for recurrent connections
/// - `bias` - Bias vector
pub struct SimpleRNNLayerWeight<'a> {
    pub kernel: &'a ndarray::Array2<f32>,
    pub recurrent_kernel: &'a ndarray::Array2<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

/// Weights for a single gate in an LSTM layer
///
/// # Fields
///
/// - `kernel` - Weight matrix for input features
/// - `recurrent_kernel` - Weight matrix for recurrent connections
/// - `bias` - Bias vector for the gate
pub struct LSTMGateWeight<'a> {
    pub kernel: &'a ndarray::Array2<f32>,
    pub recurrent_kernel: &'a ndarray::Array2<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

/// Weights for a Long Short-Term Memory (LSTM) layer
///
/// Contains weights for the four gates that control information flow in an LSTM cell:
/// input gate, forget gate, cell gate, and output gate.
///
/// # Fields
///
/// - `input` - Weights for the input gate, which controls what new information to store
/// - `forget` - Weights for the forget gate, which controls what information to discard
/// - `cell` - Weights for the cell gate, which proposes new cell state values
/// - `output` - Weights for the output gate, which controls what to output
pub struct LSTMLayerWeight<'a> {
    pub input: LSTMGateWeight<'a>,
    pub forget: LSTMGateWeight<'a>,
    pub cell: LSTMGateWeight<'a>,
    pub output: LSTMGateWeight<'a>,
}

/// Weights for a 2D convolutional layer
///
/// # Fields
///
/// - `weight` - 4D convolution kernel with shape (output_channels, input_channels, kernel_height, kernel_width)
/// - `bias` - Bias vector with shape (1, output_channels)
pub struct Conv2DLayerWeight<'a> {
    pub weight: &'a ndarray::Array4<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

/// Weights for a 1D convolutional layer
///
/// # Fields
///
/// - `weight` - 3D convolution kernel with shape (output_channels, input_channels, kernel_size)
/// - `bias` - Bias vector with shape (1, output_channels)
pub struct Conv1DLayerWeight<'a> {
    pub weight: &'a ndarray::Array3<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}
