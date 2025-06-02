/// A macro that generates the SGD (Stochastic Gradient Descent) parameter update method for convolutional layers.
///
/// This macro creates a standardized `update_parameters_sgd` function that applies SGD updates
/// to both weights and biases of convolutional layers. The generated function checks for the
/// presence of gradients and delegates the actual parameter updates to the SGD optimizer.
macro_rules! update_sgd_conv {
    () => {
        fn update_parameters_sgd(&mut self, lr: f32) {
            if let (Some(weight_grads), Some(bias_grads)) =
                (&self.weight_gradients, &self.bias_gradients)
            {
                SGD::update_sgd_parameters(
                    self.weights.as_slice_mut().unwrap(),
                    weight_grads.as_slice().unwrap(),
                    self.bias.as_slice_mut().unwrap(),
                    bias_grads.as_slice().unwrap(),
                    lr,
                )
            }
        }
    };
}

/// A macro that generates standard function implementations for neural network layers
/// without trainable parameters.
///
/// This macro expands to implementations of the following functions:
/// - `param_count`: Returns 0 as the layer has no trainable parameters
/// - `update_parameters_sgd`: Empty implementation for SGD parameter updates
/// - `update_parameters_adam`: Empty implementation for Adam parameter updates
/// - `update_parameters_rmsprop`: Empty implementation for RMSProp parameter updates
/// - `get_weights`: Returns `LayerWeight::Empty` as the layer has no weights
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> usize {
            // This layer has no trainable parameters
            0
        }

        fn update_parameters_sgd(&mut self, _lr: f32) {
            // This layer have no trainable parameters
        }

        fn update_parameters_adam(
            &mut self,
            _lr: f32,
            _beta1: f32,
            _beta2: f32,
            _epsilon: f32,
            _t: u64,
        ) {
            // This layer have no trainable parameters
        }

        fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
            // This layer have no trainable parameters
        }

        fn get_weights(&self) -> LayerWeight {
            // This layer has no weights
            LayerWeight::Empty
        }
    };
}

/// 1D Average Pooling layer for neural networks.
pub mod average_pooling_1d;
/// A 2D average pooling layer for neural networks.
pub mod average_pooling_2d;
/// 3D Average Pooling Layer
pub mod average_pooling_3d;
/// A 1D convolutional layer for neural networks.
pub mod conv1d;
/// A 2D convolutional layer for neural networks.
pub mod conv2d;
/// Dense (Fully Connected) layer implementation for neural networks.
pub mod dense;
/// A layer that flattens a 4D tensor into a 2D tensor.
pub mod flatten;
/// Global Average Pooling 1D Layer
pub mod global_average_pooling_1d;
/// Global Average Pooling 2D Layer
pub mod global_average_pooling_2d;
/// Global Max Pooling 1D Layer
mod global_max_pooling_1d;
/// Global Max Pooling Layer
pub mod global_max_pooling_2d;
/// LSTM (Long Short-Term Memory) neural network layer implementation.
pub mod lstm;
/// 1D Max Pooling layer for neural networks.
pub mod max_pooling_1d;
/// Defines a structure for max pooling operation, used to perform max pooling on 2D data.
pub mod max_pooling_2d;
/// 3D data max pooling Layer.
mod max_pooling_3d;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation.
pub mod simple_rnn;

pub use average_pooling_1d::*;
pub use average_pooling_2d::*;
pub use average_pooling_3d::*;
pub use conv1d::*;
pub use conv2d::*;
pub use dense::*;
pub use flatten::*;
pub use global_average_pooling_1d::*;
pub use global_average_pooling_2d::*;
pub use global_max_pooling_1d::*;
pub use global_max_pooling_2d::*;
pub use lstm::*;
pub use max_pooling_1d::*;
pub use max_pooling_2d::*;
pub use max_pooling_3d::*;
pub use simple_rnn::*;

/// Defines the padding method used in convolutional layers.
///
/// The padding type determines how the input is padded before applying convolution:
/// - `Valid`: No padding is applied, which reduces the output dimensions.
/// - `Same`: Padding is added to preserve the input spatial dimensions in the output.
pub enum PaddingType {
    /// No padding is applied. The convolution is only computed where the filter
    /// fully overlaps with the input, resulting in an output with reduced dimensions.
    Valid,

    /// Padding is added around the input to ensure that the output has the same
    /// spatial dimensions as the input (when stride is 1). This is done by adding
    /// zeros around the borders of the input.
    Same,
}

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

/// Calculate output shape for 1d pooling layer.
///
/// # Parameters
///
/// - `batch_size` - Number of samples in the batch
/// - `channels` - Number of channels (features) in each sample
/// - `length` - Length of the input along the dimension where pooling/convolution is applied
/// - `pool_size` - Size of the pooling/convolutional window
/// - `stride` - Step size for sliding the window across the input
///
/// # Returns
///
/// * `Vec<usize>` - A vector containing the dimensions of the output tensor in the format: `[batch_size, channels, output_length]`
fn calculate_output_shape_1d_pooling(
    batch_size: usize,
    channels: usize,
    length: usize,
    pool_size: usize,
    stride: usize,
) -> Vec<usize> {
    let output_length = (length - pool_size) / stride + 1;

    vec![batch_size, channels, output_length]
}

/// Calculates the output shape of the 2d pooling layer.
///
/// # Parameters
///
/// * `input_shape` - Shape of the input tensor, in format \[batch_size, channels, height, width\].
///
/// # Returns
///
/// * `Vec<usize>` - A vector containing the calculated output shape, in format \[batch_size, channels, output_height, output_width\].
fn calculate_output_shape_2d_pooling(
    input_shape: &[usize],
    pool_size: (usize, usize),
    strides: (usize, usize),
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    // Calculate the height and width of the output
    let output_height = (input_height - pool_size.0) / strides.0 + 1;
    let output_width = (input_width - pool_size.1) / strides.1 + 1;

    vec![batch_size, channels, output_height, output_width]
}

/// Calculates the output shape of the 3D layer.
///
/// # Parameters
///
/// * `input_shape` - The shape of the input tensor, formatted as \[batch_size, channels, depth, height, width\].
///
/// # Returns
///
/// A vector containing the calculated output shape, formatted as \[batch_size, channels, output_depth, output_height, output_width\].
fn calculate_output_shape_3d_pooling(
    input_shape: &[usize],
    pool_size: (usize, usize, usize),
    strides: (usize, usize, usize),
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    // Calculate the output depth, height, and width
    let output_depth = (input_depth - pool_size.0) / strides.0 + 1;
    let output_height = (input_height - pool_size.1) / strides.1 + 1;
    let output_width = (input_width - pool_size.2) / strides.2 + 1;

    vec![
        batch_size,
        channels,
        output_depth,
        output_height,
        output_width,
    ]
}

/// Updates parameters using the Adam optimization algorithm.
///
/// This function performs parameter updates using the Adam (Adaptive Moment Estimation) optimizer,
/// which adapts the learning rate for each parameter using estimates of first and second moments
/// of the gradients. The implementation uses parallel iteration via Rayon for improved performance.
///
/// # Parameters
///
/// * `params` - Mutable slice of model parameters to be updated
/// * `grads` - Slice of gradients corresponding to each parameter
/// * `m` - Mutable slice for first moment estimates (momentum)
/// * `v` - Mutable slice for second moment estimates (velocity/variance)
/// * `lr` - Learning rate for the update step
/// * `beta1` - Exponential decay rate for the first moment estimates (typically 0.9)
/// * `beta2` - Exponential decay rate for the second moment estimates (typically 0.999)
/// * `epsilon` - Small constant for numerical stability
/// * `bias_correction1` - Bias correction term for first moment estimate
/// * `bias_correction2` - Bias correction term for second moment estimate
fn update_adam_conv(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) {
    use rayon::prelude::*;

    params
        .par_iter_mut()
        .zip(grads.par_iter())
        .zip(m.par_iter_mut())
        .zip(v.par_iter_mut())
        .for_each(|(((param, &grad), m_val), v_val)| {
            // Update momentum and variance
            *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
            *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;

            // Calculate corrected momentum and variance
            let m_corrected = *m_val / bias_correction1;
            let v_corrected = *v_val / bias_correction2;

            // Update parameter
            *param -= lr * m_corrected / (v_corrected.sqrt() + epsilon);
        });
}
