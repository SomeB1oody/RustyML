//! 1D convolutional layer for sequential data such as time series, audio, or text

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{conv_backward, conv_forward};
use crate::neural_network::layers::convolution::validation::{
    validate_filters, validate_input_shape_1d, validate_kernel_size_1d, validate_strides_1d,
};
use crate::neural_network::layers::layer_weight::{Conv1DLayerWeight, LayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// A 1D convolutional layer for neural networks
///
/// Applies a convolution operation to sequential data such as time series, audio signals,
/// or text. Input shape is \[batch_size, channels, length\] and output shape is
/// \[batch_size, filters, output_length\], where output_length depends on input length,
/// kernel size, stride, and padding
///
/// The dimension-generic convolution math lives in
/// [`convolution_engine`](crate::neural_network::layers::convolution); this layer holds the
/// weights, activation, and caches, and delegates the forward/backward numerics to it
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array3;
///
/// // Create a simple 3D input tensor: [batch_size, channels, length]
/// // Batch size=2, 1 input channel, 10 time steps
/// let x = Array3::ones((2, 1, 10)).into_dyn();
///
/// // Create target tensor - assuming we'll have 3 filters with output length 8
/// let y = Array3::ones((2, 3, 8)).into_dyn();
///
/// // Build model: add a Conv1D layer with 3 filters and kernel size 3
/// let mut model = Sequential::new();
/// model
///     .add(Conv1D::new(
///         3,                      // Number of filters
///         3,                      // Kernel size
///         vec![2, 1, 10],         // Input shape
///         1,                      // Stride
///         PaddingType::Valid,     // No padding
///         Activation::ReLU, // ReLU activation
///         None,                   // random_state
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, None, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x).unwrap();
/// println!("Convolution layer prediction results: {:?}", prediction);
///
/// // Check if output shape is correct - should be [2, 3, 8]
/// assert_eq!(prediction.shape(), &[2, 3, 8]);
/// ```
#[derive(Debug)]
pub struct Conv1D {
    /// Number of convolution filters (output channels)
    filters: usize,
    /// Size of the convolution kernel
    kernel_size: usize,
    /// Stride value for the convolution operation
    stride: usize,
    /// Type of padding to apply (`Valid` or `Same`)
    padding: PaddingType,
    /// 3D array of filter weights with shape \[filters, channels, kernel_size\]
    weights: Array3<f32>,
    /// 2D array of bias values with shape \[1, filters\]
    bias: Array2<f32>,
    /// Activation applied to the convolution output
    activation: Activation,
    /// Cached activated output, used by the activation backward pass
    output_cache: Option<Tensor>,
    /// Cached input from the forward pass, used during backpropagation
    input_cache: Option<Tensor>,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Gradients for the weights, computed during backpropagation
    weight_gradients: Option<Array3<f32>>,
    /// Gradients for the biases, computed during backpropagation
    bias_gradients: Option<Array2<f32>>,
}

impl Conv1D {
    /// Creates a new Conv1D layer with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (channels)
    /// - `kernel_size` - Size of the convolution kernel
    /// - `input_shape` - Shape of input tensor \[batch_size, channels, length\]
    /// - `stride` - Stride for the convolution operation
    /// - `padding` - Padding type (`Valid` or `Same`)
    /// - `activation` - Activation function (ReLU, Sigmoid, Tanh, Softmax)
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global
    ///   seed or entropy. See `crate::random`
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv1D` layer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters`, `kernel_size`, or `stride` is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 3D, has 0 channels, or input length is
    ///   less than kernel size
    pub fn new(
        filters: usize,
        kernel_size: usize,
        input_shape: Vec<usize>,
        stride: usize,
        padding: PaddingType,
        activation: impl Into<Activation>,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_1d(kernel_size)?;
        validate_strides_1d(stride)?;
        validate_input_shape_1d(&input_shape, kernel_size)?;

        let input_channels = input_shape[1];

        // Xavier initialization: weight_bound = sqrt(6 / (fan_in + fan_out))
        let fan_in = input_channels * kernel_size;
        let fan_out = filters * kernel_size;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let mut rng = crate::random::make_rng(random_state);
        let weights = Array3::random_using(
            (filters, input_channels, kernel_size),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
            &mut rng,
        );

        let bias = Array2::zeros((1, filters));

        Ok(Self {
            filters,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
            activation: activation.into(),
            output_cache: None,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
        })
    }

    /// Calculates the output length after convolution
    ///
    /// # Parameters
    ///
    /// - `input_length` - Length of the input sequence
    ///
    /// # Returns
    ///
    /// - `usize` - Output length after convolution
    fn calculate_output_length(&self, input_length: usize) -> usize {
        match self.padding {
            PaddingType::Valid => (input_length - self.kernel_size) / self.stride + 1,
            PaddingType::Same => input_length.div_ceil(self.stride),
        }
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - 3D array of filter weights with shape \[filters, channels, kernel_size\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match the
    ///   layer's expected shape
    pub fn set_weights(&mut self, weights: Array3<f32>, bias: Array2<f32>) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }
}

impl Layer for Conv1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        // Cache input for backpropagation
        self.input_cache = Some(input.clone());

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[self.stride],
            self.padding,
        )?;
        let activated = self.activation.forward(&output)?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[self.stride],
            self.padding,
        )?;
        let activated = self.activation.forward(&output)?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Apply activation backward pass
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Conv1D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| Error::computation("No cached input for backward pass"))?;

        let grads = conv_backward(
            &grad_upstream,
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.stride],
            self.padding,
        )?;

        self.weight_gradients = Some(
            Array3::from_shape_vec(self.weights.raw_dim(), grads.weight_grad)
                .expect("weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(
            Array2::from_shape_vec(self.bias.raw_dim(), grads.bias_grad)
                .expect("bias gradient shape matches bias"),
        );

        Ok(grads.input_grad)
    }

    fn layer_type(&self) -> &str {
        "Conv1D"
    }

    fn output_shape(&self) -> String {
        let input_length = self.input_shape[2];
        let output_length = self.calculate_output_length(input_length);
        format!(
            "({}, {}, {})",
            self.input_shape[0], self.filters, output_length
        )
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.weights.len() + self.bias.len())
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            weights,
            bias,
            weight_gradients,
            bias_gradients,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(grad_a), Some(grad_b)) = (weight_gradients.as_ref(), bias_gradients.as_ref()) {
            params.push(ParamGrad {
                value: weights.as_slice_mut().expect("weights must be contiguous"),
                grad: grad_a
                    .as_slice()
                    .expect("weight_gradients must be contiguous"),
            });
            params.push(ParamGrad {
                value: bias.as_slice_mut().expect("bias must be contiguous"),
                grad: grad_b
                    .as_slice()
                    .expect("bias_gradients must be contiguous"),
            });
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv1D(Conv1DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
