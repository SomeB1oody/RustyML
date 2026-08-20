//! 1D transposed convolutional layer, the decoder counterpart of `Conv1D`
//!
//! Holds the layer weights, activation, and caches, and delegates the forward/backward numerics
//! to the dimension-generic transposed-convolution engine

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::conv_transpose_engine::{
    conv_transpose_backward, conv_transpose_forward, transpose_output_length,
};
use crate::neural_network::layers::convolution::validation::{
    validate_filters, validate_kernel_size_1d, validate_strides_1d, validate_transpose_input_shape,
};
use crate::neural_network::layers::layer_weight::{Conv1DTransposeLayerWeight, LayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array1, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::borrow::Cow;

/// A 1D transposed convolutional layer for neural networks
///
/// Runs a convolution backwards over its length axis, so it grows a sequence instead of
/// shrinking one. Input shape is \[batch_size, length, channels\]. Output shape is
/// \[batch_size, output_length, filters\]. This is the layer an audio or time-series decoder uses
/// to reach the length the matching [`Conv1D`](super::conv_1d::Conv1D) consumed.
///
/// The dimension-generic transposed-convolution math lives in the transposed-convolution engine.
/// This layer holds the weights, activation, and caches, and delegates the forward/backward
/// numerics to it.
///
/// # Notes
///
/// The kernel has shape \[kernel_size, filters, channels\]. The filter axis comes **before** the
/// input-channel axis, which is the reverse of the `Conv1D` kernel. A transposed convolution
/// reads `channels` and writes `filters`, so this is the order its inner product needs.
///
/// The output size follows 1 rule per padding mode:
///
/// - [`PaddingType::Valid`]: `input * stride + max(kernel - stride, 0)`
/// - [`PaddingType::Same`]: `input * stride`
///
/// Unlike `Conv1D`, this layer puts no lower bound on the input length. A length-1 input under a
/// length-3 kernel is a normal first decoder step.
///
/// A stride wider than the kernel leaves output positions that no input position reaches. These
/// positions hold exactly the bias.
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
/// // Create a simple 3D input tensor: [batch_size, length, channels]
/// // Batch size=2, 10 time steps, 1 input channel
/// let x = Array3::ones((2, 10, 1)).into_dyn();
///
/// // A length-3 kernel at stride 2 with Valid padding grows 10 steps into 21
/// let y = Array3::ones((2, 21, 3)).into_dyn();
///
/// // Build model: add a Conv1DTranspose layer with 3 filters and kernel size 3
/// let mut model = Sequential::new();
/// model
///     .add(Conv1DTranspose::new(
///         3,                      // Number of filters
///         3,                      // Kernel size
///         vec![2, 10, 1],         // Input shape
///         2,                      // Stride
///         Activation::ReLU,       // ReLU activation
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x).unwrap();
///
/// // Check if output shape is correct (should be [2, 21, 3])
/// assert_eq!(prediction.shape(), &[2, 21, 3]);
/// ```
///
/// # Performance
///
/// The pass runs 1 matrix product per batch item, then scatters its rows into the output. The
/// scatter accumulates wherever the windows overlap, so it cannot split 1 sequence across
/// threads. The pass parallelizes over the batch, and runs the per-item products in parallel
/// while the batch alone cannot fill the thread pool. The gate is the convolution engine's
/// estimated-FLOPs gate, reachable through [`crate::tuning::conv`].
#[derive(Debug)]
pub struct Conv1DTranspose {
    /// Number of transposed convolution filters (output channels)
    filters: usize,
    /// Size of the convolution kernel
    kernel_size: usize,
    /// Stride value for the transposed convolution
    stride: usize,
    /// Type of padding to apply (`Valid` or `Same`)
    padding: PaddingType,
    /// 3D array of filter weights with shape \[kernel_size, filters, channels\]
    weights: Array3<f32>,
    /// 1D array of bias values with shape \[filters\]
    bias: Array1<f32>,
    /// Activation applied to the transposed convolution output
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
    bias_gradients: Option<Array1<f32>>,
}

impl Conv1DTranspose {
    /// Creates a new 1D transposed convolutional layer with the specified parameters
    ///
    /// The constructor initializes weights with Xavier (Glorot) uniform initialization and sets
    /// biases to 0
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of transposed convolution filters (output channels)
    /// - `kernel_size` - Size of the convolution kernel
    /// - `input_shape` - Shape of the input tensor as \[batch_size, length, channels\]
    /// - `stride` - Stride for the transposed convolution
    /// - `activation` - Activation applied to the transposed convolution output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv1DTranspose` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`Conv1DTranspose::with_padding`]. By default, the layer seeds weights from the global
    /// seed or entropy. For reproducible initialization, set a seed with
    /// [`Conv1DTranspose::with_random_state`].
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters`, `kernel_size`, or `stride` is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 3D or holds a 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        filters: usize,
        kernel_size: usize,
        input_shape: Vec<usize>,
        stride: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_1d(kernel_size)?;
        validate_strides_1d(stride)?;
        validate_transpose_input_shape(&input_shape, 1, "[batch_size, length, channels]")?;
        let activation = activation.into();
        activation.validate()?;

        let channels = input_shape[2];
        let weights = Self::init_weights_array(filters, channels, kernel_size, None);
        let bias = Array1::zeros(filters);

        Ok(Conv1DTranspose {
            filters,
            kernel_size,
            stride,
            padding: PaddingType::Valid,
            weights,
            bias,
            activation,
            output_cache: None,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
        })
    }

    /// Sets the padding mode (defaults to [`PaddingType::Valid`])
    ///
    /// # Parameters
    ///
    /// - `padding` - Type of padding to apply (`Valid` or `Same`)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the seed used to initialize the filter weights and re-initializes them
    /// deterministically
    ///
    /// By default, the layer seeds weights from the global seed or entropy (see
    /// [`crate::random`]). This method re-runs Xavier/Glorot uniform initialization with
    /// `random_state`. Call it before assigning custom weights or training. The bias stays
    /// zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        let channels = self.input_shape[2];
        self.weights =
            Self::init_weights_array(self.filters, channels, self.kernel_size, Some(random_state));
        self
    }

    /// Xavier/Glorot uniform initialization of the \[kernel_size, filters, channels\] weight
    /// tensor
    fn init_weights_array(
        filters: usize,
        channels: usize,
        kernel_size: usize,
        random_state: Option<u64>,
    ) -> Array3<f32> {
        // Xavier init: bound = sqrt(6 / (fan_in + fan_out)). The transposed kernel swaps the 2
        // channel axes against the plain one, which swaps the 2 fans. Their sum, and so the
        // bound, is unchanged
        let fan_in = channels * kernel_size;
        let fan_out = filters * kernel_size;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array3::random_using(
            (kernel_size, filters, channels),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
            &mut rng,
        )
    }

    /// Calculates the output shape of the transposed convolution from the input dimensions
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        vec![
            input_shape[0],
            transpose_output_length(input_shape[1], self.kernel_size, self.stride, self.padding),
            self.filters,
        ]
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - 3D array of filter weights with shape \[kernel_size, filters, channels\]
    /// - `bias` - 1D array of bias values with shape \[filters\]
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match the
    ///   layer's expected shape
    pub fn set_weights(&mut self, weights: Array3<f32>, bias: Array1<f32>) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }
}

impl Layer for Conv1DTranspose {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }

        self.input_cache = Some(input.clone());

        let output = conv_transpose_forward(
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

        let output = conv_transpose_forward(
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
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Conv1DTranspose"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("Conv1DTranspose"))?;

        let grads = conv_transpose_backward(
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
            Array1::from_shape_vec(self.bias.raw_dim(), grads.bias_grad)
                .expect("bias gradient shape matches bias"),
        );

        Ok(grads.input_grad)
    }

    fn layer_type(&self) -> &str {
        "Conv1DTranspose"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2]
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
            params.push(ParamGrad::weight(
                weights.as_slice_mut().expect("weights must be contiguous"),
                grad_a
                    .as_slice()
                    .expect("weight_gradients must be contiguous"),
            ));
            params.push(ParamGrad::no_decay(
                bias.as_slice_mut().expect("bias must be contiguous"),
                grad_b
                    .as_slice()
                    .expect("bias_gradients must be contiguous"),
            ));
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv1DTranspose(Conv1DTransposeLayerWeight {
            weight: Cow::Borrowed(&self.weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}
