//! 1D convolutional layer for sequential data such as time series, audio, or text

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
#[doc(inline)]
pub use crate::neural_network::layers::convolution::convolution_engine::ConvPadding;
use crate::neural_network::layers::convolution::convolution_engine::{
    conv_backward, conv_forward, effective_kernel,
};
use crate::neural_network::layers::convolution::validation::{
    valid_output_size, validate_dilation, validate_filters, validate_input_shape_1d,
    validate_kernel_size_1d, validate_stride_dilation_exclusive, validate_strides_1d,
};
use crate::neural_network::layers::layer_weight::{Conv1DLayerWeight, LayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array1, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::borrow::Cow;

/// A 1D convolutional layer for neural networks
///
/// Applies a convolution operation to sequential data such as time series, audio signals,
/// or text. Input shape is \[batch_size, length, channels\] and output shape is
/// \[batch_size, output_length, filters\], where output_length depends on input length,
/// kernel size, stride, and padding
///
/// The dimension-generic convolution math lives in
/// [`convolution_engine`](crate::neural_network::layers::convolution). This layer holds the
/// weights, activation, and caches, and delegates the forward/backward numerics to it.
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
/// // Create target tensor (output length 8, 3 filters)
/// let y = Array3::ones((2, 8, 3)).into_dyn();
///
/// // Build model: add a Conv1D layer with 3 filters and kernel size 3
/// let mut model = Sequential::new();
/// model
///     .add(Conv1D::new(
///         3,                      // Number of filters
///         3,                      // Kernel size
///         vec![2, 10, 1],         // Input shape
///         1,                      // Stride
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
/// println!("Convolution layer prediction results: {:?}", prediction);
///
/// // Check if output shape is correct (should be [2, 8, 3])
/// assert_eq!(prediction.shape(), &[2, 8, 3]);
/// ```
#[derive(Debug)]
pub struct Conv1D {
    /// Number of convolution filters (output channels)
    filters: usize,
    /// Size of the convolution kernel
    kernel_size: usize,
    /// Stride value for the convolution operation
    stride: usize,
    /// Tap spacing of the kernel. 1 gives a solid kernel
    dilation_rate: usize,
    /// Type of padding to apply (`Valid`, `Same`, or `Causal`)
    padding: ConvPadding,
    /// 3D array of filter weights with shape \[kernel_size, channels, filters\]
    weights: Array3<f32>,
    /// 1D array of bias values with shape \[filters\]
    bias: Array1<f32>,
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
    bias_gradients: Option<Array1<f32>>,
}

impl Conv1D {
    /// Creates a new Conv1D layer with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (channels)
    /// - `kernel_size` - Size of the convolution kernel
    /// - `input_shape` - Shape of input tensor \[batch_size, length, channels\]
    /// - `stride` - Stride for the convolution operation
    /// - `activation` - Activation applied to the convolution output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv1D` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`ConvPadding::Valid`]. Choose [`ConvPadding::Same`] or
    /// [`ConvPadding::Causal`] with [`Conv1D::with_padding`]. The kernel is solid by default.
    /// Space its taps out with [`Conv1D::with_dilation_rate`]. By default, the layer seeds
    /// weights from the global seed or entropy. For reproducible initialization, set a seed with
    /// [`Conv1D::with_random_state`].
    ///
    /// The kernel is not bounded by the input length here. Only [`ConvPadding::Valid`] needs the
    /// effective kernel to fit. The padding mode is not final until the layer runs, so the
    /// forward pass applies that rule.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters`, `kernel_size`, or `stride` is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 3D or has 0 channels
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
        validate_input_shape_1d(&input_shape)?;
        let activation = activation.into();
        activation.validate()?;

        let input_channels = input_shape[2];
        let weights = Self::init_weights_array(filters, input_channels, kernel_size, None);
        let bias = Array1::zeros(filters);

        Ok(Self {
            filters,
            kernel_size,
            stride,
            dilation_rate: 1,
            padding: ConvPadding::Valid,
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

    /// Sets the padding mode (defaults to [`ConvPadding::Valid`])
    ///
    /// This takes a [`PaddingType`](super::PaddingType) as well, which converts to the matching
    /// [`ConvPadding`]. [`ConvPadding::Causal`] is the 1 mode no other convolution accepts. It
    /// puts every pad cell on the leading edge, so an output position reads no later input
    /// position
    ///
    /// # Parameters
    ///
    /// - `padding` - Padding mode (`Valid`, `Same`, or `Causal`)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: impl Into<ConvPadding>) -> Self {
        self.padding = padding.into();
        self
    }

    /// Sets the tap spacing of the kernel (defaults to 1)
    ///
    /// A dilation of `d` spaces the kernel taps `d` cells apart, so `kernel_size` taps span
    /// `(kernel_size - 1) * d + 1` input cells. The window still advances by the stride. A
    /// dilation of 1 gives a solid kernel and the same result as before
    ///
    /// # Parameters
    ///
    /// - `dilation_rate` - Tap spacing along the length axis
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
    ///
    /// # Notes
    ///
    /// The effective kernel is not bounded by the input length here. Only [`ConvPadding::Valid`]
    /// needs it to fit, and the forward pass applies that rule
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `dilation_rate` is 0
    /// - `Error::InvalidParameter` - If `dilation_rate` is above 1 and the stride is also above 1
    pub fn with_dilation_rate(mut self, dilation_rate: usize) -> Result<Self, Error> {
        validate_dilation(&[dilation_rate])?;
        validate_stride_dilation_exclusive(&[self.stride], &[dilation_rate])?;
        self.dilation_rate = dilation_rate;
        Ok(self)
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
        let input_channels = self.input_shape[2];
        self.weights = Self::init_weights_array(
            self.filters,
            input_channels,
            self.kernel_size,
            Some(random_state),
        );
        self
    }

    /// Xavier/Glorot uniform initialization of the \[kernel_size, channels, filters\] weight tensor
    fn init_weights_array(
        filters: usize,
        input_channels: usize,
        kernel_size: usize,
        random_state: Option<u64>,
    ) -> Array3<f32> {
        // Xavier initialization: weight_bound = sqrt(6 / (fan_in + fan_out))
        let fan_in = input_channels * kernel_size;
        let fan_out = filters * kernel_size;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array3::random_using(
            (kernel_size, input_channels, filters),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
            &mut rng,
        )
    }

    /// Calculates the output length after convolution
    ///
    /// The `Valid` rule reads the dilated extent of the kernel. `Causal` gives the same length
    /// as `Same`, because both rules round the input length up by the stride
    fn calculate_output_length(&self, input_length: usize) -> usize {
        match self.padding {
            ConvPadding::Valid => valid_output_size(
                input_length,
                effective_kernel(self.kernel_size, self.dilation_rate),
                self.stride,
            ),
            ConvPadding::Same | ConvPadding::Causal => input_length.div_ceil(self.stride),
        }
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - 3D array of filter weights with shape \[kernel_size, channels, filters\]
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
            &[self.dilation_rate],
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
            &[self.dilation_rate],
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
            .ok_or_else(|| Error::forward_pass_not_run("Conv1D"))?;

        let grads = conv_backward(
            &grad_upstream,
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.stride],
            &[self.dilation_rate],
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
        "Conv1D"
    }

    fn output_shape(&self) -> String {
        let input_length = self.input_shape[1];
        let output_length = self.calculate_output_length(input_length);
        format!(
            "({}, {}, {})",
            self.input_shape[0], output_length, self.filters
        )
    }

    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(self.weights.len() + self.bias.len())
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
        // Each tensor is pushed on its own, so a tensor without a gradient holds back no other
        if let Some(grad) = weight_gradients.as_ref() {
            params.push(ParamGrad::weight(
                "kernel",
                weights.as_slice_mut().expect("weights must be contiguous"),
                grad.as_slice()
                    .expect("weight_gradients must be contiguous"),
            ));
        }
        if let Some(grad) = bias_gradients.as_ref() {
            params.push(ParamGrad::no_decay(
                "bias",
                bias.as_slice_mut().expect("bias must be contiguous"),
                grad.as_slice().expect("bias_gradients must be contiguous"),
            ));
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv1D(Conv1DLayerWeight {
            weight: Cow::Borrowed(&self.weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}
