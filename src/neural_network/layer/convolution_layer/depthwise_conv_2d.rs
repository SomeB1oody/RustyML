use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::activation_layer::Activation;
use crate::neural_network::layer::convolution_layer::PaddingType;
use crate::neural_network::layer::convolution_layer::validation::{
    validate_filters, validate_input_shape_2d, validate_kernel_size_2d, validate_strides_2d,
};
use crate::neural_network::layer::conv_op_helpers::pad_tensor_2d;
use crate::neural_network::layer::shape_helpers::calculate_output_shape_2d;
use crate::neural_network::layer::validation::validate_weight_shape;
use crate::neural_network::layer::layer_weight::{DepthwiseConv2DLayerWeight, LayerWeight};
use crate::neural_network::neural_network_trait::{Layer, ParamGrad};
use ndarray::{Array1, Array2, Array4, ArrayView2, ArrayViewD, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Threshold for using parallel computation in forward pass.
/// If batch_size * channels * output_height * output_width < threshold, use sequential computation.
const DEPTHWISE_CONV_2D_PARALLEL_THRESHOLD: usize = 1500;

/// A 2D depthwise convolutional layer for neural networks.
///
/// Applies one convolutional filter per input channel, reducing parameter count and
/// computation while preserving spatial feature extraction. Input shape is
/// \[batch_size, channels, height, width\], and each channel is convolved independently.
///
/// # Fields
///
/// - `filters` - Number of output channels (must equal input channels for pure depthwise convolution)
/// - `kernel_size` - Size of the convolution kernel as (height, width)
/// - `strides` - Stride of the convolution as (height_stride, width_stride)
/// - `padding` - Padding strategy (Valid or Same)
/// - `weights` - 4D weight tensor with shape \[filters, 1, kernel_height, kernel_width\]
/// - `bias` - 1D bias vector with shape \[filters\]
/// - `activation` - Activation layer from activation_layer module
/// - `input` - Cached input tensor for backward pass
/// - `input_shape` - Shape of the input tensor
/// - `weight_gradients` - Gradients with respect to weights
/// - `bias_gradients` - Gradients with respect to bias
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::Array4;
///
/// // Create Sequential model
/// let mut model = Sequential::new();
///
/// // Create DepthwiseConv2D layer with ReLU activation (weights are initialized in `new`)
/// let depthwise_layer = DepthwiseConv2D::new(
///     3,                        // filters (must equal input channels)
///     (2, 2),                  // kernel_size
///     vec![1, 3, 4, 4],        // input shape [batch_size, channels, height, width]
///     (1, 1),                  // strides
///     PaddingType::Valid,      // padding
///     Activation::ReLU // activation
/// ).unwrap();
///
/// // Add layer and compile model
/// model
///     .add(depthwise_layer)
///     .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Create test input data: [batch_size, channels, height, width]
/// let batch_size = 1;
/// let input_channels = 3;
/// let height = 4;
/// let width = 4;
///
/// let mut input_data = Array4::zeros((batch_size, input_channels, height, width));
///
/// // Set different values for each channel
/// for c in 0..input_channels {
///     for h in 0..height {
///         for w in 0..width {
///             input_data[[0, c, h, w]] = (c * 10 + h * width + w) as f32;
///         }
///     }
/// }
///
/// let input = input_data.into_dyn();
///
/// // Display model structure
/// model.summary();
///
/// // Forward propagation
/// let output = model.predict(&input).unwrap();
///
/// // Verify output shape
/// // Input: [1, 3, 4, 4], kernel (2,2), stride (1,1), valid padding
/// // Output should be: [1, 3, 3, 3]
/// assert_eq!(output.shape(), &[1, 3, 3, 3]);
///
/// // Since ReLU activation is used, all output values should be non-negative
/// for value in output.iter() {
///     assert!(*value >= 0.0);
/// }
/// ```
pub struct DepthwiseConv2D {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    weights: Array4<f32>,
    bias: Array1<f32>,
    activation: Activation,
    output_cache: Option<Tensor>,
    input: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array1<f32>>,
}

impl DepthwiseConv2D {
    /// Creates a new DepthwiseConv2D layer.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (must equal input channels for pure depthwise)
    /// - `kernel_size` - Size of the convolution kernel as (height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\]
    /// - `strides` - Stride of the convolution as (height_stride, width_stride)
    /// - `padding` - Padding strategy (Valid or Same)
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `DepthwiseConv2D` instance with Xavier-initialized weights or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 4D or smaller than the kernel
    /// - `Error::InvalidParameter` - If `filters` does not equal the input channels
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        // Input validation
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        // For pure depthwise convolution every input channel is convolved by exactly one filter,
        // so the filter count is fixed by the input channels. Enforcing it here turns a silent
        // shape mismatch at the first `forward` into an explicit construction-time error.
        let channels = input_shape[1];
        if channels != filters {
            return Err(Error::invalid_parameter(
                "filters",
                "must equal the number of input channels",
            ));
        }

        let (kernel_height, kernel_width) = kernel_size;

        // Xavier (Glorot) uniform initialization. Each depthwise filter maps one input channel to
        // one output channel, so fan_in == fan_out == kernel_area. Weight shape is
        // [filters, 1, kernel_height, kernel_width]; biases start at zero.
        let fan = kernel_height * kernel_width;
        let weight_bound = (6.0 / (fan + fan) as f32).sqrt();
        let weights = Array4::random(
            (filters, 1, kernel_height, kernel_width),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
        );
        let bias = Array1::zeros(filters);

        Ok(Self {
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            activation: activation.into(),
            output_cache: None,
            input: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
        })
    }

    /// Calculates padding dimensions for Same padding mode.
    fn calculate_padding(
        &self,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
    ) -> (usize, usize) {
        match self.padding {
            PaddingType::Valid => (0, 0),
            PaddingType::Same => {
                let pad_h = ((output_height - 1) * self.strides.0 + self.kernel_size.0)
                    .saturating_sub(input_height);
                let pad_w = ((output_width - 1) * self.strides.1 + self.kernel_size.1)
                    .saturating_sub(input_width);
                (pad_h, pad_w)
            }
        }
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `weights` - 4D weight tensor with shape \[filters, 1, kernel_height, kernel_width\]
    /// - `bias` - 1D bias vector with shape \[filters\]
    pub fn set_weights(
        &mut self,
        weights: Array4<f32>,
        bias: Array1<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }

    /// Performs depthwise convolution for a single channel.
    #[allow(clippy::too_many_arguments)] // geometry params (shapes/strides/kernel/padding) are all needed
    fn convolve_channel(
        input_channel: &ArrayView2<f32>,
        kernel: &ArrayView2<f32>,
        bias: f32,
        output_shape: (usize, usize),
        strides: (usize, usize),
        kernel_size: (usize, usize),
        padding: &PaddingType,
        pad_h: usize,
        pad_w: usize,
    ) -> Array2<f32> {
        // Apply padding if needed
        let padded_input = if *padding == PaddingType::Same {
            pad_tensor_2d(&input_channel.to_owned(), pad_h, pad_w)
        } else {
            input_channel.to_owned()
        };

        let (output_height, output_width) = output_shape;
        let mut channel_output = Array2::zeros(output_shape);

        for oh in 0..output_height {
            for ow in 0..output_width {
                let start_h = oh * strides.0;
                let start_w = ow * strides.1;
                let end_h = start_h + kernel_size.0;
                let end_w = start_w + kernel_size.1;

                if end_h <= padded_input.shape()[0] && end_w <= padded_input.shape()[1] {
                    let input_patch = padded_input.slice(s![start_h..end_h, start_w..end_w]);
                    let conv_result = (&input_patch * kernel).sum();
                    channel_output[[oh, ow]] = conv_result + bias;
                }
            }
        }

        channel_output
    }

    /// Computes weight and input gradients for a single batch.
    #[allow(clippy::too_many_arguments)] // geometry params (shapes/strides/padding) are all needed
    fn compute_batch_gradients(
        &self,
        input_array: &ArrayViewD<f32>,
        grad_upstream: &Tensor,
        batch_idx: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> (Array4<f32>, Array4<f32>) {
        let mut batch_weight_grads = Array4::zeros(self.weights.raw_dim());
        // Accumulate input gradients in PADDED coordinates (matching the symmetric padding used in
        // the forward pass via `pad_tensor_2d`), then strip the padding before returning. The
        // previous code reused the padded coordinate `oh*stride+kh` directly as an unpadded index,
        // which dropped/offset contributions under `Same` padding.
        let padded_height = input_height + pad_h;
        let padded_width = input_width + pad_w;
        let mut batch_input_grads_padded =
            Array4::zeros((1, channels, padded_height, padded_width));

        for c in 0..channels {
            let input_channel = input_array.slice(s![batch_idx, c, .., ..]);
            let grad_channel = grad_upstream.slice(s![batch_idx, c, .., ..]);

            // Apply padding to input if needed
            let padded_input = if self.padding == PaddingType::Same {
                pad_tensor_2d(&input_channel.to_owned(), pad_h, pad_w)
            } else {
                input_channel.to_owned()
            };

            // Calculate weight gradients
            for kh in 0..self.kernel_size.0 {
                for kw in 0..self.kernel_size.1 {
                    let mut weight_grad = 0.0;
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let ih = oh * self.strides.0 + kh;
                            let iw = ow * self.strides.1 + kw;

                            if ih < padded_input.shape()[0] && iw < padded_input.shape()[1] {
                                weight_grad += padded_input[[ih, iw]] * grad_channel[[oh, ow]];
                            }
                        }
                    }
                    batch_weight_grads[[c, 0, kh, kw]] = weight_grad;
                }
            }

            // Calculate input gradients
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let grad_val = grad_channel[[oh, ow]];

                    for kh in 0..self.kernel_size.0 {
                        for kw in 0..self.kernel_size.1 {
                            let ih = oh * self.strides.0 + kh;
                            let iw = ow * self.strides.1 + kw;

                            if ih < padded_height && iw < padded_width {
                                batch_input_grads_padded[[0, c, ih, iw]] +=
                                    self.weights[[c, 0, kh, kw]] * grad_val;
                            }
                        }
                    }
                }
            }
        }

        // Strip the symmetric padding so the input gradient matches the original input shape.
        let pad_top = pad_h / 2;
        let pad_left = pad_w / 2;
        let batch_input_grads = batch_input_grads_padded
            .slice(s![
                ..,
                ..,
                pad_top..pad_top + input_height,
                pad_left..pad_left + input_width
            ])
            .to_owned();

        (batch_weight_grads, batch_input_grads)
    }
}

impl Layer for DepthwiseConv2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        self.input = Some(input.clone());
        self.input_shape = input.shape().to_vec();

        let input_array = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();

        let (batch_size, channels, height, width) = (
            input_array.shape()[0],
            input_array.shape()[1],
            input_array.shape()[2],
            input_array.shape()[3],
        );

        assert_eq!(
            channels, self.filters,
            "Input channels must equal number of filters for depthwise convolution"
        );

        // Calculate output dimensions
        let output_shape = calculate_output_shape_2d(
            &self.input_shape,
            self.kernel_size,
            self.strides,
            &self.padding,
        );
        let (_, _, output_height, output_width) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );

        // Calculate padding dimensions once if needed
        let (pad_h, pad_w) = self.calculate_padding(height, width, output_height, output_width);

        let mut output = Array4::zeros((batch_size, channels, output_height, output_width));

        // Determine whether to use parallel or sequential execution
        let total_elements = batch_size * channels * output_height * output_width;

        if total_elements >= DEPTHWISE_CONV_2D_PARALLEL_THRESHOLD {
            // Parallel execution for large workloads
            output
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(b, mut batch_output)| {
                    batch_output
                        .axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(c, mut channel_output)| {
                            let input_channel = input_array.slice(s![b, c, .., ..]);
                            let kernel = self.weights.slice(s![c, 0, .., ..]);
                            let result = Self::convolve_channel(
                                &input_channel,
                                &kernel,
                                self.bias[c],
                                (output_height, output_width),
                                self.strides,
                                self.kernel_size,
                                &self.padding,
                                pad_h,
                                pad_w,
                            );
                            channel_output.assign(&result);
                        });
                });
        } else {
            // Sequential execution for small workloads
            for b in 0..batch_size {
                for c in 0..channels {
                    let input_channel = input_array.slice(s![b, c, .., ..]);
                    let kernel = self.weights.slice(s![c, 0, .., ..]);
                    let result = Self::convolve_channel(
                        &input_channel,
                        &kernel,
                        self.bias[c],
                        (output_height, output_width),
                        self.strides,
                        self.kernel_size,
                        &self.padding,
                        pad_h,
                        pad_w,
                    );
                    output.slice_mut(s![b, c, .., ..]).assign(&result);
                }
            }
        }

        let output = output.into_dyn();

        // Apply activation
        let activated = self.activation.forward(&output)?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        let input_shape = input.shape().to_vec();

        let input_array = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();

        let (batch_size, channels, height, width) = (
            input_array.shape()[0],
            input_array.shape()[1],
            input_array.shape()[2],
            input_array.shape()[3],
        );

        assert_eq!(
            channels, self.filters,
            "Input channels must equal number of filters for depthwise convolution"
        );

        // Calculate output dimensions
        let output_shape = calculate_output_shape_2d(
            &input_shape,
            self.kernel_size,
            self.strides,
            &self.padding,
        );
        let (_, _, output_height, output_width) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );

        // Calculate padding dimensions once if needed
        let (pad_h, pad_w) = self.calculate_padding(height, width, output_height, output_width);

        let mut output = Array4::zeros((batch_size, channels, output_height, output_width));

        // Determine whether to use parallel or sequential execution
        let total_elements = batch_size * channels * output_height * output_width;

        if total_elements >= DEPTHWISE_CONV_2D_PARALLEL_THRESHOLD {
            // Parallel execution for large workloads
            output
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(b, mut batch_output)| {
                    batch_output
                        .axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(c, mut channel_output)| {
                            let input_channel = input_array.slice(s![b, c, .., ..]);
                            let kernel = self.weights.slice(s![c, 0, .., ..]);
                            let result = Self::convolve_channel(
                                &input_channel,
                                &kernel,
                                self.bias[c],
                                (output_height, output_width),
                                self.strides,
                                self.kernel_size,
                                &self.padding,
                                pad_h,
                                pad_w,
                            );
                            channel_output.assign(&result);
                        });
                });
        } else {
            // Sequential execution for small workloads
            for b in 0..batch_size {
                for c in 0..channels {
                    let input_channel = input_array.slice(s![b, c, .., ..]);
                    let kernel = self.weights.slice(s![c, 0, .., ..]);
                    let result = Self::convolve_channel(
                        &input_channel,
                        &kernel,
                        self.bias[c],
                        (output_height, output_width),
                        self.strides,
                        self.kernel_size,
                        &self.padding,
                        pad_h,
                        pad_w,
                    );
                    output.slice_mut(s![b, c, .., ..]).assign(&result);
                }
            }
        }

        let output = output.into_dyn();

        // Apply activation
        let activated = self.activation.forward(&output)?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Apply activation backward pass
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("DepthwiseConv2D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("DepthwiseConv2D"))?;

        let input_array = input.view();

        let (batch_size, channels, input_height, input_width) = (
            input_array.shape()[0],
            input_array.shape()[1],
            input_array.shape()[2],
            input_array.shape()[3],
        );

        let (_, _, output_height, output_width) = (
            grad_upstream.shape()[0],
            grad_upstream.shape()[1],
            grad_upstream.shape()[2],
            grad_upstream.shape()[3],
        );

        // Initialize gradients
        let mut weight_grads = Array4::zeros(self.weights.raw_dim());
        let mut bias_grads = Array1::zeros(self.bias.raw_dim());
        let mut input_grads = Array4::zeros((batch_size, channels, input_height, input_width));

        // Compute bias gradients
        for c in 0..channels {
            let mut channel_sum = 0.0;
            for b in 0..batch_size {
                channel_sum += grad_upstream.slice(s![b, c, .., ..]).sum();
            }
            bias_grads[c] = channel_sum;
        }

        // Calculate padding dimensions once if needed
        let (pad_h, pad_w) =
            self.calculate_padding(input_height, input_width, output_height, output_width);

        // Determine whether to use parallel or sequential execution
        let total_elements = batch_size * channels * output_height * output_width;

        if total_elements >= DEPTHWISE_CONV_2D_PARALLEL_THRESHOLD {
            // Parallel computation of weight gradients and input gradients per batch
            let batch_results: Vec<(Array4<f32>, Array4<f32>)> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    self.compute_batch_gradients(
                        &input_array,
                        &grad_upstream,
                        b,
                        channels,
                        input_height,
                        input_width,
                        output_height,
                        output_width,
                        pad_h,
                        pad_w,
                    )
                })
                .collect();

            // Accumulate gradients from all batches
            for (b, (batch_weight_grads, batch_input_grads)) in
                batch_results.into_iter().enumerate()
            {
                weight_grads += &batch_weight_grads;
                input_grads
                    .slice_mut(s![b, .., .., ..])
                    .assign(&batch_input_grads.slice(s![0, .., .., ..]));
            }
        } else {
            // Sequential computation for small workloads
            for b in 0..batch_size {
                let (batch_weight_grads, batch_input_grads) = self.compute_batch_gradients(
                    &input_array,
                    &grad_upstream,
                    b,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    pad_h,
                    pad_w,
                );

                weight_grads += &batch_weight_grads;
                input_grads
                    .slice_mut(s![b, .., .., ..])
                    .assign(&batch_input_grads.slice(s![0, .., .., ..]));
            }
        }

        self.weight_gradients = Some(weight_grads);
        self.bias_gradients = Some(bias_grads);

        Ok(input_grads.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "DepthwiseConv2D"
    }

    fn output_shape(&self) -> String {
        if !self.input_shape.is_empty() {
            let output_shape = calculate_output_shape_2d(
                &self.input_shape,
                self.kernel_size,
                self.strides,
                &self.padding,
            );
            format!(
                "({}, {}, {}, {})",
                output_shape[0], output_shape[1], output_shape[2], output_shape[3]
            )
        } else {
            String::from("Unknown")
        }
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
        LayerWeight::DepthwiseConv2DLayer(DepthwiseConv2DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
