//! 2D depthwise convolution layer that applies one filter per input channel

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::pad_tensor_2d;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::validation::{
    validate_filters, validate_input_shape_2d, validate_kernel_size_2d, validate_strides_2d,
};
use crate::neural_network::layers::layer_weight::{DepthwiseConv2DLayerWeight, LayerWeight};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_2d;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::parallel_gates::NAIVE_CONV_PARALLEL_MIN_FLOPS;
use ndarray::{Array1, Array2, Array4, ArrayView2, ArrayViewD, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::borrow::Cow;

/// A 2D depthwise convolutional layer for neural networks
///
/// Applies one convolutional filter per input channel, reducing parameter count and
/// computation while preserving spatial feature extraction. Input shape is
/// \[batch_size, channels, height, width\], and each channel is convolved independently
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
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
///     Activation::ReLU,        // activation
/// ).unwrap();
///
/// // Add layer and compile model
/// model
///     .add(depthwise_layer)
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
#[derive(Debug)]
pub struct DepthwiseConv2D {
    /// Number of output channels (must equal input channels for pure depthwise convolution)
    filters: usize,
    /// Size of the convolution kernel as (height, width)
    kernel_size: (usize, usize),
    /// Stride of the convolution as (height_stride, width_stride)
    strides: (usize, usize),
    /// Padding strategy (Valid or Same)
    padding: PaddingType,
    /// 4D weight tensor with shape \[filters, 1, kernel_height, kernel_width\]
    weights: Array4<f32>,
    /// 1D bias vector with shape \[filters\]
    bias: Array1<f32>,
    /// Activation applied to the convolution output
    activation: Activation,
    /// Cached post-activation output for the backward pass
    output_cache: Option<Tensor>,
    /// Cached input tensor for the backward pass
    input: Option<Tensor>,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Gradients with respect to weights
    weight_gradients: Option<Array4<f32>>,
    /// Gradients with respect to bias
    bias_gradients: Option<Array1<f32>>,
}

impl DepthwiseConv2D {
    /// Creates a new DepthwiseConv2D layer
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (must equal input channels for pure depthwise)
    /// - `kernel_size` - Size of the convolution kernel as (height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\]
    /// - `strides` - Stride of the convolution as (height_stride, width_stride)
    /// - `activation` - Activation function applied to the output (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]; choose [`PaddingType::Same`] with
    /// [`DepthwiseConv2D::with_padding`]. Weights are seeded from the global seed or entropy by
    /// default; for reproducible initialization, set a seed with
    /// [`DepthwiseConv2D::with_random_state`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `DepthwiseConv2D` instance with Xavier-initialized weights or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 4D, has 0 channels, or is smaller than the kernel
    /// - `Error::InvalidParameter` - If `filters` does not equal the input channels
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        // Pure depthwise convolution fixes the filter count to the input channels
        let channels = input_shape[1];
        if channels != filters {
            return Err(Error::invalid_parameter(
                "filters",
                "must equal the number of input channels",
            ));
        }

        let weights = Self::init_weights_array(filters, kernel_size, None);
        let bias = Array1::zeros(filters);

        Ok(Self {
            filters,
            kernel_size,
            strides,
            padding: PaddingType::Valid,
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

    /// Sets the padding mode (defaults to [`PaddingType::Valid`])
    ///
    /// # Parameters
    ///
    /// - `padding` - Padding strategy (Valid or Same)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the seed used to initialize the depthwise weights and re-initializes them deterministically
    ///
    /// By default the weights are seeded from the global seed or entropy (see [`crate::random`]). This
    /// re-runs Xavier/Glorot uniform initialization with `random_state`, so call it before assigning
    /// custom weights or training. The bias stays zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.weights = Self::init_weights_array(self.filters, self.kernel_size, Some(random_state));
        self
    }

    /// Xavier/Glorot uniform initialization of the \[filters, 1, kh, kw\] depthwise weight tensor
    fn init_weights_array(
        filters: usize,
        kernel_size: (usize, usize),
        random_state: Option<u64>,
    ) -> Array4<f32> {
        let (kernel_height, kernel_width) = kernel_size;
        let fan = kernel_height * kernel_width;
        let weight_bound = (6.0 / (fan + fan) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array4::random_using(
            (filters, 1, kernel_height, kernel_width),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
            &mut rng,
        )
    }

    /// Calculates padding dimensions for Same padding mode
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

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - 4D weight tensor with shape \[filters, 1, kernel_height, kernel_width\]
    /// - `bias` - 1D bias vector with shape \[filters\]
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match the existing shape
    pub fn set_weights(&mut self, weights: Array4<f32>, bias: Array1<f32>) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }

    /// Runs the depthwise convolution and activation for `input`, returning the activated output
    ///
    /// Shared numeric body of [`Layer::forward`] and [`Layer::predict`]; writes no caches. `forward`
    /// wraps this and records the input/output caches, `predict` returns its result directly. Each
    /// `(batch item, channel)` is convolved independently into a disjoint output plane, so the
    /// FLOPs-gated parallel and sequential paths are bitwise identical
    fn convolve(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        let input_array = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();

        let (batch_size, channels, height, width) = (
            input_array.shape()[0],
            input_array.shape()[1],
            input_array.shape()[2],
            input_array.shape()[3],
        );

        if channels != self.filters {
            return Err(Error::dimension_mismatch(self.filters, channels));
        }

        let output_shape =
            calculate_output_shape_2d(input.shape(), self.kernel_size, self.strides, &self.padding);
        let (output_height, output_width) = (output_shape[2], output_shape[3]);

        let (pad_h, pad_w) = self.calculate_padding(height, width, output_height, output_width);

        let mut output = Array4::zeros((batch_size, channels, output_height, output_width));

        let flops = 2
            * batch_size
            * channels
            * output_height
            * output_width
            * self.kernel_size.0
            * self.kernel_size.1;

        let convolve_into =
            |b: usize, c: usize, channel_output: &mut ndarray::ArrayViewMut2<f32>| {
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
            };

        if flops >= NAIVE_CONV_PARALLEL_MIN_FLOPS {
            // Parallel over (batch item, channel): a single large image still uses every core
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
                            convolve_into(b, c, &mut channel_output)
                        });
                });
        } else {
            for b in 0..batch_size {
                for c in 0..channels {
                    let mut channel_output = output.slice_mut(s![b, c, .., ..]);
                    convolve_into(b, c, &mut channel_output);
                }
            }
        }

        self.activation.forward(&output.into_dyn())
    }

    /// Performs depthwise convolution for a single channel
    #[allow(clippy::too_many_arguments)]
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
        let padded_input = if *padding == PaddingType::Same {
            pad_tensor_2d(&input_channel.to_owned(), pad_h, pad_w)
        } else {
            input_channel.to_owned()
        };

        let (output_height, output_width) = output_shape;
        let (kh_size, kw_size) = kernel_size;
        let padded_w = padded_input.shape()[1];

        // Direct convolution over the flat contiguous buffers: each output is a multiply-accumulate
        // over the kernel window, with no per-position temporary. The output geometry guarantees
        // every window fits, so no bounds guard is needed
        let src = padded_input
            .as_slice()
            .expect("padded channel plane is contiguous");
        let ker = kernel.to_owned();
        let ker = ker.as_slice().expect("kernel is contiguous after to_owned");

        let mut channel_output = Array2::zeros(output_shape);
        let out = channel_output
            .as_slice_mut()
            .expect("output plane is contiguous");

        for oh in 0..output_height {
            let row_base = oh * strides.0 * padded_w;
            let out_row = oh * output_width;
            for ow in 0..output_width {
                let base = row_base + ow * strides.1;
                let mut sum = bias;
                for kh in 0..kh_size {
                    let in_off = base + kh * padded_w;
                    let k_off = kh * kw_size;
                    for kw in 0..kw_size {
                        sum += src[in_off + kw] * ker[k_off + kw];
                    }
                }
                out[out_row + ow] = sum;
            }
        }

        channel_output
    }

    /// Computes weight and input gradients for one `(batch item, channel)` pair
    ///
    /// The depthwise channels are independent (channel `c` only touches filter `c`), so this is the
    /// natural task granularity: a single large image parallelizes across its channels even at
    /// batch == 1
    #[allow(clippy::too_many_arguments)]
    fn compute_channel_gradients(
        &self,
        input_array: &ArrayViewD<f32>,
        grad_upstream: &Tensor,
        batch_idx: usize,
        c: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> (Array2<f32>, Array2<f32>) {
        let (kh_size, kw_size) = self.kernel_size;
        // Accumulate the input gradient in PADDED coordinates
        let padded_height = input_height + pad_h;
        let padded_width = input_width + pad_w;

        let input_channel = input_array.slice(s![batch_idx, c, .., ..]);
        let grad_channel = grad_upstream.slice(s![batch_idx, c, .., ..]);

        let padded_input = if self.padding == PaddingType::Same {
            pad_tensor_2d(&input_channel.to_owned(), pad_h, pad_w)
        } else {
            input_channel.to_owned()
        };
        let src = padded_input
            .as_slice()
            .expect("padded channel plane is contiguous");
        let grad_owned = grad_channel.to_owned();
        let grad = grad_owned
            .as_slice()
            .expect("grad plane is contiguous after to_owned");
        let kview = self.weights.slice(s![c, 0, .., ..]).to_owned();
        let ker = kview
            .as_slice()
            .expect("kernel is contiguous after to_owned");

        // Single pass over output positions accumulates both gradients directly on the flat
        // buffers. For each output `(oh, ow)` the kernel window contributes `src * grad` to the
        // weight gradient and `kernel * grad` to the padded input gradient at the same offset
        let mut weight_grad = vec![0.0f32; kh_size * kw_size];
        let mut input_grad_padded = vec![0.0f32; padded_height * padded_width];
        for oh in 0..output_height {
            let row_base = oh * self.strides.0 * padded_width;
            let g_row = oh * output_width;
            for ow in 0..output_width {
                let g = grad[g_row + ow];
                let base = row_base + ow * self.strides.1;
                for kh in 0..kh_size {
                    let in_off = base + kh * padded_width;
                    let k_off = kh * kw_size;
                    for kw in 0..kw_size {
                        weight_grad[k_off + kw] += src[in_off + kw] * g;
                        input_grad_padded[in_off + kw] += ker[k_off + kw] * g;
                    }
                }
            }
        }
        let weight_grad = Array2::from_shape_vec((kh_size, kw_size), weight_grad)
            .expect("weight gradient length matches the kernel shape");

        // Strip the symmetric padding back to the input plane
        let pad_top = pad_h / 2;
        let pad_left = pad_w / 2;
        let mut input_grad = Array2::zeros((input_height, input_width));
        let ig = input_grad
            .as_slice_mut()
            .expect("input gradient plane is contiguous");
        for ih in 0..input_height {
            let src_row = (ih + pad_top) * padded_width + pad_left;
            let dst_row = ih * input_width;
            ig[dst_row..dst_row + input_width]
                .copy_from_slice(&input_grad_padded[src_row..src_row + input_width]);
        }

        (weight_grad, input_grad)
    }
}

impl Layer for DepthwiseConv2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let activated = self.convolve(input)?;
        // Cache only after a successful convolution, so a rejected input leaves no partial state
        self.input = Some(input.clone());
        self.input_shape = input.shape().to_vec();
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.convolve(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
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

        let (pad_h, pad_w) =
            self.calculate_padding(input_height, input_width, output_height, output_width);

        let flops = 2
            * batch_size
            * channels
            * output_height
            * output_width
            * self.kernel_size.0
            * self.kernel_size.1;

        // One task per (batch item, channel)
        let tasks: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..channels).map(move |c| (b, c)))
            .collect();
        let run = |&(b, c): &(usize, usize)| {
            self.compute_channel_gradients(
                &input_array,
                &grad_upstream,
                b,
                c,
                input_height,
                input_width,
                output_height,
                output_width,
                pad_h,
                pad_w,
            )
        };
        let results: Vec<(Array2<f32>, Array2<f32>)> = if flops >= NAIVE_CONV_PARALLEL_MIN_FLOPS {
            tasks.par_iter().map(run).collect()
        } else {
            tasks.iter().map(run).collect()
        };

        for (&(b, c), (weight_grad, input_grad)) in tasks.iter().zip(results) {
            let mut wg = weight_grads.slice_mut(s![c, 0, .., ..]);
            wg += &weight_grad;
            input_grads.slice_mut(s![b, c, .., ..]).assign(&input_grad);
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
        LayerWeight::DepthwiseConv2D(DepthwiseConv2DLayerWeight {
            weight: Cow::Borrowed(&self.weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}
