//! 2D depthwise separable convolution layer (depthwise stage followed by a pointwise 1x1 stage)

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::pad_tensor_4d_spatial;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{conv_backward, conv_forward};
use crate::neural_network::layers::convolution::validation::{
    validate_depth_multiplier, validate_filters, validate_input_shape_2d, validate_kernel_size_2d,
    validate_strides_2d,
};
use crate::neural_network::layers::layer_weight::{LayerWeight, SeparableConv2DLayerWeight};
use crate::neural_network::layers::shape_helpers::calculate_output_height_and_weight;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::parallel_gates::naive_conv_parallel_min_flops;
use ndarray::{Array2, Array4, ArrayD, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::borrow::Cow;

/// A 2D separable convolutional layer
///
/// Implements depthwise separable convolution with a depthwise step followed by a pointwise step,
/// this reduces parameters and computation compared to standard convolution while keeping similar
/// performance. Input shape is \[batch_size, channels, height, width\], intermediate depthwise
/// output shape is \[batch_size, channels * depth_multiplier, height', width'\], and final output
/// shape is \[batch_size, filters, height', width'\]
///
/// The separable convolution consists of:
/// 1. Depthwise convolution: each input channel is convolved with its own set of filters
/// 2. Pointwise convolution: a 1x1 convolution that combines the outputs from the depthwise step
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
/// // Create a 4D input tensor: [batch_size, channels, height, width]
/// let x = Array4::ones((2, 3, 32, 32)).into_dyn();
///
/// // Create target tensor
/// let y = Array4::ones((2, 64, 32, 32)).into_dyn();
///
/// // Build model with separable convolution
/// let mut model = Sequential::new();
/// model
///     .add(SeparableConv2D::new(
///         64,                          // Number of output filters
///         (3, 3),                      // Kernel size
///         vec![2, 3, 32, 32],          // Input shape
///         (1, 1),                      // Stride
///         1,                           // Depth multiplier
///         Activation::ReLU,            // ReLU activation
///     ).unwrap().with_padding(PaddingType::Same)) // Same padding
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// model.summary();
/// model.fit(&x, &y, 3).unwrap();
/// ```
#[derive(Debug)]
pub struct SeparableConv2D {
    /// Number of output channels from the pointwise convolution
    filters: usize,
    /// Depthwise convolution kernel size as (height, width)
    kernel_size: (usize, usize),
    /// Stride values for the convolution as (vertical, horizontal)
    strides: (usize, usize),
    /// Padding applied to the spatial dimensions (`Valid` or `Same`)
    padding: PaddingType,
    /// Number of depthwise filters per input channel
    depth_multiplier: usize,
    /// Depthwise filters with shape \[depth_multiplier, channels, kernel_height, kernel_width\]
    depthwise_weights: Array4<f32>,
    /// Pointwise filters with shape \[filters, channels * depth_multiplier, 1, 1\]
    pointwise_weights: Array4<f32>,
    /// Bias values with shape \[1, filters\]
    bias: Array2<f32>,
    /// Activation applied to the layer output
    activation: Activation,
    /// Cached activated output from the forward pass, used during backpropagation
    output_cache: Option<Tensor>,
    /// Cached input from the forward pass, used during backpropagation
    input_cache: Option<Tensor>,
    /// Cached depthwise output, used during backpropagation
    depthwise_output_cache: Option<Tensor>,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Gradients for the depthwise weights
    depthwise_weight_gradients: Option<Array4<f32>>,
    /// Gradients for the pointwise weights
    pointwise_weight_gradients: Option<Array4<f32>>,
    /// Gradients for the biases
    bias_gradients: Option<Array2<f32>>,
}

impl SeparableConv2D {
    /// Creates a new 2D separable convolutional layer
    ///
    /// Weights are initialized using Xavier (Glorot) uniform initialization; biases are zeros
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output channels from the pointwise convolution
    /// - `kernel_size` - Size of the depthwise convolution kernel as (height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\]
    /// - `strides` - Stride values for the convolution as (vertical, horizontal)
    /// - `depth_multiplier` - Number of depthwise convolution filters per input channel
    /// - `activation` - Activation applied to the output (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]; choose [`PaddingType::Same`] with
    /// [`SeparableConv2D::with_padding`]. Weights are seeded from the global seed or entropy by
    /// default; for reproducible initialization, set a seed with
    /// [`SeparableConv2D::with_random_state`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `SeparableConv2D` layer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidParameter` - If `depth_multiplier` is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 4D or has 0 channels
    /// - `Error::InvalidInput` - If input dimensions are smaller than kernel size
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        depth_multiplier: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_depth_multiplier(depth_multiplier)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        let channels = input_shape[1];
        let (depthwise_weights, pointwise_weights) =
            Self::init_weights_arrays(filters, channels, kernel_size, depth_multiplier, None);
        let bias = Array2::zeros((1, filters));

        Ok(SeparableConv2D {
            filters,
            kernel_size,
            strides,
            padding: PaddingType::Valid,
            depth_multiplier,
            depthwise_weights,
            pointwise_weights,
            bias,
            activation: activation.into(),
            output_cache: None,
            input_cache: None,
            depthwise_output_cache: None,
            input_shape,
            depthwise_weight_gradients: None,
            pointwise_weight_gradients: None,
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

    /// Sets the seed used to initialize the depthwise/pointwise weights and re-initializes them
    /// deterministically
    ///
    /// By default the weights are seeded from the global seed or entropy (see [`crate::random`]),
    /// this re-runs Xavier/Glorot uniform initialization with `random_state`, so call it before
    /// assigning custom weights or training. The bias stays zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        let channels = self.input_shape[1];
        let (depthwise_weights, pointwise_weights) = Self::init_weights_arrays(
            self.filters,
            channels,
            self.kernel_size,
            self.depth_multiplier,
            Some(random_state),
        );
        self.depthwise_weights = depthwise_weights;
        self.pointwise_weights = pointwise_weights;
        self
    }

    /// Xavier/Glorot uniform initialization of the depthwise and pointwise weight tensors
    ///
    /// Both draws share one RNG (threaded depthwise-then-pointwise) so a given seed reproduces the
    /// exact same pair of tensors
    fn init_weights_arrays(
        filters: usize,
        channels: usize,
        kernel_size: (usize, usize),
        depth_multiplier: usize,
        random_state: Option<u64>,
    ) -> (Array4<f32>, Array4<f32>) {
        // Xavier init for the depthwise weights
        let depthwise_fan_in = kernel_size.0 * kernel_size.1;
        let depthwise_fan_out = depth_multiplier * kernel_size.0 * kernel_size.1;
        let depthwise_bound = (6.0 / (depthwise_fan_in + depthwise_fan_out) as f32).sqrt();

        let mut rng = crate::random::make_rng(random_state);
        let depthwise_weights = Array4::random_using(
            (depth_multiplier, channels, kernel_size.0, kernel_size.1),
            Uniform::new(-depthwise_bound, depthwise_bound).unwrap(),
            &mut rng,
        );

        // Xavier init for the pointwise weights; the 1x1 kernel area is 1
        let pointwise_fan_in = channels * depth_multiplier;
        let pointwise_fan_out = filters;
        let pointwise_bound = (6.0 / (pointwise_fan_in + pointwise_fan_out) as f32).sqrt();

        let pointwise_weights = Array4::random_using(
            (filters, channels * depth_multiplier, 1, 1),
            Uniform::new(-pointwise_bound, pointwise_bound).unwrap(),
            &mut rng,
        );

        (depthwise_weights, pointwise_weights)
    }

    /// Calculates the output shape of the separable convolutional layer
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// Performs the depthwise convolution stage
    fn depthwise_convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_depthwise_output_shape(input_shape);

        // Zero-pad the spatial dims for `Same`
        let (pad_h, pad_w) = self.calculate_padding(
            input_shape[2],
            input_shape[3],
            output_shape[2],
            output_shape[3],
        );
        let padded_storage = if pad_h == 0 && pad_w == 0 {
            None
        } else {
            Some(pad_tensor_4d_spatial(input, pad_h, pad_w))
        };
        let conv_input: &Tensor = padded_storage.as_ref().unwrap_or(input);

        // Run per-batch in parallel above the FLOPs gate, sequentially below it
        let flops = 2
            * batch_size
            * channels
            * self.depth_multiplier
            * output_shape[2]
            * output_shape[3]
            * self.kernel_size.0
            * self.kernel_size.1;

        // One task per (batch item, output channel)
        let out_channels = channels * self.depth_multiplier;
        let tasks: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..out_channels).map(move |oc| (b, oc)))
            .collect();
        let run = |&(b, oc): &(usize, usize)| {
            let c = oc / self.depth_multiplier;
            let m = oc % self.depth_multiplier;
            self.compute_depthwise_channel(b, c, m, conv_input, &output_shape)
        };
        let results: Vec<Array2<f32>> = if flops >= naive_conv_parallel_min_flops() {
            tasks.par_iter().map(run).collect()
        } else {
            tasks.iter().map(run).collect()
        };

        let mut output: Tensor = ArrayD::zeros(output_shape);
        for (&(b, oc), plane) in tasks.iter().zip(results) {
            output.slice_mut(s![b, oc, .., ..]).assign(&plane);
        }
        output
    }

    /// Computes one depthwise output plane for a single (batch item, input channel,
    /// depth-multiplier) combination - the per-task unit of the depthwise stage
    fn compute_depthwise_channel(
        &self,
        b: usize,
        c: usize,
        m: usize,
        input: &Tensor,
        output_shape: &[usize],
    ) -> Array2<f32> {
        let (out_h, out_w) = (output_shape[2], output_shape[3]);
        let (kh_size, kw_size) = self.kernel_size;
        let in_w = input.shape()[3];

        // Direct convolution over the flat channel and kernel buffers, no per-element 4-D dynamic
        // indexing. The output geometry guarantees the kernel window fits, so no boundary clamp is
        // needed
        let channel = input.slice(s![b, c, .., ..]);
        let chan_std = channel.as_standard_layout();
        let src = chan_std
            .as_slice()
            .expect("standard-layout channel plane is contiguous");
        let kview = self.depthwise_weights.slice(s![m, c, .., ..]);
        let ker = kview.as_slice().expect("depthwise kernel is contiguous");

        let mut plane = Array2::zeros((out_h, out_w));
        let out = plane.as_slice_mut().expect("output plane is contiguous");
        for i in 0..out_h {
            let row_base = i * self.strides.0 * in_w;
            let out_row = i * out_w;
            for j in 0..out_w {
                let base = row_base + j * self.strides.1;
                let mut sum = 0.0;
                for ki in 0..kh_size {
                    let in_off = base + ki * in_w;
                    let k_off = ki * kw_size;
                    for kj in 0..kw_size {
                        sum += src[in_off + kj] * ker[k_off + kj];
                    }
                }
                out[out_row + j] = sum;
            }
        }

        plane
    }

    /// Performs the pointwise (1x1) convolution stage
    ///
    /// A 1x1 convolution is a per-position cross-channel matrix multiply, so this delegates to the
    /// shared [`conv_forward`] engine (im2col + gemm) rather than a hand-rolled loop nest. The
    /// pointwise weights `[filters, C*dm, 1, 1]` already match the engine's flat `[F, Cin, k...]`
    /// layout, and the bias `[1, filters]` is its per-filter `[F]` vector
    fn pointwise_convolve(&self, input: &Tensor) -> Tensor {
        conv_forward(
            input,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[1, 1],
            PaddingType::Valid,
        )
        // A 1x1 kernel under Valid padding can never exceed the input (every spatial dim >= 1)
        .expect("1x1 pointwise convolution geometry is always valid")
    }

    /// Calculates the output shape after the depthwise convolution stage
    fn calculate_depthwise_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![
            batch_size,
            channels * self.depth_multiplier,
            output_height,
            output_width,
        ]
    }

    /// Calculates the symmetric zero-padding (total height/width pad) for the depthwise stage
    ///
    /// Returns `(0, 0)` for `Valid` padding. For `Same`, returns the total padding along each
    /// spatial axis required so a stride-`s` convolution yields the given output size; the padding
    /// is split with `pad / 2` on the leading edge (see [`pad_tensor_4d_spatial`])
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
    /// - `depthwise_weights` - 4D array for depthwise filters with shape \[depth_multiplier, channels, kernel_height, kernel_width\]
    /// - `pointwise_weights` - 4D array for pointwise filters with shape \[filters, channels * depth_multiplier, 1, 1\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    ///
    /// # Errors
    ///
    /// - `Error` - If any supplied array shape does not match the existing layer weights
    pub fn set_weights(
        &mut self,
        depthwise_weights: Array4<f32>,
        pointwise_weights: Array4<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape(
            "depthwise_weight",
            self.depthwise_weights.shape(),
            depthwise_weights.shape(),
        )?;
        validate_weight_shape(
            "pointwise_weight",
            self.pointwise_weights.shape(),
            pointwise_weights.shape(),
        )?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.depthwise_weights = depthwise_weights;
        self.pointwise_weights = pointwise_weights;
        self.bias = bias;
        Ok(())
    }
}

impl Layer for SeparableConv2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Cache input for backpropagation
        self.input_cache = Some(input.clone());

        // Depthwise convolution (each channel independently), then pointwise (1x1) to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        // Cache the depthwise output; only backward needs it
        self.depthwise_output_cache = Some(depthwise_output);

        let activated = self.activation.forward(&output.into_dyn())?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches); see [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Depthwise convolution (each channel independently), then pointwise (1x1) to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        let activated = self.activation.forward(&output.into_dyn())?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Backward through the activation first
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("SeparableConv2D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        if let (Some(input), Some(depthwise_output)) =
            (&self.input_cache, &self.depthwise_output_cache)
        {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let depthwise_shape = depthwise_output.shape();

            // Re-create the zero-padded input from `depthwise_convolve`
            let (pad_h, pad_w) = self.calculate_padding(
                input_shape[2],
                input_shape[3],
                depthwise_shape[2],
                depthwise_shape[3],
            );
            let padded_storage = if pad_h == 0 && pad_w == 0 {
                None
            } else {
                Some(pad_tensor_4d_spatial(input, pad_h, pad_w))
            };
            let padded_input: &Tensor = padded_storage.as_ref().unwrap_or(input);
            let padded_shape = padded_input.shape().to_vec();

            // Pointwise (1x1) backward via the shared engine (im2col + gemm)
            let pw_grads = conv_backward(
                &grad_upstream,
                depthwise_output,
                self.pointwise_weights
                    .as_slice()
                    .expect("pointwise weights must be contiguous"),
                self.pointwise_weights.shape(),
                &[1, 1],
                PaddingType::Valid,
            )
            // 1x1 Valid geometry is always valid (see `pointwise_convolve`)
            .expect("1x1 pointwise convolution geometry is always valid");
            let pointwise_weight_grads =
                Array4::from_shape_vec(self.pointwise_weights.raw_dim(), pw_grads.weight_grad)
                    .expect("pointwise weight gradient shape matches weights");
            let bias_grads = Array2::from_shape_vec(self.bias.raw_dim(), pw_grads.bias_grad)
                .expect("bias gradient shape matches bias");
            // Gradient w.r.t. the depthwise output, shape [batch, C*dm, H', W']
            let depthwise_grad = pw_grads.input_grad;

            // Estimated FLOPs of the depthwise backward, for the parallel/serial gates below
            let bwd_flops = 2
                * batch_size
                * channels
                * self.depth_multiplier
                * depthwise_shape[2]
                * depthwise_shape[3]
                * self.kernel_size.0
                * self.kernel_size.1;
            let parallel = bwd_flops >= naive_conv_parallel_min_flops();

            // Depthwise weight gradients
            let mut depthwise_weight_grads = Array4::zeros(self.depthwise_weights.dim());
            let wg_tasks: Vec<(usize, usize)> = (0..self.depth_multiplier)
                .flat_map(|m| (0..channels).map(move |c| (m, c)))
                .collect();
            let (kh_size, kw_size) = self.kernel_size;
            let (out_h, out_w) = (depthwise_shape[2], depthwise_shape[3]);
            let (padded_h, padded_w) = (padded_shape[2], padded_shape[3]);
            // Weight gradient for one `(m, c)`: a fused pass over output positions accumulating
            // `padded_input * grad` on flat buffers, summed across the batch (no 4-D indexing)
            let wg_run = |&(m, c): &(usize, usize)| -> Array2<f32> {
                let oc = c * self.depth_multiplier + m;
                let mut wg = vec![0.0f32; kh_size * kw_size];
                for b in 0..batch_size {
                    let g = depthwise_grad.slice(s![b, oc, .., ..]);
                    let g = g.as_standard_layout();
                    let g = g.as_slice().expect("grad plane is contiguous");
                    let inp = padded_input.slice(s![b, c, .., ..]);
                    let inp = inp.as_standard_layout();
                    let inp = inp.as_slice().expect("padded input plane is contiguous");
                    for i in 0..out_h {
                        let row_base = i * self.strides.0 * padded_w;
                        let g_row = i * out_w;
                        for j in 0..out_w {
                            let gv = g[g_row + j];
                            let base = row_base + j * self.strides.1;
                            for h in 0..kh_size {
                                let in_off = base + h * padded_w;
                                let k_off = h * kw_size;
                                for w in 0..kw_size {
                                    wg[k_off + w] += inp[in_off + w] * gv;
                                }
                            }
                        }
                    }
                }
                Array2::from_shape_vec((kh_size, kw_size), wg)
                    .expect("weight gradient length matches the kernel shape")
            };
            let wg_results: Vec<Array2<f32>> = if parallel {
                wg_tasks.par_iter().map(wg_run).collect()
            } else {
                wg_tasks.iter().map(wg_run).collect()
            };
            for (&(m, c), wg) in wg_tasks.iter().zip(wg_results) {
                depthwise_weight_grads
                    .slice_mut(s![m, c, .., ..])
                    .assign(&wg);
            }

            // Input gradients
            let mut input_gradients = ArrayD::zeros(padded_input.dim());
            let ig_tasks: Vec<(usize, usize)> = (0..batch_size)
                .flat_map(|b| (0..channels).map(move |c| (b, c)))
                .collect();
            // Input gradient for one `(b, c)`: scatter each output position's `kernel * grad` into
            // the padded input-gradient plane (the transpose of the forward overlap-add), summed
            // over the depth multiplier - flat buffers, no per-padded-position gather with modulo
            let ig_run = |&(b, c): &(usize, usize)| -> Array2<f32> {
                let mut plane = vec![0.0f32; padded_h * padded_w];
                for m in 0..self.depth_multiplier {
                    let oc = c * self.depth_multiplier + m;
                    let g = depthwise_grad.slice(s![b, oc, .., ..]);
                    let g = g.as_standard_layout();
                    let g = g.as_slice().expect("grad plane is contiguous");
                    let kview = self.depthwise_weights.slice(s![m, c, .., ..]);
                    let ker = kview.as_slice().expect("depthwise kernel is contiguous");
                    for i in 0..out_h {
                        let row_base = i * self.strides.0 * padded_w;
                        let g_row = i * out_w;
                        for j in 0..out_w {
                            let gv = g[g_row + j];
                            let base = row_base + j * self.strides.1;
                            for h in 0..kh_size {
                                let off = base + h * padded_w;
                                let k_off = h * kw_size;
                                for w in 0..kw_size {
                                    plane[off + w] += ker[k_off + w] * gv;
                                }
                            }
                        }
                    }
                }
                Array2::from_shape_vec((padded_h, padded_w), plane)
                    .expect("input gradient length matches the padded plane")
            };
            let ig_results: Vec<Array2<f32>> = if parallel {
                ig_tasks.par_iter().map(ig_run).collect()
            } else {
                ig_tasks.iter().map(ig_run).collect()
            };
            for (&(b, c), plane) in ig_tasks.iter().zip(ig_results) {
                input_gradients.slice_mut(s![b, c, .., ..]).assign(&plane);
            }

            let input_gradients = if pad_h == 0 && pad_w == 0 {
                input_gradients
            } else {
                let pad_top = pad_h / 2;
                let pad_left = pad_w / 2;
                input_gradients
                    .slice(s![
                        ..,
                        ..,
                        pad_top..pad_top + input_shape[2],
                        pad_left..pad_left + input_shape[3]
                    ])
                    .to_owned()
                    .into_dyn()
            };

            self.depthwise_weight_gradients = Some(depthwise_weight_grads);
            self.pointwise_weight_gradients = Some(pointwise_weight_grads);
            self.bias_gradients = Some(bias_grads);

            Ok(input_gradients)
        } else {
            Err(Error::forward_pass_not_run("SeparableConv2D"))
        }
    }

    fn layer_type(&self) -> &str {
        "SeparableConv2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        )
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            self.depthwise_weights.len() + self.pointwise_weights.len() + self.bias.len(),
        )
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            depthwise_weights,
            pointwise_weights,
            bias,
            depthwise_weight_gradients,
            pointwise_weight_gradients,
            bias_gradients,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(gd), Some(gp), Some(gb)) = (
            depthwise_weight_gradients.as_ref(),
            pointwise_weight_gradients.as_ref(),
            bias_gradients.as_ref(),
        ) {
            params.push(ParamGrad::weight(
                depthwise_weights
                    .as_slice_mut()
                    .expect("depthwise weights must be contiguous"),
                gd.as_slice()
                    .expect("depthwise weight gradient must be contiguous"),
            ));
            params.push(ParamGrad::weight(
                pointwise_weights
                    .as_slice_mut()
                    .expect("pointwise weights must be contiguous"),
                gp.as_slice()
                    .expect("pointwise weight gradient must be contiguous"),
            ));
            params.push(ParamGrad::no_decay(
                bias.as_slice_mut().expect("bias must be contiguous"),
                gb.as_slice().expect("bias gradient must be contiguous"),
            ));
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::SeparableConv2D(SeparableConv2DLayerWeight {
            depthwise_weight: Cow::Borrowed(&self.depthwise_weights),
            pointwise_weight: Cow::Borrowed(&self.pointwise_weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}
