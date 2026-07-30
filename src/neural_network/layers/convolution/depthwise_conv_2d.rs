//! 2D depthwise convolution layer that gives each input channel its own kernel or kernels

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, DepthwiseGradients, depthwise_forward_row, depthwise_item_gradients,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::validation::{
    validate_depth_multiplier, validate_input_shape_2d, validate_kernel_size_2d,
    validate_strides_2d,
};
use crate::neural_network::layers::layer_weight::{DepthwiseConv2DLayerWeight, LayerWeight};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_2d;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::parallel_gates::naive_conv_parallel_min_flops;
use ndarray::{Array1, Array4};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::prelude::*;
use std::borrow::Cow;

/// A 2D depthwise convolutional layer
///
/// Applies its own kernel to each input channel. This lowers the parameter count and the
/// computation compared to a standard convolution. It still extracts spatial features per
/// channel. Input shape is \[batch_size, height, width, channels\] and output shape is
/// \[batch_size, out_height, out_width, channels * depth_multiplier\]. No kernel mixes channels
///
/// `depth_multiplier` (default 1, set with [`DepthwiseConv2D::with_depth_multiplier`]) is how many
/// kernels each input channel gets. The output channel for input channel `c` and multiplier index
/// `m` is `c * depth_multiplier + m`, which matches Keras
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
/// let mut model = Sequential::new();
///
/// // DepthwiseConv2D with ReLU (new initializes the weights)
/// let depthwise_layer = DepthwiseConv2D::new(
///     (2, 2),                  // kernel_size
///     vec![1, 4, 4, 3],        // input shape [batch_size, height, width, channels]
///     (1, 1),                  // strides
///     Activation::ReLU,        // activation
/// ).unwrap();
///
/// model
///     .add(depthwise_layer)
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Input data shape: [batch_size, height, width, channels]
/// let batch_size = 1;
/// let input_channels = 3;
/// let height = 4;
/// let width = 4;
///
/// let mut input_data = Array4::zeros((batch_size, height, width, input_channels));
///
/// // Give each channel distinct values
/// for c in 0..input_channels {
///     for h in 0..height {
///         for w in 0..width {
///             input_data[[0, h, w, c]] = (c * 10 + h * width + w) as f32;
///         }
///     }
/// }
///
/// let input = input_data.into_dyn();
///
/// model.summary();
///
/// let output = model.predict(&input).unwrap();
///
/// // Input [1, 4, 4, 3] with kernel (2, 2), stride (1, 1), and Valid padding gives output
/// // [1, 3, 3, 3]
/// assert_eq!(output.shape(), &[1, 3, 3, 3]);
///
/// // ReLU makes every output value non-negative
/// for value in output.iter() {
///     assert!(*value >= 0.0);
/// }
/// ```
#[derive(Debug)]
pub struct DepthwiseConv2D {
    /// Number of input channels, read from the declared input shape
    channels: usize,
    /// Kernels per input channel. The output carries `channels * depth_multiplier` of them
    depth_multiplier: usize,
    /// Size of the convolution kernel as (height, width)
    kernel_size: (usize, usize),
    /// Stride of the convolution as (height_stride, width_stride)
    strides: (usize, usize),
    /// Padding strategy (Valid or Same)
    padding: PaddingType,
    /// 4D weight tensor with shape \[kernel_height, kernel_width, channels, depth_multiplier\]
    weights: Array4<f32>,
    /// 1D bias vector with shape \[channels * depth_multiplier\]
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
    /// - `kernel_size` - Size of the convolution kernel as (height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, height, width, channels\]
    /// - `strides` - Stride of the convolution as (height_stride, width_stride)
    /// - `activation` - Activation function applied to the output (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `DepthwiseConv2D` instance with Xavier-initialized
    ///   weights, or an error
    ///
    /// # Notes
    ///
    /// There is no `filters` argument. A depthwise convolution derives its output channel count
    /// from the input, as `channels * depth_multiplier`. `depth_multiplier` defaults to 1. Set it
    /// with [`DepthwiseConv2D::with_depth_multiplier`]. Padding defaults to
    /// [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`DepthwiseConv2D::with_padding`]. The layer seeds weights from the global seed or entropy
    /// by default. For reproducible initialization, set a seed with
    /// [`DepthwiseConv2D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 4D, has 0 channels, or is smaller
    ///   than the kernel
    pub fn new(
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        let channels = input_shape[3];
        let weights = Self::init_weights_array(channels, 1, kernel_size, None);
        let bias = Array1::zeros(channels);

        Ok(Self {
            channels,
            depth_multiplier: 1,
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

    /// Sets how many kernels each input channel gets (defaults to 1)
    ///
    /// The layer then produces `channels * depth_multiplier` output channels. Input channel `c`'s
    /// multiplier `m` lands at output channel `c * depth_multiplier + m`. This re-runs weight
    /// initialization at the new shape and re-zeros the bias. Call it before you assign custom
    /// weights or start training
    ///
    /// # Parameters
    ///
    /// - `depth_multiplier` - Kernels per input channel
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `depth_multiplier` is 0
    pub fn with_depth_multiplier(mut self, depth_multiplier: usize) -> Result<Self, Error> {
        validate_depth_multiplier(depth_multiplier)?;
        self.depth_multiplier = depth_multiplier;
        self.weights =
            Self::init_weights_array(self.channels, depth_multiplier, self.kernel_size, None);
        self.bias = Array1::zeros(self.channels * depth_multiplier);
        Ok(self)
    }

    /// Sets the seed for the depthwise weights and re-initializes them deterministically
    ///
    /// By default, the layer seeds weights from the global seed or entropy (see
    /// [`crate::random`]). This re-runs Xavier/Glorot uniform initialization with `random_state`.
    /// Call it before you assign custom weights or start training. The bias stays zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.weights = Self::init_weights_array(
            self.channels,
            self.depth_multiplier,
            self.kernel_size,
            Some(random_state),
        );
        self
    }

    /// Xavier/Glorot uniform initialization of the \[kh, kw, channels, depth_multiplier\] tensor
    fn init_weights_array(
        channels: usize,
        depth_multiplier: usize,
        kernel_size: (usize, usize),
        random_state: Option<u64>,
    ) -> Array4<f32> {
        let (kernel_height, kernel_width) = kernel_size;
        // Keras' `compute_fans` reads only the kernel tensor's last 2 axes. For shape
        // [kh, kw, channels, depth_multiplier] this gives `fan_in = channels * kh * kw` and
        // `fan_out = depth_multiplier * kh * kw`. A depthwise unit sees only 1 input channel.
        // This fan_in is therefore `channels` times the true receptive field. The resulting
        // bound is narrower by about sqrt(channels) than a per-channel count would give. This
        // matches `Conv2D` and Keras rather than a depthwise-specific formula
        let fan_in = channels * kernel_height * kernel_width;
        let fan_out = depth_multiplier * kernel_height * kernel_width;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array4::random_using(
            (kernel_height, kernel_width, channels, depth_multiplier),
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
    /// - `weights` - 4D weight tensor with shape
    ///   \[kernel_height, kernel_width, channels, depth_multiplier\]
    /// - `bias` - 1D bias vector with shape \[channels * depth_multiplier\]
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match
    ///   the existing shape
    pub fn set_weights(&mut self, weights: Array4<f32>, bias: Array1<f32>) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }

    /// The layer's geometry for a given input, as the shared kernel wants it
    fn geometry(&self, input_shape: &[usize]) -> DepthwiseGeometry {
        let (height, width) = (input_shape[1], input_shape[2]);
        let output_shape =
            calculate_output_shape_2d(input_shape, self.kernel_size, self.strides, &self.padding);
        let (out_height, out_width) = (output_shape[1], output_shape[2]);
        let (pad_h, pad_w) = self.calculate_padding(height, width, out_height, out_width);
        DepthwiseGeometry {
            input: (height, width),
            output: (out_height, out_width),
            channels: self.channels,
            depth_multiplier: self.depth_multiplier,
            kernel: self.kernel_size,
            strides: self.strides,
            pad_before: (pad_h / 2, pad_w / 2),
        }
    }

    /// Depthwise convolution over a channels-last tensor, followed by the activation
    ///
    /// Shared numeric body of [`Layer::forward`] and [`Layer::predict`]. `forward` wraps this and
    /// records the input/output caches. `predict` returns the result directly
    fn convolve(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }
        let channels = input.shape()[3];
        if channels != self.channels {
            return Err(Error::dimension_mismatch(self.channels, channels));
        }

        let g = self.geometry(input.shape());
        let batch_size = input.shape()[0];
        let out_channels = g.out_channels();

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self.weights.as_slice().expect("weights must be contiguous");
        let bias = Some(self.bias.as_slice().expect("bias must be contiguous"));

        let mut output = Array4::<f32>::zeros((batch_size, g.output.0, g.output.1, out_channels));
        let flops =
            2 * batch_size * out_channels * g.output.0 * g.output.1 * g.kernel.0 * g.kernel.1;
        let row_len = g.output.1 * out_channels;
        let out_flat = output.as_slice_mut().expect("output is contiguous");

        // 1 task per (batch item, output row). Output rows are disjoint, so this needs no halo
        // and no merge. It keeps every core busy even at batch == 1
        if flops >= naive_conv_parallel_min_flops() {
            out_flat
                .par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| {
                    depthwise_forward_row(&g, src, ker, bias, i / g.output.0, i % g.output.0, row)
                });
        } else {
            for (i, row) in out_flat.chunks_mut(row_len).enumerate() {
                depthwise_forward_row(&g, src, ker, bias, i / g.output.0, i % g.output.0, row);
            }
        }

        self.activation.forward(&output.into_dyn())
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

        let batch_size = input.shape()[0];
        let g = self.geometry(input.shape());
        let out_channels = g.out_channels();

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let grad_std = grad_upstream.as_standard_layout();
        let grad = grad_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self.weights.as_slice().expect("weights must be contiguous");

        let flops =
            2 * batch_size * out_channels * g.output.0 * g.output.1 * g.kernel.0 * g.kernel.1;

        let run = |b: usize| depthwise_item_gradients(&g, src, grad, ker, b);
        let per_b: Vec<DepthwiseGradients> = if flops >= naive_conv_parallel_min_flops() {
            (0..batch_size).into_par_iter().map(run).collect()
        } else {
            (0..batch_size).map(run).collect()
        };

        // Sum the weight and bias partials in batch order, so the result does not depend on
        // which branch above ran
        let mut weight_grads = vec![0.0f32; self.weights.len()];
        let mut bias_grads = vec![0.0f32; out_channels];
        let mut input_grads = Vec::with_capacity(batch_size * g.input.0 * g.input.1 * g.channels);
        for part in per_b {
            for (acc, v) in weight_grads.iter_mut().zip(part.weight) {
                *acc += v;
            }
            for (acc, v) in bias_grads.iter_mut().zip(part.bias) {
                *acc += v;
            }
            input_grads.extend(part.input);
        }

        self.weight_gradients = Some(
            Array4::from_shape_vec(self.weights.raw_dim(), weight_grads)
                .expect("weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(Array1::from_vec(bias_grads));

        Ok(
            Array4::from_shape_vec((batch_size, g.input.0, g.input.1, g.channels), input_grads)
                .expect("input gradient shape matches input")
                .into_dyn(),
        )
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

/// Unit tests for `DepthwiseConv2D`
#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::activation::linear::Linear;
    use ndarray::ArrayD;

    /// Channels never mix. The values match a hand-worked cross-correlation
    ///
    /// Channel 0 carries 1..9 against an all-ones kernel. Channel 1 carries all ones against an
    /// all-twos kernel. Channel 1's output is therefore the same constant at every position,
    /// while channel 0's output varies. This only holds if the 2 kernels stay on their own
    /// channels
    #[test]
    fn depthwise_forward_keeps_channels_separate_hand_derived() {
        let mut layer =
            DepthwiseConv2D::new((2, 2), vec![1, 3, 3, 2], (1, 1), Linear::new()).unwrap();
        // [kh, kw, C, dm] with dm == 1: channel 0 all ones, channel 1 all twos
        let weights =
            Array4::from_shape_vec((2, 2, 2, 1), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
                .unwrap();
        layer.set_weights(weights, Array1::zeros(2)).unwrap();

        // [1, 3, 3, 2]: channel 0 is 1..9, channel 1 is all ones
        let input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 3, 2]),
            vec![
                1.0, 1.0, 2.0, 1.0, 3.0, 1.0, // row 0
                4.0, 1.0, 5.0, 1.0, 6.0, 1.0, // row 1
                7.0, 1.0, 8.0, 1.0, 9.0, 1.0, // row 2
            ],
        )
        .unwrap();

        let out = layer.predict(&input).unwrap();
        assert_eq!(out.shape(), &[1, 2, 2, 2]);
        // Channel 0: the 4 window sums from the 2x2 kernel. Channel 1: 2 * 4 ones at every position
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![12.0, 8.0, 16.0, 8.0, 24.0, 8.0, 28.0, 8.0]
        );
    }

    /// With `depth_multiplier`, input channel `c`'s multiplier `m` lands at output channel
    /// `c * depth_multiplier + m`, the Keras ordering
    #[test]
    fn depthwise_depth_multiplier_output_channel_order() {
        let mut layer = DepthwiseConv2D::new((1, 1), vec![1, 1, 1, 2], (1, 1), Linear::new())
            .unwrap()
            .with_depth_multiplier(2)
            .unwrap();
        assert_eq!(layer.weights.shape(), &[1, 1, 2, 2]);

        // [kh=1, kw=1, C=2, dm=2] flat as (c, m): c0 -> [1, 10], c1 -> [100, 1000]
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
        layer.set_weights(weights, Array1::zeros(4)).unwrap();

        // 1 position holding [2, 3]
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 1, 2]), vec![2.0, 3.0]).unwrap();

        let out = layer.predict(&input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 1, 4]);
        // [c0m0, c0m1, c1m0, c1m1] = [2*1, 2*10, 3*100, 3*1000]
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 20.0, 300.0, 3000.0]
        );
    }
}
