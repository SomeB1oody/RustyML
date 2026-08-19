//! 1D depthwise convolution layer that gives each input channel its own kernel or kernels

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, depthwise_backward, depthwise_forward,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::validation::{
    validate_depth_multiplier, validate_input_shape_1d, validate_kernel_size_1d,
    validate_strides_1d,
};
use crate::neural_network::layers::layer_weight::{DepthwiseConv1DLayerWeight, LayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array1, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::borrow::Cow;

/// A 1D depthwise convolutional layer
///
/// Applies its own kernel to each input channel of a sequence. This lowers the parameter count
/// and the computation compared to a standard convolution. It still extracts features along the
/// length axis per channel. Input shape is \[batch_size, length, channels\] and output shape is
/// \[batch_size, out_length, channels * depth_multiplier\]. No kernel mixes channels
///
/// `depth_multiplier` (default 1, set with [`DepthwiseConv1D::with_depth_multiplier`]) is how many
/// kernels each input channel gets. The output channel for input channel `c` and multiplier index
/// `m` is `c * depth_multiplier + m`, which matches Keras
///
/// This is the 1D form of
/// [`DepthwiseConv2D`](crate::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D).
/// Both layers run the same shared kernel, because a sequence is a height-1 image
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
/// let mut model = Sequential::new();
///
/// // DepthwiseConv1D with ReLU (new initializes the weights)
/// let depthwise_layer = DepthwiseConv1D::new(
///     3,                       // kernel_size
///     vec![2, 10, 4],          // input shape [batch_size, length, channels]
///     1,                       // stride
///     Activation::ReLU,        // activation
/// ).unwrap();
///
/// model
///     .add(depthwise_layer)
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// model.summary();
///
/// // Input data shape: [batch_size, length, channels]
/// let input = Array3::<f32>::ones((2, 10, 4)).into_dyn();
/// let output = model.predict(&input).unwrap();
///
/// // Input [2, 10, 4] with kernel 3, stride 1, and Valid padding gives output [2, 8, 4]
/// assert_eq!(output.shape(), &[2, 8, 4]);
/// ```
#[derive(Debug)]
pub struct DepthwiseConv1D {
    /// Number of input channels, read from the declared input shape
    channels: usize,
    /// Kernels per input channel. The output carries `channels * depth_multiplier` of them
    depth_multiplier: usize,
    /// Size of the convolution kernel along the length axis
    kernel_size: usize,
    /// Stride of the convolution along the length axis
    stride: usize,
    /// Padding strategy (Valid or Same)
    padding: PaddingType,
    /// 3D weight tensor with shape \[kernel_size, channels, depth_multiplier\]
    weights: Array3<f32>,
    /// 1D bias vector with shape \[channels * depth_multiplier\]
    bias: Array1<f32>,
    /// Activation applied to the convolution output
    activation: Activation,
    /// Cached post-activation output for the backward pass
    output_cache: Option<Tensor>,
    /// Cached input tensor for the backward pass
    input_cache: Option<Tensor>,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Gradients with respect to weights
    weight_gradients: Option<Array3<f32>>,
    /// Gradients with respect to bias
    bias_gradients: Option<Array1<f32>>,
}

impl DepthwiseConv1D {
    /// Creates a new DepthwiseConv1D layer
    ///
    /// # Parameters
    ///
    /// - `kernel_size` - Size of the convolution kernel along the length axis
    /// - `input_shape` - Shape of the input tensor as \[batch_size, length, channels\]
    /// - `stride` - Stride of the convolution along the length axis
    /// - `activation` - Activation function applied to the output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `DepthwiseConv1D` instance with Xavier-initialized
    ///   weights, or an error
    ///
    /// # Notes
    ///
    /// There is no `filters` argument. A depthwise convolution derives its output channel count
    /// from the input, as `channels * depth_multiplier`. `depth_multiplier` defaults to 1. Set it
    /// with [`DepthwiseConv1D::with_depth_multiplier`]. Padding defaults to
    /// [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`DepthwiseConv1D::with_padding`]. The layer seeds weights from the global seed or entropy
    /// by default. For reproducible initialization, set a seed with
    /// [`DepthwiseConv1D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `kernel_size` or `stride` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    /// - `Error::InvalidInput` - If `input_shape` is not 3D, has 0 channels, or is shorter
    ///   than the kernel
    pub fn new(
        kernel_size: usize,
        input_shape: Vec<usize>,
        stride: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_kernel_size_1d(kernel_size)?;
        validate_strides_1d(stride)?;
        validate_input_shape_1d(&input_shape, kernel_size)?;
        let activation = activation.into();
        activation.validate()?;

        let channels = input_shape[2];
        let weights = Self::init_weights_array(channels, 1, kernel_size, None);
        let bias = Array1::zeros(channels);

        Ok(Self {
            channels,
            depth_multiplier: 1,
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

    /// Xavier/Glorot uniform initialization of the \[kernel_size, channels, depth_multiplier\]
    /// tensor
    fn init_weights_array(
        channels: usize,
        depth_multiplier: usize,
        kernel_size: usize,
        random_state: Option<u64>,
    ) -> Array3<f32> {
        // Keras' `compute_fans` reads only the kernel tensor's last 2 axes. For shape
        // [kernel_size, channels, depth_multiplier] this gives `fan_in = channels * kernel_size`
        // and `fan_out = depth_multiplier * kernel_size`. A depthwise unit sees only 1 input
        // channel. This fan_in is therefore `channels` times the true receptive field. The
        // resulting bound is narrower by about sqrt(channels) than a per-channel count would
        // give. This matches `Conv1D` and Keras rather than a depthwise-specific formula
        let fan_in = channels * kernel_size;
        let fan_out = depth_multiplier * kernel_size;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array3::random_using(
            (kernel_size, channels, depth_multiplier),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
            &mut rng,
        )
    }

    /// Calculates the output length after convolution
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
    /// - `weights` - 3D weight tensor with shape
    ///   \[kernel_size, channels, depth_multiplier\]
    /// - `bias` - 1D bias vector with shape \[channels * depth_multiplier\]
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match
    ///   the existing shape
    pub fn set_weights(&mut self, weights: Array3<f32>, bias: Array1<f32>) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }

    /// The layer's geometry for a given input, as the shared kernel wants it
    ///
    /// The shared kernel names 2 spatial axes, so this fixes the height at 1 and puts the length
    /// on the width axis. A `[batch, length, channels]` tensor and a
    /// `[kernel_size, channels, depth_multiplier]` weight already hold the values in that order,
    /// so neither one needs a copy
    fn geometry(&self, input_shape: &[usize]) -> DepthwiseGeometry {
        let length = input_shape[1];
        let out_length = self.calculate_output_length(length);
        let pad = match self.padding {
            PaddingType::Valid => 0,
            PaddingType::Same => {
                ((out_length - 1) * self.stride + self.kernel_size).saturating_sub(length)
            }
        };
        DepthwiseGeometry {
            input: (1, length),
            output: (1, out_length),
            channels: self.channels,
            depth_multiplier: self.depth_multiplier,
            kernel: (1, self.kernel_size),
            strides: (1, self.stride),
            pad_before: (0, pad / 2),
        }
    }

    /// Depthwise convolution over a channels-last sequence, followed by the activation
    ///
    /// Shared numeric body of [`Layer::forward`] and [`Layer::predict`]. `forward` wraps this and
    /// records the input/output caches. `predict` returns the result directly
    fn convolve(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }
        let channels = input.shape()[2];
        if channels != self.channels {
            return Err(Error::dimension_mismatch(self.channels, channels));
        }

        let g = self.geometry(input.shape());
        let batch_size = input.shape()[0];

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self.weights.as_slice().expect("weights must be contiguous");
        let bias = Some(self.bias.as_slice().expect("bias must be contiguous"));

        let mut output = Array3::<f32>::zeros((batch_size, g.output.1, g.out_channels()));
        depthwise_forward(
            &g,
            src,
            ker,
            bias,
            output.as_slice_mut().expect("output is contiguous"),
        );

        self.activation.forward(&output.into_dyn())
    }
}

impl Layer for DepthwiseConv1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let activated = self.convolve(input)?;
        // Cache only after a successful convolution, so a rejected input leaves no partial state
        self.input_cache = Some(input.clone());
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
            .ok_or_else(|| Error::forward_pass_not_run("DepthwiseConv1D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("DepthwiseConv1D"))?;

        let batch_size = input.shape()[0];
        let g = self.geometry(input.shape());

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let grad_std = grad_upstream.as_standard_layout();
        let grad = grad_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self.weights.as_slice().expect("weights must be contiguous");

        let grads = depthwise_backward(&g, src, grad, ker, batch_size);

        self.weight_gradients = Some(
            Array3::from_shape_vec(self.weights.raw_dim(), grads.weight)
                .expect("weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(Array1::from_vec(grads.bias));

        Ok(
            Array3::from_shape_vec((batch_size, g.input.1, g.channels), grads.input)
                .expect("input gradient shape matches input")
                .into_dyn(),
        )
    }

    fn layer_type(&self) -> &str {
        "DepthwiseConv1D"
    }

    fn output_shape(&self) -> String {
        let output_length = self.calculate_output_length(self.input_shape[1]);
        format!(
            "({}, {}, {})",
            self.input_shape[0],
            output_length,
            self.channels * self.depth_multiplier
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
        LayerWeight::DepthwiseConv1D(DepthwiseConv1DLayerWeight {
            weight: Cow::Borrowed(&self.weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}

/// Unit tests for `DepthwiseConv1D`
#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::activation::linear::Linear;
    use ndarray::ArrayD;

    /// Channels never mix. The values match a hand-worked cross-correlation
    ///
    /// Channel 0 carries 1..5 against an all-ones kernel. Channel 1 carries all ones against an
    /// all-twos kernel. Channel 1's output is therefore the same constant at every position,
    /// while channel 0's output varies. This only holds if the 2 kernels stay on their own
    /// channels
    #[test]
    fn depthwise_1d_forward_keeps_channels_separate_hand_derived() {
        let mut layer = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();
        // [k, C, dm] with dm == 1: channel 0 all ones, channel 1 all twos
        let weights = Array3::from_shape_vec((2, 2, 1), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
        layer.set_weights(weights, Array1::zeros(2)).unwrap();

        // [1, 5, 2]: channel 0 is 1..5, channel 1 is all ones
        let input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 5, 2]),
            vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0],
        )
        .unwrap();

        let out = layer.predict(&input).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        // Channel 0: the 4 window sums from the width-2 kernel. Channel 1: 2 * 2 ones everywhere
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![3.0, 4.0, 5.0, 4.0, 7.0, 4.0, 9.0, 4.0]
        );
    }

    /// With `depth_multiplier`, input channel `c`'s multiplier `m` lands at output channel
    /// `c * depth_multiplier + m`, the Keras ordering
    #[test]
    fn depthwise_1d_depth_multiplier_output_channel_order() {
        let mut layer = DepthwiseConv1D::new(1, vec![1, 1, 2], 1, Linear::new())
            .unwrap()
            .with_depth_multiplier(2)
            .unwrap();
        assert_eq!(layer.weights.shape(), &[1, 2, 2]);

        // [k=1, C=2, dm=2] flat as (c, m): c0 -> [1, 10], c1 -> [100, 1000]
        let weights = Array3::from_shape_vec((1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
        layer.set_weights(weights, Array1::zeros(4)).unwrap();

        // 1 position holding [2, 3]
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2]), vec![2.0, 3.0]).unwrap();

        let out = layer.predict(&input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4]);
        // [c0m0, c0m1, c1m0, c1m1] = [2*1, 2*10, 3*100, 3*1000]
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 20.0, 300.0, 3000.0]
        );
    }
}
