//! 2D depthwise convolution layer that gives each input channel its own kernel or kernels

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, depthwise_backward, depthwise_forward,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::effective_kernel;
use crate::neural_network::layers::convolution::validation::{
    valid_output_size, validate_depth_multiplier, validate_dilation, validate_input_shape_2d,
    validate_kernel_size_2d, validate_strides_2d, validate_valid_kernel_fits,
};
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::shape_helpers::calculate_output_shape_2d;
use crate::neural_network::layers::validation::{validate_optional_weight, validate_weight_shape};
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::neural_network::{Fans, Initializer, Shape, Tensor};
use ndarray::{Array1, Array4};

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
    /// Tap spacing of the kernel as (height, width). 1 gives a solid kernel on that axis
    dilation_rate: (usize, usize),
    /// Padding strategy (Valid or Same)
    padding: PaddingType,
    /// 4D weight tensor with shape \[kernel_height, kernel_width, channels, depth_multiplier\]
    weights: Array4<f32>,
    /// 1D bias vector with shape \[channels * depth_multiplier\]
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The forward pass adds nothing, `weights` hides the array, and `parameters` never yields
    /// it, so a bias-free layer holds it and no more
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
    /// Whether the layer adds a bias to the convolution output
    use_bias: bool,
}

impl DepthwiseConv2D {
    /// Creates a new DepthwiseConv2D layer
    ///
    /// # Parameters
    ///
    /// - `kernel_size` - Size of the convolution kernel as (height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, height, width, channels\]
    /// - `strides` - Stride of the convolution as (height_stride, width_stride)
    /// - `activation` - Activation function applied to the output
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
    /// [`DepthwiseConv2D::with_padding`]. The kernel is solid by default. Space its taps out
    /// with [`DepthwiseConv2D::with_dilation_rate`]. The layer seeds weights from the global seed
    /// or entropy by default. For reproducible initialization, set a seed with
    /// [`DepthwiseConv2D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    /// - `Error::InvalidInput` - If `input_shape` is not 4D or has 0 channels
    pub fn new(
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_input_shape_2d(&input_shape)?;
        let activation = activation.into();
        activation.validate()?;

        let channels = input_shape[3];
        let weights = Self::init_weights_array(channels, 1, kernel_size, None);
        let bias = Array1::zeros(channels);

        Ok(Self {
            channels,
            depth_multiplier: 1,
            kernel_size,
            strides,
            dilation_rate: (1, 1),
            padding: PaddingType::Valid,
            weights,
            bias,
            activation,
            output_cache: None,
            input: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
            use_bias: true,
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

    /// Sets the tap spacing of the kernel (defaults to `(1, 1)`)
    ///
    /// A dilation of `d` on an axis spaces the kernel taps `d` cells apart, so `k` taps span
    /// `(k - 1) * d + 1` input cells of that axis. The window still advances by the stride. A
    /// dilation of 1 on both axes gives a solid kernel and the same result as before
    ///
    /// # Parameters
    ///
    /// - `dilation_rate` - Tap spacing as (height, width)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
    ///
    /// # Notes
    ///
    /// A depthwise convolution takes a stride above 1 and a dilation above 1 together. Only the
    /// plain and the transposed convolutions reject that pair
    ///
    /// The effective kernel is not bounded by the input axis here. Only [`PaddingType::Valid`]
    /// needs it to fit, and the forward pass applies that rule
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If either dilation is 0
    pub fn with_dilation_rate(mut self, dilation_rate: (usize, usize)) -> Result<Self, Error> {
        let dilation = [dilation_rate.0, dilation_rate.1];
        validate_dilation(&dilation)?;
        self.dilation_rate = dilation_rate;
        Ok(self)
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
    ///
    /// The depth multiplier takes the place of the filter count in the fan pair, so `fan_in`
    /// counts every input channel although a depthwise unit reads only 1 of them. See
    /// [`Fans::conv`]
    fn init_weights_array(
        channels: usize,
        depth_multiplier: usize,
        kernel_size: (usize, usize),
        random_state: Option<u64>,
    ) -> Array4<f32> {
        let (kernel_height, kernel_width) = kernel_size;
        let mut rng = crate::random::make_rng(random_state);
        Initializer::GlorotUniform.draw(
            (kernel_height, kernel_width, channels, depth_multiplier),
            Fans::conv(channels, depth_multiplier, kernel_height * kernel_width),
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
                let (keff_h, keff_w) = self.effective_kernel_size();
                let pad_h =
                    ((output_height - 1) * self.strides.0 + keff_h).saturating_sub(input_height);
                let pad_w =
                    ((output_width - 1) * self.strides.1 + keff_w).saturating_sub(input_width);
                (pad_h, pad_w)
            }
        }
    }

    /// Sets whether the layer adds a bias to the convolution output (defaults to `true`)
    ///
    /// With `use_bias` set to false the layer holds the kernel alone: `param_count` counts the
    /// kernel alone, `parameters` yields the kernel alone, and a checkpoint of the layer holds
    /// 1 array under the path `<position>.kernel`. A checkpoint written by a layer that has a
    /// bias therefore fails to load into a layer that has none, and the refusal names the path
    ///
    /// # Parameters
    ///
    /// - `use_bias` - `true` to add a bias, `false` to leave it out
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        if !use_bias {
            // Drop any gradient a previous backward pass left, so the bias cannot reach
            // `parameters` after the layer stops holding it
            self.bias_gradients = None;
        }
        self
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - 4D weight tensor with shape
    ///   \[kernel_height, kernel_width, channels, depth_multiplier\]
    /// - `bias` - 1D bias vector with shape \[channels * depth_multiplier\], or `None`
    ///   for a layer built with [`with_use_bias(false)`](Self::with_use_bias)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match
    ///   the existing shape
    /// - `Error::InvalidParameter` - If a bias is given to a layer that holds none, or none
    ///   is given to a layer that holds one
    pub fn set_weights(
        &mut self,
        weights: Array4<f32>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Result<(), Error> {
        validate_weight_shape("kernel", self.weights.shape(), weights.shape())?;
        let bias = validate_optional_weight("bias", "use_bias", self.use_bias, bias.into())?;
        if let Some(bias) = bias.as_ref() {
            validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        }
        self.weights = weights;
        if let Some(bias) = bias {
            self.bias = bias;
        }
        Ok(())
    }

    /// The layer's geometry for a given input, as the shared kernel wants it
    fn geometry(&self, input_shape: &[usize]) -> DepthwiseGeometry {
        let (height, width) = (input_shape[1], input_shape[2]);
        let output_shape = calculate_output_shape_2d(
            input_shape,
            self.effective_kernel_size(),
            self.strides,
            &self.padding,
        );
        let (out_height, out_width) = (output_shape[1], output_shape[2]);
        let (pad_h, pad_w) = self.calculate_padding(height, width, out_height, out_width);
        DepthwiseGeometry {
            input: (height, width),
            output: (out_height, out_width),
            channels: self.channels,
            depth_multiplier: self.depth_multiplier,
            kernel: self.kernel_size,
            strides: self.strides,
            dilation: self.dilation_rate,
            pad_before: (pad_h / 2, pad_w / 2),
        }
    }

    /// The extent the dilated taps span on each axis
    ///
    /// Every output-size and padding rule reads this rather than the kernel size. The 2 agree at
    /// a dilation of 1
    fn effective_kernel_size(&self) -> (usize, usize) {
        (
            effective_kernel(self.kernel_size.0, self.dilation_rate.0),
            effective_kernel(self.kernel_size.1, self.dilation_rate.1),
        )
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
        validate_valid_kernel_fits(
            self.padding.into(),
            &[self.kernel_size.0, self.kernel_size.1],
            &[self.dilation_rate.0, self.dilation_rate.1],
            &input.shape()[1..3],
        )?;

        let g = self.geometry(input.shape());
        let batch_size = input.shape()[0];
        let out_channels = g.out_channels();

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self.weights.as_slice().expect("weights must be contiguous");
        let bias = self
            .use_bias
            .then(|| self.bias.as_slice().expect("bias must be contiguous"));

        let mut output = Array4::<f32>::zeros((batch_size, g.output.0, g.output.1, out_channels));
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
            Array4::from_shape_vec(self.weights.raw_dim(), grads.weight)
                .expect("weight gradient shape matches weights"),
        );
        // A bias-free layer keeps no bias gradient, so `parameters` yields none and no
        // optimizer state is ever keyed on a bias that the layer does not hold
        self.bias_gradients = self.use_bias.then(|| Array1::from_vec(grads.bias));

        Ok(
            Array4::from_shape_vec((batch_size, g.input.0, g.input.1, g.channels), grads.input)
                .expect("input gradient shape matches input")
                .into_dyn(),
        )
    }

    fn layer_type(&self) -> &str {
        "DepthwiseConv2D"
    }

    fn known_input_shape(&self) -> Option<Shape> {
        (!self.input_shape.is_empty()).then(|| Shape::known(&self.input_shape))
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("DepthwiseConv2D", 4)?;
        let (batch, tail) = input.split_batch("DepthwiseConv2D")?;
        let (height, width) = (tail[0], tail[1]);
        if tail[2] != self.channels {
            return Err(Error::dimension_mismatch(self.channels, tail[2]));
        }
        let (keff_h, keff_w) = self.effective_kernel_size();
        // A `Valid` layer whose effective kernel is longer than an input axis is legal until the
        // forward pass rejects it. `valid_output_size` reports 0 positions there instead of
        // subtracting past 0, which `calculate_output_shape_2d` would do
        let (out_height, out_width) = match self.padding {
            PaddingType::Valid => (
                valid_output_size(height, keff_h, self.strides.0),
                valid_output_size(width, keff_w, self.strides.1),
            ),
            PaddingType::Same => (
                height.div_ceil(self.strides.0),
                width.div_ceil(self.strides.1),
            ),
        };
        // A depthwise convolution emits `channels * depth_multiplier` channels
        Ok(Shape::from_batch(
            batch,
            &[out_height, out_width, self.channels * self.depth_multiplier],
        ))
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so dropping the
        // bias corrects the count with no second formula to keep in step
        let bias = if self.use_bias { self.bias.len() } else { 0 };
        ParamCounts::trainable(self.weights.len() + bias)
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

    named_weight_layer_functions!(
        trainable "kernel" => weights,
        trainable "bias" => bias if use_bias,
    );
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
