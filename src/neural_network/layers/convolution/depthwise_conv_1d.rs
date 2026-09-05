//! 1D depthwise convolution layer that gives each input channel its own kernel or kernels

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, depthwise_backward, depthwise_forward,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::effective_kernel;
use crate::neural_network::layers::convolution::validation::{
    valid_output_size, validate_depth_multiplier, validate_dilation, validate_input_shape_1d,
    validate_kernel_size_1d, validate_strides_1d, validate_valid_kernel_fits,
};
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Fans, Initializer, Shape, Tensor};
use ndarray::{Array1, Array3};

/// A 1D depthwise convolutional layer
///
/// Applies its own kernel to each input channel of a sequence. This lowers the parameter count
/// and the computation compared to a standard convolution. It still extracts features along the
/// length axis per channel. Input shape is \[batch_size, length, channels\] and output shape is
/// \[batch_size, out_length, channels * depth_multiplier\]. No kernel mixes channels
///
/// `depth_multiplier` (default 1, set with [`DepthwiseConv1D::with_depth_multiplier`]) is how many
/// kernels each input channel gets. The output channel for input channel `c` and multiplier index
/// `m` is `c * depth_multiplier + m`
///
/// This is the 1D form of
/// [`DepthwiseConv2D`](crate::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D).
/// Both layers run the same shared kernel, because a sequence is a height-1 image
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array3;
///
/// // Input data shape: [batch_size, length, channels]
/// let input = Array3::<f32>::ones((2, 10, 4)).into_dyn();
///
/// // DepthwiseConv1D with ReLU. The build draws the weights
/// let depthwise_layer = DepthwiseConv1D::new(
///     3,                       // kernel_size
///     1,                       // stride
///     Activation::ReLU,        // activation
/// ).unwrap();
///
/// let mut model = SequentialBuilder::new()
///     .add(depthwise_layer)
///     .build(&Shape::known(input.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// model.summary();
///
/// let output = model.predict(&input).unwrap();
///
/// // Input [2, 10, 4] with kernel 3, stride 1, and Valid padding gives output [2, 8, 4]
/// assert_eq!(output.shape(), &[2, 8, 4]);
/// ```
#[derive(Debug)]
pub struct DepthwiseConv1D {
    /// Number of input channels, which [`UnaryLayer::build`] reads from the input shape
    channels: usize,
    /// Kernels per input channel. The output carries `channels * depth_multiplier` of them
    depth_multiplier: usize,
    /// Size of the convolution kernel along the length axis
    kernel_size: usize,
    /// Stride of the convolution along the length axis
    stride: usize,
    /// Tap spacing of the kernel along the length axis. 1 gives a solid kernel
    dilation_rate: usize,
    /// Padding strategy (Valid or Same)
    padding: PaddingType,
    /// 3D weight tensor with shape \[kernel_size, channels, depth_multiplier\]
    weights: Array3<f32>,
    /// 1D bias vector with shape \[channels * depth_multiplier\]
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The forward pass adds nothing, `weights` hides the array, and `parameters` never yields
    /// it, so a bias-free layer holds it and no more
    bias: Array1<f32>,
    /// Activation applied to the convolution output
    activation: Activation,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Whether the layer adds a bias to the convolution output
    use_bias: bool,
}

impl DepthwiseConv1D {
    /// Creates a new DepthwiseConv1D layer
    ///
    /// # Parameters
    ///
    /// - `kernel_size` - Size of the convolution kernel along the length axis
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
    /// with [`DepthwiseConv1D::with_depth_multiplier`].
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`DepthwiseConv1D::with_padding`]. The kernel is solid by default. Space its taps out
    /// with [`DepthwiseConv1D::with_dilation_rate`].
    ///
    /// The layer seeds weights from the global seed or entropy by default. For reproducible
    /// initialization, set a seed with [`DepthwiseConv1D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `kernel_size` or `stride` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        kernel_size: usize,
        stride: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_kernel_size_1d(kernel_size)?;
        validate_strides_1d(stride)?;
        let activation = activation.into();
        activation.validate()?;

        Ok(Self {
            channels: 0,
            depth_multiplier: 1,
            kernel_size,
            stride,
            dilation_rate: 1,
            padding: PaddingType::Valid,
            weights: Array3::zeros((0, 0, 0)),
            bias: Array1::zeros(0),
            activation,
            built: None,
            random_state: None,
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
    /// A depthwise convolution takes a stride above 1 and a dilation above 1 together. Only the
    /// plain and the transposed convolutions reject that pair
    ///
    /// The effective kernel is not bounded by the input length here. Only [`PaddingType::Valid`]
    /// needs it to fit, and the build applies that rule
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `dilation_rate` is 0
    pub fn with_dilation_rate(mut self, dilation_rate: usize) -> Result<Self, Error> {
        validate_dilation(&[dilation_rate])?;
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
        if self.built.is_some() {
            self.draw_parameters();
        }
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
        self.random_state = Some(random_state);
        if self.built.is_some() {
            self.draw_parameters();
        }
        self
    }

    /// Xavier/Glorot uniform initialization of the \[kernel_size, channels, depth_multiplier\]
    /// tensor
    ///
    /// The depth multiplier takes the place of the filter count in the fan pair, so `fan_in`
    /// counts every input channel although a depthwise unit reads only 1 of them. See
    /// [`Fans::conv`]
    fn draw_parameters(&mut self) {
        let mut rng = crate::random::make_rng(self.random_state);
        // The depth multiplier takes the place of the filter count. See `Fans::conv`
        self.weights = Initializer::GlorotUniform.draw(
            (self.kernel_size, self.channels, self.depth_multiplier),
            Fans::conv(self.channels, self.depth_multiplier, self.kernel_size),
            &mut rng,
        );
        self.bias = Array1::zeros(self.channels * self.depth_multiplier);
    }

    /// Calculates the output length after convolution
    ///
    /// The `Valid` rule reads the extent the dilated taps span, not the tap count, and it refuses
    /// an extent longer than the input length
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the padding is `Valid` and the effective kernel is longer than
    ///   the input length
    fn calculate_output_length(&self, input_length: usize) -> Result<usize, Error> {
        match self.padding {
            PaddingType::Valid => valid_output_size(
                "DepthwiseConv1D",
                "length",
                input_length,
                effective_kernel(self.kernel_size, self.dilation_rate),
                self.stride,
            ),
            PaddingType::Same => Ok(input_length.div_ceil(self.stride)),
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
        self
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - 3D weight tensor with shape
    ///   \[kernel_size, channels, depth_multiplier\]
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
        weights: Array3<f32>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("DepthwiseConv1D"));
        }
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
    ///
    /// The shared kernel names 2 spatial axes, so this fixes the height at 1 and puts the length
    /// on the width axis. A `[batch, length, channels]` tensor and a
    /// `[kernel_size, channels, depth_multiplier]` weight already hold the values in that order,
    /// so neither one needs a copy
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the padding is `Valid` and the effective kernel is longer than
    ///   the input length. The build applies the same rule, so a built layer never meets it
    fn geometry(&self, input_shape: &[usize]) -> Result<DepthwiseGeometry, Error> {
        let length = input_shape[1];
        let out_length = self.calculate_output_length(length)?;
        let keff = effective_kernel(self.kernel_size, self.dilation_rate);
        let pad = match self.padding {
            PaddingType::Valid => 0,
            PaddingType::Same => ((out_length - 1) * self.stride + keff).saturating_sub(length),
        };
        Ok(DepthwiseGeometry {
            input: (1, length),
            output: (1, out_length),
            channels: self.channels,
            depth_multiplier: self.depth_multiplier,
            kernel: (1, self.kernel_size),
            strides: (1, self.stride),
            // The height axis is the placeholder axis, so it stays solid at 1
            dilation: (1, self.dilation_rate),
            pad_before: (0, pad / 2),
        })
    }

    /// Depthwise convolution over a channels-last sequence, followed by the activation
    ///
    /// The numeric body of [`UnaryLayer::forward`]. A training pass parks the input and the
    /// output of this call in the context, and an inference pass parks nothing
    fn convolve(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "DepthwiseConv1D", input.shape())?;
        validate_valid_kernel_fits(
            self.padding.into(),
            &[self.kernel_size],
            &[self.dilation_rate],
            &input.shape()[1..2],
        )?;

        let g = self.geometry(input.shape())?;
        let batch_size = input.shape()[0];

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self.weights.as_slice().expect("weights must be contiguous");
        let bias = self
            .use_bias
            .then(|| self.bias.as_slice().expect("bias must be contiguous"));

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

/// What the forward pass of [`DepthwiseConv1D`] parks for its backward pass
struct DepthwiseConv1DCache {
    /// The tensor the forward pass received
    input: Tensor,
    /// The activated output, to backpropagate through the activation
    output: Tensor,
}

impl LayerBase for DepthwiseConv1D {
    fn layer_type(&self) -> &str {
        "DepthwiseConv1D"
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so dropping the
        // bias corrects the count with no second formula to keep in step
        let bias = if self.use_bias { self.bias.len() } else { 0 };
        ParamCounts::trainable(self.weights.len() + bias)
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let Self {
            weights,
            bias,
            use_bias,
            ..
        } = self;
        let mut params = vec![ParamRef::weight(
            "kernel",
            weights.as_slice_mut().expect("weights must be contiguous"),
        )];
        if *use_bias {
            params.push(ParamRef::no_decay(
                "bias",
                bias.as_slice_mut().expect("bias must be contiguous"),
            ));
        }
        params
    }

    built_layer_shape_functions!();

    named_weight_layer_functions!(
        trainable "kernel" => weights,
        trainable "bias" => bias if use_bias,
    );
}

impl UnaryLayer for DepthwiseConv1D {
    /// Reads the channel count from the input shape, and draws the kernel and the bias
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "DepthwiseConv1D", input)? else {
            return Ok(());
        };
        built.check_rank("DepthwiseConv1D", 3)?;
        let (batch, tail) = built.split_batch("DepthwiseConv1D")?;
        // The family validators read a full extent list, and the batch extent is not part of
        // what they check
        let mut dims = vec![batch.unwrap_or(1)];
        dims.extend(tail);
        validate_input_shape_1d(&dims)?;
        // The shape algebra holds every rule the geometry has, so a stack that cannot run is
        // refused here, before the layer draws a single weight
        self.compute_output_shape(&built)?;
        self.channels = dims[2];
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            return Err(Error::not_built("DepthwiseConv1D"));
        }
        let activated = self.convolve(input)?;
        // Park only after a successful convolution, so a rejected input leaves no partial state
        if ctx.is_training() {
            ctx.push_cache(
                "DepthwiseConv1D",
                DepthwiseConv1DCache {
                    input: input.clone(),
                    output: activated.clone(),
                },
            );
        }
        Ok(activated)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: DepthwiseConv1DCache = ctx.pop_cache("DepthwiseConv1D")?;
        let grad_upstream = self.activation.backward(&cache.output, grad_output)?;

        let input = &cache.input;

        let batch_size = input.shape()[0];
        let g = self.geometry(input.shape())?;

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

        ctx.add_grad(
            "kernel",
            Array3::from_shape_vec(self.weights.raw_dim(), grads.weight)
                .expect("weight gradient shape matches weights")
                .into_dyn(),
        )?;
        // A bias-free layer writes no bias gradient, so the store holds none and no
        // optimizer state is ever keyed on a bias that the layer does not hold
        if self.use_bias {
            ctx.add_grad("bias", Array1::from_vec(grads.bias).into_dyn())?;
        }

        Ok(
            Array3::from_shape_vec((batch_size, g.input.1, g.channels), grads.input)
                .expect("input gradient shape matches input")
                .into_dyn(),
        )
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("DepthwiseConv1D", 3)?;
        let (batch, tail) = input.split_batch("DepthwiseConv1D")?;
        let channels = tail[1];
        // The layer keeps 1 kernel group per channel of the shape it was built for, so a built
        // layer refuses any other channel count. An unbuilt layer holds no group yet, and the
        // answer needs the channel count of the argument alone
        if self.built.is_some() && channels != self.channels {
            return Err(Error::dimension_mismatch(self.channels, channels));
        }
        // A depthwise convolution emits `channels * depth_multiplier` channels
        Ok(Shape::from_batch(
            batch,
            &[
                self.calculate_output_length(tail[0])?,
                channels * self.depth_multiplier,
            ],
        ))
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
        let mut layer = DepthwiseConv1D::new(2, 1, Linear::new()).unwrap();
        layer.build(&Shape::known(&[1, 5, 2])).unwrap();
        // [k, C, dm] with dm == 1: channel 0 all ones, channel 1 all twos
        let weights = Array3::from_shape_vec((2, 2, 1), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
        layer.set_weights(weights, Array1::zeros(2)).unwrap();

        // [1, 5, 2]: channel 0 is 1..5, channel 1 is all ones
        let input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 5, 2]),
            vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0],
        )
        .unwrap();

        let out = layer.forward(&input, &mut Ctx::inference()).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        // Channel 0: the 4 window sums from the width-2 kernel. Channel 1: 2 * 2 ones everywhere
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![3.0, 4.0, 5.0, 4.0, 7.0, 4.0, 9.0, 4.0]
        );
    }

    /// With `depth_multiplier`, input channel `c`'s multiplier `m` lands at output channel
    /// `c * depth_multiplier + m`
    #[test]
    fn depthwise_1d_depth_multiplier_output_channel_order() {
        let mut layer = DepthwiseConv1D::new(1, 1, Linear::new())
            .unwrap()
            .with_depth_multiplier(2)
            .unwrap();
        layer.build(&Shape::known(&[1, 1, 2])).unwrap();
        assert_eq!(layer.weights.shape(), &[1, 2, 2]);

        // [k=1, C=2, dm=2] flat as (c, m): c0 -> [1, 10], c1 -> [100, 1000]
        let weights = Array3::from_shape_vec((1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
        layer.set_weights(weights, Array1::zeros(4)).unwrap();

        // 1 position holding [2, 3]
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2]), vec![2.0, 3.0]).unwrap();

        let out = layer.forward(&input, &mut Ctx::inference()).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4]);
        // [c0m0, c0m1, c1m0, c1m1] = [2*1, 2*10, 3*100, 3*1000]
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 20.0, 300.0, 3000.0]
        );
    }
}
