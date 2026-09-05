//! 2D depthwise separable convolution layer (depthwise stage followed by a pointwise 1x1 stage)

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, depthwise_backward, depthwise_forward,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{
    ConvPadding, conv_backward, conv_forward, effective_kernel,
};
use crate::neural_network::layers::convolution::validation::{
    valid_output_size, validate_depth_multiplier, validate_dilation, validate_filters,
    validate_input_shape_2d, validate_kernel_size_2d, validate_strides_2d,
    validate_valid_kernel_fits,
};
use crate::neural_network::layers::shape_helpers::calculate_output_height_and_weight;
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Fans, Initializer, Shape, Tensor};
use ndarray::{Array1, Array4};

/// A 2D separable convolutional layer
///
/// Runs a depthwise step followed by a pointwise step. This uses fewer parameters and less
/// computation than a standard convolution. Input shape is \[batch_size, height, width, channels\].
/// Intermediate depthwise output shape is
/// \[batch_size, height', width', channels * depth_multiplier\]. Final output shape is
/// \[batch_size, height', width', filters\]
///
/// The intermediate channel for input channel `c` and multiplier index `m` is
/// `c * depth_multiplier + m`, which is Keras' ordering. It is also the row order of the
/// pointwise weight `\[1, 1, channels * depth_multiplier, filters\]`. This lets the 2 stages
/// line up with no repacking between them
///
/// The separable convolution runs 2 steps:
/// 1. Depthwise convolution: convolves each input channel with its own set of filters
/// 2. Pointwise convolution: a 1x1 convolution that combines the depthwise outputs
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
///
/// // input tensor: [batch_size, height, width, channels]
/// let x = Array4::ones((2, 32, 32, 3)).into_dyn();
///
/// // target tensor
/// let y = Array4::ones((2, 32, 32, 64)).into_dyn();
///
/// let mut model = SequentialBuilder::new()
///     .add(SeparableConv2D::new(
///         64,                          // filters
///         (3, 3),                      // kernel_size
///         (1, 1),                      // strides
///         1,                           // depth_multiplier
///         Activation::ReLU,            // activation
///     ).unwrap().with_padding(PaddingType::Same))
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Tap spacing of the depthwise kernel as (vertical, horizontal). 1 gives a solid axis
    dilation_rate: (usize, usize),
    /// Padding applied to the spatial dimensions (`Valid` or `Same`)
    padding: PaddingType,
    /// Number of depthwise filters per input channel
    depth_multiplier: usize,
    /// Depthwise filters with shape \[kernel_height, kernel_width, channels, depth_multiplier\]
    depthwise_weights: Array4<f32>,
    /// Pointwise filters with shape \[1, 1, channels * depth_multiplier, filters\]
    pointwise_weights: Array4<f32>,
    /// Bias vector with shape \[filters\]
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The pointwise stage adds nothing, `weights` hides the array, and `parameters` never
    /// yields it, so a bias-free layer holds it and no more
    bias: Array1<f32>,
    /// Activation applied to the layer output
    activation: Activation,
    /// Number of input channels, which [`UnaryLayer::build`] reads from the input shape
    channels: usize,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Whether the layer adds a bias to the pointwise output
    use_bias: bool,
}

impl SeparableConv2D {
    /// Creates a new 2D separable convolutional layer
    ///
    /// The layer initializes weights with Xavier (Glorot) uniform initialization. Biases start
    /// at 0
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output channels from the pointwise convolution
    /// - `kernel_size` - Size of the depthwise convolution kernel as (height, width)
    /// - `strides` - Stride values for the convolution as (vertical, horizontal)
    /// - `depth_multiplier` - Number of depthwise convolution filters per input channel
    /// - `activation` - Activation applied to the output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `SeparableConv2D` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`SeparableConv2D::with_padding`]. The depthwise kernel is solid by default. Space its
    /// taps out with [`SeparableConv2D::with_dilation_rate`].
    ///
    /// The layer seeds weights from the global seed or entropy
    /// by default. For reproducible initialization, set a seed with
    /// [`SeparableConv2D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidParameter` - If `depth_multiplier` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        depth_multiplier: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_depth_multiplier(depth_multiplier)?;
        let activation = activation.into();
        activation.validate()?;

        Ok(SeparableConv2D {
            filters,
            kernel_size,
            strides,
            dilation_rate: (1, 1),
            padding: PaddingType::Valid,
            depth_multiplier,
            depthwise_weights: Array4::zeros((0, 0, 0, 0)),
            pointwise_weights: Array4::zeros((0, 0, 0, 0)),
            bias: Array1::zeros(0),
            activation,
            channels: 0,
            built: None,
            random_state: None,
            use_bias: true,
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

    /// Sets the tap spacing of the depthwise kernel (defaults to `(1, 1)`)
    ///
    /// A dilation of `d` on an axis spaces the depthwise taps `d` cells apart, so `k` taps span
    /// `(k - 1) * d + 1` input cells of that axis. The window still advances by the stride. The
    /// pointwise stage reads 1 tap, so no dilation can reach it. A dilation of 1 on both axes
    /// gives a solid kernel and the same result as before
    ///
    /// # Parameters
    ///
    /// - `dilation_rate` - Tap spacing as (vertical, horizontal)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
    ///
    /// # Notes
    ///
    /// A separable convolution takes a stride above 1 and a dilation above 1 together. Only the
    /// plain and the transposed convolutions reject that pair
    ///
    /// The effective kernel is not bounded by the input axis here. Only [`PaddingType::Valid`]
    /// needs it to fit, and the build applies that rule
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

    /// The extent the dilated depthwise taps span on each axis
    ///
    /// Every output-size and padding rule reads this rather than the kernel size. The 2 agree at
    /// a dilation of 1
    fn effective_kernel_size(&self) -> (usize, usize) {
        (
            effective_kernel(self.kernel_size.0, self.dilation_rate.0),
            effective_kernel(self.kernel_size.1, self.dilation_rate.1),
        )
    }

    /// Sets the seed for the depthwise and pointwise weights, and re-initializes them
    /// deterministically
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

    /// Xavier/Glorot uniform initialization of the depthwise and pointwise weight tensors
    ///
    /// Both draws share 1 RNG (threaded depthwise-then-pointwise) so a given seed reproduces the
    /// exact same pair of tensors
    fn draw_parameters(&mut self) {
        let (kernel_height, kernel_width) = self.kernel_size;
        let mut rng = crate::random::make_rng(self.random_state);

        // The depth multiplier takes the place of the filter count. See `Fans::conv`
        self.depthwise_weights = Initializer::GlorotUniform.draw(
            (
                kernel_height,
                kernel_width,
                self.channels,
                self.depth_multiplier,
            ),
            Fans::conv(
                self.channels,
                self.depth_multiplier,
                kernel_height * kernel_width,
            ),
            &mut rng,
        );

        // The stored width of the pointwise kernel is already the fan-in, and its 1 tap adds no
        // factor to the fan-out
        let width = self.channels * self.depth_multiplier;
        self.pointwise_weights = Initializer::GlorotUniform.draw(
            (1, 1, width, self.filters),
            Fans::new(width, self.filters),
            &mut rng,
        );

        self.bias = Array1::zeros(self.filters);
    }

    /// Calculates the output shape of the separable convolutional layer
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the padding is `Valid` and an effective kernel is longer than
    ///   the input axis it runs on
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, Error> {
        let batch_size = input_shape[0];
        let input_height = input_shape[1];
        let input_width = input_shape[2];

        let (keff_h, keff_w) = self.effective_kernel_size();
        // A `Valid` layer whose effective kernel is longer than an input axis is refused here,
        // rather than reported as an axis of 0 positions that every later layer would carry
        let (output_height, output_width) = match self.padding {
            PaddingType::Valid => (
                valid_output_size(
                    "SeparableConv2D",
                    "height",
                    input_height,
                    keff_h,
                    self.strides.0,
                )?,
                valid_output_size(
                    "SeparableConv2D",
                    "width",
                    input_width,
                    keff_w,
                    self.strides.1,
                )?,
            ),
            PaddingType::Same => (
                input_height.div_ceil(self.strides.0),
                input_width.div_ceil(self.strides.1),
            ),
        };

        Ok(vec![batch_size, output_height, output_width, self.filters])
    }

    /// The depthwise stage's geometry for a given input, as the shared kernel wants it
    fn depthwise_geometry(&self, input_shape: &[usize]) -> DepthwiseGeometry {
        let (height, width) = (input_shape[1], input_shape[2]);
        let depthwise_shape = self.calculate_depthwise_output_shape(input_shape);
        let (out_height, out_width) = (depthwise_shape[1], depthwise_shape[2]);
        let (pad_h, pad_w) = self.calculate_padding(height, width, out_height, out_width);
        DepthwiseGeometry {
            input: (height, width),
            output: (out_height, out_width),
            channels: input_shape[3],
            depth_multiplier: self.depth_multiplier,
            kernel: self.kernel_size,
            strides: self.strides,
            dilation: self.dilation_rate,
            pad_before: (pad_h / 2, pad_w / 2),
        }
    }

    /// Checks a runtime input against the rank, the channel count, and the padding rule the
    /// layer was built for
    ///
    /// The depthwise kernel is sized from the declared channel count, so an input carrying more
    /// channels would read past the end of it. This turns that into an error at the layer
    /// boundary. It also applies the `Valid` fit rule, because a `Valid` window that is longer
    /// than the input spatial gives no complete window
    fn validate_input(&self, input: &Tensor) -> Result<(), Error> {
        validate_built_input(&self.built, "SeparableConv2D", input.shape())?;
        validate_valid_kernel_fits(
            self.padding.into(),
            &[self.kernel_size.0, self.kernel_size.1],
            &[self.dilation_rate.0, self.dilation_rate.1],
            &input.shape()[1..3],
        )?;
        Ok(())
    }

    /// Performs the depthwise convolution stage
    ///
    /// Carries no bias and no activation. Both belong to the pointwise stage that follows, so
    /// this passes `None` for the bias to the shared kernel
    fn depthwise_convolve(&self, input: &Tensor) -> Tensor {
        let g = self.depthwise_geometry(input.shape());
        let batch_size = input.shape()[0];
        let out_channels = g.out_channels();

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self
            .depthwise_weights
            .as_slice()
            .expect("depthwise weights must be contiguous");

        let mut output = Array4::<f32>::zeros((batch_size, g.output.0, g.output.1, out_channels));
        depthwise_forward(
            &g,
            src,
            ker,
            None,
            output.as_slice_mut().expect("output is contiguous"),
        );

        output.into_dyn()
    }

    /// Performs the pointwise (1x1) convolution stage
    ///
    /// A 1x1 convolution is a per-position cross-channel matrix multiply. This delegates to the
    /// shared [`conv_forward`] engine (im2col + gemm) rather than a hand-rolled loop nest. The
    /// pointwise weights `[1, 1, C*dm, filters]` already match the engine's flat `[k..., Cin, F]`
    /// layout. The bias is already its per-filter `[F]` vector. The depthwise stage emits its
    /// channels in `c * depth_multiplier + m` order, which is exactly the row order the pointwise
    /// weight is indexed by. Nothing repacks the data between the stages
    fn pointwise_convolve(&self, input: &Tensor) -> Tensor {
        conv_forward(
            input,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            self.use_bias
                .then(|| self.bias.as_slice().expect("bias must be contiguous")),
            &[1, 1],
            &[1, 1],
            ConvPadding::Valid,
        )
        // A 1x1 kernel under Valid padding can never exceed the input (every spatial dim >= 1)
        .expect("1x1 pointwise convolution geometry is always valid")
    }

    /// Calculates the output shape after the depthwise convolution stage
    fn calculate_depthwise_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[1];
        let input_width = input_shape[2];
        let channels = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.effective_kernel_size(),
            self.strides,
        );

        vec![
            batch_size,
            output_height,
            output_width,
            channels * self.depth_multiplier,
        ]
    }

    /// Calculates the symmetric zero-padding (total height/width pad) for the depthwise stage
    ///
    /// Returns `(0, 0)` for `Valid` padding. For `Same`, returns the total padding along each
    /// spatial axis required so a stride-`s` convolution yields the given output size. The
    /// caller splits this total with `pad / 2` on the leading edge, matching the convolution
    /// engine
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

    /// Sets whether the layer adds a bias to the pointwise output (defaults to `true`)
    ///
    /// With `use_bias` set to false the layer holds the 2 kernels alone: `param_count` counts
    /// the 2 kernels, `parameters` yields the 2 kernels, and a checkpoint of the layer holds
    /// the paths `<position>.depthwise_kernel` and `<position>.pointwise_kernel`. A checkpoint
    /// written by a layer that has a bias therefore fails to load into a layer that has none,
    /// and the refusal names the path
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
    /// - `depthwise_weights` - 4D array for depthwise filters with shape
    ///   \[kernel_height, kernel_width, channels, depth_multiplier\]
    /// - `pointwise_weights` - 4D array for pointwise filters with shape
    ///   \[1, 1, channels * depth_multiplier, filters\]
    /// - `bias` - 1D bias vector with shape \[filters\], or `None` for a layer built with
    ///   [`with_use_bias(false)`](Self::with_use_bias)
    ///
    /// # Errors
    ///
    /// - `Error` - If any supplied array shape does not match the existing layer weights
    /// - `Error::InvalidParameter` - If a bias is given to a layer that holds none, or none
    ///   is given to a layer that holds one
    pub fn set_weights(
        &mut self,
        depthwise_weights: Array4<f32>,
        pointwise_weights: Array4<f32>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("SeparableConv2D"));
        }
        validate_weight_shape(
            "depthwise_kernel",
            self.depthwise_weights.shape(),
            depthwise_weights.shape(),
        )?;
        validate_weight_shape(
            "pointwise_kernel",
            self.pointwise_weights.shape(),
            pointwise_weights.shape(),
        )?;
        let bias = validate_optional_weight("bias", "use_bias", self.use_bias, bias.into())?;
        if let Some(bias) = bias.as_ref() {
            validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        }
        self.depthwise_weights = depthwise_weights;
        self.pointwise_weights = pointwise_weights;
        if let Some(bias) = bias {
            self.bias = bias;
        }
        Ok(())
    }
}

/// What the forward pass of [`SeparableConv2D`] parks for its backward pass
struct SeparableConv2DCache {
    /// The tensor the forward pass received
    input: Tensor,
    /// The output of the depthwise stage, which the pointwise backward pass reads
    depthwise_output: Tensor,
    /// The activated output, to backpropagate through the activation
    output: Tensor,
}

impl LayerBase for SeparableConv2D {
    fn layer_type(&self) -> &str {
        "SeparableConv2D"
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so dropping the
        // bias corrects the count with no second formula to keep in step
        let bias = if self.use_bias { self.bias.len() } else { 0 };
        ParamCounts::trainable(self.depthwise_weights.len() + self.pointwise_weights.len() + bias)
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let Self {
            depthwise_weights,
            pointwise_weights,
            bias,
            use_bias,
            ..
        } = self;
        let mut params = vec![
            ParamRef::weight(
                "depthwise_kernel",
                depthwise_weights
                    .as_slice_mut()
                    .expect("depthwise weights must be contiguous"),
            ),
            ParamRef::weight(
                "pointwise_kernel",
                pointwise_weights
                    .as_slice_mut()
                    .expect("pointwise weights must be contiguous"),
            ),
        ];
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
        trainable "depthwise_kernel" => depthwise_weights,
        trainable "pointwise_kernel" => pointwise_weights,
        trainable "bias" => bias if use_bias,
    );
}

impl UnaryLayer for SeparableConv2D {
    /// Reads the channel count from the input shape, and draws both kernels and the bias
    ///
    /// 1 generator threads the depthwise draw and then the pointwise draw, in that order. A
    /// second generator, or the other order, changes every value of the second draw
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "SeparableConv2D", input)? else {
            return Ok(());
        };
        built.check_rank("SeparableConv2D", 4)?;
        let (batch, tail) = built.split_batch("SeparableConv2D")?;
        // The family validators read a full extent list, and the batch extent is not part of
        // what they check
        let mut dims = vec![batch.unwrap_or(1)];
        dims.extend(tail);
        validate_input_shape_2d(&dims)?;
        // The shape algebra holds every rule the geometry has, so a stack that cannot run is
        // refused here, before the layer draws a single weight
        self.compute_output_shape(&built)?;
        self.channels = dims[3];
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            return Err(Error::not_built("SeparableConv2D"));
        }
        self.validate_input(input)?;

        // Depthwise convolution (each channel independently), then pointwise (1x1) to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        let activated = self.activation.forward(&output.into_dyn())?;

        // Park only after a successful pass, so a rejected input leaves no partial state. The
        // depthwise output reaches the backward pass alone
        if ctx.is_training() {
            ctx.push_cache(
                "SeparableConv2D",
                SeparableConv2DCache {
                    input: input.clone(),
                    depthwise_output,
                    output: activated.clone(),
                },
            );
        }
        Ok(activated)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: SeparableConv2DCache = ctx.pop_cache("SeparableConv2D")?;

        // Backward through the activation first
        let grad_upstream = self.activation.backward(&cache.output, grad_output)?;

        let input = &cache.input;
        let depthwise_output = &cache.depthwise_output;

        let batch_size = input.shape()[0];
        let g = self.depthwise_geometry(input.shape());

        // Pointwise (1x1) backward via the shared engine (im2col + gemm). Its input gradient is
        // the gradient with respect to the depthwise output, with shape [batch, H', W', C*dm]
        let pw_grads = conv_backward(
            &grad_upstream,
            depthwise_output,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            &[1, 1],
            &[1, 1],
            ConvPadding::Valid,
        )
        // 1x1 Valid geometry is always valid (see `pointwise_convolve`)
        .expect("1x1 pointwise convolution geometry is always valid");
        ctx.add_grad(
            "pointwise_kernel",
            Array4::from_shape_vec(self.pointwise_weights.raw_dim(), pw_grads.weight_grad)
                .expect("pointwise weight gradient shape matches weights")
                .into_dyn(),
        )?;
        // A bias-free layer writes no bias gradient, so the store holds none and no
        // optimizer state is ever keyed on a bias that the layer does not hold
        if self.use_bias {
            ctx.add_grad("bias", Array1::from_vec(pw_grads.bias_grad).into_dyn())?;
        }
        let depthwise_grad = pw_grads.input_grad;

        // Depthwise backward through the shared driver
        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let grad_std = depthwise_grad.as_standard_layout();
        let grad = grad_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self
            .depthwise_weights
            .as_slice()
            .expect("depthwise weights must be contiguous");

        // This stage has no bias of its own, so backward drops the bias field the shared driver
        // returns
        let dw_grads = depthwise_backward(&g, src, grad, ker, batch_size);

        ctx.add_grad(
            "depthwise_kernel",
            Array4::from_shape_vec(self.depthwise_weights.raw_dim(), dw_grads.weight)
                .expect("depthwise weight gradient shape matches weights")
                .into_dyn(),
        )?;

        Ok(Array4::from_shape_vec(
            (batch_size, g.input.0, g.input.1, g.channels),
            dw_grads.input,
        )
        .expect("input gradient shape matches input")
        .into_dyn())
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("SeparableConv2D", 4)?;
        let (batch, tail) = input.split_batch("SeparableConv2D")?;
        // `calculate_output_shape` reads the batch axis, so the list it takes starts with one
        let mut dims = vec![0];
        dims.extend(tail);
        Ok(Shape::from_batch(
            batch,
            &self.calculate_output_shape(&dims)?[1..],
        ))
    }
}

/// Unit tests for `SeparableConv2D`
#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::activation::linear::Linear;
    use ndarray::ArrayD;

    /// The 2 stages agree on the intermediate channel order
    ///
    /// A 1x1 kernel at a single spatial position reduces the layer to arithmetic a reader can
    /// write out by hand. The depthwise weights give each `(channel, multiplier)` pair a
    /// distinct power of 10. The pointwise weights give each intermediate channel a distinct
    /// power of 2. The single output value only comes out right if the depthwise stage emits its
    /// channels in `c * depth_multiplier + m` order. It also needs the pointwise weight rows
    /// indexed in that same order. Any transposition of either would change the total
    #[test]
    fn separable_stage_channel_order_hand_derived() {
        let mut layer = SeparableConv2D::new(1, (1, 1), (1, 1), 2, Linear::new()).unwrap();
        layer.build(&Shape::known(&[1, 1, 1, 2])).unwrap();
        assert_eq!(layer.depthwise_weights.shape(), &[1, 1, 2, 2]);
        assert_eq!(layer.pointwise_weights.shape(), &[1, 1, 4, 1]);

        // [kh, kw, C, dm] as (c, m): c0 -> [1, 10], c1 -> [100, 1000]
        let depthwise =
            Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
        // [1, 1, C*dm, F]: 1 distinct weight per intermediate channel
        let pointwise = Array4::from_shape_vec((1, 1, 4, 1), vec![1.0, 2.0, 4.0, 8.0]).unwrap();
        layer
            .set_weights(depthwise, pointwise, Array1::zeros(1))
            .unwrap();

        // 1 position holding [2, 3]
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 1, 2]), vec![2.0, 3.0]).unwrap();
        let out = layer.forward(&input, &mut Ctx::inference()).unwrap();

        // Intermediate = [2*1, 2*10, 3*100, 3*1000] = [2, 20, 300, 3000]
        // Output = 2*1 + 20*2 + 300*4 + 3000*8 = 25242
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.iter().copied().collect::<Vec<f32>>(), vec![25242.0]);
    }

    /// Spatially, the depthwise stage is an ordinary per-channel cross-correlation. The pointwise
    /// stage is a per-position channel mix. A 2x2 kernel over a 3x3 input reduces to the 4 window
    /// sums, scaled by the pointwise weight
    #[test]
    fn separable_spatial_pass_hand_derived() {
        let mut layer = SeparableConv2D::new(1, (2, 2), (1, 1), 1, Linear::new()).unwrap();
        layer.build(&Shape::known(&[1, 3, 3, 1])).unwrap();

        // Single channel, all-ones depthwise kernel. Pointwise scales by 3
        let depthwise = Array4::from_shape_vec((2, 2, 1, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let pointwise = Array4::from_shape_vec((1, 1, 1, 1), vec![3.0]).unwrap();
        layer
            .set_weights(depthwise, pointwise, Array1::zeros(1))
            .unwrap();

        // [1, 3, 3, 1] holding 1..9
        let input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 3, 1]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let out = layer.forward(&input, &mut Ctx::inference()).unwrap();

        // Window sums 12, 16, 24, 28, each tripled
        assert_eq!(out.shape(), &[1, 2, 2, 1]);
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![36.0, 48.0, 72.0, 84.0]
        );
    }
}
