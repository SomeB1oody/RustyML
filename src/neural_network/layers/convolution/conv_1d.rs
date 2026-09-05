//! 1D convolutional layer for sequential data such as time series, audio, or text

use crate::error::Error;
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
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Fans, Initializer, Shape, Tensor};
use ndarray::{Array1, Array3};

/// A 1D convolutional layer for neural networks
///
/// Applies a convolution operation to sequential data such as time series, audio signals,
/// or text. Input shape is \[batch_size, length, channels\] and output shape is
/// \[batch_size, output_length, filters\], where output_length depends on input length,
/// kernel size, stride, and padding
///
/// The dimension-generic convolution math lives in
/// [`convolution_engine`](crate::neural_network::layers::convolution). This layer holds the
/// weights and the activation, and delegates the forward/backward numerics to it.
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
/// // Create a simple 3D input tensor: [batch_size, length, channels]
/// // Batch size=2, 10 time steps, 1 input channel
/// let x = Array3::ones((2, 10, 1)).into_dyn();
///
/// // Create target tensor (output length 8, 3 filters)
/// let y = Array3::ones((2, 8, 3)).into_dyn();
///
/// // Build model: add a Conv1D layer with 3 filters and kernel size 3
/// let mut model = SequentialBuilder::new()
///     .add(Conv1D::new(
///         3,                      // Number of filters
///         3,                      // Kernel size
///         1,                      // Stride
///         Activation::ReLU,       // ReLU activation
///     ).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
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
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The forward pass adds nothing, `weights` hides the array, and `parameters` never yields
    /// it, so a bias-free layer holds it and no more
    bias: Array1<f32>,
    /// Activation applied to the convolution output
    activation: Activation,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Input channels, which [`UnaryLayer::build`] reads from the input shape
    channels: usize,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Whether the layer adds a bias to the convolution output
    use_bias: bool,
}

impl Conv1D {
    /// Creates a new Conv1D layer with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (channels)
    /// - `kernel_size` - Size of the convolution kernel
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
    /// effective kernel to fit. The padding mode is not final until the build, so the build
    /// applies that rule.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters`, `kernel_size`, or `stride` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        filters: usize,
        kernel_size: usize,
        stride: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_1d(kernel_size)?;
        validate_strides_1d(stride)?;
        let activation = activation.into();
        activation.validate()?;

        Ok(Self {
            filters,
            kernel_size,
            stride,
            dilation_rate: 1,
            padding: ConvPadding::Valid,
            weights: Array3::zeros((0, 0, 0)),
            bias: Array1::zeros(0),
            activation,
            built: None,
            channels: 0,
            random_state: None,
            use_bias: true,
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
    /// needs it to fit, and the build applies that rule
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
    /// By default, the draw takes the global seed or entropy (see [`crate::random`]). An
    /// unbuilt layer holds no kernel, so this records the seed and draws nothing. A layer that
    /// is already built draws its kernel again from the new seed, so the order of the 2 calls
    /// does not matter. The bias stays zero-initialized
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

    /// Draws the kernel and zeroes the bias, at the extents the build settled
    ///
    /// The layer names its own channel count and filter count, so the fan pair does not depend
    /// on the order of the 2 channel axes in the stored kernel
    fn draw_parameters(&mut self) {
        let mut rng = crate::random::make_rng(self.random_state);
        self.weights = Initializer::GlorotUniform.draw(
            (self.kernel_size, self.channels, self.filters),
            Fans::conv(self.channels, self.filters, self.kernel_size),
            &mut rng,
        );
        self.bias = Array1::zeros(self.filters);
    }

    /// Calculates the output length after convolution
    ///
    /// The `Valid` rule reads the dilated extent of the kernel, and it refuses an extent longer
    /// than the input length. `Causal` gives the same length as `Same`, because both rules round
    /// the input length up by the stride, and both accept every kernel
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the padding is `Valid` and the effective kernel is longer than
    ///   the input length
    fn calculate_output_length(&self, input_length: usize) -> Result<usize, Error> {
        match self.padding {
            ConvPadding::Valid => valid_output_size(
                "Conv1D",
                "length",
                input_length,
                effective_kernel(self.kernel_size, self.dilation_rate),
                self.stride,
            ),
            ConvPadding::Same | ConvPadding::Causal => Ok(input_length.div_ceil(self.stride)),
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
    /// - `weights` - 3D array of filter weights with shape \[kernel_size, channels, filters\]
    /// - `bias` - 1D array of bias values with shape \[filters\], or `None` for a layer
    ///   built with [`with_use_bias(false)`](Self::with_use_bias)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match the
    ///   layer's expected shape
    pub fn set_weights(
        &mut self,
        weights: Array3<f32>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Conv1D"));
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
}

/// What the forward pass of [`Conv1D`] parks for its backward pass
struct Conv1DCache {
    /// The input tensor the forward pass received
    input: Tensor,
    /// The activated output, to backpropagate through the activation
    output: Tensor,
}

impl LayerBase for Conv1D {
    fn layer_type(&self) -> &str {
        "Conv1D"
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

impl UnaryLayer for Conv1D {
    /// Reads the channel count from the input shape, and draws the kernel and the bias
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Conv1D", input)? else {
            return Ok(());
        };
        built.check_rank("Conv1D", 3)?;
        let (batch, tail) = built.split_batch("Conv1D")?;
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
        validate_built_input(&self.built, "Conv1D", input.shape())?;

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.use_bias
                .then(|| self.bias.as_slice().expect("bias must be contiguous")),
            &[self.stride],
            &[self.dilation_rate],
            self.padding,
        )?;
        let activated = self.activation.forward(&output)?;

        if ctx.is_training() {
            ctx.push_cache(
                "Conv1D",
                Conv1DCache {
                    input: input.clone(),
                    output: activated.clone(),
                },
            );
        }

        Ok(activated)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: Conv1DCache = ctx.pop_cache("Conv1D")?;

        // Apply activation backward pass
        let grad_upstream = self.activation.backward(&cache.output, grad_output)?;

        let grads = conv_backward(
            &grad_upstream,
            &cache.input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.stride],
            &[self.dilation_rate],
            self.padding,
        )?;

        ctx.add_grad(
            "kernel",
            Array3::from_shape_vec(self.weights.raw_dim(), grads.weight_grad)
                .expect("weight gradient shape matches weights")
                .into_dyn(),
        )?;
        // A bias-free layer computes no bias gradient, so the store holds none and no
        // optimizer state is ever keyed on a bias that the layer does not hold
        if self.use_bias {
            ctx.add_grad(
                "bias",
                Array1::from_shape_vec(self.bias.raw_dim(), grads.bias_grad)
                    .expect("bias gradient shape matches bias")
                    .into_dyn(),
            )?;
        }

        Ok(grads.input_grad)
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("Conv1D", 3)?;
        let (batch, tail) = input.split_batch("Conv1D")?;
        Ok(Shape::from_batch(
            batch,
            &[self.calculate_output_length(tail[0])?, self.filters],
        ))
    }
}
