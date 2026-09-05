//! 3D convolutional layer that applies a 3D convolution over volumetric input
//! and delegates the forward/backward numerics to the dimension-generic convolution engine

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{
    conv_backward, conv_forward, effective_kernel,
};
use crate::neural_network::layers::convolution::validation::{
    valid_output_size, validate_dilation, validate_filters, validate_input_shape_3d,
    validate_kernel_size_3d, validate_stride_dilation_exclusive, validate_strides_3d,
};
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Fans, Initializer, Shape, Tensor};
use ndarray::{Array1, Array5};

/// A 3D convolutional layer for neural networks
///
/// Applies a 3D convolution to volumetric data such as medical images, 3D models, or video
/// sequences. Input shape is \[batch_size, depth, height, width, channels\]. Output shape is
/// \[batch_size, output_depth, output_height, output_width, filters\]. Output dimensions depend
/// on input size, kernel size, strides, and padding
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
/// use ndarray::Array5;
///
/// // 5D input tensor: [batch_size, depth, height, width, channels]
/// // Batch size 2, 8x8x8 voxels, 1 input channel
/// let x = Array5::ones((2, 8, 8, 8, 1)).into_dyn();
///
/// // Target tensor for output size 6x6x6 with 3 filters
/// let y = Array5::ones((2, 6, 6, 6, 3)).into_dyn();
///
/// // Build model: add a Conv3D layer with 3 filters and 3x3x3 kernel
/// let mut model = SequentialBuilder::new()
///     .add(Conv3D::new(
///         3,                         // Number of filters
///         (3, 3, 3),                 // Kernel size
///         (1, 1, 1),                 // Stride
///         Activation::ReLU,          // ReLU activation
///     ).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model for a few epochs
/// model.fit(&x, &y, 3).unwrap();
///
/// // Forward-pass prediction
/// let prediction = model.predict(&x).unwrap();
/// println!("3D Convolution layer prediction results: {:?}", prediction);
///
/// // Output shape should be [2, 6, 6, 6, 3]
/// assert_eq!(prediction.shape(), &[2, 6, 6, 6, 3]);
/// ```
#[derive(Debug)]
pub struct Conv3D {
    /// Number of convolution filters (output channels)
    filters: usize,
    /// Size of the convolution kernel as (depth, height, width)
    kernel_size: (usize, usize, usize),
    /// Stride values as (depth_stride, height_stride, width_stride)
    strides: (usize, usize, usize),
    /// Tap spacing of the kernel as (depth, height, width). 1 gives a solid kernel on that axis
    dilation_rate: (usize, usize, usize),
    /// Type of padding to apply (`Valid` or `Same`)
    padding: PaddingType,
    /// 5D filter weights with shape
    /// \[kernel_depth, kernel_height, kernel_width, channels, filters\]
    weights: Array5<f32>,
    /// 1D bias values with shape \[filters\]
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

impl Conv3D {
    /// Creates a new Conv3D layer with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters
    /// - `kernel_size` - Size of the convolution kernel as (depth, height, width)
    /// - `strides` - Stride values as (depth_stride, height_stride, width_stride)
    /// - `activation` - Activation applied to the convolution output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv3D` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`Conv3D::with_padding`]. The kernel is solid by default. Space its taps out with
    /// [`Conv3D::with_dilation_rate`]. By default, the layer seeds weights from the global seed
    /// or entropy. For reproducible initialization, set a seed with
    /// [`Conv3D::with_random_state`].
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize, usize),
        strides: (usize, usize, usize),
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_3d(kernel_size)?;
        validate_strides_3d(strides)?;
        let activation = activation.into();
        activation.validate()?;

        Ok(Conv3D {
            filters,
            kernel_size,
            strides,
            dilation_rate: (1, 1, 1),
            padding: PaddingType::Valid,
            weights: Array5::zeros((0, 0, 0, 0, 0)),
            bias: Array1::zeros(0),
            activation,
            built: None,
            channels: 0,
            random_state: None,
            use_bias: true,
        })
    }

    /// Sets the padding mode (defaults to [`PaddingType::Valid`])
    ///
    /// # Parameters
    ///
    /// - `padding` - Padding type (`Valid` or `Same`)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the tap spacing of the kernel (defaults to `(1, 1, 1)`)
    ///
    /// A dilation of `d` on an axis spaces the kernel taps `d` cells apart, so `k` taps span
    /// `(k - 1) * d + 1` input cells of that axis. The window still advances by the stride. A
    /// dilation of 1 on every axis gives a solid kernel and the same result as before
    ///
    /// # Parameters
    ///
    /// - `dilation_rate` - Tap spacing as (depth, height, width)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
    ///
    /// # Notes
    ///
    /// The effective kernel is not bounded by the input axis here. Only [`PaddingType::Valid`]
    /// needs it to fit, and the build applies that rule
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any dilation is 0
    /// - `Error::InvalidParameter` - If any dilation is above 1 and any stride is also above 1
    pub fn with_dilation_rate(
        mut self,
        dilation_rate: (usize, usize, usize),
    ) -> Result<Self, Error> {
        let dilation = [dilation_rate.0, dilation_rate.1, dilation_rate.2];
        validate_dilation(&dilation)?;
        validate_stride_dilation_exclusive(
            &[self.strides.0, self.strides.1, self.strides.2],
            &dilation,
        )?;
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
        let (kd, kh, kw) = self.kernel_size;
        let mut rng = crate::random::make_rng(self.random_state);
        self.weights = Initializer::GlorotUniform.draw(
            (kd, kh, kw, self.channels, self.filters),
            Fans::conv(self.channels, self.filters, kd * kh * kw),
            &mut rng,
        );
        self.bias = Array1::zeros(self.filters);
    }

    /// Calculates the output shape for 3D convolution
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the padding is `Valid` and an effective kernel is longer than
    ///   the input axis it runs on
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, Error> {
        let (batch_size, depth, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.strides;

        let (dd, dh, dw) = self.dilation_rate;
        let (output_depth, output_height, output_width) = match self.padding {
            // The `Valid` rule reads the extent the dilated taps span, not the tap count
            PaddingType::Valid => (
                valid_output_size("Conv3D", "depth", depth, effective_kernel(kd, dd), sd)?,
                valid_output_size("Conv3D", "height", height, effective_kernel(kh, dh), sh)?,
                valid_output_size("Conv3D", "width", width, effective_kernel(kw, dw), sw)?,
            ),
            PaddingType::Same => (depth.div_ceil(sd), height.div_ceil(sh), width.div_ceil(sw)),
        };

        Ok(vec![
            batch_size,
            output_depth,
            output_height,
            output_width,
            self.filters,
        ])
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
    /// - `weights` - 5D array of filter weights with shape \[kernel_depth, kernel_height,
    ///   kernel_width, channels, filters\]
    /// - `bias` - 1D array of bias values with shape \[filters\], or `None` for a layer
    ///   built with [`with_use_bias(false)`](Self::with_use_bias)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match the
    ///   layer's expected shape
    pub fn set_weights(
        &mut self,
        weights: Array5<f32>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Conv3D"));
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

/// What the forward pass of [`Conv3D`] parks for its backward pass
struct Conv3DCache {
    /// The input tensor the forward pass received
    input: Tensor,
    /// The activated output, to backpropagate through the activation
    output: Tensor,
}

impl LayerBase for Conv3D {
    fn layer_type(&self) -> &str {
        "Conv3D"
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

impl UnaryLayer for Conv3D {
    /// Reads the channel count from the input shape, and draws the kernel and the bias
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Conv3D", input)? else {
            return Ok(());
        };
        built.check_rank("Conv3D", 5)?;
        let (batch, tail) = built.split_batch("Conv3D")?;
        // The family validators read a full extent list, and the batch extent is not part of
        // what they check
        let mut dims = vec![batch.unwrap_or(1)];
        dims.extend(tail);
        validate_input_shape_3d(&dims)?;
        // The shape algebra holds every rule the geometry has, so a stack that cannot run is
        // refused here, before the layer draws a single weight
        self.compute_output_shape(&built)?;
        self.channels = dims[4];
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "Conv3D", input.shape())?;

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.use_bias
                .then(|| self.bias.as_slice().expect("bias must be contiguous")),
            &[self.strides.0, self.strides.1, self.strides.2],
            &[
                self.dilation_rate.0,
                self.dilation_rate.1,
                self.dilation_rate.2,
            ],
            self.padding.into(),
        )?;
        let activated = self.activation.forward(&output)?;

        if ctx.is_training() {
            ctx.push_cache(Conv3DCache {
                input: input.clone(),
                output: activated.clone(),
            });
        }

        Ok(activated)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: Conv3DCache = ctx.pop_cache("Conv3D")?;

        let grad_upstream = self.activation.backward(&cache.output, grad_output)?;

        let grads = conv_backward(
            &grad_upstream,
            &cache.input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.strides.0, self.strides.1, self.strides.2],
            &[
                self.dilation_rate.0,
                self.dilation_rate.1,
                self.dilation_rate.2,
            ],
            self.padding.into(),
        )?;

        ctx.add_grad(
            "kernel",
            Array5::from_shape_vec(self.weights.raw_dim(), grads.weight_grad)
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
        input.check_rank("Conv3D", 5)?;
        let (batch, tail) = input.split_batch("Conv3D")?;
        // `calculate_output_shape` reads the batch axis, so the list it takes starts with one
        let mut dims = vec![0];
        dims.extend(tail);
        Ok(Shape::from_batch(
            batch,
            &self.calculate_output_shape(&dims)?[1..],
        ))
    }
}
