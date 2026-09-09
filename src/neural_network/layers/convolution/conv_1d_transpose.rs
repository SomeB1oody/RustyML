//! 1D transposed convolutional layer, the decoder counterpart of `Conv1D`
//!
//! Holds the layer weights and the activation, and delegates the forward/backward numerics
//! to the dimension-generic transposed-convolution engine

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::conv_transpose_engine::{
    conv_transpose_backward, conv_transpose_forward, transpose_output_length,
};
use crate::neural_network::layers::convolution::validation::{
    validate_dilation, validate_filters, validate_kernel_size_1d,
    validate_stride_dilation_exclusive, validate_strides_1d, validate_transpose_input_shape,
};
use crate::neural_network::layers::validation::{
    start_build, validate_built_input, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Fans, Initializer, Shape, Tensor};
use ndarray::{Array1, Array3};

/// A 1D transposed convolutional layer for neural networks
///
/// Runs a convolution backwards over its length axis, so it grows a sequence instead of
/// shrinking one. Input shape is \[batch_size, length, channels\]. Output shape is
/// \[batch_size, output_length, filters\]. This is the layer a decoder uses to reach the length
/// the matching [`Conv1D`](super::conv_1d::Conv1D) consumed.
///
/// The dimension-generic transposed-convolution math lives in the transposed-convolution engine.
/// This layer holds the weights and the activation, and delegates the forward/backward
/// numerics to it.
///
/// # Notes
///
/// The kernel has shape \[kernel_size, filters, channels\]. The filter axis comes **before** the
/// input-channel axis, which is the reverse of the `Conv1D` kernel. A transposed convolution
/// reads `channels` and writes `filters`, so this is the order its inner product needs.
///
/// The output size follows 1 rule per padding mode. `keff` is the kernel size after dilation,
/// defined in [`Conv1DTranspose::with_dilation_rate`]:
///
/// - [`PaddingType::Valid`]: `input * stride + max(keff - stride, 0)`
/// - [`PaddingType::Same`]: `input * stride`
///
/// Unlike `Conv1D`, this layer puts no lower bound on the input length. A length-1 input under a
/// length-3 kernel is a normal first decoder step.
///
/// A stride wider than the kernel leaves output positions that no input position reaches. These
/// positions hold exactly the bias.
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
/// // A length-3 kernel at stride 2 with Valid padding grows 10 steps into 21
/// let y = Array3::ones((2, 21, 3)).into_dyn();
///
/// // Build model: add a Conv1DTranspose layer with 3 filters and kernel size 3
/// let mut model = SequentialBuilder::new()
///     .add(Conv1DTranspose::new(
///         3,                      // Number of filters
///         3,                      // Kernel size
///         2,                      // Stride
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
///
/// // Check if output shape is correct (should be [2, 21, 3])
/// assert_eq!(prediction.shape(), &[2, 21, 3]);
/// ```
///
/// # Performance
///
/// The pass runs 1 matrix product per batch item, then scatters its rows into the output. The
/// scatter accumulates wherever the windows overlap, so it cannot split 1 sequence across
/// threads. The pass parallelizes over the batch, and runs the per-item products in parallel
/// while the batch alone cannot fill the thread pool. The gate is the convolution engine's
/// estimated-FLOPs gate, reachable through [`crate::tuning::conv`].
#[derive(Debug)]
pub struct Conv1DTranspose {
    /// Number of transposed convolution filters (output channels)
    filters: usize,
    /// Size of the convolution kernel
    kernel_size: usize,
    /// Stride value for the transposed convolution
    stride: usize,
    /// Tap spacing of the kernel. 1 gives a solid kernel
    dilation_rate: usize,
    /// Type of padding to apply (`Valid` or `Same`)
    padding: PaddingType,
    /// 3D array of filter weights with shape \[kernel_size, filters, channels\]
    weights: Array3<f32>,
    /// 1D array of bias values with shape \[filters\]
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The forward pass adds nothing, `param_count` hides the array, and `parameters_mut` never
    /// yields it, so a bias-free layer holds it and no more
    bias: Array1<f32>,
    /// Activation applied to the transposed convolution output
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

impl Conv1DTranspose {
    /// Creates a new 1D transposed convolutional layer with the specified parameters
    ///
    /// The constructor draws nothing. [`UnaryLayer::build`] reads the channel count from the input
    /// shape, draws the kernel with Xavier (Glorot) uniform initialization, and sets the bias
    /// to 0
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of transposed convolution filters (output channels)
    /// - `kernel_size` - Size of the convolution kernel
    /// - `stride` - Stride for the transposed convolution
    /// - `activation` - Activation applied to the transposed convolution output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv1DTranspose` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`Conv1DTranspose::with_padding`]. The kernel is solid by default. Space its taps out
    /// with [`Conv1DTranspose::with_dilation_rate`]. By default, the layer seeds weights from the
    /// global
    /// seed or entropy. For reproducible initialization, set a seed with
    /// [`Conv1DTranspose::with_random_state`].
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

        Ok(Conv1DTranspose {
            filters,
            kernel_size,
            stride,
            dilation_rate: 1,
            padding: PaddingType::Valid,
            weights: Array3::zeros((0, 0, 0)),
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
    /// - `padding` - Type of padding to apply (`Valid` or `Same`)
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
    /// A dilation of `d` on an axis spaces the kernel taps `d` cells apart, so `k` taps span
    /// `(k - 1) * d + 1` output cells of that axis. The window still advances by the stride. A
    /// dilation of 1 gives a solid kernel, with no gap between taps
    ///
    /// # Parameters
    ///
    /// - `dilation_rate` - Tap spacing along the length axis
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any dilation is 0
    /// - `Error::InvalidParameter` - If any dilation is above 1 and any stride is also above 1
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
    /// The stored kernel puts the filter axis before the channel axis, and the fan pair does
    /// not follow that order. The layer names the 2 counts, so `fan_in` stays the channel side
    fn draw_parameters(&mut self) {
        let mut rng = crate::random::make_rng(self.random_state);
        self.weights = Initializer::GlorotUniform.draw(
            (self.kernel_size, self.filters, self.channels),
            Fans::conv(self.channels, self.filters, self.kernel_size),
            &mut rng,
        );
        self.bias = Array1::zeros(self.filters);
    }

    /// Calculates the output shape of the transposed convolution from the input dimensions
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        vec![
            input_shape[0],
            transpose_output_length(
                input_shape[1],
                self.kernel_size,
                self.stride,
                self.dilation_rate,
                self.padding,
            ),
            self.filters,
        ]
    }

    /// Sets whether the layer adds a bias to the convolution output (defaults to `true`)
    ///
    /// With `use_bias` set to false the layer holds the kernel alone: `param_count` counts the
    /// kernel alone, `parameters_mut` yields the kernel alone, and a checkpoint of the layer
    /// holds 1 array under the path `<position>.kernel`. A checkpoint written by a layer that has
    /// a bias therefore fails to load into a layer that has none, and the refusal names the path
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
    /// - `weights` - 3D array of filter weights with shape \[kernel_size, filters, channels\]
    /// - `bias` - 1D array of bias values with shape \[filters\], or `None` for a layer
    ///   built with [`with_use_bias(false)`](Self::with_use_bias)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer has no build yet
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` does not match the
    ///   layer's expected shape
    /// - `Error::InvalidParameter` - If `bias` is given for a layer built with `use_bias(false)`,
    ///   or left out for a layer built with `use_bias(true)`
    pub fn set_weights(
        &mut self,
        weights: Array3<f32>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Conv1DTranspose"));
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

/// What the forward pass of [`Conv1DTranspose`] parks for its backward pass
struct Conv1DTransposeCache {
    /// The input the forward pass received, which the weight gradient reads
    input: Tensor,
    /// The activated output, to backpropagate through the activation
    output: Tensor,
}

impl LayerBase for Conv1DTranspose {
    fn layer_type(&self) -> &str {
        "Conv1DTranspose"
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

impl UnaryLayer for Conv1DTranspose {
    /// Reads the channel count from the input shape, and draws the kernel and the bias
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Conv1DTranspose", input)? else {
            return Ok(());
        };
        built.check_rank("Conv1DTranspose", 3)?;
        let (batch, tail) = built.split_batch("Conv1DTranspose")?;
        // The family validators read a full extent list, and the batch extent is not part of
        // what they check
        let mut dims = vec![batch.unwrap_or(1)];
        dims.extend(tail);
        validate_transpose_input_shape(&dims, 1, "[batch_size, length, channels]")?;
        self.channels = dims[2];
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "Conv1DTranspose", input.shape())?;

        let output = conv_transpose_forward(
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
                "Conv1DTranspose",
                Conv1DTransposeCache {
                    input: input.clone(),
                    output: activated.clone(),
                },
            );
        }

        Ok(activated)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: Conv1DTransposeCache = ctx.pop_cache("Conv1DTranspose")?;
        let grad_upstream = self.activation.backward(&cache.output, grad_output)?;

        let grads = conv_transpose_backward(
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
        input.check_rank("Conv1DTranspose", 3)?;
        let (batch, tail) = input.split_batch("Conv1DTranspose")?;
        // `calculate_output_shape` indexes from the batch axis, so the list it takes needs a
        // placeholder value first
        let mut dims = vec![0];
        dims.extend(tail);
        Ok(Shape::from_batch(
            batch,
            &self.calculate_output_shape(&dims)[1..],
        ))
    }
}
