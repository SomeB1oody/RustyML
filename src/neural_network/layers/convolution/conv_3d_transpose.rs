//! 3D transposed convolutional layer, the decoder counterpart of `Conv3D`
//!
//! Holds the layer weights, activation, and caches, and delegates the forward/backward numerics
//! to the dimension-generic transposed-convolution engine

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::conv_transpose_engine::{
    conv_transpose_backward, conv_transpose_forward, transpose_output_length,
};
use crate::neural_network::layers::convolution::validation::{
    validate_dilation, validate_filters, validate_kernel_size_3d,
    validate_stride_dilation_exclusive, validate_strides_3d, validate_transpose_input_shape,
};
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::validation::{validate_optional_weight, validate_weight_shape};
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::neural_network::{Shape, Tensor};
use ndarray::{Array1, Array5};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// A 3D transposed convolutional layer for neural networks
///
/// Runs a convolution backwards over its 3 spatial axes, so it grows a volume instead of
/// shrinking one. Input shape is \[batch_size, depth, height, width, channels\]. Output shape is
/// \[batch_size, output_depth, output_height, output_width, filters\]. This is the layer a
/// volumetric decoder uses to reach the resolution the matching
/// [`Conv3D`](super::conv_3d::Conv3D) consumed.
///
/// The dimension-generic transposed-convolution math lives in the transposed-convolution engine.
/// This layer holds the weights, activation, and caches, and delegates the forward/backward
/// numerics to it.
///
/// # Notes
///
/// The kernel has shape \[kernel_depth, kernel_height, kernel_width, filters, channels\]. The
/// filter axis comes **before** the input-channel axis, which is the reverse of the `Conv3D`
/// kernel. A transposed convolution reads `channels` and writes `filters`, so this is the order
/// its inner product needs.
///
/// The output size follows 1 rule per padding mode, on each axis on its own:
///
/// - [`PaddingType::Valid`]: `input * stride + max(kernel - stride, 0)`
/// - [`PaddingType::Same`]: `input * stride`
///
/// Unlike `Conv3D`, this layer puts no lower bound on the input size. A 1x1x1 input under a
/// 3x3x3 kernel is a normal first decoder step.
///
/// A stride wider than the kernel leaves output positions that no input position reaches. These
/// positions hold exactly the bias.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array5;
///
/// // Create a simple 5D input tensor: [batch_size, depth, height, width, channels]
/// let x = Array5::ones((1, 3, 3, 3, 1)).into_dyn();
///
/// // A 2x2x2 kernel at stride 1 with Valid padding grows 3x3x3 into 4x4x4
/// let y = Array5::ones((1, 4, 4, 4, 2)).into_dyn();
///
/// // Build model: add a Conv3DTranspose layer with 2 filters and a 2x2x2 kernel
/// let mut model = Sequential::new();
/// model
///     .add(Conv3DTranspose::new(
///         2,                      // Number of filters
///         (2, 2, 2),              // Kernel size
///         vec![1, 3, 3, 3, 1],    // Input shape
///         (1, 1, 1),              // Stride
///         Activation::ReLU,       // ReLU activation
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
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
/// // Check if output shape is correct (should be [1, 4, 4, 4, 2])
/// assert_eq!(prediction.shape(), &[1, 4, 4, 4, 2]);
/// ```
///
/// # Performance
///
/// The pass runs 1 matrix product per batch item, then scatters its rows into the output. The
/// scatter accumulates wherever the windows overlap, so it cannot split 1 volume across threads.
/// The pass parallelizes over the batch, and runs the per-item products in parallel while the
/// batch alone cannot fill the thread pool. The gate is the convolution engine's estimated-FLOPs
/// gate, reachable through [`crate::tuning::conv`].
#[derive(Debug)]
pub struct Conv3DTranspose {
    /// Number of transposed convolution filters (output channels)
    filters: usize,
    /// Size of the convolution kernel as (depth, height, width)
    kernel_size: (usize, usize, usize),
    /// Stride values for the transposed convolution as (depth, height, width)
    strides: (usize, usize, usize),
    /// Tap spacing of the kernel as (depth, height, width). 1 gives a solid axis
    dilation_rate: (usize, usize, usize),
    /// Type of padding to apply (`Valid` or `Same`)
    padding: PaddingType,
    /// 5D array of filter weights with shape \[kernel_depth, kernel_height, kernel_width,
    /// filters, channels\]
    weights: Array5<f32>,
    /// 1D array of bias values with shape \[filters\]
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The forward pass adds nothing, `weights` hides the array, and `parameters` never yields
    /// it, so a bias-free layer holds it and no more
    bias: Array1<f32>,
    /// Activation applied to the transposed convolution output
    activation: Activation,
    /// Cached activated output, used by the activation backward pass
    output_cache: Option<Tensor>,
    /// Cached input from the forward pass, used during backpropagation
    input_cache: Option<Tensor>,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Gradients for the weights, computed during backpropagation
    weight_gradients: Option<Array5<f32>>,
    /// Gradients for the biases, computed during backpropagation
    bias_gradients: Option<Array1<f32>>,
    /// Whether the layer adds a bias to the convolution output
    use_bias: bool,
}

impl Conv3DTranspose {
    /// Creates a new 3D transposed convolutional layer with the specified parameters
    ///
    /// The constructor initializes weights with Xavier (Glorot) uniform initialization and sets
    /// biases to 0
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of transposed convolution filters (output channels)
    /// - `kernel_size` - Size of the convolution kernel as (depth, height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, depth, height, width,
    ///   channels\]
    /// - `strides` - Stride values as (depth_stride, height_stride, width_stride)
    /// - `activation` - Activation applied to the transposed convolution output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv3DTranspose` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`Conv3DTranspose::with_padding`]. By default, the layer seeds weights from the global
    /// seed or entropy. For reproducible initialization, set a seed with
    /// [`Conv3DTranspose::with_random_state`].
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 5D or holds a 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize, usize),
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_3d(kernel_size)?;
        validate_strides_3d(strides)?;
        validate_transpose_input_shape(
            &input_shape,
            3,
            "[batch_size, depth, height, width, channels]",
        )?;
        let activation = activation.into();
        activation.validate()?;

        let channels = input_shape[4];
        let weights = Self::init_weights_array(filters, channels, kernel_size, None);
        let bias = Array1::zeros(filters);

        Ok(Conv3DTranspose {
            filters,
            kernel_size,
            strides,
            dilation_rate: (1, 1, 1),
            padding: PaddingType::Valid,
            weights,
            bias,
            activation,
            output_cache: None,
            input_cache: None,
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
    /// - `padding` - Type of padding to apply (`Valid` or `Same`)
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
    /// `(k - 1) * d + 1` output cells of that axis. The window still advances by the stride. A
    /// dilation of 1 gives a solid kernel and the same result as before
    ///
    /// # Parameters
    ///
    /// - `dilation_rate` - Tap spacing as (depth, height, width)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error
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
    /// By default, the layer seeds weights from the global seed or entropy (see
    /// [`crate::random`]). This method re-runs Xavier/Glorot uniform initialization with
    /// `random_state`. Call it before assigning custom weights or training. The bias stays
    /// zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        let channels = self.input_shape[4];
        self.weights =
            Self::init_weights_array(self.filters, channels, self.kernel_size, Some(random_state));
        self
    }

    /// Xavier/Glorot uniform initialization of the \[kd, kh, kw, filters, channels\] weight tensor
    fn init_weights_array(
        filters: usize,
        channels: usize,
        kernel_size: (usize, usize, usize),
        random_state: Option<u64>,
    ) -> Array5<f32> {
        // Xavier init: bound = sqrt(6 / (fan_in + fan_out)). The transposed kernel swaps the 2
        // channel axes against the plain one, which swaps the 2 fans. Their sum, and so the
        // bound, is unchanged
        let k_plane = kernel_size.0 * kernel_size.1 * kernel_size.2;
        let fan_in = channels * k_plane;
        let fan_out = filters * k_plane;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array5::random_using(
            (
                kernel_size.0,
                kernel_size.1,
                kernel_size.2,
                filters,
                channels,
            ),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
            &mut rng,
        )
    }

    /// Calculates the output shape of the transposed convolution from the input dimensions
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        vec![
            input_shape[0],
            transpose_output_length(
                input_shape[1],
                self.kernel_size.0,
                self.strides.0,
                self.dilation_rate.0,
                self.padding,
            ),
            transpose_output_length(
                input_shape[2],
                self.kernel_size.1,
                self.strides.1,
                self.dilation_rate.1,
                self.padding,
            ),
            transpose_output_length(
                input_shape[3],
                self.kernel_size.2,
                self.strides.2,
                self.dilation_rate.2,
                self.padding,
            ),
            self.filters,
        ]
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
    /// - `weights` - 5D array of filter weights with shape \[kernel_depth, kernel_height,
    ///   kernel_width, filters, channels\]
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

impl Layer for Conv3DTranspose {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        self.input_cache = Some(input.clone());

        let output = conv_transpose_forward(
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
            self.padding,
        )?;
        let activated = self.activation.forward(&output)?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        let output = conv_transpose_forward(
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
            self.padding,
        )?;
        let activated = self.activation.forward(&output)?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Conv3DTranspose"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("Conv3DTranspose"))?;

        let grads = conv_transpose_backward(
            &grad_upstream,
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.strides.0, self.strides.1, self.strides.2],
            &[
                self.dilation_rate.0,
                self.dilation_rate.1,
                self.dilation_rate.2,
            ],
            self.padding,
        )?;

        self.weight_gradients = Some(
            Array5::from_shape_vec(self.weights.raw_dim(), grads.weight_grad)
                .expect("weight gradient shape matches weights"),
        );
        // A bias-free layer keeps no bias gradient, so `parameters` yields none and no
        // optimizer state is ever keyed on a bias that the layer does not hold
        self.bias_gradients = self.use_bias.then(|| {
            Array1::from_shape_vec(self.bias.raw_dim(), grads.bias_grad)
                .expect("bias gradient shape matches bias")
        });

        Ok(grads.input_grad)
    }

    fn layer_type(&self) -> &str {
        "Conv3DTranspose"
    }

    fn known_input_shape(&self) -> Option<Shape> {
        Some(Shape::known(&self.input_shape))
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("Conv3DTranspose", 5)?;
        let (batch, tail) = input.split_batch("Conv3DTranspose")?;
        // `calculate_output_shape` reads the batch axis, so the list it takes starts with one
        let mut dims = vec![0];
        dims.extend(tail);
        Ok(Shape::from_batch(
            batch,
            &self.calculate_output_shape(&dims)[1..],
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
