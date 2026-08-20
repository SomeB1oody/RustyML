//! 1D depthwise separable convolution layer (depthwise stage followed by a pointwise 1-tap stage)

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, depthwise_backward, depthwise_forward,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{conv_backward, conv_forward};
use crate::neural_network::layers::convolution::validation::{
    validate_depth_multiplier, validate_filters, validate_input_shape_1d, validate_kernel_size_1d,
    validate_strides_1d,
};
use crate::neural_network::layers::layer_weight::{LayerWeight, SeparableConv1DLayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array1, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::borrow::Cow;

/// A 1D separable convolutional layer
///
/// Runs a depthwise step followed by a pointwise step. This uses fewer parameters and less
/// computation than a standard convolution. Input shape is \[batch_size, length, channels\].
/// Intermediate depthwise output shape is \[batch_size, length', channels * depth_multiplier\].
/// Final output shape is \[batch_size, length', filters\]
///
/// The intermediate channel for input channel `c` and multiplier index `m` is
/// `c * depth_multiplier + m`. It is also the row order of the
/// pointwise weight `\[1, channels * depth_multiplier, filters\]`. This aligns the 2 stages
/// with no repacking between them
///
/// The separable convolution runs 2 steps:
/// 1. Depthwise convolution: convolves each input channel with its own set of filters
/// 2. Pointwise convolution: a 1-tap convolution that combines the depthwise outputs
///
/// This is the 1D form of
/// [`SeparableConv2D`](crate::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D)
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
/// // input tensor: [batch_size, length, channels]
/// let x = Array3::ones((2, 32, 3)).into_dyn();
///
/// // target tensor
/// let y = Array3::ones((2, 32, 16)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(SeparableConv1D::new(
///         16,                          // filters
///         3,                           // kernel_size
///         vec![2, 32, 3],              // input_shape
///         1,                           // stride
///         1,                           // depth_multiplier
///         Activation::ReLU,            // activation
///     ).unwrap().with_padding(PaddingType::Same))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// model.summary();
/// model.fit(&x, &y, 3).unwrap();
/// ```
#[derive(Debug)]
pub struct SeparableConv1D {
    /// Number of output channels from the pointwise convolution
    filters: usize,
    /// Number of input channels, read from the declared input shape
    channels: usize,
    /// Depthwise convolution kernel size along the length axis
    kernel_size: usize,
    /// Stride of the convolution along the length axis
    stride: usize,
    /// Padding applied to the length axis (`Valid` or `Same`)
    padding: PaddingType,
    /// Number of depthwise filters per input channel
    depth_multiplier: usize,
    /// Depthwise filters with shape \[kernel_size, channels, depth_multiplier\]
    depthwise_weights: Array3<f32>,
    /// Pointwise filters with shape \[1, channels * depth_multiplier, filters\]
    pointwise_weights: Array3<f32>,
    /// Bias vector with shape \[filters\]
    bias: Array1<f32>,
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
    depthwise_weight_gradients: Option<Array3<f32>>,
    /// Gradients for the pointwise weights
    pointwise_weight_gradients: Option<Array3<f32>>,
    /// Gradients for the biases
    bias_gradients: Option<Array1<f32>>,
}

impl SeparableConv1D {
    /// Creates a new 1D separable convolutional layer
    ///
    /// The layer initializes weights with Xavier (Glorot) uniform initialization. Biases start
    /// at 0
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output channels from the pointwise convolution
    /// - `kernel_size` - Size of the depthwise convolution kernel along the length axis
    /// - `input_shape` - Shape of the input tensor as \[batch_size, length, channels\]
    /// - `stride` - Stride of the convolution along the length axis
    /// - `depth_multiplier` - Number of depthwise convolution filters per input channel
    /// - `activation` - Activation applied to the output
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `SeparableConv1D` layer instance or an error
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]. Choose [`PaddingType::Same`] with
    /// [`SeparableConv1D::with_padding`].
    ///
    /// The layer seeds weights from the global seed or entropy by default. For reproducible
    /// initialization, set a seed with [`SeparableConv1D::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If `kernel_size` or `stride` is 0
    /// - `Error::InvalidParameter` - If `depth_multiplier` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    /// - `Error::InvalidInput` - If `input_shape` is not 3D or has 0 channels
    /// - `Error::InvalidInput` - If the input length is smaller than the kernel size
    pub fn new(
        filters: usize,
        kernel_size: usize,
        input_shape: Vec<usize>,
        stride: usize,
        depth_multiplier: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_1d(kernel_size)?;
        validate_strides_1d(stride)?;
        validate_depth_multiplier(depth_multiplier)?;
        validate_input_shape_1d(&input_shape, kernel_size)?;
        let activation = activation.into();
        activation.validate()?;

        let channels = input_shape[2];
        let (depthwise_weights, pointwise_weights) =
            Self::init_weights_arrays(filters, channels, kernel_size, depth_multiplier, None);
        let bias = Array1::zeros(filters);

        Ok(SeparableConv1D {
            filters,
            channels,
            kernel_size,
            stride,
            padding: PaddingType::Valid,
            depth_multiplier,
            depthwise_weights,
            pointwise_weights,
            bias,
            activation,
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
        let (depthwise_weights, pointwise_weights) = Self::init_weights_arrays(
            self.filters,
            self.channels,
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
    /// Both draws share 1 RNG (threaded depthwise-then-pointwise) so a given seed reproduces the
    /// exact same pair of tensors
    fn init_weights_arrays(
        filters: usize,
        channels: usize,
        kernel_size: usize,
        depth_multiplier: usize,
        random_state: Option<u64>,
    ) -> (Array3<f32>, Array3<f32>) {
        // Xavier init for the depthwise weights. The fan calculation derives both fans from the
        // kernel tensor's last 2 axes. For shape [kernel_size, channels, dm] this makes the
        // depthwise kernel count `channels` in its fan_in, even though a depthwise unit sees only
        // 1 channel
        let depthwise_fan_in = channels * kernel_size;
        let depthwise_fan_out = depth_multiplier * kernel_size;
        let depthwise_bound = (6.0 / (depthwise_fan_in + depthwise_fan_out) as f32).sqrt();

        let mut rng = crate::random::make_rng(random_state);
        let depthwise_weights = Array3::random_using(
            (kernel_size, channels, depth_multiplier),
            Uniform::new(-depthwise_bound, depthwise_bound).unwrap(),
            &mut rng,
        );

        // Xavier init for the pointwise weights. The 1-tap kernel length is 1
        let pointwise_fan_in = channels * depth_multiplier;
        let pointwise_fan_out = filters;
        let pointwise_bound = (6.0 / (pointwise_fan_in + pointwise_fan_out) as f32).sqrt();

        let pointwise_weights = Array3::random_using(
            (1, channels * depth_multiplier, filters),
            Uniform::new(-pointwise_bound, pointwise_bound).unwrap(),
            &mut rng,
        );

        (depthwise_weights, pointwise_weights)
    }

    /// Calculates the output length of both stages, which share 1 length
    ///
    /// The pointwise stage has a 1-tap kernel under `Valid` padding, so it leaves the length
    /// unchanged. The depthwise stage alone therefore sets the output length
    fn calculate_output_length(&self, input_length: usize) -> usize {
        match self.padding {
            PaddingType::Valid => (input_length - self.kernel_size) / self.stride + 1,
            PaddingType::Same => input_length.div_ceil(self.stride),
        }
    }

    /// The depthwise stage's geometry for a given input, as the shared kernel wants it
    ///
    /// The shared kernel names 2 spatial axes, so this fixes the height at 1 and puts the length
    /// on the width axis. A `[batch, length, channels]` tensor and a
    /// `[kernel_size, channels, depth_multiplier]` weight already hold the values in that order,
    /// so neither one needs a copy
    fn depthwise_geometry(&self, input_shape: &[usize]) -> DepthwiseGeometry {
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
            channels: input_shape[2],
            depth_multiplier: self.depth_multiplier,
            kernel: (1, self.kernel_size),
            strides: (1, self.stride),
            pad_before: (0, pad / 2),
        }
    }

    /// Checks a runtime input against the rank and the channel count the layer was built for
    ///
    /// The depthwise kernel is sized from the declared channel count, so an input carrying more
    /// channels would read past the end of it. This turns that into an error at the layer
    /// boundary
    fn validate_input(&self, input: &Tensor) -> Result<(), Error> {
        if input.ndim() != 3 {
            return Err(Error::invalid_input("input tensor is not 3D"));
        }
        let channels = input.shape()[2];
        if channels != self.channels {
            return Err(Error::dimension_mismatch(self.channels, channels));
        }
        Ok(())
    }

    /// Performs the depthwise convolution stage
    ///
    /// Carries no bias and no activation. Both belong to the pointwise stage that follows, so
    /// this passes `None` for the bias to the shared kernel
    fn depthwise_convolve(&self, input: &Tensor) -> Tensor {
        let g = self.depthwise_geometry(input.shape());
        let batch_size = input.shape()[0];

        let input_std = input.as_standard_layout();
        let src = input_std
            .as_slice()
            .expect("standard-layout array is contiguous");
        let ker = self
            .depthwise_weights
            .as_slice()
            .expect("depthwise weights must be contiguous");

        let mut output = Array3::<f32>::zeros((batch_size, g.output.1, g.out_channels()));
        depthwise_forward(
            &g,
            src,
            ker,
            None,
            output.as_slice_mut().expect("output is contiguous"),
        );

        output.into_dyn()
    }

    /// Performs the pointwise (1-tap) convolution stage
    ///
    /// A 1-tap convolution is a per-position cross-channel matrix multiply. This delegates to the
    /// shared [`conv_forward`] engine (im2col + gemm) rather than a hand-rolled loop nest. The
    /// pointwise weights `[1, C*dm, filters]` already match the engine's flat `[k..., Cin, F]`
    /// layout. The bias is already its per-filter `[F]` vector. The depthwise stage emits its
    /// channels in `c * depth_multiplier + m` order, which is exactly the row order the pointwise
    /// weight uses. Nothing repacks the data between the stages
    fn pointwise_convolve(&self, input: &Tensor) -> Tensor {
        conv_forward(
            input,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[1],
            PaddingType::Valid,
        )
        // A 1-tap kernel under Valid padding can never exceed the input (every length >= 1)
        .expect("1-tap pointwise convolution geometry is always valid")
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `depthwise_weights` - 3D array for depthwise filters with shape
    ///   \[kernel_size, channels, depth_multiplier\]
    /// - `pointwise_weights` - 3D array for pointwise filters with shape
    ///   \[1, channels * depth_multiplier, filters\]
    /// - `bias` - 1D bias vector with shape \[filters\]
    ///
    /// # Errors
    ///
    /// - `Error` - If any supplied array shape does not match the existing layer weights
    pub fn set_weights(
        &mut self,
        depthwise_weights: Array3<f32>,
        pointwise_weights: Array3<f32>,
        bias: Array1<f32>,
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

impl Layer for SeparableConv1D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate_input(input)?;

        // Cache input for backpropagation
        self.input_cache = Some(input.clone());

        // Depthwise convolution (each channel independently), then pointwise to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        // Cache the depthwise output. Only backward needs it
        self.depthwise_output_cache = Some(depthwise_output);

        let activated = self.activation.forward(&output)?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate_input(input)?;

        // Depthwise convolution (each channel independently), then pointwise to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        self.activation.forward(&output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Backward through the activation first
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("SeparableConv1D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let (Some(input), Some(depthwise_output)) =
            (&self.input_cache, &self.depthwise_output_cache)
        else {
            return Err(Error::forward_pass_not_run("SeparableConv1D"));
        };

        let batch_size = input.shape()[0];
        let g = self.depthwise_geometry(input.shape());

        // Pointwise (1-tap) backward via the shared engine (im2col + gemm). Its input gradient is
        // the gradient with respect to the depthwise output, with shape [batch, L', C*dm]
        let pw_grads = conv_backward(
            &grad_upstream,
            depthwise_output,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            &[1],
            PaddingType::Valid,
        )
        // 1-tap Valid geometry is always valid (see `pointwise_convolve`)
        .expect("1-tap pointwise convolution geometry is always valid");
        self.pointwise_weight_gradients = Some(
            Array3::from_shape_vec(self.pointwise_weights.raw_dim(), pw_grads.weight_grad)
                .expect("pointwise weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(Array1::from_vec(pw_grads.bias_grad));
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

        self.depthwise_weight_gradients = Some(
            Array3::from_shape_vec(self.depthwise_weights.raw_dim(), dw_grads.weight)
                .expect("depthwise weight gradient shape matches weights"),
        );

        Ok(
            Array3::from_shape_vec((batch_size, g.input.1, g.channels), dw_grads.input)
                .expect("input gradient shape matches input")
                .into_dyn(),
        )
    }

    fn layer_type(&self) -> &str {
        "SeparableConv1D"
    }

    fn output_shape(&self) -> String {
        let output_length = self.calculate_output_length(self.input_shape[1]);
        format!(
            "({}, {}, {})",
            self.input_shape[0], output_length, self.filters
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
        LayerWeight::SeparableConv1D(SeparableConv1DLayerWeight {
            depthwise_weight: Cow::Borrowed(&self.depthwise_weights),
            pointwise_weight: Cow::Borrowed(&self.pointwise_weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}

/// Unit tests for `SeparableConv1D`
#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::activation::linear::Linear;
    use ndarray::ArrayD;

    /// The 2 stages agree on the intermediate channel order
    ///
    /// A 1-tap kernel at a single position reduces the layer to arithmetic a reader can write out
    /// by hand. The depthwise weights give each `(channel, multiplier)` pair a distinct power of
    /// 10. The pointwise weights give each intermediate channel a distinct power of 2. The single
    /// output value only comes out right if the depthwise stage emits its channels in
    /// `c * depth_multiplier + m` order. It also needs the pointwise weight rows indexed in that
    /// same order. Any transposition of either would change the total
    #[test]
    fn separable_1d_stage_channel_order_hand_derived() {
        let mut layer = SeparableConv1D::new(1, 1, vec![1, 1, 2], 1, 2, Linear::new()).unwrap();
        assert_eq!(layer.depthwise_weights.shape(), &[1, 2, 2]);
        assert_eq!(layer.pointwise_weights.shape(), &[1, 4, 1]);

        // [k, C, dm] as (c, m): c0 -> [1, 10], c1 -> [100, 1000]
        let depthwise = Array3::from_shape_vec((1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
        // [1, C*dm, F]: 1 distinct weight per intermediate channel
        let pointwise = Array3::from_shape_vec((1, 4, 1), vec![1.0, 2.0, 4.0, 8.0]).unwrap();
        layer
            .set_weights(depthwise, pointwise, Array1::zeros(1))
            .unwrap();

        // 1 position holding [2, 3]
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2]), vec![2.0, 3.0]).unwrap();
        let out = layer.predict(&input).unwrap();

        // Intermediate = [2*1, 2*10, 3*100, 3*1000] = [2, 20, 300, 3000]
        // Output = 2*1 + 20*2 + 300*4 + 3000*8 = 25242
        assert_eq!(out.shape(), &[1, 1, 1]);
        assert_eq!(out.iter().copied().collect::<Vec<f32>>(), vec![25242.0]);
    }

    /// Along the length axis, the depthwise stage is an ordinary per-channel cross-correlation.
    /// The pointwise stage is a per-position channel mix. A width-2 kernel over a length-4 input
    /// reduces to the 3 window sums, scaled by the pointwise weight
    #[test]
    fn separable_1d_length_pass_hand_derived() {
        let mut layer = SeparableConv1D::new(1, 2, vec![1, 4, 1], 1, 1, Linear::new()).unwrap();

        // Single channel, all-ones depthwise kernel. Pointwise scales by 3
        let depthwise = Array3::from_shape_vec((2, 1, 1), vec![1.0, 1.0]).unwrap();
        let pointwise = Array3::from_shape_vec((1, 1, 1), vec![3.0]).unwrap();
        layer
            .set_weights(depthwise, pointwise, Array1::zeros(1))
            .unwrap();

        // [1, 4, 1] holding 1..4
        let input =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 1]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = layer.predict(&input).unwrap();

        // Window sums 3, 5, 7, each tripled
        assert_eq!(out.shape(), &[1, 3, 1]);
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![9.0, 15.0, 21.0]
        );
    }
}
