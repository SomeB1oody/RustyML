//! 2D depthwise separable convolution layer (depthwise stage followed by a pointwise 1x1 stage)

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    DepthwiseGeometry, DepthwiseGradients, depthwise_forward_row, depthwise_item_gradients,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{conv_backward, conv_forward};
use crate::neural_network::layers::convolution::validation::{
    validate_depth_multiplier, validate_filters, validate_input_shape_2d, validate_kernel_size_2d,
    validate_strides_2d,
};
use crate::neural_network::layers::layer_weight::{LayerWeight, SeparableConv2DLayerWeight};
use crate::neural_network::layers::shape_helpers::calculate_output_height_and_weight;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::parallel_gates::naive_conv_parallel_min_flops;
use ndarray::{Array1, Array4};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::prelude::*;
use std::borrow::Cow;

/// A 2D separable convolutional layer
///
/// Implements depthwise separable convolution with a depthwise step followed by a pointwise step,
/// this reduces parameters and computation compared to standard convolution while keeping similar
/// performance. Input shape is \[batch_size, height, width, channels\], intermediate depthwise
/// output shape is \[batch_size, height', width', channels * depth_multiplier\], and final output
/// shape is \[batch_size, height', width', filters\]
///
/// The intermediate channel for input channel `c` and multiplier index `m` is
/// `c * depth_multiplier + m`, which is Keras' ordering and also the row order the pointwise
/// weight `\[1, 1, channels * depth_multiplier, filters\]` is indexed by - so the two stages line
/// up with no repacking between them
///
/// The separable convolution consists of:
/// 1. Depthwise convolution: each input channel is convolved with its own set of filters
/// 2. Pointwise convolution: a 1x1 convolution that combines the outputs from the depthwise step
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
/// // Create a 4D input tensor: [batch_size, height, width, channels]
/// let x = Array4::ones((2, 32, 32, 3)).into_dyn();
///
/// // Create target tensor
/// let y = Array4::ones((2, 32, 32, 64)).into_dyn();
///
/// // Build model with separable convolution
/// let mut model = Sequential::new();
/// model
///     .add(SeparableConv2D::new(
///         64,                          // Number of output filters
///         (3, 3),                      // Kernel size
///         vec![2, 32, 32, 3],          // Input shape
///         (1, 1),                      // Stride
///         1,                           // Depth multiplier
///         Activation::ReLU,            // ReLU activation
///     ).unwrap().with_padding(PaddingType::Same)) // Same padding
///     .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Padding applied to the spatial dimensions (`Valid` or `Same`)
    padding: PaddingType,
    /// Number of depthwise filters per input channel
    depth_multiplier: usize,
    /// Depthwise filters with shape \[kernel_height, kernel_width, channels, depth_multiplier\]
    depthwise_weights: Array4<f32>,
    /// Pointwise filters with shape \[1, 1, channels * depth_multiplier, filters\]
    pointwise_weights: Array4<f32>,
    /// Bias values with shape \[1, filters\]
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
    depthwise_weight_gradients: Option<Array4<f32>>,
    /// Gradients for the pointwise weights
    pointwise_weight_gradients: Option<Array4<f32>>,
    /// Gradients for the biases
    bias_gradients: Option<Array1<f32>>,
}

impl SeparableConv2D {
    /// Creates a new 2D separable convolutional layer
    ///
    /// Weights are initialized using Xavier (Glorot) uniform initialization; biases are zeros
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output channels from the pointwise convolution
    /// - `kernel_size` - Size of the depthwise convolution kernel as (height, width)
    /// - `input_shape` - Shape of the input tensor as \[batch_size, height, width, channels\]
    /// - `strides` - Stride values for the convolution as (vertical, horizontal)
    /// - `depth_multiplier` - Number of depthwise convolution filters per input channel
    /// - `activation` - Activation applied to the output (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Notes
    ///
    /// Padding defaults to [`PaddingType::Valid`]; choose [`PaddingType::Same`] with
    /// [`SeparableConv2D::with_padding`]. Weights are seeded from the global seed or entropy by
    /// default; for reproducible initialization, set a seed with
    /// [`SeparableConv2D::with_random_state`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `SeparableConv2D` layer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidParameter` - If `depth_multiplier` is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 4D or has 0 channels
    /// - `Error::InvalidInput` - If input dimensions are smaller than kernel size
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        depth_multiplier: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_depth_multiplier(depth_multiplier)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        let channels = input_shape[3];
        let (depthwise_weights, pointwise_weights) =
            Self::init_weights_arrays(filters, channels, kernel_size, depth_multiplier, None);
        let bias = Array1::zeros(filters);

        Ok(SeparableConv2D {
            filters,
            kernel_size,
            strides,
            padding: PaddingType::Valid,
            depth_multiplier,
            depthwise_weights,
            pointwise_weights,
            bias,
            activation: activation.into(),
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

    /// Sets the seed used to initialize the depthwise/pointwise weights and re-initializes them
    /// deterministically
    ///
    /// By default the weights are seeded from the global seed or entropy (see [`crate::random`]),
    /// this re-runs Xavier/Glorot uniform initialization with `random_state`, so call it before
    /// assigning custom weights or training. The bias stays zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        let channels = self.input_shape[3];
        let (depthwise_weights, pointwise_weights) = Self::init_weights_arrays(
            self.filters,
            channels,
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
    /// Both draws share one RNG (threaded depthwise-then-pointwise) so a given seed reproduces the
    /// exact same pair of tensors
    fn init_weights_arrays(
        filters: usize,
        channels: usize,
        kernel_size: (usize, usize),
        depth_multiplier: usize,
        random_state: Option<u64>,
    ) -> (Array4<f32>, Array4<f32>) {
        // Xavier init for the depthwise weights
        let depthwise_fan_in = kernel_size.0 * kernel_size.1;
        let depthwise_fan_out = depth_multiplier * kernel_size.0 * kernel_size.1;
        let depthwise_bound = (6.0 / (depthwise_fan_in + depthwise_fan_out) as f32).sqrt();

        let mut rng = crate::random::make_rng(random_state);
        let depthwise_weights = Array4::random_using(
            (kernel_size.0, kernel_size.1, channels, depth_multiplier),
            Uniform::new(-depthwise_bound, depthwise_bound).unwrap(),
            &mut rng,
        );

        // Xavier init for the pointwise weights; the 1x1 kernel area is 1
        let pointwise_fan_in = channels * depth_multiplier;
        let pointwise_fan_out = filters;
        let pointwise_bound = (6.0 / (pointwise_fan_in + pointwise_fan_out) as f32).sqrt();

        let pointwise_weights = Array4::random_using(
            (1, 1, channels * depth_multiplier, filters),
            Uniform::new(-pointwise_bound, pointwise_bound).unwrap(),
            &mut rng,
        );

        (depthwise_weights, pointwise_weights)
    }

    /// Calculates the output shape of the separable convolutional layer
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[1];
        let input_width = input_shape[2];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![batch_size, output_height, output_width, self.filters]
    }

    /// Performs the depthwise convolution stage
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
            pad_before: (pad_h / 2, pad_w / 2),
        }
    }

    /// Performs the depthwise convolution stage
    ///
    /// Carries no bias and no activation: both belong to the pointwise stage that follows, so the
    /// shared kernel is called with `None` for the bias seed
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
        let flops =
            2 * batch_size * out_channels * g.output.0 * g.output.1 * g.kernel.0 * g.kernel.1;
        let row_len = g.output.1 * out_channels;
        let out_flat = output.as_slice_mut().expect("output is contiguous");

        // One task per (batch item, output row); rows are disjoint so this needs no merge
        if flops >= naive_conv_parallel_min_flops() {
            out_flat
                .par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| {
                    depthwise_forward_row(&g, src, ker, None, i / g.output.0, i % g.output.0, row)
                });
        } else {
            for (i, row) in out_flat.chunks_mut(row_len).enumerate() {
                depthwise_forward_row(&g, src, ker, None, i / g.output.0, i % g.output.0, row);
            }
        }

        output.into_dyn()
    }

    /// Performs the pointwise (1x1) convolution stage
    ///
    /// A 1x1 convolution is a per-position cross-channel matrix multiply, so this delegates to the
    /// shared [`conv_forward`] engine (im2col + gemm) rather than a hand-rolled loop nest. The
    /// pointwise weights `[1, 1, C*dm, filters]` already match the engine's flat `[k..., Cin, F]`
    /// layout, and the bias is already its per-filter `[F]` vector - the depthwise stage emits its
    /// channels in `c * depth_multiplier + m` order, which is exactly the row order the pointwise
    /// weight is indexed by, so nothing is repacked between the stages
    fn pointwise_convolve(&self, input: &Tensor) -> Tensor {
        conv_forward(
            input,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[1, 1],
            PaddingType::Valid,
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
            self.kernel_size,
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
    /// spatial axis required so a stride-`s` convolution yields the given output size; the padding
    /// is split with `pad / 2` on the leading edge, matching the convolution engine
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
    /// - `depthwise_weights` - 4D array for depthwise filters with shape \[kernel_height, kernel_width, channels, depth_multiplier\]
    /// - `pointwise_weights` - 4D array for pointwise filters with shape \[1, 1, channels * depth_multiplier, filters\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    ///
    /// # Errors
    ///
    /// - `Error` - If any supplied array shape does not match the existing layer weights
    pub fn set_weights(
        &mut self,
        depthwise_weights: Array4<f32>,
        pointwise_weights: Array4<f32>,
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

impl Layer for SeparableConv2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Cache input for backpropagation
        self.input_cache = Some(input.clone());

        // Depthwise convolution (each channel independently), then pointwise (1x1) to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        // Cache the depthwise output; only backward needs it
        self.depthwise_output_cache = Some(depthwise_output);

        let activated = self.activation.forward(&output.into_dyn())?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches); see [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Depthwise convolution (each channel independently), then pointwise (1x1) to combine
        let depthwise_output = self.depthwise_convolve(input);
        let output = self.pointwise_convolve(&depthwise_output);

        let activated = self.activation.forward(&output.into_dyn())?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Backward through the activation first
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("SeparableConv2D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let (Some(input), Some(depthwise_output)) =
            (&self.input_cache, &self.depthwise_output_cache)
        else {
            return Err(Error::forward_pass_not_run("SeparableConv2D"));
        };

        let batch_size = input.shape()[0];
        let g = self.depthwise_geometry(input.shape());

        // Pointwise (1x1) backward via the shared engine (im2col + gemm). Its input gradient is
        // the gradient w.r.t. the depthwise output, shape [batch, H', W', C*dm]
        let pw_grads = conv_backward(
            &grad_upstream,
            depthwise_output,
            self.pointwise_weights
                .as_slice()
                .expect("pointwise weights must be contiguous"),
            self.pointwise_weights.shape(),
            &[1, 1],
            PaddingType::Valid,
        )
        // 1x1 Valid geometry is always valid (see `pointwise_convolve`)
        .expect("1x1 pointwise convolution geometry is always valid");
        self.pointwise_weight_gradients = Some(
            Array4::from_shape_vec(self.pointwise_weights.raw_dim(), pw_grads.weight_grad)
                .expect("pointwise weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(Array1::from_vec(pw_grads.bias_grad));
        let depthwise_grad = pw_grads.input_grad;

        // Depthwise backward through the shared kernel, one task per batch item. The bias partial
        // it returns is discarded: this stage has no bias of its own
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

        let flops =
            2 * batch_size * g.out_channels() * g.output.0 * g.output.1 * g.kernel.0 * g.kernel.1;
        let run = |b: usize| depthwise_item_gradients(&g, src, grad, ker, b);
        let per_b: Vec<DepthwiseGradients> = if flops >= naive_conv_parallel_min_flops() {
            (0..batch_size).into_par_iter().map(run).collect()
        } else {
            (0..batch_size).map(run).collect()
        };

        // Sum the weight partials in batch order so the result does not depend on scheduling
        let mut depthwise_weight_grads = vec![0.0f32; self.depthwise_weights.len()];
        let mut input_gradients =
            Vec::with_capacity(batch_size * g.input.0 * g.input.1 * g.channels);
        for part in per_b {
            for (acc, v) in depthwise_weight_grads.iter_mut().zip(part.weight) {
                *acc += v;
            }
            input_gradients.extend(part.input);
        }

        self.depthwise_weight_gradients = Some(
            Array4::from_shape_vec(self.depthwise_weights.raw_dim(), depthwise_weight_grads)
                .expect("depthwise weight gradient shape matches weights"),
        );

        Ok(Array4::from_shape_vec(
            (batch_size, g.input.0, g.input.1, g.channels),
            input_gradients,
        )
        .expect("input gradient shape matches input")
        .into_dyn())
    }

    fn layer_type(&self) -> &str {
        "SeparableConv2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
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
        LayerWeight::SeparableConv2D(SeparableConv2DLayerWeight {
            depthwise_weight: Cow::Borrowed(&self.depthwise_weights),
            pointwise_weight: Cow::Borrowed(&self.pointwise_weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::layers::activation::linear::Linear;
    use crate::neural_network::traits::Layer;
    use ndarray::ArrayD;

    /// The two stages agree on the intermediate channel order
    ///
    /// A 1x1 kernel at a single spatial position reduces the layer to arithmetic that can be
    /// written out by hand. The depthwise weights give each `(channel, multiplier)` pair a distinct
    /// power of ten and the pointwise weights give each intermediate channel a distinct power of
    /// two, so the single output value only comes out right if the depthwise stage emits its
    /// channels in `c * depth_multiplier + m` order *and* the pointwise weight rows are indexed in
    /// that same order. Any transposition of either would change the total
    #[test]
    fn separable_stage_channel_order_hand_derived() {
        let mut layer =
            SeparableConv2D::new(1, (1, 1), vec![1, 1, 1, 2], (1, 1), 2, Linear::new()).unwrap();
        assert_eq!(layer.depthwise_weights.shape(), &[1, 1, 2, 2]);
        assert_eq!(layer.pointwise_weights.shape(), &[1, 1, 4, 1]);

        // [kh, kw, C, dm] as (c, m): c0 -> [1, 10], c1 -> [100, 1000]
        let depthwise =
            Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
        // [1, 1, C*dm, F]: one distinct weight per intermediate channel
        let pointwise = Array4::from_shape_vec((1, 1, 4, 1), vec![1.0, 2.0, 4.0, 8.0]).unwrap();
        layer
            .set_weights(depthwise, pointwise, Array1::zeros(1))
            .unwrap();

        // One position holding [2, 3]
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 1, 2]), vec![2.0, 3.0]).unwrap();
        let out = layer.predict(&input).unwrap();

        // Intermediate = [2*1, 2*10, 3*100, 3*1000] = [2, 20, 300, 3000]
        // Output = 2*1 + 20*2 + 300*4 + 3000*8 = 25242
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.iter().copied().collect::<Vec<f32>>(), vec![25242.0]);
    }

    /// Spatially, the depthwise stage is an ordinary per-channel cross-correlation and the
    /// pointwise stage a per-position channel mix, so a 2x2 kernel over a 3x3 input reduces to the
    /// four window sums scaled by the pointwise weight
    #[test]
    fn separable_spatial_pass_hand_derived() {
        let mut layer =
            SeparableConv2D::new(1, (2, 2), vec![1, 3, 3, 1], (1, 1), 1, Linear::new()).unwrap();

        // Single channel, all-ones depthwise kernel; pointwise scales by 3
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
        let out = layer.predict(&input).unwrap();

        // Window sums 12, 16, 24, 28, each tripled
        assert_eq!(out.shape(), &[1, 2, 2, 1]);
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![36.0, 48.0, 72.0, 84.0]
        );
    }
}
