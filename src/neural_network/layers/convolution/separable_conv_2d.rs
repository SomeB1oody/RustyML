use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::conv_op_helpers::{
    compute_row_gradient_sum, merge_results, pad_tensor_4d_spatial,
};
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::validation::{
    validate_depth_multiplier, validate_filters, validate_input_shape_2d, validate_kernel_size_2d,
    validate_strides_2d,
};
use crate::neural_network::layers::layer_weight::{LayerWeight, SeparableConv2DLayerWeight};
use crate::neural_network::layers::shape_helpers::calculate_output_height_and_weight;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3, Array4, ArrayD, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Threshold for deciding between parallel and sequential execution.
/// When `batch_size * output_channels * output_height * output_width >= SEPARABLE_CONV_2D_PARALLEL_THRESHOLD`,
/// parallel processing is used for better performance.
const SEPARABLE_CONV_2D_PARALLEL_THRESHOLD: usize = 5000;

/// A 2D separable convolutional layer for neural networks.
///
/// Implements depthwise separable convolution with a depthwise step followed by a pointwise step.
/// This reduces parameters and computation compared to standard convolution while maintaining
/// similar performance. Input shape is \[batch_size, channels, height, width\], intermediate
/// depthwise output shape is \[batch_size, channels * depth_multiplier, height', width'\], and
/// final output shape is \[batch_size, filters, height', width'\].
///
/// The separable convolution consists of:
/// 1. **Depthwise Convolution**: Each input channel is convolved with its own set of filters
/// 2. **Pointwise Convolution**: A 1x1 convolution that combines the outputs from the depthwise step
///
/// # Fields
///
/// - `filters` - Number of output channels from the pointwise convolution.
/// - `kernel_size` - Size of the depthwise convolution kernel as (height, width).
/// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `depth_multiplier` - Number of depthwise convolution filters per input channel.
/// - `depthwise_weights` - 4D array for depthwise filters with shape \[depth_multiplier, channels, kernel_height, kernel_width\].
/// - `pointwise_weights` - 4D array for pointwise filters with shape \[filters, channels * depth_multiplier, 1, 1\].
/// - `bias` - 2D array of bias values with shape \[1, filters\].
/// - `activation` - Activation layer from activation_layer module.
/// - `input_cache` - Cached input from the forward pass, used during backpropagation.
/// - `depthwise_output_cache` - Cached depthwise output, used during backpropagation.
/// - `input_shape` - Shape of the input tensor.
/// - `depthwise_weight_gradients` - Gradients for the depthwise weights.
/// - `pointwise_weight_gradients` - Gradients for the pointwise weights.
/// - `bias_gradients` - Gradients for the biases.
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
///
/// // Create a simple 4D input tensor: [batch_size, channels, height, width]
/// let x = Array4::ones((2, 3, 32, 32)).into_dyn();
///
/// // Create target tensor
/// let y = Array4::ones((2, 64, 32, 32)).into_dyn();
///
/// // Build model with separable convolution
/// let mut model = Sequential::new();
/// model
///     .add(SeparableConv2D::new(
///         64,                          // Number of output filters
///         (3, 3),                      // Kernel size
///         vec![2, 3, 32, 32],          // Input shape
///         (1, 1),                      // Stride
///         PaddingType::Same,           // Same padding
///         1,                           // Depth multiplier
///         Activation::ReLU, // ReLU activation
///         None,                        // random_state
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// model.summary();
/// model.fit(&x, &y, 3).unwrap();
/// ```
#[derive(Debug)]
pub struct SeparableConv2D {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    depth_multiplier: usize,
    depthwise_weights: Array4<f32>,
    pointwise_weights: Array4<f32>,
    bias: Array2<f32>,
    activation: Activation,
    output_cache: Option<Tensor>,
    input_cache: Option<Tensor>,
    depthwise_output_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    depthwise_weight_gradients: Option<Array4<f32>>,
    pointwise_weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array2<f32>>,
}

impl SeparableConv2D {
    /// Creates a new 2D separable convolutional layer with the specified parameters.
    ///
    /// Weights are initialized using Xavier (Glorot) uniform initialization.
    /// Biases are initialized to zeros.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output channels from the pointwise convolution.
    /// - `kernel_size` - Size of the depthwise convolution kernel as (height, width).
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\].
    /// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
    /// - `padding` - Type of padding to apply (`Valid` or `Same`).
    /// - `depth_multiplier` - Number of depthwise convolution filters per input channel.
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax).
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random.
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
    // Eight positional parameters mirror the other convolution constructors (a config struct would be
    // inconsistent with the crate's layer API); the trailing `random_state` pushes this past clippy's
    // 7-argument threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        depth_multiplier: usize,
        activation: impl Into<Activation>,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        // Validate input parameters
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_depth_multiplier(depth_multiplier)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        let channels = input_shape[1];

        // Initialize depthwise weights using Xavier initialization
        // For depthwise convolution, each filter only operates on one channel
        let depthwise_fan_in = kernel_size.0 * kernel_size.1;
        let depthwise_fan_out = depth_multiplier * kernel_size.0 * kernel_size.1;
        let depthwise_bound = (6.0 / (depthwise_fan_in + depthwise_fan_out) as f32).sqrt();

        let mut rng = crate::random::make_rng(random_state);
        let depthwise_weights = Array4::random_using(
            (depth_multiplier, channels, kernel_size.0, kernel_size.1),
            Uniform::new(-depthwise_bound, depthwise_bound).unwrap(),
            &mut rng,
        );

        // Initialize pointwise weights using Xavier initialization
        // For pointwise convolution (1x1), the kernel area is 1
        let pointwise_fan_in = channels * depth_multiplier;
        let pointwise_fan_out = filters;
        let pointwise_bound = (6.0 / (pointwise_fan_in + pointwise_fan_out) as f32).sqrt();

        let pointwise_weights = Array4::random_using(
            (filters, channels * depth_multiplier, 1, 1),
            Uniform::new(-pointwise_bound, pointwise_bound).unwrap(),
            &mut rng,
        );

        // Initialize biases to zero
        let bias = Array2::zeros((1, filters));

        Ok(SeparableConv2D {
            filters,
            kernel_size,
            strides,
            padding,
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

    /// Calculates the output shape of the separable convolutional layer.
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// Performs depthwise convolution operation.
    fn depthwise_convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_depthwise_output_shape(input_shape);

        // Zero-pad the spatial dims for `Same` so the depthwise stage convolves over a padded
        // tensor. Without this, `compute_depthwise_batch`'s boundary clipping silently degrades
        // `Same` into a `Valid`-style, top-left-aligned result at the borders. `DepthwiseConv2D`
        // and the convolution engine pad identically; `Valid` is a no-op (`None`, no copy).
        let (pad_h, pad_w) = self.calculate_padding(
            input_shape[2],
            input_shape[3],
            output_shape[2],
            output_shape[3],
        );
        let padded_storage = if pad_h == 0 && pad_w == 0 {
            None
        } else {
            Some(pad_tensor_4d_spatial(input, pad_h, pad_w))
        };
        let conv_input: &Tensor = padded_storage.as_ref().unwrap_or(input);

        // Calculate workload size to decide between parallel and sequential execution
        let workload_size =
            batch_size * channels * self.depth_multiplier * output_shape[2] * output_shape[3];

        // Compute batch convolution results (parallel or sequential based on workload)
        let results: Vec<_> = if workload_size >= SEPARABLE_CONV_2D_PARALLEL_THRESHOLD {
            // Use parallel processing for large workloads
            (0..batch_size)
                .into_par_iter()
                .map(|b| self.compute_depthwise_batch(b, conv_input, &output_shape, channels))
                .collect()
        } else {
            // Use sequential processing for small workloads
            (0..batch_size)
                .map(|b| self.compute_depthwise_batch(b, conv_input, &output_shape, channels))
                .collect()
        };

        // Use merge_results function to combine batch results
        merge_results(output_shape, results)
    }

    /// Computes depthwise convolution for a single batch.
    fn compute_depthwise_batch(
        &self,
        b: usize,
        input: &Tensor,
        output_shape: &[usize],
        channels: usize,
    ) -> (usize, Array3<f32>) {
        let input_shape = input.shape();
        let mut batch_output = Array3::zeros((
            channels * self.depth_multiplier,
            output_shape[2],
            output_shape[3],
        ));

        for c in 0..channels {
            for m in 0..self.depth_multiplier {
                let output_channel = c * self.depth_multiplier + m;

                for i in 0..output_shape[2] {
                    let i_base = i * self.strides.0;

                    for j in 0..output_shape[3] {
                        let j_base = j * self.strides.1;
                        let mut sum = 0.0;

                        let max_ki = input_shape[2]
                            .saturating_sub(i_base)
                            .min(self.kernel_size.0);
                        let max_kj = input_shape[3]
                            .saturating_sub(j_base)
                            .min(self.kernel_size.1);

                        for ki in 0..max_ki {
                            let i_pos = i_base + ki;
                            for kj in 0..max_kj {
                                let j_pos = j_base + kj;
                                sum += input[[b, c, i_pos, j_pos]]
                                    * self.depthwise_weights[[m, c, ki, kj]];
                            }
                        }

                        batch_output[[output_channel, i, j]] = sum;
                    }
                }
            }
        }

        (b, batch_output)
    }

    /// Performs pointwise convolution (1x1 convolution).
    fn pointwise_convolve(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let output_shape = vec![batch_size, self.filters, input_shape[2], input_shape[3]];

        // Calculate workload size to decide between parallel and sequential execution
        let workload_size = batch_size * self.filters * input_shape[2] * input_shape[3];

        // Compute batch convolution results (parallel or sequential based on workload)
        let results: Vec<_> = if workload_size >= SEPARABLE_CONV_2D_PARALLEL_THRESHOLD {
            // Use parallel processing for large workloads
            (0..batch_size)
                .into_par_iter()
                .map(|b| self.compute_pointwise_batch(b, input, input_shape))
                .collect()
        } else {
            // Use sequential processing for small workloads
            (0..batch_size)
                .map(|b| self.compute_pointwise_batch(b, input, input_shape))
                .collect()
        };

        // Use merge_results function to combine batch results
        merge_results(output_shape, results)
    }

    /// Computes pointwise convolution for a single batch.
    fn compute_pointwise_batch(
        &self,
        b: usize,
        input: &Tensor,
        input_shape: &[usize],
    ) -> (usize, Array3<f32>) {
        let mut batch_output = Array3::zeros((self.filters, input_shape[2], input_shape[3]));

        for f in 0..self.filters {
            for i in 0..input_shape[2] {
                for j in 0..input_shape[3] {
                    let mut sum = 0.0;

                    for c in 0..input_shape[1] {
                        sum += input[[b, c, i, j]] * self.pointwise_weights[[f, c, 0, 0]];
                    }

                    sum += self.bias[[0, f]];
                    batch_output[[f, i, j]] = sum;
                }
            }
        }

        (b, batch_output)
    }

    /// Calculates the output shape after depthwise convolution.
    fn calculate_depthwise_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = calculate_output_height_and_weight(
            self.padding,
            input_height,
            input_width,
            self.kernel_size,
            self.strides,
        );

        vec![
            batch_size,
            channels * self.depth_multiplier,
            output_height,
            output_width,
        ]
    }

    /// Calculates the symmetric zero-padding (total height/width pad) for the depthwise stage.
    ///
    /// Returns `(0, 0)` for `Valid` padding. For `Same`, returns the total padding along each
    /// spatial axis required so a stride-`s` convolution yields the given output size; the
    /// padding is split with `pad / 2` on the leading edge (see [`pad_tensor_4d_spatial`]).
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

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `depthwise_weights` - 4D array for depthwise filters with shape \[depth_multiplier, channels, kernel_height, kernel_width\]
    /// - `pointwise_weights` - 4D array for pointwise filters with shape \[filters, channels * depth_multiplier, 1, 1\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    pub fn set_weights(
        &mut self,
        depthwise_weights: Array4<f32>,
        pointwise_weights: Array4<f32>,
        bias: Array2<f32>,
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
        // Validate input is 4D tensor
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Step 1: Depthwise convolution - each input channel convolved independently
        let depthwise_output = self.depthwise_convolve(input);

        // Step 2: Pointwise convolution (1x1) - combines depthwise outputs
        let output = self.pointwise_convolve(&depthwise_output);

        // Cache depthwise output after pointwise conv to save memory if no activation
        // (activation derivative doesn't need depthwise output, only backward does)
        self.depthwise_output_cache = Some(depthwise_output);

        // Apply activation
        let activated = self.activation.forward(&output.into_dyn())?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D tensor
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Step 1: Depthwise convolution - each input channel convolved independently
        let depthwise_output = self.depthwise_convolve(input);

        // Step 2: Pointwise convolution (1x1) - combines depthwise outputs
        let output = self.pointwise_convolve(&depthwise_output);

        // Apply activation
        let activated = self.activation.forward(&output.into_dyn())?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Apply activation backward pass
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("SeparableConv2D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        if let (Some(input), Some(depthwise_output)) =
            (&self.input_cache, &self.depthwise_output_cache)
        {
            let input_shape = input.shape();
            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let depthwise_shape = depthwise_output.shape();
            let grad_shape = grad_output.shape();

            // Re-create the zero-padded input the forward pass convolved over (see
            // `depthwise_convolve`) so the depthwise weight and input gradients are accumulated in
            // padded coordinates. The padding is stripped from the input gradient before returning.
            let (pad_h, pad_w) = self.calculate_padding(
                input_shape[2],
                input_shape[3],
                depthwise_shape[2],
                depthwise_shape[3],
            );
            let padded_storage = if pad_h == 0 && pad_w == 0 {
                None
            } else {
                Some(pad_tensor_4d_spatial(input, pad_h, pad_w))
            };
            let padded_input: &Tensor = padded_storage.as_ref().unwrap_or(input);
            let padded_shape = padded_input.shape().to_vec();

            let gradient = grad_upstream;

            // Initialize gradients
            let mut pointwise_weight_grads = Array4::zeros(self.pointwise_weights.dim());
            let mut depthwise_weight_grads = Array4::zeros(self.depthwise_weights.dim());
            let mut bias_grads = Array2::zeros((1, self.filters));

            // Each gradient below is parallelized over an axis whose entries are written
            // independently (disjoint mutable sub-views), so the parallel result is identical to
            // the sequential one. The forward pass is already parallel; this stops the backward
            // pass from being a serial bottleneck.

            // Bias gradients: one independent sum per output filter.
            bias_grads
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut bias)| {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        for i in 0..grad_shape[2] {
                            for j in 0..grad_shape[3] {
                                sum += gradient[[b, f, i, j]];
                            }
                        }
                    }
                    *bias.first_mut().unwrap() = sum;
                });

            // Pointwise weight gradients: parallel over output filters.
            pointwise_weight_grads
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut pw_grad)| {
                    for c in 0..depthwise_shape[1] {
                        let mut sum = 0.0;
                        for b in 0..batch_size {
                            for i in 0..depthwise_shape[2] {
                                for j in 0..depthwise_shape[3] {
                                    sum += gradient[[b, f, i, j]] * depthwise_output[[b, c, i, j]];
                                }
                            }
                        }
                        pw_grad[[c, 0, 0]] = sum;
                    }
                });

            // Gradient w.r.t. the depthwise output: parallel over the batch axis.
            let mut depthwise_grad = ArrayD::zeros(depthwise_output.dim());
            depthwise_grad
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(b, mut dw_grad_b)| {
                    for c in 0..depthwise_shape[1] {
                        for i in 0..depthwise_shape[2] {
                            for j in 0..depthwise_shape[3] {
                                let mut sum = 0.0;
                                for f in 0..self.filters {
                                    sum += gradient[[b, f, i, j]]
                                        * self.pointwise_weights[[f, c, 0, 0]];
                                }
                                dw_grad_b[[c, i, j]] = sum;
                            }
                        }
                    }
                });

            // Depthwise weight gradients: parallel over the depth-multiplier axis.
            depthwise_weight_grads
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(m, mut dw_wgrad_m)| {
                    for c in 0..channels {
                        for h in 0..self.kernel_size.0 {
                            for w in 0..self.kernel_size.1 {
                                let mut sum = 0.0;
                                let output_channel = c * self.depth_multiplier + m;
                                for b in 0..batch_size {
                                    for i in 0..depthwise_shape[2] {
                                        let i_pos = i * self.strides.0 + h;
                                        if i_pos < padded_shape[2] {
                                            sum += compute_row_gradient_sum(
                                                &depthwise_grad,
                                                padded_input,
                                                b,
                                                output_channel,
                                                c,
                                                i,
                                                i_pos,
                                                w,
                                                depthwise_shape,
                                                &padded_shape,
                                                self.strides.1,
                                            );
                                        }
                                    }
                                }
                                dw_wgrad_m[[c, h, w]] = sum;
                            }
                        }
                    }
                });

            // Input gradients: parallel over the batch axis. Accumulated in PADDED coordinates
            // (matching the padded forward pass), then sliced back to the original shape below.
            let mut input_gradients = ArrayD::zeros(padded_input.dim());
            input_gradients
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(b, mut in_grad_b)| {
                    for c in 0..channels {
                        for i in 0..padded_shape[2] {
                            for j in 0..padded_shape[3] {
                                let mut sum = 0.0;
                                for m in 0..self.depth_multiplier {
                                    let output_channel = c * self.depth_multiplier + m;
                                    for h in 0..self.kernel_size.0 {
                                        for w in 0..self.kernel_size.1 {
                                            if i >= h && j >= w {
                                                let grad_i = (i - h) / self.strides.0;
                                                let grad_j = (j - w) / self.strides.1;
                                                if grad_i < depthwise_shape[2]
                                                    && grad_j < depthwise_shape[3]
                                                    && (i - h) % self.strides.0 == 0
                                                    && (j - w) % self.strides.1 == 0
                                                {
                                                    sum += depthwise_grad
                                                        [[b, output_channel, grad_i, grad_j]]
                                                        * self.depthwise_weights[[m, c, h, w]];
                                                }
                                            }
                                        }
                                    }
                                }
                                in_grad_b[[c, i, j]] = sum;
                            }
                        }
                    }
                });

            // Strip the symmetric padding so the returned gradient matches the original input
            // shape (no-op for `Valid`, where no padding was added).
            let input_gradients = if pad_h == 0 && pad_w == 0 {
                input_gradients
            } else {
                let pad_top = pad_h / 2;
                let pad_left = pad_w / 2;
                input_gradients
                    .slice(s![
                        ..,
                        ..,
                        pad_top..pad_top + input_shape[2],
                        pad_left..pad_left + input_shape[3]
                    ])
                    .to_owned()
                    .into_dyn()
            };

            // Store gradients
            self.depthwise_weight_gradients = Some(depthwise_weight_grads);
            self.pointwise_weight_gradients = Some(pointwise_weight_grads);
            self.bias_gradients = Some(bias_grads);

            Ok(input_gradients)
        } else {
            Err(Error::forward_pass_not_run("SeparableConv2D"))
        }
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
            params.push(ParamGrad {
                value: depthwise_weights
                    .as_slice_mut()
                    .expect("depthwise weights must be contiguous"),
                grad: gd
                    .as_slice()
                    .expect("depthwise weight gradient must be contiguous"),
            });
            params.push(ParamGrad {
                value: pointwise_weights
                    .as_slice_mut()
                    .expect("pointwise weights must be contiguous"),
                grad: gp
                    .as_slice()
                    .expect("pointwise weight gradient must be contiguous"),
            });
            params.push(ParamGrad {
                value: bias.as_slice_mut().expect("bias must be contiguous"),
                grad: gb.as_slice().expect("bias gradient must be contiguous"),
            });
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::SeparableConv2DLayer(SeparableConv2DLayerWeight {
            depthwise_weight: &self.depthwise_weights,
            pointwise_weight: &self.pointwise_weights,
            bias: &self.bias,
        })
    }
}
