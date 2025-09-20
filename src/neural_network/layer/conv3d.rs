use super::super::*;
use super::helper_functions::*;
use super::*;
use ndarray::{Array3, Array5, ArrayD, ArrayView5, Axis, Zip};
use ndarray_rand::RandomExt;
use rayon::prelude::*;

/// A 3D convolutional layer for neural networks.
///
/// This layer applies a 3D convolution operation to input data, which is fundamental
/// for processing volumetric data such as medical images (CT scans, MRI), 3D models,
/// or video sequences where temporal information is important.
///
/// # Fields
///
/// - `filters` - Number of convolution filters (output channels).
/// - `kernel_size` - Size of the convolution kernel as (depth, height, width).
/// - `strides` - Stride values for the convolution operation as (depth_stride, height_stride, width_stride).
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `weights` - 5D array of filter weights with shape \[filters, channels, kernel_depth, kernel_height, kernel_width\].
/// - `bias` - 3D array of bias values with shape \[1, 1, filters\].
/// - `activation` - Optional activation function applied after the convolution.
/// - `input_cache` - Cached input from the forward pass, used during backpropagation.
/// - `input_shape` - Shape of the input tensor.
/// - `weight_gradients` - Gradients for the weights, computed during backpropagation.
/// - `bias_gradients` - Gradients for the biases, computed during backpropagation.
/// - `optimizer_cache` - Cache for optimizer-specific state (e.g., momentum values for Adam).
///
/// # Shape Information
///
/// Input shape: \[batch_size, channels, depth, height, width\]
/// Output shape: \[batch_size, filters, output_depth, output_height, output_width\]
///
/// The output dimensions (output_depth, output_height, output_width) depend on:
/// - Input dimensions
/// - Kernel size
/// - Stride values
/// - Padding type
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array5;
///
/// // Create a simple 5D input tensor: [batch_size, channels, depth, height, width]
/// // Batch size=2, 1 input channel, 8x8x8 voxels
/// let x = Array5::ones((2, 1, 8, 8, 8)).into_dyn();
///
/// // Create target tensor - assuming we'll have 3 filters with output size 6x6x6
/// let y = Array5::ones((2, 3, 6, 6, 6)).into_dyn();
///
/// // Build model: add a Conv3D layer with 3 filters and 3x3x3 kernel
/// let mut model = Sequential::new();
/// model
///     .add(Conv3D::new(
///         3,                         // Number of filters
///         (3, 3, 3),                 // Kernel size
///         vec![2, 1, 8, 8, 8],       // Input shape
///         (1, 1, 1),                 // Stride
///         PaddingType::Valid,        // No padding
///         Some(Activation::ReLU),    // ReLU activation function
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("3D Convolution layer prediction results: {:?}", prediction);
///
/// // Check if output shape is correct - should be [2, 3, 6, 6, 6]
/// assert_eq!(prediction.shape(), &[2, 3, 6, 6, 6]);
/// ```
pub struct Conv3D {
    filters: usize,
    kernel_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    padding: PaddingType,
    weights: Array5<f32>,
    bias: Array3<f32>,
    activation: Option<Activation>,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array5<f32>>,
    bias_gradients: Option<Array3<f32>>,
    optimizer_cache: OptimizerCacheConv3D,
}

impl Conv3D {
    /// Creates a new Conv3D layer.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters
    /// - `kernel_size` - Size of the convolution kernel as (depth, height, width)
    /// - `input_shape` - Expected input shape as \[batch_size, channels, depth, height, width\]
    /// - `strides` - Stride values as (depth_stride, height_stride, width_stride)
    /// - `padding` - Padding type (Valid or Same)
    /// - `activation` - Optional activation function
    ///
    /// # Returns
    ///
    /// * `Conv3D` - A new `Conv3D` layer instance
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize, usize),
        padding: PaddingType,
        activation: Option<Activation>,
    ) -> Self {
        // verify input is 5D: [batch_size, channels, depth, height, width]
        assert_eq!(
            input_shape.len(),
            5,
            "Input tensor must be 5-dimensional: [batch_size, channels, depth, height, width]"
        );

        let channels = input_shape[1];
        let (kd, kh, kw) = kernel_size;

        // Xavier/Glorot initialization
        let fan_in = channels * kd * kh * kw;
        let fan_out = filters * kd * kh * kw;
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        // Initialize weights with Xavier initialization
        let weights = Array5::random(
            (filters, channels, kd, kh, kw),
            ndarray_rand::rand_distr::Uniform::new(-limit, limit),
        );

        // Initialize bias to zeros
        let bias = Array3::zeros((1, 1, filters));

        Self {
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            activation,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
            optimizer_cache: OptimizerCacheConv3D::default(),
        }
    }

    /// Calculates the output shape for 3D convolution.
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let (batch_size, _, depth, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.strides;

        let (output_depth, output_height, output_width) = match self.padding {
            PaddingType::Valid => (
                (depth - kd) / sd + 1,
                (height - kh) / sh + 1,
                (width - kw) / sw + 1,
            ),
            PaddingType::Same => (
                (depth + sd - 1) / sd,
                (height + sh - 1) / sh,
                (width + sw - 1) / sw,
            ),
        };

        vec![
            batch_size,
            self.filters,
            output_depth,
            output_height,
            output_width,
        ]
    }

    /// Applies 3D convolution operation to the input.
    fn conv3d(&self, input: ArrayView5<f32>) -> Array5<f32> {
        let input_shape = input.shape();
        let output_shape = self.calculate_output_shape(input_shape);

        let (batch_size, _, out_depth, out_height, out_width) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
            output_shape[4],
        );

        let mut output =
            Array5::zeros((batch_size, self.filters, out_depth, out_height, out_width));

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.strides;

        // First level parallelization: batch level
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut batch_output)| {
                // Second level parallelization: filter level
                batch_output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(f, mut filter_output)| {
                        // Third level parallelization: output depth dimension
                        filter_output
                            .axis_iter_mut(Axis(0))
                            .into_par_iter()
                            .enumerate()
                            .for_each(|(od, mut depth_slice)| {
                                // Fourth level parallelization: output height dimension
                                depth_slice
                                    .axis_iter_mut(Axis(0))
                                    .into_par_iter()
                                    .enumerate()
                                    .for_each(|(oh, mut height_slice)| {
                                        // Inner loop: convolution computation for width dimension
                                        for ow in 0..out_width {
                                            let mut sum = 0.0;

                                            // Convolution kernel computation
                                            for c in 0..input_shape[1] {
                                                for kd_idx in 0..kd {
                                                    for kh_idx in 0..kh {
                                                        for kw_idx in 0..kw {
                                                            let id = od * sd + kd_idx;
                                                            let ih = oh * sh + kh_idx;
                                                            let iw = ow * sw + kw_idx;

                                                            if id < input_shape[2]
                                                                && ih < input_shape[3]
                                                                && iw < input_shape[4]
                                                            {
                                                                sum += input[[b, c, id, ih, iw]]
                                                                    * self.weights[[
                                                                        f, c, kd_idx, kh_idx,
                                                                        kw_idx,
                                                                    ]];
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            height_slice[ow] = sum + self.bias[[0, 0, f]];
                                        }
                                    });
                            });
                    });
            });

        output
    }

    /// Computes gradients during backward propagation.
    /// Parallel gradient computation
    fn compute_gradients(&mut self, input: &Array5<f32>, grad_output: &ArrayD<f32>) -> Array5<f32> {
        let input_shape = input.shape();
        let grad_shape = grad_output.shape();
        let (batch_size, _, depth, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        );

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.strides;

        // Initialize gradients
        self.weight_gradients = Some(Array5::zeros(self.weights.raw_dim()));
        self.bias_gradients = Some(Array3::zeros(self.bias.raw_dim()));

        let mut grad_input = Array5::zeros(input.raw_dim());

        // Parallel computation of weight gradients
        if let Some(ref mut weight_grads) = self.weight_gradients {
            // Use parallel iterators to compute weight gradients
            weight_grads
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut filter_grads)| {
                    filter_grads
                        .axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(c, mut channel_grads)| {
                            for kd_idx in 0..kd {
                                for kh_idx in 0..kh {
                                    for kw_idx in 0..kw {
                                        let mut grad_sum = 0.0;

                                        for b in 0..batch_size {
                                            for od in 0..grad_shape[2] {
                                                for oh in 0..grad_shape[3] {
                                                    for ow in 0..grad_shape[4] {
                                                        let id = od * sd + kd_idx;
                                                        let ih = oh * sh + kh_idx;
                                                        let iw = ow * sw + kw_idx;

                                                        if id < depth && ih < height && iw < width {
                                                            grad_sum += input[[b, c, id, ih, iw]]
                                                                * grad_output[[b, f, od, oh, ow]];
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        channel_grads[[kd_idx, kh_idx, kw_idx]] = grad_sum;
                                    }
                                }
                            }
                        });
                });
        }

        // Parallel computation of bias gradients
        if let Some(ref mut bias_grads) = self.bias_gradients {
            bias_grads
                .axis_iter_mut(Axis(2))
                .into_par_iter()
                .enumerate()
                .for_each(|(f, mut filter_bias)| {
                    let mut bias_sum = 0.0;
                    for b in 0..batch_size {
                        for od in 0..grad_shape[2] {
                            for oh in 0..grad_shape[3] {
                                for ow in 0..grad_shape[4] {
                                    bias_sum += grad_output[[b, f, od, oh, ow]];
                                }
                            }
                        }
                    }
                    filter_bias[[0, 0]] = bias_sum;
                });
        }

        // Parallel computation of input gradients
        grad_input
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut batch_grad)| {
                batch_grad
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(c, mut channel_grad)| {
                        for id in 0..depth {
                            for ih in 0..height {
                                for iw in 0..width {
                                    let mut grad_sum = 0.0;

                                    for f in 0..self.filters {
                                        for kd_idx in 0..kd {
                                            for kh_idx in 0..kh {
                                                for kw_idx in 0..kw {
                                                    if id >= kd_idx && ih >= kh_idx && iw >= kw_idx
                                                    {
                                                        let od = (id - kd_idx) / sd;
                                                        let oh = (ih - kh_idx) / sh;
                                                        let ow = (iw - kw_idx) / sw;

                                                        if od < grad_shape[2]
                                                            && oh < grad_shape[3]
                                                            && ow < grad_shape[4]
                                                        {
                                                            if (id - kd_idx) % sd == 0
                                                                && (ih - kh_idx) % sh == 0
                                                                && (iw - kw_idx) % sw == 0
                                                            {
                                                                grad_sum += self.weights[[
                                                                    f, c, kd_idx, kh_idx, kw_idx,
                                                                ]] * grad_output
                                                                    [[b, f, od, oh, ow]];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    channel_grad[[id, ih, iw]] = grad_sum;
                                }
                            }
                        }
                    });
            });

        grad_input
    }
}

impl Layer for Conv3D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Convert input to Array5
        let input_array = input
            .view()
            .into_dimensionality::<ndarray::Ix5>()
            .expect("Input must be 5-dimensional for Conv3D");

        // Cache input for backward pass
        self.input_cache = Some(input.clone());

        // Perform convolution
        let mut output = self.conv3d(input_array).into_dyn();

        // Apply activation function if specified
        if let Some(activation) = &self.activation {
            apply_activation_conv(activation, &mut output);
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let input = self.input_cache.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Cannot use Softmax for convolution".to_string())
        })?;

        let input_array = input
            .view()
            .into_dimensionality::<ndarray::Ix5>()
            .unwrap()
            .to_owned();

        let mut grad_output_array = grad_output
            .view()
            .into_dimensionality::<ndarray::Ix5>()
            .unwrap()
            .to_owned()
            .into_dyn();

        // Apply activation function derivative if specified
        if let Some(activation) = &self.activation {
            activation_derivative_conv(activation, &mut grad_output_array);
        }

        let grad_input = self.compute_gradients(&input_array, &grad_output_array);

        Ok(grad_input.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "Conv3D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]
        )
    }

    fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    update_sgd_conv!();

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        if self.optimizer_cache.adam_states.is_none() {
            self.optimizer_cache.adam_states = Some(AdamStatesConv3D {
                m: Array5::zeros(self.weights.raw_dim()),
                v: Array5::zeros(self.weights.raw_dim()),
                m_bias: Array3::zeros(self.bias.raw_dim()),
                v_bias: Array3::zeros(self.bias.raw_dim()),
            });
        }

        if let (Some(weight_grads), Some(bias_grads), Some(adam_states)) = (
            &self.weight_gradients,
            &self.bias_gradients,
            &mut self.optimizer_cache.adam_states,
        ) {
            // Adam update for weights
            adam_states.m.par_mapv_inplace(|m| m * beta1);
            adam_states.v.par_mapv_inplace(|v| v * beta2);

            // Add gradient terms
            Zip::from(&mut adam_states.m)
                .and(weight_grads)
                .par_for_each(|m, &g| *m += g * (1.0 - beta1));

            Zip::from(&mut adam_states.v)
                .and(weight_grads)
                .par_for_each(|v, &g| *v += g * g * (1.0 - beta2));

            // Compute bias correction
            let beta1_t = 1.0 - beta1.powi(t as i32);
            let beta2_t = 1.0 - beta2.powi(t as i32);

            // Parallel weight updates
            Zip::from(&mut self.weights)
                .and(&adam_states.m)
                .and(&adam_states.v)
                .par_for_each(|w, &m, &v| {
                    let m_hat = m / beta1_t;
                    let v_hat = v / beta2_t;
                    *w -= lr * m_hat / (v_hat.sqrt() + epsilon);
                });

            // Adam update for bias
            adam_states.m_bias.par_mapv_inplace(|m| m * beta1);
            adam_states.v_bias.par_mapv_inplace(|v| v * beta2);

            Zip::from(&mut adam_states.m_bias)
                .and(bias_grads)
                .par_for_each(|m, &g| *m += g * (1.0 - beta1));

            Zip::from(&mut adam_states.v_bias)
                .and(bias_grads)
                .par_for_each(|v, &g| *v += g * g * (1.0 - beta2));

            // Parallel bias updates
            Zip::from(&mut self.bias)
                .and(&adam_states.m_bias)
                .and(&adam_states.v_bias)
                .par_for_each(|b, &m, &v| {
                    let m_hat = m / beta1_t;
                    let v_hat = v / beta2_t;
                    *b -= lr * m_hat / (v_hat.sqrt() + epsilon);
                });
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if self.optimizer_cache.rmsprop_cache.is_none() {
            self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheConv3D {
                cache: Array5::zeros(self.weights.raw_dim()),
                bias: Array3::zeros(self.bias.raw_dim()),
            });
        }

        if let (Some(weight_grads), Some(bias_grads), Some(rmsprop_cache)) = (
            &self.weight_gradients,
            &self.bias_gradients,
            &mut self.optimizer_cache.rmsprop_cache,
        ) {
            // Parallel weight cache updates
            Zip::from(&mut rmsprop_cache.cache)
                .and(weight_grads)
                .par_for_each(|cache, &grad| {
                    *cache = *cache * rho + grad * grad * (1.0 - rho);
                });

            // Parallel weight updates
            Zip::from(&mut self.weights)
                .and(weight_grads)
                .and(&rmsprop_cache.cache)
                .par_for_each(|w, &g, &c| {
                    *w -= lr * g / (c.sqrt() + epsilon);
                });

            // Parallel bias cache updates
            Zip::from(&mut rmsprop_cache.bias)
                .and(bias_grads)
                .par_for_each(|cache, &grad| {
                    *cache = *cache * rho + grad * grad * (1.0 - rho);
                });

            // Parallel bias updates
            Zip::from(&mut self.bias)
                .and(bias_grads)
                .and(&rmsprop_cache.bias)
                .par_for_each(|b, &g, &c| {
                    *b -= lr * g / (c.sqrt() + epsilon);
                });
        }
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv3D(Conv3DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
