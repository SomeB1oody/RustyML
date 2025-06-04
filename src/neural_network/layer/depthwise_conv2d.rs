use crate::ModelError;
use crate::neural_network::Tensor;
use crate::neural_network::layer::helper_functions::*;
use crate::neural_network::layer::{DepthwiseConv2DLayerWeight, LayerWeight, PaddingType};
use crate::neural_network::optimizer::*;
use crate::traits::Layer;
use ndarray::{Array1, Array2, Array4, s};
use rayon::prelude::*;

/// A 2D depthwise convolutional layer for neural networks.
///
/// Depthwise convolution applies a single convolutional filter per input channel,
/// unlike standard convolution which mixes channels. This operation significantly
/// reduces the number of parameters and computational cost while maintaining
/// spatial feature extraction capabilities.
///
/// The depthwise convolution operates on 4D tensors with shape \[batch_size, channels, height, width\].
/// Each channel is convolved independently with its own set of filters.
///
/// # Fields
///
/// - `filters` - Number of output channels (must equal input channels for pure depthwise convolution)
/// - `kernel_size` - Size of the convolution kernel as (height, width)
/// - `strides` - Stride of the convolution as (height_stride, width_stride)
/// - `padding` - Padding strategy (Valid or Same)
/// - `weights` - 4D weight tensor with shape \[filters, 1, kernel_height, kernel_width\]
/// - `bias` - 1D bias vector with shape \[filters\]
/// - `input` - Cached input tensor for backward pass
/// - `input_shape` - Shape of the input tensor
/// - `weight_gradients` - Gradients with respect to weights
/// - `bias_gradients` - Gradients with respect to bias
/// - `optimizer_cache` - Cache for optimizer state
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::prelude::*;
///
/// // Create test input data: [batch_size, channels, height, width]
/// let batch_size = 1;
/// let input_channels = 3;
/// let height = 4;
/// let width = 4;
///
/// let mut input_data = Array4::zeros((batch_size, input_channels, height, width));
///
/// // Set different values for each channel to verify that depthwise convolution processes channels separately
/// for c in 0..input_channels {
///     for h in 0..height {
///         for w in 0..width {
///             input_data[[0, c, h, w]] = (c * 10 + h * width + w) as f32;
///         }
///     }
/// }
///
/// let input = input_data.into_dyn();
///
/// // Test DepthwiseConv2D using Sequential model
/// let mut model = Sequential::new();
///
/// // Create layer and initialize weights
/// let mut depthwise_layer = DepthwiseConv2D::new(
///     input_channels, // filters must equal input_channels
///     (2, 2),         // kernel_size
///     (1, 1),         // strides
///     PaddingType::Valid,
/// );
/// depthwise_layer.initialize_weights(input_channels);
///
/// model
///     .add(depthwise_layer)
///     .compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // Forward propagation
/// let output = model.predict(&input);
///
/// // Check output shape
/// // Input: [1, 3, 4, 4], kernel (2,2), stride (1,1), valid padding
/// // Output should be: [1, 3, 3, 3]
/// assert_eq!(output.shape(), &[1, 3, 3, 3]);
///
/// // Print model structure
/// model.summary();
/// ```
pub struct DepthwiseConv2D {
    pub filters: usize,
    pub kernel_size: (usize, usize),
    pub strides: (usize, usize),
    pub padding: PaddingType,
    pub weights: Array4<f32>,
    pub bias: Array1<f32>,
    pub input: Option<Tensor>,
    pub input_shape: Vec<usize>,
    pub weight_gradients: Option<Array4<f32>>,
    pub bias_gradients: Option<Array1<f32>>,
    pub optimizer_cache: OptimizerCacheConv2D,
}

impl DepthwiseConv2D {
    /// Creates a new DepthwiseConv2D layer.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters (should equal input channels for pure depthwise)
    /// - `kernel_size` - Size of the convolution kernel as (height, width)
    /// - `strides` - Stride of the convolution as (height_stride, width_stride)
    /// - `padding` - Padding strategy (Valid or Same)
    ///
    /// # Returns
    ///
    /// * `Self` - A new DepthwiseConv2D instance with randomly initialized weights
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: PaddingType,
    ) -> Self {
        let (kernel_height, kernel_width) = kernel_size;

        // For depthwise convolution, each filter processes one input channel
        // Weight shape: [filters, 1, kernel_height, kernel_width]
        let weights = Array4::zeros((filters, 1, kernel_height, kernel_width));
        let bias = Array1::zeros(filters);

        Self {
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            input: None,
            input_shape: Vec::new(),
            weight_gradients: None,
            bias_gradients: None,
            optimizer_cache: OptimizerCacheConv2D::default(),
        }
    }

    /// Initializes the layer with random weights using Xavier initialization.
    ///
    /// # Parameters
    ///
    /// * `input_channels` - Number of input channels
    pub fn initialize_weights(&mut self, input_channels: usize) {
        assert_eq!(
            self.filters, input_channels,
            "For depthwise convolution, number of filters must equal input channels"
        );

        let (kernel_height, kernel_width) = self.kernel_size;
        let fan_in = kernel_height * kernel_width;
        let fan_out = kernel_height * kernel_width; // Each filter outputs to 1 channel
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        // Xavier initialization
        self.weights
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .for_each(|mut filter| {
                filter
                    .slice_mut(s![0, .., ..])
                    .par_mapv_inplace(|_| (rand::random::<f32>() - 0.5) * 2.0 * limit);
            });

        // bias initialization
        self.bias
            .par_mapv_inplace(|_| (rand::random::<f32>() - 0.5) * 0.1);
    }
}

impl Layer for DepthwiseConv2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());
        self.input_shape = input.shape().to_vec();

        let input_array = input.view();
        let (batch_size, channels, height, width) = (
            input_array.shape()[0],
            input_array.shape()[1],
            input_array.shape()[2],
            input_array.shape()[3],
        );

        assert_eq!(
            channels, self.filters,
            "Input channels must equal number of filters for depthwise convolution"
        );

        // Calculate output dimensions
        let output_shape = calculate_output_shape_2d(
            &self.input_shape,
            self.kernel_size,
            self.strides,
            &self.padding,
        );
        let (_, _, output_height, output_width) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );

        let mut output = Array4::zeros((batch_size, channels, output_height, output_width));

        output
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut batch_output)| {
                batch_output
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(c, mut channel_output)| {
                        // Extract single channel
                        let input_channel = input_array.slice(s![b, c, .., ..]).to_owned();

                        // Apply padding if needed
                        let padded_input = match self.padding {
                            PaddingType::Valid => input_channel,
                            PaddingType::Same => {
                                let pad_h = ((output_height - 1) * self.strides.0
                                    + self.kernel_size.0)
                                    .saturating_sub(height);
                                let pad_w = ((output_width - 1) * self.strides.1
                                    + self.kernel_size.1)
                                    .saturating_sub(width);
                                pad_tensor_2d(&input_channel, pad_h, pad_w)
                            }
                        };

                        // Perform depthwise convolution for this channel
                        let kernel = self.weights.slice(s![c, 0, .., ..]);

                        channel_output.indexed_iter_mut().par_bridge().for_each(
                            |((oh, ow), output_val)| {
                                let start_h = oh * self.strides.0;
                                let start_w = ow * self.strides.1;
                                let end_h = start_h + self.kernel_size.0;
                                let end_w = start_w + self.kernel_size.1;

                                if end_h <= padded_input.shape()[0]
                                    && end_w <= padded_input.shape()[1]
                                {
                                    let input_patch =
                                        padded_input.slice(s![start_h..end_h, start_w..end_w]);
                                    let conv_result = (&input_patch * &kernel).sum();
                                    *output_val = conv_result + self.bias[c];
                                }
                            },
                        );
                    });
            });

        output.into_dyn()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let input = self.input.as_ref().ok_or_else(|| {
            ModelError::ProcessingError("Forward pass has not been run".to_string())
        })?;

        let input_array = input.view();
        let grad_output_array = grad_output.view();

        let (batch_size, channels, input_height, input_width) = (
            input_array.shape()[0],
            input_array.shape()[1],
            input_array.shape()[2],
            input_array.shape()[3],
        );

        let (_, _, output_height, output_width) = (
            grad_output_array.shape()[0],
            grad_output_array.shape()[1],
            grad_output_array.shape()[2],
            grad_output_array.shape()[3],
        );

        // Initialize gradients
        let mut weight_grads = Array4::zeros(self.weights.raw_dim());
        let mut bias_grads = Array1::zeros(self.bias.raw_dim());
        let mut input_grads = Array4::zeros((batch_size, channels, input_height, input_width));

        // Parallel computation of bias gradients
        let bias_grads_vec: Vec<f32> = (0..channels)
            .into_par_iter()
            .map(|c| {
                let mut channel_sum = 0.0;
                for b in 0..batch_size {
                    channel_sum += grad_output_array.slice(s![b, c, .., ..]).sum();
                }
                channel_sum
            })
            .collect();

        // Copy results to bias gradients array
        for (c, &grad) in bias_grads_vec.iter().enumerate() {
            bias_grads[c] = grad;
        }

        // Parallel computation of weight gradients and input gradients
        let batch_channel_results: Vec<(Array4<f32>, Array4<f32>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_weight_grads = Array4::zeros(self.weights.raw_dim());
                let mut batch_input_grads = Array4::zeros((1, channels, input_height, input_width));

                for c in 0..channels {
                    // Weight and input gradients
                    let input_channel = input_array.slice(s![b, c, .., ..]);
                    let grad_channel = grad_output_array.slice(s![b, c, .., ..]);

                    // Apply padding to input as needed
                    let padded_input = match self.padding {
                        PaddingType::Valid => input_channel.to_owned(),
                        PaddingType::Same => {
                            let pad_h = ((output_height - 1) * self.strides.0 + self.kernel_size.0)
                                .saturating_sub(input_height);
                            let pad_w = ((output_width - 1) * self.strides.1 + self.kernel_size.1)
                                .saturating_sub(input_width);
                            pad_tensor_2d(&input_channel.to_owned(), pad_h, pad_w)
                        }
                    };

                    // Calculate weight gradients
                    for kh in 0..self.kernel_size.0 {
                        for kw in 0..self.kernel_size.1 {
                            let kernel_pos = (kh, kw);

                            // Parallelize inner loop
                            let weight_grad: f32 = (0..output_height)
                                .into_par_iter()
                                .map(|oh| {
                                    let mut row_sum = 0.0;
                                    for ow in 0..output_width {
                                        let ih = oh * self.strides.0 + kernel_pos.0;
                                        let iw = ow * self.strides.1 + kernel_pos.1;

                                        if ih < padded_input.shape()[0]
                                            && iw < padded_input.shape()[1]
                                        {
                                            row_sum +=
                                                padded_input[[ih, iw]] * grad_channel[[oh, ow]];
                                        }
                                    }
                                    row_sum
                                })
                                .sum();

                            batch_weight_grads[[c, 0, kh, kw]] += weight_grad;
                        }
                    }

                    // Calculate input gradients
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let grad_val = grad_channel[[oh, ow]];

                            // Parallel processing of kernel positions
                            let grad_updates: Vec<((usize, usize), f32)> = (0..self.kernel_size.0)
                                .into_par_iter()
                                .flat_map(|kh| {
                                    let mut row_updates = Vec::new();
                                    for kw in 0..self.kernel_size.1 {
                                        let ih = oh * self.strides.0 + kh;
                                        let iw = ow * self.strides.1 + kw;

                                        if ih < input_height && iw < input_width {
                                            row_updates.push((
                                                (ih, iw),
                                                self.weights[[c, 0, kh, kw]] * grad_val,
                                            ));
                                        }
                                    }
                                    row_updates
                                })
                                .collect();

                            // Apply gradient updates
                            for ((ih, iw), grad_update) in grad_updates {
                                batch_input_grads[[0, c, ih, iw]] += grad_update;
                            }
                        }
                    }
                }

                (batch_weight_grads, batch_input_grads)
            })
            .collect();

        // Merge batch results
        for (batch_weight_grads, batch_input_grads) in batch_channel_results {
            // Merge weight gradients
            weight_grads += &batch_weight_grads;

            // Merge input gradients
            for b in 0..batch_size {
                if b < batch_input_grads.shape()[0] {
                    for c in 0..channels {
                        for h in 0..input_height {
                            for w in 0..input_width {
                                input_grads[[b, c, h, w]] += batch_input_grads[[0, c, h, w]];
                            }
                        }
                    }
                }
            }
        }

        self.weight_gradients = Some(weight_grads);
        self.bias_gradients = Some(bias_grads);

        Ok(input_grads.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "DepthwiseConv2D"
    }

    fn output_shape(&self) -> String {
        if !self.input_shape.is_empty() {
            let output_shape = calculate_output_shape_2d(
                &self.input_shape,
                self.kernel_size,
                self.strides,
                &self.padding,
            );
            format!(
                "({}, {}, {}, {})",
                output_shape[0], output_shape[1], output_shape[2], output_shape[3]
            )
        } else {
            String::from("Unknown")
        }
    }

    fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    update_sgd_conv!();

    fn update_parameters_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: u64) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize Adam state (if not already initialized)
            if self.optimizer_cache.adam_states.is_none() {
                self.optimizer_cache.adam_states = Some(AdamStatesConv2D {
                    m: Array4::zeros(self.weights.raw_dim()),
                    v: Array4::zeros(self.weights.raw_dim()),
                    m_bias: Array2::zeros((1, self.bias.len())),
                    v_bias: Array2::zeros((1, self.bias.len())),
                });
            }

            if let Some(ref mut adam_states) = self.optimizer_cache.adam_states {
                // Calculate bias correction factors
                let bias_correction1 = 1.0 - beta1.powi(t as i32);
                let bias_correction2 = 1.0 - beta2.powi(t as i32);

                // Update weights using the update_adam_conv function from helper_functions
                if let (Some(weights_slice), Some(grads_slice), Some(m_slice), Some(v_slice)) = (
                    self.weights.as_slice_mut(),
                    weight_grads.as_slice(),
                    adam_states.m.as_slice_mut(),
                    adam_states.v.as_slice_mut(),
                ) {
                    update_adam_conv(
                        weights_slice,
                        grads_slice,
                        m_slice,
                        v_slice,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        bias_correction1,
                        bias_correction2,
                    );
                }

                // Update bias using the update_adam_conv function from helper_functions
                if let (
                    Some(bias_slice),
                    Some(bias_grads_slice),
                    Some(m_bias_slice),
                    Some(v_bias_slice),
                ) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    adam_states.m_bias.as_slice_mut(),
                    adam_states.v_bias.as_slice_mut(),
                ) {
                    update_adam_conv(
                        bias_slice,
                        bias_grads_slice,
                        m_bias_slice,
                        v_bias_slice,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        bias_correction1,
                        bias_correction2,
                    );
                }
            }
        }
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, epsilon: f32) {
        if let (Some(weight_grads), Some(bias_grads)) =
            (&self.weight_gradients, &self.bias_gradients)
        {
            // Initialize RMSprop cache (if not already initialized)
            if self.optimizer_cache.rmsprop_cache.is_none() {
                self.optimizer_cache.rmsprop_cache = Some(RMSpropCacheConv2D {
                    cache: Array4::zeros(self.weights.raw_dim()),
                    bias: Array2::zeros((1, self.bias.len())),
                });
            }

            if let Some(ref mut rmsprop_cache) = self.optimizer_cache.rmsprop_cache {
                // Update weights using the update_rmsprop function from helper_functions
                if let (Some(weights_slice), Some(grads_slice), Some(cache_slice)) = (
                    self.weights.as_slice_mut(),
                    weight_grads.as_slice(),
                    rmsprop_cache.cache.as_slice_mut(),
                ) {
                    update_rmsprop(weights_slice, grads_slice, cache_slice, rho, epsilon, lr);
                }

                // Update bias using the update_rmsprop function from helper_functions
                if let (Some(bias_slice), Some(bias_grads_slice), Some(bias_cache_slice)) = (
                    self.bias.as_slice_mut(),
                    bias_grads.as_slice(),
                    rmsprop_cache.bias.as_slice_mut(),
                ) {
                    update_rmsprop(
                        bias_slice,
                        bias_grads_slice,
                        bias_cache_slice,
                        rho,
                        epsilon,
                        lr,
                    );
                }
            }
        }
    }

    fn get_weights(&self) -> LayerWeight {
        LayerWeight::DepthwiseConv2DLayer(DepthwiseConv2DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
