use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::convolution::PaddingType;
use crate::neural_network::layers::convolution::convolution_engine::{
    conv_backward, conv_forward,
};
use crate::neural_network::layers::convolution::validation::{
    validate_filters, validate_input_shape_3d, validate_kernel_size_3d, validate_strides_3d,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::layers::layer_weight::{Conv3DLayerWeight, LayerWeight};
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array5};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// A 3D convolutional layer for neural networks.
///
/// Applies a 3D convolution operation to volumetric data such as medical images, 3D models,
/// or video sequences. Input shape is \[batch_size, channels, depth, height, width\] and
/// output shape is \[batch_size, filters, output_depth, output_height, output_width\], where
/// output dimensions depend on input size, kernel size, strides, and padding.
///
/// The dimension-generic convolution math lives in
/// [`convolution_engine`](crate::neural_network::layers::convolution); this layer holds the
/// weights, activation, and caches, and delegates the forward/backward numerics to it.
///
/// # Fields
///
/// - `filters` - Number of convolution filters (output channels).
/// - `kernel_size` - Size of the convolution kernel as (depth, height, width).
/// - `strides` - Stride values for the convolution operation as (depth_stride, height_stride, width_stride).
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `weights` - 5D array of filter weights with shape \[filters, channels, kernel_depth, kernel_height, kernel_width\].
/// - `bias` - 2D array of bias values with shape \[1, filters\].
/// - `activation` - Activation applied to the convolution output.
/// - `output_cache` - Cached activated output, used by the activation backward pass.
/// - `input_cache` - Cached input from the forward pass, used during backpropagation.
/// - `input_shape` - Shape of the input tensor.
/// - `weight_gradients` - Gradients for the weights, computed during backpropagation.
/// - `bias_gradients` - Gradients for the biases, computed during backpropagation.
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
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
///         Activation::ReLU, // ReLU activation
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x).unwrap();
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
    bias: Array2<f32>,
    activation: Activation,
    output_cache: Option<Tensor>,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array5<f32>>,
    bias_gradients: Option<Array2<f32>>,
}

impl Conv3D {
    /// Creates a new Conv3D layer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of output filters
    /// - `kernel_size` - Size of the convolution kernel as (depth, height, width)
    /// - `input_shape` - Expected input shape as \[batch_size, channels, depth, height, width\]
    /// - `strides` - Stride values as (depth_stride, height_stride, width_stride)
    /// - `padding` - Padding type (Valid or Same)
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv3D` layer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 5D or has 0 dimensions
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize, usize),
        padding: PaddingType,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        // Validate input arguments
        validate_filters(filters)?;
        validate_kernel_size_3d(kernel_size)?;
        validate_strides_3d(strides)?;
        validate_input_shape_3d(&input_shape)?;

        let channels = input_shape[1];
        let (kd, kh, kw) = kernel_size;

        // Xavier/Glorot initialization
        let fan_in = channels * kd * kh * kw;
        let fan_out = filters * kd * kh * kw;
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        // Initialize weights with Xavier initialization
        let weights = Array5::random(
            (filters, channels, kd, kh, kw),
            Uniform::new(-limit, limit).unwrap(),
        );

        // Initialize bias to zeros
        let bias = Array2::zeros((1, filters));

        Ok(Conv3D {
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            activation: activation.into(),
            output_cache: None,
            input_cache: None,
            input_shape,
            weight_gradients: None,
            bias_gradients: None,
        })
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
            PaddingType::Same => (depth.div_ceil(sd), height.div_ceil(sh), width.div_ceil(sw)),
        };

        vec![
            batch_size,
            self.filters,
            output_depth,
            output_height,
            output_width,
        ]
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `weights` - 5D array of filter weights with shape \[filters, channels, kernel_depth, kernel_height, kernel_width\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    pub fn set_weights(
        &mut self,
        weights: Array5<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }
}

impl Layer for Conv3D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        // Cache input for backward pass
        self.input_cache = Some(input.clone());

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[self.strides.0, self.strides.1, self.strides.2],
            self.padding,
        );
        let activated = self.activation.forward(&output)?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 5D
        if input.ndim() != 5 {
            return Err(Error::invalid_input("input tensor is not 5D"));
        }

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[self.strides.0, self.strides.1, self.strides.2],
            self.padding,
        );
        let activated = self.activation.forward(&output)?;
        Ok(activated)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Apply activation backward pass
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Conv3D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| Error::computation("Input cache not available for backward pass"))?;

        let grads = conv_backward(
            &grad_upstream,
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.strides.0, self.strides.1, self.strides.2],
            self.padding,
        );

        self.weight_gradients = Some(
            Array5::from_shape_vec(self.weights.raw_dim(), grads.weight_grad)
                .expect("weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(
            Array2::from_shape_vec(self.bias.raw_dim(), grads.bias_grad)
                .expect("bias gradient shape matches bias"),
        );

        Ok(grads.input_grad)
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

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.weights.len() + self.bias.len())
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
        if let (Some(grad_a), Some(grad_b)) = (weight_gradients.as_ref(), bias_gradients.as_ref()) {
            params.push(ParamGrad {
                value: weights.as_slice_mut().expect("weights must be contiguous"),
                grad: grad_a
                    .as_slice()
                    .expect("weight_gradients must be contiguous"),
            });
            params.push(ParamGrad {
                value: bias.as_slice_mut().expect("bias must be contiguous"),
                grad: grad_b
                    .as_slice()
                    .expect("bias_gradients must be contiguous"),
            });
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Conv3D(Conv3DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
