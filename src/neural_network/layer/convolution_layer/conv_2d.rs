use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::activation_layer::Activation;
use crate::neural_network::layer::convolution_layer::PaddingType;
use crate::neural_network::layer::convolution_layer::convolution_engine::{
    conv_backward, conv_forward,
};
use crate::neural_network::layer::convolution_layer::validation::{
    validate_filters, validate_input_shape_2d, validate_kernel_size_2d, validate_strides_2d,
};
use crate::neural_network::layer::validation::validate_weight_shape;
use crate::neural_network::layer::layer_weight::{Conv2DLayerWeight, LayerWeight};
use crate::neural_network::neural_network_trait::{Layer, ParamGrad};
use ndarray::{Array2, Array4};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// A 2D convolutional layer for neural networks.
///
/// Applies a convolution operation to grid-like data such as images. Input shape is
/// \[batch_size, channels, height, width\] and output shape is
/// \[batch_size, filters, output_height, output_width\], where output dimensions depend on
/// input size, kernel size, strides, and padding.
///
/// The dimension-generic convolution math lives in
/// [`convolution_engine`](crate::neural_network::layer::convolution_layer); this layer holds the
/// weights, activation, and caches, and delegates the forward/backward numerics to it.
///
/// # Fields
///
/// - `filters` - Number of convolution filters (output channels).
/// - `kernel_size` - Size of the convolution kernel as (height, width).
/// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
/// - `padding` - Type of padding to apply (`Valid` or `Same`).
/// - `weights` - 4D array of filter weights with shape \[filters, channels, kernel_height, kernel_width\].
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
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::Array4;
///
/// // Create a simple 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 1 input channel, 5x5 pixels
/// let x = Array4::ones((2, 1, 5, 5)).into_dyn();
///
/// // Create target tensor - assuming we'll have 3 filters with output size 3x3
/// let y = Array4::ones((2, 3, 3, 3)).into_dyn();
///
/// // Build model: add a Conv2D layer with 3 filters and 3x3 kernel
/// let mut model = Sequential::new();
/// model
///     .add(Conv2D::new(
///         3,                      // Number of filters
///         (3, 3),                 // Kernel size
///         vec![2, 1, 5, 5],       // Input shape
///         (1, 1),                 // Stride
///         PaddingType::Valid,     // No padding
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
/// println!("Convolution layer prediction results: {:?}", prediction);
///
/// // Check if output shape is correct - should be [2, 3, 3, 3]
/// assert_eq!(prediction.shape(), &[2, 3, 3, 3]);
/// ```
pub struct Conv2D {
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: PaddingType,
    weights: Array4<f32>,
    bias: Array2<f32>,
    activation: Activation,
    output_cache: Option<Tensor>,
    input_cache: Option<Tensor>,
    input_shape: Vec<usize>,
    weight_gradients: Option<Array4<f32>>,
    bias_gradients: Option<Array2<f32>>,
}

impl Conv2D {
    /// Creates a new 2D convolutional layer with the specified parameters.
    ///
    /// Weights are initialized using Xavier (Glorot) uniform initialization.
    /// Biases are initialized to zeros.
    ///
    /// # Parameters
    ///
    /// - `filters` - Number of convolution filters (output channels).
    /// - `kernel_size` - Size of the convolution kernel as (height, width).
    /// - `input_shape` - Shape of the input tensor as \[batch_size, channels, height, width\].
    /// - `strides` - Stride values for the convolution operation as (vertical, horizontal).
    /// - `padding` - Type of padding to apply (`Valid` or `Same`).
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `Conv2D` layer instance with randomly initialized weights or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `filters` is 0
    /// - `Error::InvalidParameter` - If any kernel dimension or stride is 0
    /// - `Error::InvalidInput` - If `input_shape` is not 4D or has 0 channels
    /// - `Error::InvalidInput` - If input dimensions are smaller than kernel size
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: (usize, usize),
        padding: PaddingType,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_filters(filters)?;
        validate_kernel_size_2d(kernel_size)?;
        validate_strides_2d(strides)?;
        validate_input_shape_2d(&input_shape, kernel_size)?;

        // Shape is [batch_size, channels, height, width]
        let channels = input_shape[1];

        // Initialize weights using Xavier initialization for convolutional layers
        // Formula: sqrt(6 / (input_channels * kernel_area + filters * kernel_area))
        let fan_in = channels * kernel_size.0 * kernel_size.1;
        let fan_out = filters * kernel_size.0 * kernel_size.1;
        let weight_bound = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let weights = Array4::random(
            (filters, channels, kernel_size.0, kernel_size.1),
            Uniform::new(-weight_bound, weight_bound).unwrap(),
        );

        // Initialize biases to zero
        let bias = Array2::zeros((1, filters));

        Ok(Conv2D {
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

    /// Calculates the output shape of the convolutional layer based on input dimensions.
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let (output_height, output_width) = match self.padding {
            PaddingType::Valid => {
                let out_height = (input_height - self.kernel_size.0) / self.strides.0 + 1;
                let out_width = (input_width - self.kernel_size.1) / self.strides.1 + 1;
                (out_height, out_width)
            }
            PaddingType::Same => {
                let out_height = input_height.div_ceil(self.strides.0);
                let out_width = input_width.div_ceil(self.strides.1);
                (out_height, out_width)
            }
        };

        vec![batch_size, self.filters, output_height, output_width]
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `weights` - 4D array of filter weights with shape \[filters, channels, kernel_height, kernel_width\]
    /// - `bias` - 2D array of bias values with shape \[1, filters\]
    pub fn set_weights(
        &mut self,
        weights: Array4<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[self.strides.0, self.strides.1],
            self.padding,
        );
        let activated = self.activation.forward(&output)?;
        self.output_cache = Some(activated.clone());
        Ok(activated)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`](crate::neural_network::neural_network_trait::Layer::predict).
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Convolution (dimension-generic engine), then activation
        let output = conv_forward(
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            self.bias.as_slice().expect("bias must be contiguous"),
            &[self.strides.0, self.strides.1],
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
            .ok_or_else(|| Error::forward_pass_not_run("Conv2D"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("Conv2D"))?;

        let grads = conv_backward(
            &grad_upstream,
            input,
            self.weights.as_slice().expect("weights must be contiguous"),
            self.weights.shape(),
            &[self.strides.0, self.strides.1],
            self.padding,
        );

        self.weight_gradients = Some(
            Array4::from_shape_vec(self.weights.raw_dim(), grads.weight_grad)
                .expect("weight gradient shape matches weights"),
        );
        self.bias_gradients = Some(
            Array2::from_shape_vec(self.bias.raw_dim(), grads.bias_grad)
                .expect("bias gradient shape matches bias"),
        );

        Ok(grads.input_grad)
    }

    fn layer_type(&self) -> &str {
        "Conv2D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.calculate_output_shape(&self.input_shape);
        format!(
            "({}, {}, {}, {})",
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
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
        LayerWeight::Conv2D(Conv2DLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
