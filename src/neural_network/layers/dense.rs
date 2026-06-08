use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::layers::layer_weight::{DenseLayerWeight, LayerWeight};
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Dense (Fully Connected) layer implementation for neural networks.
///
/// This layer performs a linear transformation using a weight matrix and bias vector, optionally
/// followed by an activation function: output = activation(input * weights + bias). Input shape
/// is (batch_size, input_dim) and output shape is (batch_size, output_dim).
///
/// Weights are initialized with Xavier/Glorot initialization and biases start at zeros. During
/// training, the layer stores intermediate values for backpropagation and supports multiple
/// optimization algorithms including SGD, Adam, and RMSprop.
///
/// # Fields
///
/// - `input_dim` - Input dimension size
/// - `output_dim` - Output dimension size
/// - `weights` - Weight matrix with shape (input_dim, output_dim)
/// - `bias` - Bias vector with shape (1, output_dim)
/// - `input_cache` - Cache of the input from forward pass for use in backward pass
/// - `output_cache` - Cache of the activated output, used to backprop through the activation
/// - `grad_weights` - Stored weight gradients
/// - `grad_bias` - Stored bias gradients
/// - `activation` - Activation function applied to the linear output
///
/// # Examples
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::{Activation, Dense};
/// use rustyml::neural_network::optimizers::SGD;
/// use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
///
/// // Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 3, Activation::ReLU).unwrap())
///     .add(Dense::new(3, 1, Activation::ReLU).unwrap());
/// model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());
///
/// // Print model structure (summary)
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("Prediction results: {:?}", prediction);
/// ```
pub struct Dense {
    input_dim: usize,
    output_dim: usize,
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_cache: Option<Array2<f32>>,
    output_cache: Option<Tensor>,
    grad_weights: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,
    activation: Activation,
}

impl Dense {
    /// Creates a new dense layer with an activation function.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation applied to the linear output (any value convertible into
    ///   [`Activation`], e.g. `Activation::ReLU` or a standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Dense` layer instance with initialized parameters
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `input_dim` or `units` is zero
    pub fn new(
        input_dim: usize,
        units: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        // Validate that dimensions are greater than zero
        if input_dim == 0 {
            return Err(Error::invalid_parameter(
                "input_dim",
                "must be greater than 0",
            ));
        }
        if units == 0 {
            return Err(Error::invalid_parameter("units", "must be greater than 0"));
        }

        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let weights = Array::random((input_dim, units), Uniform::new(-limit, limit).unwrap());
        let bias = Array::zeros((1, units));
        Ok(Self {
            input_dim,
            output_dim: units,
            weights,
            bias,
            input_cache: None,
            output_cache: None,
            grad_weights: None,
            grad_bias: None,
            activation: activation.into(),
        })
    }

    /// Sets the weights and bias for this layer.
    ///
    /// # Parameters
    ///
    /// - `weights` - Weight matrix with shape (input_dim, output_dim)
    /// - `bias` - Bias vector with shape (1, output_dim)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` do not match the
    ///   layer's configured shape
    pub fn set_weights(
        &mut self,
        weights: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        // Force a contiguous standard layout: `parameters()` exposes the weights as a flat
        // mutable slice (`as_slice_mut().expect(..)`), which panics on a non-contiguous array.
        // A caller could pass e.g. a transposed/sliced view materialized in non-standard order.
        self.weights = weights.as_standard_layout().into_owned();
        self.bias = bias.as_standard_layout().into_owned();
        Ok(())
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 2D
        if input.ndim() != 2 {
            return Err(Error::invalid_input("input tensor is not 2D"));
        }

        let input_2d = input.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        // Input shape is [batch_size, input_dim]
        self.input_cache = Some(input_2d.to_owned());

        // Linear transform. ndarray's `dot` dispatches to the cache-blocked, SIMD `matrixmultiply`
        // kernel, which beats a hand-rolled parallel triple loop across all sizes.
        let z = input_2d.dot(&self.weights) + &self.bias;

        // Apply activation and cache the activated output for backpropagation
        let output = self.activation.forward(&z.into_dyn())?;
        self.output_cache = Some(output.clone());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 2D
        if input.ndim() != 2 {
            return Err(Error::invalid_input("input tensor is not 2D"));
        }

        let input_2d = input.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        // Linear transform (see `forward` for why `dot` rather than a hand-rolled loop).
        let z = input_2d.dot(&self.weights) + &self.bias;

        // Apply activation (no cache writes during inference)
        self.activation.forward(&z.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Backprop through the activation using the cached activated output
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Dense"))?;
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        // Convert gradient to 2D array with shape [batch_size, output_dim]
        let grad_upstream_2d = grad_upstream.into_dimensionality::<ndarray::Ix2>().unwrap();

        // Get input cache
        let input = self
            .input_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Dense"))?;

        // Weight gradients: grad_w = input^T · grad_upstream (matmul via `dot`).
        let grad_w = input.t().dot(&grad_upstream_2d);

        // Calculate bias gradients by summing over batch dimension
        let grad_b = grad_upstream_2d.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Ensure arrays are contiguous before storing (as_standard_layout() ensures contiguous memory layout)
        self.grad_weights = Some(grad_w.as_standard_layout().to_owned());
        self.grad_bias = Some(grad_b.as_standard_layout().to_owned());

        // Gradient w.r.t. the input: grad_input = grad_upstream · weights^T.
        let grad_input = grad_upstream_2d.dot(&self.weights.t());

        Ok(grad_input.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn output_shape(&self) -> String {
        // Returns only (None, output_dim)
        format!("(None, {})", self.output_dim)
    }

    fn param_count(&self) -> TrainingParameters {
        // Parameter count = number of weight parameters + number of bias parameters
        TrainingParameters::Trainable(self.input_dim * self.output_dim + self.output_dim)
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            weights,
            bias,
            grad_weights,
            grad_bias,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(grad_a), Some(grad_b)) = (grad_weights.as_ref(), grad_bias.as_ref()) {
            params.push(ParamGrad {
                value: weights.as_slice_mut().expect("weights must be contiguous"),
                grad: grad_a.as_slice().expect("grad_weights must be contiguous"),
            });
            params.push(ParamGrad {
                value: bias.as_slice_mut().expect("bias must be contiguous"),
                grad: grad_b.as_slice().expect("grad_bias must be contiguous"),
            });
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Dense(DenseLayerWeight {
            weight: &self.weights,
            bias: &self.bias,
        })
    }
}
