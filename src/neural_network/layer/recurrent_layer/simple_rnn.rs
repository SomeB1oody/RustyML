use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::activation_layer::Activation;
use crate::neural_network::layer::validation::validate_weight_shape;
use crate::neural_network::layer::layer_weight::{LayerWeight, SimpleRNNLayerWeight};
use crate::neural_network::layer::recurrent_layer::validation::{
    validate_input_3d, validate_recurrent_dimensions,
};
use crate::neural_network::layer::recurrent_layer::{GRADIENT_CLIP_VALUE, orthogonal_init};
use crate::neural_network::neural_network_trait::{Layer, ParamGrad};
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Simple Recurrent Neural Network (SimpleRNN) layer implementation.
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). The activation is provided by
/// the activation_layer module.
///
/// # Fields
///
/// - `input_dim` - Number of input features
/// - `units` - Number of output units (neurons)
/// - `kernel` - Weight matrix connecting inputs to the layer with shape (input_dim, units)
/// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape (units, units)
/// - `bias` - Bias vector for the layer with shape (1, units)
/// - `input_cache` - Cached input tensor from the forward pass
/// - `hidden_state_cache` - Cached hidden states from the forward pass
/// - `grad_kernel` - Gradient of the kernel weights
/// - `grad_recurrent_kernel` - Gradient of the recurrent kernel weights
/// - `grad_bias` - Gradient of the bias
/// - `activation` - Activation function applied at each timestep of the recurrence
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layer::*;
/// use rustyml::neural_network::optimizer::*;
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::Array;
///
/// // Create input with batch_size=2, timesteps=5, input_dim=4,
/// // and target with batch_size=2, units=3 (same dimension as the last hidden state)
/// let x = Array::ones((2, 5, 4)).into_dyn();
/// let y = Array::ones((2, 3)).into_dyn();
///
/// // Build model: one SimpleRnn layer with Tanh activation
/// let mut model = Sequential::new();
/// model
/// .add(SimpleRNN::new(4, 3, Activation::Tanh).unwrap())
/// .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Print structure
/// model.summary();
///
/// // Train for 1 epoch
/// model.fit(&x, &y, 1).unwrap();
///
/// // Predict
/// let pred = model.predict(&x);
/// println!("SimpleRnn prediction:\n{:#?}\n", pred);
/// ```
pub struct SimpleRNN {
    input_dim: usize,
    units: usize,
    kernel: Array2<f32>,
    recurrent_kernel: Array2<f32>,
    bias: Array2<f32>,
    input_cache: Option<Array3<f32>>,
    hidden_state_cache: Option<Vec<Array2<f32>>>,
    grad_kernel: Option<Array2<f32>>,
    grad_recurrent_kernel: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,
    activation: Activation,
}

impl SimpleRNN {
    /// Creates a SimpleRNN layer with the specified dimensions and activation.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Size of each input sample
    /// - `units` - Number of output units
    /// - `activation` - Activation layer from activation_layer module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SimpleRNN layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `input_dim` or `units` is 0
    pub fn new(
        input_dim: usize,
        units: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        validate_recurrent_dimensions(input_dim, units)?;

        // Xavier/Glorot initialization for input kernel
        let limit = (6.0_f32 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random((input_dim, units), Uniform::new(-limit, limit).unwrap());

        // Orthogonal initialization for recurrent kernel to maintain gradient flow
        let recurrent_kernel = orthogonal_init(units);

        let bias = Array::zeros((1, units));
        Ok(SimpleRNN {
            input_dim,
            units,
            kernel,
            recurrent_kernel,
            bias,
            input_cache: None,
            hidden_state_cache: None,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
            activation: activation.into(),
        })
    }

    /// Sets the weights for this layer.
    ///
    /// # Parameters
    ///
    /// - `kernel` - Weight matrix connecting inputs to the layer with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    pub fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape("kernel", self.kernel.shape(), kernel.shape())?;
        validate_weight_shape(
            "recurrent_kernel",
            self.recurrent_kernel.shape(),
            recurrent_kernel.shape(),
        )?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.kernel = kernel;
        self.recurrent_kernel = recurrent_kernel;
        self.bias = bias;
        Ok(())
    }
}

impl Layer for SimpleRNN {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape=(batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut hs = Vec::with_capacity(timesteps + 1);
        hs.push(h_prev.clone());

        // Sequential timestep processing (required for RNN)
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t); // (batch, input_dim)

            // Compute: z = x_t @ W + h_{t-1} @ U + b
            let z = x_t.dot(&self.kernel) + h_prev.dot(&self.recurrent_kernel) + &self.bias;

            // Apply activation
            let h_t = self
                .activation
                .forward(&z.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            h_prev = h_t.clone();
            hs.push(h_prev.clone());
        }
        self.hidden_state_cache = Some(hs);
        Ok(h_prev.into_dyn()) // Return hidden state of the last timestep
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 3D
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape=(batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));

        // Sequential timestep processing (required for RNN)
        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t); // (batch, input_dim)

            // Compute: z = x_t @ W + h_{t-1} @ U + b
            let z = x_t.dot(&self.kernel) + h_prev.dot(&self.recurrent_kernel) + &self.bias;

            // Apply activation
            let h_t = self
                .activation
                .forward(&z.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            h_prev = h_t;
        }
        Ok(h_prev.into_dyn()) // Return hidden state of the last timestep
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let grad_h_t = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        fn take_cache<T>(cache: &mut Option<T>, layer: &'static str) -> Result<T, Error> {
            cache
                .take()
                .ok_or_else(|| Error::forward_pass_not_run(layer))
        }

        let x3 = take_cache(&mut self.input_cache, "SimpleRNN")?;
        let hs = take_cache(&mut self.hidden_state_cache, "SimpleRNN")?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // Initialize or reuse existing gradients for accumulation
        let mut grad_k = self
            .grad_kernel
            .take()
            .unwrap_or_else(|| Array2::<f32>::zeros((self.input_dim, self.units)));
        let mut grad_rk = self
            .grad_recurrent_kernel
            .take()
            .unwrap_or_else(|| Array2::<f32>::zeros((self.units, self.units)));
        let mut grad_b = self
            .grad_bias
            .take()
            .unwrap_or_else(|| Array2::<f32>::zeros((1, self.units)));
        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        let mut grad_h = grad_h_t;
        // Backpropagation Through Time (BPTT)
        for t in (0..timesteps).rev() {
            let h_tm1 = &hs[t];

            // Backprop through the activation using THIS timestep's cached output `h_t`
            // (`hs[t + 1]`); every supported activation's derivative is a function of its output.
            let d_z = {
                let h_t = hs[t + 1].clone().into_dyn();
                let grad_h_dyn = grad_h.clone().into_dyn();
                let grad_z_dyn = self.activation.backward(&h_t, &grad_h_dyn)?;
                grad_z_dyn.into_dimensionality::<ndarray::Ix2>().unwrap()
            };

            // Accumulate gradients for weights
            let x_t = x3.index_axis(Axis(1), t);
            grad_k += &x_t.t().dot(&d_z);
            grad_rk += &h_tm1.t().dot(&d_z);
            grad_b += &d_z.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Gradient w.r.t. input at timestep t
            grad_x3
                .index_axis_mut(Axis(1), t)
                .assign(&d_z.dot(&self.kernel.t()));

            // Gradient w.r.t. previous hidden state (for next iteration)
            grad_h = d_z.dot(&self.recurrent_kernel.t());
        }

        // Apply gradient clipping to prevent exploding gradients
        grad_k.mapv_inplace(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        grad_rk.mapv_inplace(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        grad_b.mapv_inplace(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));

        self.grad_kernel = Some(grad_k);
        self.grad_recurrent_kernel = Some(grad_rk);
        self.grad_bias = Some(grad_b);

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "SimpleRNN"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            self.input_dim * self.units + self.units * self.units + self.units,
        )
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            kernel,
            recurrent_kernel,
            bias,
            grad_kernel,
            grad_recurrent_kernel,
            grad_bias,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(gk), Some(grk), Some(gb)) = (
            grad_kernel.as_ref(),
            grad_recurrent_kernel.as_ref(),
            grad_bias.as_ref(),
        ) {
            params.push(ParamGrad {
                value: kernel.as_slice_mut().expect("kernel must be contiguous"),
                grad: gk.as_slice().expect("kernel gradient must be contiguous"),
            });
            params.push(ParamGrad {
                value: recurrent_kernel
                    .as_slice_mut()
                    .expect("recurrent kernel must be contiguous"),
                grad: grk
                    .as_slice()
                    .expect("recurrent kernel gradient must be contiguous"),
            });
            params.push(ParamGrad {
                value: bias.as_slice_mut().expect("bias must be contiguous"),
                grad: gb.as_slice().expect("bias gradient must be contiguous"),
            });
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::SimpleRNN(SimpleRNNLayerWeight {
            kernel: &self.kernel,
            recurrent_kernel: &self.recurrent_kernel,
            bias: &self.bias,
        })
    }
}
