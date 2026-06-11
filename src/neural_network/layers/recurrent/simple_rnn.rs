//! SimpleRNN layer: a basic recurrent layer that returns the last hidden state

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::layer_weight::{LayerWeight, SimpleRNNLayerWeight};
use crate::neural_network::layers::recurrent::validation::{
    validate_input_3d, validate_recurrent_dimensions,
};
use crate::neural_network::layers::recurrent::orthogonal_init;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Simple Recurrent Neural Network (SimpleRNN) layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). The activation is provided by
/// the activation module
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
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
/// .add(SimpleRNN::new(4, 3, Activation::Tanh, None).unwrap())
/// .compile(RMSprop::new(0.001, 0.9, 1e-8, None).unwrap(), MeanSquaredError::new());
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
#[derive(Debug)]
pub struct SimpleRNN {
    /// Number of input features
    input_dim: usize,
    /// Number of output units (neurons)
    units: usize,
    /// Weight matrix connecting inputs to the layer with shape (input_dim, units)
    kernel: Array2<f32>,
    /// Weight matrix connecting previous hidden states with shape (units, units)
    recurrent_kernel: Array2<f32>,
    /// Bias vector for the layer with shape (1, units)
    bias: Array2<f32>,
    /// Cached input tensor from the forward pass
    input_cache: Option<Array3<f32>>,
    /// Cached hidden states from the forward pass
    hidden_state_cache: Option<Vec<Array2<f32>>>,
    /// Gradient of the kernel weights
    grad_kernel: Option<Array2<f32>>,
    /// Gradient of the recurrent kernel weights
    grad_recurrent_kernel: Option<Array2<f32>>,
    /// Gradient of the bias
    grad_bias: Option<Array2<f32>>,
    /// Activation function applied at each timestep of the recurrence
    activation: Activation,
}

impl SimpleRNN {
    /// Creates a SimpleRNN layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Size of each input sample
    /// - `units` - Number of output units
    /// - `activation` - Activation function from the activation module (ReLU, Sigmoid, Tanh, Softmax)
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random
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
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        validate_recurrent_dimensions(input_dim, units)?;

        let mut rng = crate::random::make_rng(random_state);

        // Xavier/Glorot initialization for input kernel
        let limit = (6.0_f32 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random_using(
            (input_dim, units),
            Uniform::new(-limit, limit).unwrap(),
            &mut rng,
        );

        // Orthogonal initialization for recurrent kernel to maintain gradient flow
        let recurrent_kernel = orthogonal_init(units, &mut rng);

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

    /// Sets the weights for this layer
    ///
    /// # Parameters
    ///
    /// - `kernel` - Weight matrix connecting inputs to the layer with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any supplied matrix does not match the layer's existing shape
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

    /// Batched input projection: `x3 [batch, timesteps, input_dim] @ kernel` for all timesteps in a
    /// single gemm, returning `[batch, timesteps, units]`. Collapsing the (batch, timesteps) axes
    /// into one matmul beats `timesteps` separate small gemms on cache/SIMD utilization
    fn project_input(&self, x3: &ndarray::ArrayView3<f32>) -> Array3<f32> {
        let (batch, timesteps) = (x3.shape()[0], x3.shape()[1]);
        let x2 = x3
            .to_shape((batch * timesteps, self.input_dim))
            .expect("contiguous [batch*timesteps, input_dim] reshape");
        x2.dot(&self.kernel)
            .into_shape_with_order((batch, timesteps, self.units))
            .expect("reshape projection to [batch, timesteps, units]")
    }
}

impl Layer for SimpleRNN {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // input shape (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        // The input projection X @ W does not depend on the recurrence, so compute it for every
        // timestep in one gemm: [batch*timesteps, input_dim] @ [input_dim, units]. `xw[:, t, :]` is
        // x_t @ W. Only h_{t-1} @ U below has to stay sequential
        let xw = self.project_input(&x3);

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut hs = Vec::with_capacity(timesteps + 1);
        hs.push(h_prev.clone());

        // sequential timestep processing is required for an RNN
        for t in 0..timesteps {
            // z = x_t @ W + h_{t-1} @ U + b
            let z = h_prev.dot(&self.recurrent_kernel) + xw.index_axis(Axis(1), t) + &self.bias;

            let h_t = self
                .activation
                .forward(&z.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            h_prev = h_t.clone();
            hs.push(h_prev.clone());
        }
        self.hidden_state_cache = Some(hs);
        Ok(h_prev.into_dyn()) // last timestep's hidden state
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // input shape (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);

        // Batched input projection (see `forward`); only the recurrence stays sequential
        let xw = self.project_input(&x3);

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));

        // sequential timestep processing is required for an RNN
        for t in 0..timesteps {
            // z = x_t @ W + h_{t-1} @ U + b
            let z = h_prev.dot(&self.recurrent_kernel) + xw.index_axis(Axis(1), t) + &self.bias;

            let h_t = self
                .activation
                .forward(&z.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            h_prev = h_t;
        }
        Ok(h_prev.into_dyn()) // last timestep's hidden state
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

        // Fresh per-call gradient buffers (replace semantics, matching Dense/LSTM/GRU). The
        // optimizer reads grads without clearing them and there is no zero_grad step, so reusing
        // the previous call's buffers would sum gradients across every batch/epoch
        let mut grad_k = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_rk = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_b = Array2::<f32>::zeros((1, self.units));
        // Per-timestep d_z, stored so the input-side reductions can batch into single gemms. Only
        // `grad_h = d_z @ U^T` (the recurrence) has to be computed step by step
        let mut dz_all = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut grad_h = grad_h_t;
        // backpropagation through time (BPTT)
        for t in (0..timesteps).rev() {
            // backprop through activation using this timestep's cached output `hs[t + 1]`
            // every supported activation's derivative is a function of its output
            let d_z = {
                let h_t = hs[t + 1].clone().into_dyn();
                let grad_h_dyn = grad_h.clone().into_dyn();
                let grad_z_dyn = self.activation.backward(&h_t, &grad_h_dyn)?;
                grad_z_dyn.into_dimensionality::<ndarray::Ix2>().unwrap()
            };

            // gradient w.r.t. previous hidden state, used by the next iteration (sequential)
            grad_h = d_z.dot(&self.recurrent_kernel.t());
            dz_all.index_axis_mut(Axis(1), t).assign(&d_z);
        }

        // Batched reductions over all timesteps. With DZ and X stacked as [batch*timesteps, *]:
        //   grad_k = X^T @ DZ, grad_x = DZ @ W^T, grad_b = sum_rows(DZ); and with H_prev stacked
        //   from hs[0..timesteps], grad_rk = H_prev^T @ DZ
        let dz_flat = dz_all
            .to_shape((batch * timesteps, self.units))
            .expect("contiguous DZ reshape");
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");

        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, self.units));
        for (mut dst, h) in h_prev3.axis_iter_mut(Axis(1)).zip(hs.iter()) {
            dst.assign(h);
        }
        let h_prev_flat = h_prev3
            .to_shape((batch * timesteps, self.units))
            .expect("contiguous H_prev reshape");

        grad_k += &x_flat.t().dot(&dz_flat);
        grad_rk += &h_prev_flat.t().dot(&dz_flat);
        grad_b += &dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_x3 = dz_flat
            .dot(&self.kernel.t())
            .into_shape_with_order((batch, timesteps, feat))
            .expect("reshape grad_x to [batch, timesteps, feat]");

        // Gradient clipping is no longer applied here; configure clip-by-global-norm on the
        // optimizer (the `clip_norm` constructor argument) if exploding gradients need taming
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
