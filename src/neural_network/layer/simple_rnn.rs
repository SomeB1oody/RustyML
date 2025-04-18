use crate::ModelError;
use crate::neural_network::{Activation, Layer, Tensor};
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// SimpleRNN layer
pub struct SimpleRNN {
    input_dim: usize,
    units: usize,
    kernel: Array2<f32>,           // (input_dim, units)
    recurrent_kernel: Array2<f32>, // (units, units)
    bias: Array2<f32>,             // (1, units)

    // Cache
    input_cache: Option<Array3<f32>>, // (batch, timesteps, input_dim)
    hidden_state_cache: Option<Vec<Array2<f32>>>, // len = timesteps+1

    // Gradients
    grad_kernel: Option<Array2<f32>>,
    grad_recurrent_kernel: Option<Array2<f32>>,
    grad_bias: Option<Array2<f32>>,

    // Adam states
    m_kernel: Option<Array2<f32>>,
    v_kernel: Option<Array2<f32>>,
    m_recurrent_kernel: Option<Array2<f32>>,
    v_recurrent_kernel: Option<Array2<f32>>,
    m_bias: Option<Array2<f32>>,
    v_bias: Option<Array2<f32>>,

    // RMSprop cache
    cache_kernel: Option<Array2<f32>>,
    cache_recurrent_kernel: Option<Array2<f32>>,
    cache_bias: Option<Array2<f32>>,

    activation: Activation, // default tanh
}

impl SimpleRNN {
    pub fn new(input_dim: usize, units: usize) -> Self {
        Self::new_with_activation(input_dim, units, Activation::Tanh)
    }
    pub fn new_with_activation(input_dim: usize, units: usize, activation: Activation) -> Self {
        let kernel = Array::random((input_dim, units), Uniform::new(-0.05, 0.05));
        let recurrent_kernel = Array::random((units, units), Uniform::new(-0.05, 0.05));
        let bias = Array::zeros((1, units));
        SimpleRNN {
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
            m_kernel: None,
            v_kernel: None,
            m_recurrent_kernel: None,
            v_recurrent_kernel: None,
            m_bias: None,
            v_bias: None,
            cache_kernel: None,
            cache_recurrent_kernel: None,
            cache_bias: None,
            activation,
        }
    }
}

impl Layer for SimpleRNN {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Input shape=(batch, timesteps, input_dim)
        let x3 = input.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.clone());

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut hs = Vec::with_capacity(timesteps + 1);
        hs.push(h_prev.clone());

        for t in 0..timesteps {
            let x_t = x3.index_axis(Axis(1), t).to_owned(); // (batch, input_dim)
            let z = x_t.dot(&self.kernel) + h_prev.dot(&self.recurrent_kernel) + &self.bias;
            let h_t = Activation::apply_activation(&z, &self.activation);
            hs.push(h_t.clone());
            h_prev = h_t;
        }
        self.hidden_state_cache = Some(hs);
        h_prev.into_dyn() // Return hidden state of the last timestep
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        let grad_h_t = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let x3 = match self.input_cache.take() {
            Some(x3) => x3,
            None => Err(ModelError::ProcessingError(String::from(
                "Forward pass has not been run",
            )))?,
        };
        let hs = match self.hidden_state_cache.take() {
            Some(x3) => x3,
            None => Err(ModelError::ProcessingError(String::from(
                "Forward pass has not been run",
            )))?,
        };
        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        let mut grad_k = Array2::<f32>::zeros((self.input_dim, self.units));
        let mut grad_rk = Array2::<f32>::zeros((self.units, self.units));
        let mut grad_b = Array2::<f32>::zeros((1, self.units));
        let mut grad_x3 = Array3::<f32>::zeros((batch, timesteps, feat));

        let mut grad_h = grad_h_t;
        // BPTT
        for t in (0..timesteps).rev() {
            let h_t = hs[t + 1].clone();
            let h_tm1 = hs[t].clone();
            let d_z = if self.activation == Activation::Softmax {
                Activation::softmax_backward(&h_t, &grad_h)
            } else {
                let d_act = Activation::activation_derivative(&h_t, &self.activation);
                d_act * &grad_h
            };
            let x_t = x3.index_axis(Axis(1), t).to_owned();
            grad_k = grad_k + &x_t.t().dot(&d_z);
            grad_rk = grad_rk + &h_tm1.t().dot(&d_z);
            let db = d_z.sum_axis(Axis(0)).insert_axis(Axis(0));
            grad_b = grad_b + &db;
            let dx = d_z.dot(&self.kernel.t());
            grad_x3.index_axis_mut(Axis(1), t).assign(&dx);
            grad_h = d_z.dot(&self.recurrent_kernel.t());
        }

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
    fn param_count(&self) -> usize {
        self.input_dim * self.units + self.units * self.units + self.units
    }

    fn update_parameters(&mut self, lr: f32) {
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            self.kernel = &self.kernel - &(lr * gk);
            self.recurrent_kernel = &self.recurrent_kernel - &(lr * grk);
            self.bias = &self.bias - &(lr * gb);
        }
    }

    fn update_parameters_adam(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, t: u64) {
        // Same Adam implementation as in Dense, but for kernel/recurrent_kernel/bias
        if self.m_kernel.is_none() {
            let dims_k = (self.input_dim, self.units);
            let dims_r = (self.units, self.units);
            self.m_kernel = Some(Array2::zeros(dims_k));
            self.v_kernel = Some(Array2::zeros(dims_k));
            self.m_recurrent_kernel = Some(Array2::zeros(dims_r));
            self.v_recurrent_kernel = Some(Array2::zeros(dims_r));
            self.m_bias = Some(Array2::zeros((1, self.units)));
            self.v_bias = Some(Array2::zeros((1, self.units)));
        }
        let m_k = self.m_kernel.as_mut().unwrap();
        let v_k = self.v_kernel.as_mut().unwrap();
        let m_rk = self.m_recurrent_kernel.as_mut().unwrap();
        let v_rk = self.v_recurrent_kernel.as_mut().unwrap();
        let m_b = self.m_bias.as_mut().unwrap();
        let v_b = self.v_bias.as_mut().unwrap();

        let gk = self.grad_kernel.as_ref().unwrap();
        let grk = self.grad_recurrent_kernel.as_ref().unwrap();
        let gb = self.grad_bias.as_ref().unwrap();

        *m_k = m_k.mapv(|x| x * b1) + &(gk * (1.0 - b1));
        *m_rk = m_rk.mapv(|x| x * b1) + &(grk * (1.0 - b1));
        *m_b = m_b.mapv(|x| x * b1) + &(gb * (1.0 - b1));

        *v_k = v_k.mapv(|x| x * b2) + &(gk.mapv(|x| x * x) * (1.0 - b2));
        *v_rk = v_rk.mapv(|x| x * b2) + &(grk.mapv(|x| x * x) * (1.0 - b2));
        *v_b = v_b.mapv(|x| x * b2) + &(gb.mapv(|x| x * x) * (1.0 - b2));

        let mhat_k = m_k.mapv(|x| x / (1.0 - b1.powi(t as i32)));
        let mhat_rk = m_rk.mapv(|x| x / (1.0 - b1.powi(t as i32)));
        let mhat_b = m_b.mapv(|x| x / (1.0 - b1.powi(t as i32)));
        let vhat_k = v_k.mapv(|x| x / (1.0 - b2.powi(t as i32)));
        let vhat_rk = v_rk.mapv(|x| x / (1.0 - b2.powi(t as i32)));
        let vhat_b = v_b.mapv(|x| x / (1.0 - b2.powi(t as i32)));

        self.kernel = &self.kernel - &(lr * &mhat_k / &(vhat_k.mapv(f32::sqrt) + eps));
        self.recurrent_kernel =
            &self.recurrent_kernel - &(lr * &mhat_rk / &(vhat_rk.mapv(f32::sqrt) + eps));
        self.bias = &self.bias - &(lr * &mhat_b / &(vhat_b.mapv(f32::sqrt) + eps));
    }

    fn update_parameters_rmsprop(&mut self, lr: f32, rho: f32, eps: f32) {
        if let (Some(gk), Some(grk), Some(gb)) = (
            &self.grad_kernel,
            &self.grad_recurrent_kernel,
            &self.grad_bias,
        ) {
            if self.cache_kernel.is_none() {
                self.cache_kernel = Some(Array2::zeros((self.input_dim, self.units)));
            }
            if self.cache_recurrent_kernel.is_none() {
                self.cache_recurrent_kernel = Some(Array2::zeros((self.units, self.units)));
            }
            if self.cache_bias.is_none() {
                self.cache_bias = Some(Array2::zeros((1, self.units)));
            }
            let ck = self.cache_kernel.as_mut().unwrap();
            let crk = self.cache_recurrent_kernel.as_mut().unwrap();
            let cb = self.cache_bias.as_mut().unwrap();

            *ck = ck.mapv(|x| x * rho) + &(gk.mapv(|x| x * x) * (1.0 - rho));
            *crk = crk.mapv(|x| x * rho) + &(grk.mapv(|x| x * x) * (1.0 - rho));
            *cb = cb.mapv(|x| x * rho) + &(gb.mapv(|x| x * x) * (1.0 - rho));

            self.kernel = &self.kernel - &(lr * gk / &(ck.mapv(f32::sqrt) + eps));
            self.recurrent_kernel =
                &self.recurrent_kernel - &(lr * grk / &(crk.mapv(f32::sqrt) + eps));
            self.bias = &self.bias - &(lr * gb / &(cb.mapv(f32::sqrt) + eps));
        }
    }
}
