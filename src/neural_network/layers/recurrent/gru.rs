//! Gated Recurrent Unit (GRU) layer with reset, update, and candidate gates

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::layer_weight::{GRUGateWeight, GRULayerWeight, LayerWeight};
use crate::neural_network::layers::recurrent::apply_sigmoid;
use crate::neural_network::layers::recurrent::gate::{
    Gate, gate_value_from_projection, project_gate_input, store_gate_gradients, take_cache,
};
use crate::neural_network::layers::recurrent::validation::{
    validate_input_3d, validate_recurrent_dimensions,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3, Axis};

/// Threshold for switching the GRU layer to parallel computation
///
/// Parallel execution kicks in once `batch_size * units >= this value`, below which
/// sequential execution avoids rayon's thread-pool overhead. Tuned from benchmarks
const GRU_PARALLEL_THRESHOLD: usize = 1024;

/// Gated Recurrent Unit (GRU) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses reset, update, and candidate
/// gates to control information flow and mitigate vanishing gradients
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
/// // Create input data: batch_size=2, timesteps=5, features=4
/// let input = Array::ones((2, 5, 4)).into_dyn();
/// let target = Array::ones((2, 3)).into_dyn(); // batch_size=2, units=3
///
/// // Create GRU layer with 4 input features, 3 units, Tanh activation
/// let mut model = Sequential::new();
/// model.add(GRU::new(4, 3, Activation::Tanh, None).unwrap())
///      .compile(RMSprop::new(0.001, 0.9, 1e-8, None, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input).unwrap();
/// println!("GRU output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
#[derive(Debug)]
pub struct GRU {
    /// Dimensionality of input features
    input_dim: usize,
    /// Number of GRU units (neurons) in the layer
    units: usize,

    /// Gate controlling what information to forget from the previous hidden state
    reset_gate: Gate,
    /// Gate controlling how much to update the hidden state
    update_gate: Gate,
    /// Gate proposing new candidate values for the hidden state
    candidate_gate: Gate,

    /// Cached input tensor for backward propagation
    input_cache: Option<Array3<f32>>,
    /// Cached hidden states h_t for each timestep
    hidden_cache: Option<Vec<Array2<f32>>>,

    /// Cached reset gate activations (sigmoid applied) for each timestep
    r_cache: Option<Vec<Array2<f32>>>,
    /// Cached update gate activations (sigmoid applied) for each timestep
    z_cache: Option<Vec<Array2<f32>>>,
    /// Cached candidate hidden states (activation applied) for each timestep
    h_candidate_cache: Option<Vec<Array2<f32>>>,
    /// Cached r_t .* h_{t-1} values for each timestep
    rh_cache: Option<Vec<Array2<f32>>>,

    /// Activation applied to the candidate hidden state each timestep (Keras-style)
    activation: Activation,
}

impl GRU {
    /// Creates a GRU layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of GRU units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation layer from the activation module (ReLU, Sigmoid, Tanh, Softmax)
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new GRU layer instance
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

        // One RNG threaded through all three gates so the layer shares a single
        // reproducible initialization stream
        let mut rng = crate::random::make_rng(random_state);

        Ok(Self {
            input_dim,
            units,
            reset_gate: Gate::new(input_dim, units, 0.0, &mut rng)?,
            update_gate: Gate::new(input_dim, units, 0.0, &mut rng)?,
            candidate_gate: Gate::new(input_dim, units, 0.0, &mut rng)?,
            input_cache: None,
            hidden_cache: None,
            r_cache: None,
            z_cache: None,
            h_candidate_cache: None,
            rh_cache: None,
            activation: activation.into(),
        })
    }

    /// Sets the weights for all three gates in this GRU layer
    ///
    /// # Parameters
    ///
    /// - `reset_kernel` - Input kernel for the reset gate with shape (input_dim, units)
    /// - `reset_recurrent_kernel` - Recurrent kernel for the reset gate with shape (units, units)
    /// - `reset_bias` - Bias for the reset gate with shape (1, units)
    /// - `update_kernel` - Input kernel for the update gate with shape (input_dim, units)
    /// - `update_recurrent_kernel` - Recurrent kernel for the update gate with shape (units, units)
    /// - `update_bias` - Bias for the update gate with shape (1, units)
    /// - `candidate_kernel` - Input kernel for the candidate gate with shape (input_dim, units)
    /// - `candidate_recurrent_kernel` - Recurrent kernel for the candidate gate with shape (units, units)
    /// - `candidate_bias` - Bias for the candidate gate with shape (1, units)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any supplied weight shape does not match the gate
    #[allow(clippy::too_many_arguments)] // three gates x (kernel, recurrent_kernel, bias)
    pub fn set_weights(
        &mut self,
        reset_kernel: Array2<f32>,
        reset_recurrent_kernel: Array2<f32>,
        reset_bias: Array2<f32>,
        update_kernel: Array2<f32>,
        update_recurrent_kernel: Array2<f32>,
        update_bias: Array2<f32>,
        candidate_kernel: Array2<f32>,
        candidate_recurrent_kernel: Array2<f32>,
        candidate_bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape(
            "reset_kernel",
            self.reset_gate.kernel.shape(),
            reset_kernel.shape(),
        )?;
        validate_weight_shape(
            "reset_recurrent_kernel",
            self.reset_gate.recurrent_kernel.shape(),
            reset_recurrent_kernel.shape(),
        )?;
        validate_weight_shape(
            "reset_bias",
            self.reset_gate.bias.shape(),
            reset_bias.shape(),
        )?;
        validate_weight_shape(
            "update_kernel",
            self.update_gate.kernel.shape(),
            update_kernel.shape(),
        )?;
        validate_weight_shape(
            "update_recurrent_kernel",
            self.update_gate.recurrent_kernel.shape(),
            update_recurrent_kernel.shape(),
        )?;
        validate_weight_shape(
            "update_bias",
            self.update_gate.bias.shape(),
            update_bias.shape(),
        )?;
        validate_weight_shape(
            "candidate_kernel",
            self.candidate_gate.kernel.shape(),
            candidate_kernel.shape(),
        )?;
        validate_weight_shape(
            "candidate_recurrent_kernel",
            self.candidate_gate.recurrent_kernel.shape(),
            candidate_recurrent_kernel.shape(),
        )?;
        validate_weight_shape(
            "candidate_bias",
            self.candidate_gate.bias.shape(),
            candidate_bias.shape(),
        )?;

        self.reset_gate.kernel = reset_kernel;
        self.reset_gate.recurrent_kernel = reset_recurrent_kernel;
        self.reset_gate.bias = reset_bias;

        self.update_gate.kernel = update_kernel;
        self.update_gate.recurrent_kernel = update_recurrent_kernel;
        self.update_gate.bias = update_bias;

        self.candidate_gate.kernel = candidate_kernel;
        self.candidate_gate.recurrent_kernel = candidate_recurrent_kernel;
        self.candidate_gate.bias = candidate_bias;

        Ok(())
    }
}

impl Layer for GRU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));

        // Per-timestep storage for the backward pass
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut r_vals = Vec::with_capacity(timesteps);
        let mut z_vals = Vec::with_capacity(timesteps);
        let mut h_candidate_vals = Vec::with_capacity(timesteps);
        let mut rh_vals = Vec::with_capacity(timesteps);

        hs.push(h_prev.clone());

        // Use parallel execution once the computational load clears the threshold
        let use_parallel = batch * self.units >= GRU_PARALLEL_THRESHOLD;

        // Configurable activation (Copy) applied to the candidate hidden state
        let act = self.activation;

        // Batched input projection for all 3 gates (one big gemm each)
        let xw_r = project_gate_input(&self.reset_gate, &x3);
        let xw_z = project_gate_input(&self.update_gate, &x3);
        let xw_c = project_gate_input(&self.candidate_gate, &x3);

        for t in 0..timesteps {
            let xw_r_t = xw_r.index_axis(Axis(1), t);
            let xw_z_t = xw_z.index_axis(Axis(1), t);
            let xw_c_t = xw_c.index_axis(Axis(1), t);

            // Reset and update gate values (parallel or sequential)
            let (r_raw, z_raw) = if use_parallel {
                rayon::join(
                    || gate_value_from_projection(&self.reset_gate, &xw_r_t, &h_prev),
                    || gate_value_from_projection(&self.update_gate, &xw_z_t, &h_prev),
                )
            } else {
                (
                    gate_value_from_projection(&self.reset_gate, &xw_r_t, &h_prev),
                    gate_value_from_projection(&self.update_gate, &xw_z_t, &h_prev),
                )
            };

            // Sigmoid on the gate values (parallel or sequential)
            let (r_t, z_t) = if use_parallel {
                rayon::join(|| apply_sigmoid(r_raw), || apply_sigmoid(z_raw))
            } else {
                (apply_sigmoid(r_raw), apply_sigmoid(z_raw))
            };

            // r_t .* h_{t-1}
            let r_h = &r_t * &h_prev;

            // Candidate hidden state: activation(W_h^T [r_t .* h_{t-1}, x_t] + b_h)
            let h_candidate_raw =
                r_h.dot(&self.candidate_gate.recurrent_kernel) + xw_c_t + &self.candidate_gate.bias;
            let h_candidate = act
                .forward(&h_candidate_raw.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Hidden state update: h_t = (1 - z_t) .* h_{t-1} + z_t .* h_candidate
            let h_t = &(1.0 - &z_t) * &h_prev + &z_t * &h_candidate;

            r_vals.push(r_t);
            z_vals.push(z_t);
            h_candidate_vals.push(h_candidate);
            rh_vals.push(r_h);
            hs.push(h_t.clone());

            h_prev = h_t;
        }

        self.hidden_cache = Some(hs);
        self.r_cache = Some(r_vals);
        self.z_cache = Some(z_vals);
        self.h_candidate_cache = Some(h_candidate_vals);
        self.rh_cache = Some(rh_vals);

        // The hidden state already went through the configurable activation at each timestep
        // (Keras-style), so return the last hidden state directly
        Ok(h_prev.into_dyn())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));

        // Use parallel execution once the computational load clears the threshold
        let use_parallel = batch * self.units >= GRU_PARALLEL_THRESHOLD;

        // Configurable activation (Copy) applied to the candidate hidden state
        let act = self.activation;

        // Batched input projection for all 3 gates (one big gemm each)
        let xw_r = project_gate_input(&self.reset_gate, &x3);
        let xw_z = project_gate_input(&self.update_gate, &x3);
        let xw_c = project_gate_input(&self.candidate_gate, &x3);

        for t in 0..timesteps {
            let xw_r_t = xw_r.index_axis(Axis(1), t);
            let xw_z_t = xw_z.index_axis(Axis(1), t);
            let xw_c_t = xw_c.index_axis(Axis(1), t);

            // Reset and update gate values (parallel or sequential)
            let (r_raw, z_raw) = if use_parallel {
                rayon::join(
                    || gate_value_from_projection(&self.reset_gate, &xw_r_t, &h_prev),
                    || gate_value_from_projection(&self.update_gate, &xw_z_t, &h_prev),
                )
            } else {
                (
                    gate_value_from_projection(&self.reset_gate, &xw_r_t, &h_prev),
                    gate_value_from_projection(&self.update_gate, &xw_z_t, &h_prev),
                )
            };

            // Sigmoid on the gate values (parallel or sequential)
            let (r_t, z_t) = if use_parallel {
                rayon::join(|| apply_sigmoid(r_raw), || apply_sigmoid(z_raw))
            } else {
                (apply_sigmoid(r_raw), apply_sigmoid(z_raw))
            };

            // r_t .* h_{t-1}
            let r_h = &r_t * &h_prev;

            // Candidate hidden state: activation(W_h^T [r_t .* h_{t-1}, x_t] + b_h)
            let h_candidate_raw =
                r_h.dot(&self.candidate_gate.recurrent_kernel) + xw_c_t + &self.candidate_gate.bias;
            let h_candidate = act
                .forward(&h_candidate_raw.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Hidden state update: h_t = (1 - z_t) .* h_{t-1} + z_t .* h_candidate
            let h_t = &(1.0 - &z_t) * &h_prev + &z_t * &h_candidate;

            h_prev = h_t;
        }

        // The hidden state already went through the configurable activation at each timestep
        // (Keras-style), so return the last hidden state directly
        Ok(h_prev.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // The upstream gradient is dL/dh_T directly (no extra output activation)
        let grad_h_t = grad_output
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "GRU backward expects a 2D gradient [batch, units], got shape {:?}",
                    grad_output.shape()
                ))
            })?;

        // Configurable activation (Copy) used for the candidate derivative
        let act = self.activation;

        let x3 = take_cache(&mut self.input_cache, "GRU")?;
        let hs = take_cache(&mut self.hidden_cache, "GRU")?;
        let r_vals = take_cache(&mut self.r_cache, "GRU")?;
        let z_vals = take_cache(&mut self.z_cache, "GRU")?;
        let h_candidate_vals = take_cache(&mut self.h_candidate_cache, "GRU")?;
        let rh_vals = take_cache(&mut self.rh_cache, "GRU")?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // Per-gate pre-activation gradients for every timestep
        let mut dz_r = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut dz_z = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut dz_h = Array3::<f32>::zeros((batch, timesteps, self.units));

        let mut grad_h = grad_h_t;

        // Backpropagation through time
        for t in (0..timesteps).rev() {
            let h_prev = &hs[t];
            let r_t = &r_vals[t];
            let z_t = &z_vals[t];
            let h_candidate = &h_candidate_vals[t];

            // Gradient through h_t = (1 - z_t) .* h_{t-1} + z_t .* h_candidate
            let grad_z_t = &grad_h * (h_candidate - h_prev);
            let grad_h_candidate = &grad_h * z_t;
            let grad_h_prev_from_update = &grad_h * &(1.0 - z_t);

            // Gradient through h_candidate = activation(...), via the activation backward
            let grad_h_candidate_raw = act
                .backward(
                    &h_candidate.clone().into_dyn(),
                    &grad_h_candidate.into_dyn(),
                )?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Gradient through r_h = r_t .* h_{t-1} (one recurrent matmul shared by both terms)
            let grad_rh = grad_h_candidate_raw.dot(&self.candidate_gate.recurrent_kernel.t());
            let grad_r_t = &grad_rh * h_prev;
            let grad_h_prev_from_reset = &grad_rh * r_t;

            // Gate pre-activation gradients (sigmoid derivative)
            let grad_z_raw = &grad_z_t * z_t * &(1.0 - z_t);
            let grad_r_raw = &grad_r_t * r_t * &(1.0 - r_t);

            // Gradient w.r.t. the previous hidden state (sequential recurrence - the per-step gemms
            // that have to stay in the loop)
            grad_h = grad_r_raw.dot(&self.reset_gate.recurrent_kernel.t())
                + grad_z_raw.dot(&self.update_gate.recurrent_kernel.t())
                + &grad_h_prev_from_reset
                + &grad_h_prev_from_update;

            dz_r.index_axis_mut(Axis(1), t).assign(&grad_r_raw);
            dz_z.index_axis_mut(Axis(1), t).assign(&grad_z_raw);
            dz_h.index_axis_mut(Axis(1), t)
                .assign(&grad_h_candidate_raw);
        }

        // Batched reductions over all timesteps
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");
        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut rh3 = Array3::<f32>::zeros((batch, timesteps, self.units));
        for t in 0..timesteps {
            h_prev3.index_axis_mut(Axis(1), t).assign(&hs[t]);
            rh3.index_axis_mut(Axis(1), t).assign(&rh_vals[t]);
        }
        let h_prev_flat = h_prev3
            .to_shape((batch * timesteps, self.units))
            .expect("contiguous H_prev reshape");
        let rh_flat = rh3
            .to_shape((batch * timesteps, self.units))
            .expect("contiguous RH reshape");

        let flat = |dz: &Array3<f32>| {
            dz.to_shape((batch * timesteps, self.units))
                .expect("contiguous DZ reshape")
                .to_owned()
        };
        let dz_r = flat(&dz_r);
        let dz_z = flat(&dz_z);
        let dz_h = flat(&dz_h);

        let grad_r_kernel = x_flat.t().dot(&dz_r);
        let grad_r_recurrent = h_prev_flat.t().dot(&dz_r);
        let grad_r_bias = dz_r.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_z_kernel = x_flat.t().dot(&dz_z);
        let grad_z_recurrent = h_prev_flat.t().dot(&dz_z);
        let grad_z_bias = dz_z.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_h_kernel = x_flat.t().dot(&dz_h);
        let grad_h_recurrent = rh_flat.t().dot(&dz_h);
        let grad_h_bias = dz_h.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_x3 = (dz_r.dot(&self.reset_gate.kernel.t())
            + dz_z.dot(&self.update_gate.kernel.t())
            + dz_h.dot(&self.candidate_gate.kernel.t()))
        .into_shape_with_order((batch, timesteps, feat))
        .expect("reshape grad_x to [batch, timesteps, feat]");

        store_gate_gradients(
            &mut self.reset_gate,
            grad_r_kernel,
            grad_r_recurrent,
            grad_r_bias,
        );
        store_gate_gradients(
            &mut self.update_gate,
            grad_z_kernel,
            grad_z_recurrent,
            grad_z_bias,
        );
        store_gate_gradients(
            &mut self.candidate_gate,
            grad_h_kernel,
            grad_h_recurrent,
            grad_h_bias,
        );

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "GRU"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            3 * (self.input_dim * self.units + self.units * self.units + self.units),
        )
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let mut params = self.reset_gate.parameters();
        params.extend(self.update_gate.parameters());
        params.extend(self.candidate_gate.parameters());
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::GRU(GRULayerWeight {
            reset: GRUGateWeight {
                kernel: &self.reset_gate.kernel,
                recurrent_kernel: &self.reset_gate.recurrent_kernel,
                bias: &self.reset_gate.bias,
            },
            update: GRUGateWeight {
                kernel: &self.update_gate.kernel,
                recurrent_kernel: &self.update_gate.recurrent_kernel,
                bias: &self.update_gate.bias,
            },
            candidate: GRUGateWeight {
                kernel: &self.candidate_gate.kernel,
                recurrent_kernel: &self.candidate_gate.recurrent_kernel,
                bias: &self.candidate_gate.bias,
            },
        })
    }
}
