//! Gated Recurrent Unit (GRU) layer with reset, update, and candidate gates

use crate::error::Error;
use crate::math::matmul::gemm_internal;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::layer_weight::{GRULayerWeight, LayerWeight};
use crate::neural_network::layers::recurrent::apply_sigmoid;
use crate::neural_network::layers::recurrent::gate::{FusedGates, project_input, take_cache};
use crate::neural_network::layers::recurrent::validation::{
    validate_input_3d, validate_recurrent_dimensions,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use std::borrow::Cow;

/// Gated Recurrent Unit (GRU) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses reset, update, and candidate
/// gates to control information flow and mitigate vanishing gradients
///
/// All three gates are stored fused: the kernels are packed side by side into single matrices
/// with column blocks in the order `[reset | update | candidate]` (`[r | z | h]`), so the input
/// projection and the gradient reductions each run as one GEMM instead of three. Per timestep
/// the reset and update recurrent projections fuse into one GEMM; only the candidate's recurrent
/// projection stays separate because its input `r_t .* h_{t-1}` depends on the freshly computed
/// reset gate
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

    /// Fused gate weights, column blocks in the order `[r | z | h]`
    gates: FusedGates,

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

        // 1 RNG threaded through all three gate blocks
        let mut rng = crate::random::make_rng(random_state);

        Ok(Self {
            input_dim,
            units,
            // Gate blocks [r | z | h], all biases start at 0.0
            gates: FusedGates::new(input_dim, units, &[0.0, 0.0, 0.0], &mut rng)?,
            input_cache: None,
            hidden_cache: None,
            r_cache: None,
            z_cache: None,
            h_candidate_cache: None,
            rh_cache: None,
            activation: activation.into(),
        })
    }

    /// Sets the fused weights for this GRU layer
    ///
    /// # Parameters
    ///
    /// - `kernel` - Fused input kernel with shape (input_dim, 3 * units), gate column blocks in
    ///   the order `[r | z | h]` (reset, update, candidate)
    /// - `recurrent_kernel` - Fused recurrent kernel with shape (units, 3 * units), same block order
    /// - `bias` - Fused bias with shape (1, 3 * units), same block order
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   expected fused shape
    pub fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        validate_weight_shape("kernel", self.gates.kernel.shape(), kernel.shape())?;
        validate_weight_shape(
            "recurrent_kernel",
            self.gates.recurrent_kernel.shape(),
            recurrent_kernel.shape(),
        )?;
        validate_weight_shape("bias", self.gates.bias.shape(), bias.shape())?;

        // Force standard layout: `parameters()` exposes the weights as flat slices
        self.gates.kernel = kernel.as_standard_layout().into_owned();
        self.gates.recurrent_kernel = recurrent_kernel.as_standard_layout().into_owned();
        self.gates.bias = bias.as_standard_layout().into_owned();
        Ok(())
    }

    /// Sets the weights gate by gate, packing them into the fused `[r | z | h]` layout
    ///
    /// Convenience wrapper over [`GRU::set_weights`] for callers that hold per-gate matrices
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
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any supplied weight shape does not
    ///   match the expected per-gate shape
    #[allow(clippy::too_many_arguments)]
    pub fn set_gate_weights(
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
        let per_gate_kernel = [self.input_dim, self.units];
        let per_gate_recurrent = [self.units, self.units];
        let per_gate_bias = [1, self.units];
        for (name, expected, got) in [
            ("reset_kernel", &per_gate_kernel, reset_kernel.shape()),
            (
                "reset_recurrent_kernel",
                &per_gate_recurrent,
                reset_recurrent_kernel.shape(),
            ),
            ("reset_bias", &per_gate_bias, reset_bias.shape()),
            ("update_kernel", &per_gate_kernel, update_kernel.shape()),
            (
                "update_recurrent_kernel",
                &per_gate_recurrent,
                update_recurrent_kernel.shape(),
            ),
            ("update_bias", &per_gate_bias, update_bias.shape()),
            (
                "candidate_kernel",
                &per_gate_kernel,
                candidate_kernel.shape(),
            ),
            (
                "candidate_recurrent_kernel",
                &per_gate_recurrent,
                candidate_recurrent_kernel.shape(),
            ),
            ("candidate_bias", &per_gate_bias, candidate_bias.shape()),
        ] {
            validate_weight_shape(name, expected, got)?;
        }

        let kernel = concatenate(
            Axis(1),
            &[
                reset_kernel.view(),
                update_kernel.view(),
                candidate_kernel.view(),
            ],
        )
        .expect("per-gate kernels share [input_dim, units]");
        let recurrent_kernel = concatenate(
            Axis(1),
            &[
                reset_recurrent_kernel.view(),
                update_recurrent_kernel.view(),
                candidate_recurrent_kernel.view(),
            ],
        )
        .expect("per-gate recurrent kernels share [units, units]");
        let bias = concatenate(
            Axis(1),
            &[reset_bias.view(), update_bias.view(), candidate_bias.view()],
        )
        .expect("per-gate biases share [1, units]");

        self.set_weights(kernel, recurrent_kernel, bias)
    }
}

impl Layer for GRU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        let u = self.units;
        let mut h_prev = Array2::<f32>::zeros((batch, u));

        // Per-timestep storage for the backward pass
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut r_vals = Vec::with_capacity(timesteps);
        let mut z_vals = Vec::with_capacity(timesteps);
        let mut h_candidate_vals = Vec::with_capacity(timesteps);
        let mut rh_vals = Vec::with_capacity(timesteps);

        hs.push(h_prev.clone());

        // Configurable activation (Copy) applied to the candidate hidden state
        let act = self.activation;

        // Batched fused input projection for all 3 gates
        let xw = project_input(&self.gates.kernel, &x3);

        for t in 0..timesteps {
            let xw_t = xw.index_axis(Axis(1), t); // [batch, 3*units]

            // Reset and update share h_prev, so their recurrent projections fuse into one GEMM
            let rz_raw = gemm_internal(
                &h_prev,
                &self.gates.recurrent_kernel.slice(s![.., 0..2 * u]),
            ) + xw_t.slice(s![.., 0..2 * u])
                + self.gates.bias.slice(s![.., 0..2 * u]);
            let rz = apply_sigmoid(rz_raw);
            let r_t = rz.slice(s![.., 0..u]).to_owned();
            let z_t = rz.slice(s![.., u..2 * u]).to_owned();

            // r_t .* h_{t-1}
            let r_h = &r_t * &h_prev;

            // Candidate hidden state
            let h_candidate_raw =
                gemm_internal(&r_h, &self.gates.recurrent_kernel.slice(s![.., 2 * u..]))
                    + xw_t.slice(s![.., 2 * u..])
                    + self.gates.bias.slice(s![.., 2 * u..]);
            let h_candidate = act
                .forward(&h_candidate_raw.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Hidden state update
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

        Ok(h_prev.into_dyn())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();

        // (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);

        let u = self.units;
        let mut h_prev = Array2::<f32>::zeros((batch, u));

        // Configurable activation (Copy) applied to the candidate hidden state
        let act = self.activation;

        // Batched fused input projection for all 3 gates
        let xw = project_input(&self.gates.kernel, &x3);

        for t in 0..timesteps {
            let xw_t = xw.index_axis(Axis(1), t); // [batch, 3*units]

            // Reset and update share h_prev, so their recurrent projections fuse into one GEMM
            let rz_raw = gemm_internal(
                &h_prev,
                &self.gates.recurrent_kernel.slice(s![.., 0..2 * u]),
            ) + xw_t.slice(s![.., 0..2 * u])
                + self.gates.bias.slice(s![.., 0..2 * u]);
            let rz = apply_sigmoid(rz_raw);
            let r_t = rz.slice(s![.., 0..u]).to_owned();
            let z_t = rz.slice(s![.., u..2 * u]).to_owned();

            // r_t .* h_{t-1}
            let r_h = &r_t * &h_prev;

            // Candidate hidden state
            let h_candidate_raw =
                gemm_internal(&r_h, &self.gates.recurrent_kernel.slice(s![.., 2 * u..]))
                    + xw_t.slice(s![.., 2 * u..])
                    + self.gates.bias.slice(s![.., 2 * u..]);
            let h_candidate = act
                .forward(&h_candidate_raw.into_dyn())?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Hidden state update
            let h_t = &(1.0 - &z_t) * &h_prev + &z_t * &h_candidate;

            h_prev = h_t;
        }

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
        let u = self.units;

        // Fused pre-activation gradients for every timestep, gate blocks [r | z | h]
        let mut dz3 = Array3::<f32>::zeros((batch, timesteps, 3 * u));

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
            let grad_rh = gemm_internal(
                &grad_h_candidate_raw,
                &self.gates.recurrent_kernel.slice(s![.., 2 * u..]).t(),
            );
            let grad_r_t = &grad_rh * h_prev;
            let grad_h_prev_from_reset = &grad_rh * r_t;

            // Gate pre-activation gradients (sigmoid derivative)
            let grad_z_raw = &grad_z_t * z_t * &(1.0 - z_t);
            let grad_r_raw = &grad_r_t * r_t * &(1.0 - r_t);

            // Assemble the fused reset+update dz for this timestep
            let mut dz_rz_t = Array2::<f32>::zeros((batch, 2 * u));
            dz_rz_t.slice_mut(s![.., 0..u]).assign(&grad_r_raw);
            dz_rz_t.slice_mut(s![.., u..2 * u]).assign(&grad_z_raw);

            // Gradient w.r.t. the previous hidden state
            grad_h = gemm_internal(
                &dz_rz_t,
                &self.gates.recurrent_kernel.slice(s![.., 0..2 * u]).t(),
            ) + &grad_h_prev_from_reset
                + &grad_h_prev_from_update;

            let mut dz_t3 = dz3.index_axis_mut(Axis(1), t);
            dz_t3.slice_mut(s![.., 0..2 * u]).assign(&dz_rz_t);
            dz_t3
                .slice_mut(s![.., 2 * u..])
                .assign(&grad_h_candidate_raw);
        }

        // Batched reductions over all timesteps
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");
        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, u));
        let mut rh3 = Array3::<f32>::zeros((batch, timesteps, u));
        for t in 0..timesteps {
            h_prev3.index_axis_mut(Axis(1), t).assign(&hs[t]);
            rh3.index_axis_mut(Axis(1), t).assign(&rh_vals[t]);
        }
        let h_prev_flat = h_prev3
            .to_shape((batch * timesteps, u))
            .expect("contiguous H_prev reshape");
        let rh_flat = rh3
            .to_shape((batch * timesteps, u))
            .expect("contiguous RH reshape");
        let dz_flat = dz3
            .to_shape((batch * timesteps, 3 * u))
            .expect("contiguous DZ reshape");

        // Input-kernel gradient for all 3 gates in one GEMM
        let grad_kernel = gemm_internal(&x_flat.t(), &dz_flat);
        let grad_bias = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Recurrent gradient
        let mut grad_recurrent = Array2::<f32>::zeros((u, 3 * u));
        grad_recurrent
            .slice_mut(s![.., 0..2 * u])
            .assign(&gemm_internal(
                &h_prev_flat.t(),
                &dz_flat.slice(s![.., 0..2 * u]),
            ));
        grad_recurrent
            .slice_mut(s![.., 2 * u..])
            .assign(&gemm_internal(
                &rh_flat.t(),
                &dz_flat.slice(s![.., 2 * u..]),
            ));

        // Input gradient for all 3 gates in one GEMM
        let grad_x3 = crate::neural_network::layers::recurrent::gate::reshape_2d_to_3d(
            gemm_internal(&dz_flat, &self.gates.kernel.t()),
            (batch, timesteps, feat),
        );

        self.gates
            .store_gradients(grad_kernel, grad_recurrent, grad_bias);

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
        self.gates.parameters()
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::GRU(GRULayerWeight {
            kernel: Cow::Borrowed(&self.gates.kernel),
            recurrent_kernel: Cow::Borrowed(&self.gates.recurrent_kernel),
            bias: Cow::Borrowed(&self.gates.bias),
        })
    }
}
