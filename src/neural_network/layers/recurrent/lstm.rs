//! Long Short-Term Memory (LSTM) recurrent layer with input, forget, cell, and output gates

use crate::error::Error;
use crate::math::matmul::gemm_par_auto;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::layer_weight::{LSTMLayerWeight, LayerWeight};
use crate::neural_network::layers::recurrent::apply_sigmoid;
use crate::neural_network::layers::recurrent::gate::{FusedGates, project_input, take_cache};
use crate::neural_network::layers::recurrent::validation::{
    validate_input_3d, validate_recurrent_dimensions,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3, ArrayView3, Axis, Ix2, Ix3, concatenate, s};
use std::borrow::Cow;

/// Long Short-Term Memory (LSTM) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses input, forget, cell, and
/// output gates to control memory flow and mitigate vanishing gradients
///
/// All 4 gates are stored fused: the kernels are packed side by side into single matrices
/// with column blocks in the order `[input | forget | cell | output]` (`[i | f | g | o]`, the
/// Keras LSTM layout), so each projection runs as one GEMM instead of 4
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
/// // Create LSTM layer with 4 input features, 3 units, Tanh activation
/// let mut model = Sequential::new();
/// model.add(LSTM::new(4, 3, Activation::Tanh).unwrap())
///      .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input).unwrap();
/// println!("LSTM output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
#[derive(Debug)]
pub struct LSTM {
    /// Dimensionality of input features
    input_dim: usize,
    /// Number of LSTM units (neurons) in the layer
    units: usize,

    /// Fused gate weights, column blocks in the order `[i | f | g | o]`
    gates: FusedGates,

    /// Cached input tensor for backward propagation
    input_cache: Option<Array3<f32>>,
    /// Per-timestep forward values recorded by `forward` for the backward pass (`None` until the
    /// first training forward; `predict` never sets it)
    caches: Option<LstmCaches>,

    /// Activation applied to the candidate and to the cell state each timestep (Keras-style)
    activation: Activation,
}

/// Per-timestep forward values an [`LSTM`] records so its backward pass can recompute the gate
/// gradients without re-running the forward recurrence
#[derive(Debug)]
struct LstmCaches {
    /// Hidden states `h_t`, with `h_0 = 0` prepended (length `timesteps + 1`)
    hs: Vec<Array2<f32>>,
    /// Cell states `c_t`, with `c_0 = 0` prepended (length `timesteps + 1`)
    cs: Vec<Array2<f32>>,
    /// `activation(c_t)` per timestep
    cs_activated: Vec<Array2<f32>>,
    /// Input-gate activations (sigmoid) per timestep
    i: Vec<Array2<f32>>,
    /// Forget-gate activations (sigmoid) per timestep
    f: Vec<Array2<f32>>,
    /// Candidate (cell-gate) activations per timestep
    g: Vec<Array2<f32>>,
    /// Output-gate activations (sigmoid) per timestep
    o: Vec<Array2<f32>>,
}

impl LSTM {
    /// Creates an LSTM layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of LSTM units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation from the activation module (ReLU, Sigmoid, Tanh, Softmax)
    ///
    /// # Notes
    ///
    /// Weights are seeded from the global seed or entropy by default. For reproducible
    /// initialization, set a seed with [`LSTM::with_random_state`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new LSTM layer instance
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

        Ok(Self {
            input_dim,
            units,
            gates: Self::init_gates(input_dim, units, None)?,
            input_cache: None,
            caches: None,
            activation: activation.into(),
        })
    }

    /// Sets the seed used to initialize the gate weights and re-initializes them deterministically
    ///
    /// By default the weights are seeded from the global seed or entropy (see [`crate::random`])
    /// This re-runs the gate initialization with `random_state`, so call it before assigning custom
    /// weights or training
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        // Dimensions were validated in `new`, so re-initialization cannot fail
        self.gates = Self::init_gates(self.input_dim, self.units, Some(random_state))
            .expect("LSTM dimensions were validated in new()");
        self
    }

    /// Initializes the fused `[i | f | g | o]` gate blocks from the given seed
    ///
    /// One RNG is threaded through all 4 gate blocks; the forget gate bias starts at 1.0
    fn init_gates(
        input_dim: usize,
        units: usize,
        random_state: Option<u64>,
    ) -> Result<FusedGates, Error> {
        let mut rng = crate::random::make_rng(random_state);
        FusedGates::new(input_dim, units, &[0.0, 1.0, 0.0, 0.0], &mut rng)
    }

    /// Sets the fused weights for this LSTM layer (Keras-style layout)
    ///
    /// # Parameters
    ///
    /// - `kernel` - Fused input kernel with shape (input_dim, 4 * units), gate column blocks in
    ///   the order `[i | f | g | o]` (input, forget, cell, output)
    /// - `recurrent_kernel` - Fused recurrent kernel with shape (units, 4 * units), same block order
    /// - `bias` - Fused bias with shape (1, 4 * units), same block order
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

    /// Sets the weights gate by gate, packing them into the fused `[i | f | g | o]` layout
    ///
    /// Convenience wrapper over [`LSTM::set_weights`] for callers that hold per-gate matrices
    ///
    /// # Parameters
    ///
    /// - `input_kernel` - Input kernel for the input gate with shape (input_dim, units)
    /// - `input_recurrent_kernel` - Recurrent kernel for the input gate with shape (units, units)
    /// - `input_bias` - Bias for the input gate with shape (1, units)
    /// - `forget_kernel` - Input kernel for the forget gate with shape (input_dim, units)
    /// - `forget_recurrent_kernel` - Recurrent kernel for the forget gate with shape (units, units)
    /// - `forget_bias` - Bias for the forget gate with shape (1, units)
    /// - `cell_kernel` - Input kernel for the cell gate with shape (input_dim, units)
    /// - `cell_recurrent_kernel` - Recurrent kernel for the cell gate with shape (units, units)
    /// - `cell_bias` - Bias for the cell gate with shape (1, units)
    /// - `output_kernel` - Input kernel for the output gate with shape (input_dim, units)
    /// - `output_recurrent_kernel` - Recurrent kernel for the output gate with shape (units, units)
    /// - `output_bias` - Bias for the output gate with shape (1, units)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   expected per-gate shape
    #[allow(clippy::too_many_arguments)] // 4 gates x (kernel, recurrent_kernel, bias)
    pub fn set_gate_weights(
        &mut self,
        input_kernel: Array2<f32>,
        input_recurrent_kernel: Array2<f32>,
        input_bias: Array2<f32>,
        forget_kernel: Array2<f32>,
        forget_recurrent_kernel: Array2<f32>,
        forget_bias: Array2<f32>,
        cell_kernel: Array2<f32>,
        cell_recurrent_kernel: Array2<f32>,
        cell_bias: Array2<f32>,
        output_kernel: Array2<f32>,
        output_recurrent_kernel: Array2<f32>,
        output_bias: Array2<f32>,
    ) -> Result<(), Error> {
        let per_gate_kernel = [self.input_dim, self.units];
        let per_gate_recurrent = [self.units, self.units];
        let per_gate_bias = [1, self.units];
        for (name, expected, got) in [
            ("input_kernel", &per_gate_kernel, input_kernel.shape()),
            (
                "input_recurrent_kernel",
                &per_gate_recurrent,
                input_recurrent_kernel.shape(),
            ),
            ("input_bias", &per_gate_bias, input_bias.shape()),
            ("forget_kernel", &per_gate_kernel, forget_kernel.shape()),
            (
                "forget_recurrent_kernel",
                &per_gate_recurrent,
                forget_recurrent_kernel.shape(),
            ),
            ("forget_bias", &per_gate_bias, forget_bias.shape()),
            ("cell_kernel", &per_gate_kernel, cell_kernel.shape()),
            (
                "cell_recurrent_kernel",
                &per_gate_recurrent,
                cell_recurrent_kernel.shape(),
            ),
            ("cell_bias", &per_gate_bias, cell_bias.shape()),
            ("output_kernel", &per_gate_kernel, output_kernel.shape()),
            (
                "output_recurrent_kernel",
                &per_gate_recurrent,
                output_recurrent_kernel.shape(),
            ),
            ("output_bias", &per_gate_bias, output_bias.shape()),
        ] {
            validate_weight_shape(name, expected, got)?;
        }

        let kernel = concatenate(
            Axis(1),
            &[
                input_kernel.view(),
                forget_kernel.view(),
                cell_kernel.view(),
                output_kernel.view(),
            ],
        )
        .expect("per-gate kernels share [input_dim, units]");
        let recurrent_kernel = concatenate(
            Axis(1),
            &[
                input_recurrent_kernel.view(),
                forget_recurrent_kernel.view(),
                cell_recurrent_kernel.view(),
                output_recurrent_kernel.view(),
            ],
        )
        .expect("per-gate recurrent kernels share [units, units]");
        let bias = concatenate(
            Axis(1),
            &[
                input_bias.view(),
                forget_bias.view(),
                cell_bias.view(),
                output_bias.view(),
            ],
        )
        .expect("per-gate biases share [1, units]");

        self.set_weights(kernel, recurrent_kernel, bias)
    }

    /// Runs the recurrence and returns the last hidden state - the shared numeric body of
    /// [`Layer::forward`] and [`Layer::predict`]
    ///
    /// When `caches` is `Some`, every per-timestep value the backward pass needs (hidden/cell
    /// states, `activation(c_t)`, and the 4 gate activations) is recorded; `predict` passes
    /// `None` and skips both the recording and its clones
    fn run(
        &self,
        x3: &ArrayView3<f32>,
        mut caches: Option<&mut LstmCaches>,
    ) -> Result<Array2<f32>, Error> {
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        let u = self.units;
        let act = self.activation;

        let mut h_prev = Array2::<f32>::zeros((batch, u));
        let mut c_prev = Array2::<f32>::zeros((batch, u));
        if let Some(c) = caches.as_deref_mut() {
            c.hs.push(h_prev.clone());
            c.cs.push(c_prev.clone());
        }

        // Batched fused input projection for all 4 gates
        let xw = project_input(&self.gates.kernel, x3);

        for t in 0..timesteps {
            let xw_t = xw.index_axis(Axis(1), t); // [batch, 4*units]

            // All 4 gate pre-activations in one fused recurrent GEMM
            let z_all =
                gemm_par_auto(&h_prev, &self.gates.recurrent_kernel) + xw_t + &self.gates.bias;

            // Gates use the recurrent activation (sigmoid); the candidate uses `act`
            let i_t = apply_sigmoid(z_all.slice(s![.., 0..u]).to_owned());
            let f_t = apply_sigmoid(z_all.slice(s![.., u..2 * u]).to_owned());
            let g_t = act
                .forward(&z_all.slice(s![.., 2 * u..3 * u]).to_owned().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();
            let o_t = apply_sigmoid(z_all.slice(s![.., 3 * u..4 * u]).to_owned());

            // Update cell state, then apply the configurable activation to it
            let c_t = &f_t * &c_prev + &i_t * &g_t;
            let c_t_activated = act
                .forward(&c_t.clone().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update hidden state
            let h_t = &o_t * &c_t_activated;

            if let Some(c) = caches.as_deref_mut() {
                c.i.push(i_t);
                c.f.push(f_t);
                c.g.push(g_t);
                c.o.push(o_t);
                c.cs.push(c_t.clone());
                c.cs_activated.push(c_t_activated);
                c.hs.push(h_t.clone());
            }

            h_prev = h_t;
            c_prev = c_t;
        }

        Ok(h_prev)
    }
}

impl Layer for LSTM {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;
        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();
        let timesteps = x3.shape()[1];
        self.input_cache = Some(x3.to_owned());

        let mut caches = LstmCaches {
            hs: Vec::with_capacity(timesteps + 1),
            cs: Vec::with_capacity(timesteps + 1),
            cs_activated: Vec::with_capacity(timesteps),
            i: Vec::with_capacity(timesteps),
            f: Vec::with_capacity(timesteps),
            g: Vec::with_capacity(timesteps),
            o: Vec::with_capacity(timesteps),
        };
        let h_last = self.run(&x3, Some(&mut caches))?;
        self.caches = Some(caches);
        Ok(h_last.into_dyn())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;
        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();
        Ok(self.run(&x3, None)?.into_dyn())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // The upstream gradient is dL/dh_T directly (no extra output activation)
        let grad_h_t = grad_output
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "LSTM backward expects a 2D gradient [batch, units], got shape {:?}",
                    grad_output.shape()
                ))
            })?;

        // Configurable activation, used for the candidate and cell-state derivatives
        let act = self.activation;

        let x3 = take_cache(&mut self.input_cache, "LSTM")?;
        let LstmCaches {
            hs,
            cs,
            cs_activated,
            i: i_vals,
            f: f_vals,
            g: g_vals,
            o: o_vals,
        } = take_cache(&mut self.caches, "LSTM")?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];
        let u = self.units;

        // Fused pre-activation gradients for every timestep, gate blocks [i | f | g | o]
        let mut dz3 = Array3::<f32>::zeros((batch, timesteps, 4 * u));

        let mut grad_h = grad_h_t;
        let mut grad_c = Array2::<f32>::zeros((batch, u));

        // Backpropagation through time
        for t in (0..timesteps).rev() {
            let c_prev = &cs[t];
            let c_t_activated = &cs_activated[t];
            let i_t = &i_vals[t];
            let f_t = &f_vals[t];
            let g_t = &g_vals[t];
            let o_t = &o_vals[t];

            // Gradient through h_t = o_t * activation(c_t)
            let grad_o_t = &grad_h * c_t_activated;
            // dL/dc_t += activation'(c_t) * (grad_h * o_t), via the activation backward
            let grad_cell_act = act
                .backward(
                    &c_t_activated.clone().into_dyn(),
                    &(&grad_h * o_t).into_dyn(),
                )?
                .into_dimensionality::<Ix2>()
                .unwrap();
            grad_c += &grad_cell_act;

            // Gradient through c_t = f_t * c_prev + i_t * g_t
            let grad_f_t = &grad_c * c_prev;
            let grad_i_t = &grad_c * g_t;
            let grad_g_t = &grad_c * i_t;
            let grad_c_prev = &grad_c * f_t;

            // Gates use the sigmoid (recurrent activation) derivative
            let grad_o_raw = &grad_o_t * o_t * &(1.0 - o_t);
            let grad_f_raw = &grad_f_t * f_t * &(1.0 - f_t);
            let grad_i_raw = &grad_i_t * i_t * &(1.0 - i_t);
            let grad_g_raw = act
                .backward(&g_t.clone().into_dyn(), &grad_g_t.into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Assemble the fused dz for this timestep, blocks [i | f | g | o]
            let mut dz_t = Array2::<f32>::zeros((batch, 4 * u));
            dz_t.slice_mut(s![.., 0..u]).assign(&grad_i_raw);
            dz_t.slice_mut(s![.., u..2 * u]).assign(&grad_f_raw);
            dz_t.slice_mut(s![.., 2 * u..3 * u]).assign(&grad_g_raw);
            dz_t.slice_mut(s![.., 3 * u..4 * u]).assign(&grad_o_raw);

            // Gradient w.r.t. the previous hidden state: one fused GEMM instead of 4
            grad_h = gemm_par_auto(&dz_t, &self.gates.recurrent_kernel.t());

            dz3.index_axis_mut(Axis(1), t).assign(&dz_t);

            // Gradient with respect to previous cell state
            grad_c = grad_c_prev;
        }

        // Batched reductions over all timesteps, one fused GEMM each
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");
        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, u));
        for (mut dst, h) in h_prev3.axis_iter_mut(Axis(1)).zip(hs.iter()) {
            dst.assign(h);
        }
        let h_prev_flat = h_prev3
            .to_shape((batch * timesteps, u))
            .expect("contiguous H_prev reshape");
        let dz_flat = dz3
            .to_shape((batch * timesteps, 4 * u))
            .expect("contiguous DZ reshape");

        let grad_kernel = gemm_par_auto(&x_flat.t(), &dz_flat);
        let grad_recurrent = gemm_par_auto(&h_prev_flat.t(), &dz_flat);
        let grad_bias = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_x3 = crate::neural_network::layers::recurrent::gate::reshape_2d_to_3d(
            gemm_par_auto(&dz_flat, &self.gates.kernel.t()),
            (batch, timesteps, feat),
        );

        self.gates
            .store_gradients(grad_kernel, grad_recurrent, grad_bias);

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "LSTM"
    }

    fn output_shape(&self) -> String {
        format!("(None, {})", self.units)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(
            4 * (self.input_dim * self.units + self.units * self.units + self.units),
        )
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        self.gates.parameters()
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::LSTM(LSTMLayerWeight {
            kernel: Cow::Borrowed(&self.gates.kernel),
            recurrent_kernel: Cow::Borrowed(&self.gates.recurrent_kernel),
            bias: Cow::Borrowed(&self.gates.bias),
        })
    }
}
