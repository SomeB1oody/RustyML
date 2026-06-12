//! Long Short-Term Memory (LSTM) recurrent layer with input, forget, cell, and output gates

use crate::error::Error;
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
use crate::math::matmul::par_matmul;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3, Axis, Ix2, Ix3, concatenate, s};

/// Long Short-Term Memory (LSTM) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses input, forget, cell, and
/// output gates to control memory flow and mitigate vanishing gradients
///
/// All four gates are stored fused: the kernels are packed side by side into single matrices
/// with column blocks in the order `[input | forget | cell | output]` (`[i | f | g | o]`, the
/// Keras LSTM layout), so each projection runs as one GEMM instead of four
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
/// model.add(LSTM::new(4, 3, Activation::Tanh, None).unwrap())
///      .compile(RMSprop::new(0.001, 0.9, 1e-8, None, 0.0).unwrap(), MeanSquaredError::new());
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
    /// Cached hidden states h_t for each timestep
    hidden_cache: Option<Vec<Array2<f32>>>,
    /// Cached cell states c_t for each timestep
    cell_cache: Option<Vec<Array2<f32>>>,
    /// Cached activation(c_t) values for each timestep
    cell_activated_cache: Option<Vec<Array2<f32>>>,

    /// Cached input gate activations (sigmoid applied) for each timestep
    i_cache: Option<Vec<Array2<f32>>>,
    /// Cached forget gate activations (sigmoid applied) for each timestep
    f_cache: Option<Vec<Array2<f32>>>,
    /// Cached candidate (cell gate) activations for each timestep
    g_cache: Option<Vec<Array2<f32>>>,
    /// Cached output gate activations (sigmoid applied) for each timestep
    o_cache: Option<Vec<Array2<f32>>>,

    /// Activation applied to the candidate and to the cell state each timestep (Keras-style)
    activation: Activation,
}

impl LSTM {
    /// Creates an LSTM layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of LSTM units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation from the activation module (ReLU, Sigmoid, Tanh, Softmax)
    /// - `random_state` - Optional seed for reproducible initialization; falls back to the global seed or entropy. See crate::random
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
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        validate_recurrent_dimensions(input_dim, units)?;

        // One RNG threaded through all 4 gate blocks
        let mut rng = crate::random::make_rng(random_state);

        Ok(Self {
            input_dim,
            units,
            // Gate blocks [i | f | g | o]; the forget gate bias starts at 1.0
            gates: FusedGates::new(input_dim, units, &[0.0, 1.0, 0.0, 0.0], &mut rng)?,
            input_cache: None,
            hidden_cache: None,
            cell_cache: None,
            cell_activated_cache: None,
            i_cache: None,
            f_cache: None,
            g_cache: None,
            o_cache: None,
            activation: activation.into(),
        })
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
    #[allow(clippy::too_many_arguments)] // four gates x (kernel, recurrent_kernel, bias)
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
}

impl Layer for LSTM {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        let u = self.units;
        let mut h_prev = Array2::<f32>::zeros((batch, u));
        let mut c_prev = Array2::<f32>::zeros((batch, u));

        // Storage for all timesteps
        let mut hs = Vec::with_capacity(timesteps + 1);
        let mut cs = Vec::with_capacity(timesteps + 1);
        let mut cs_activated = Vec::with_capacity(timesteps);
        let mut i_vals = Vec::with_capacity(timesteps);
        let mut f_vals = Vec::with_capacity(timesteps);
        let mut g_vals = Vec::with_capacity(timesteps);
        let mut o_vals = Vec::with_capacity(timesteps);

        hs.push(h_prev.clone());
        cs.push(c_prev.clone());

        // Configurable activation, applied to the candidate and the cell state
        let act = self.activation;

        // Batched fused input projection for all 4 gates
        let xw = project_input(&self.gates.kernel, &x3);

        for t in 0..timesteps {
            let xw_t = xw.index_axis(Axis(1), t); // [batch, 4*units]

            // All 4 gate pre-activations in one fused recurrent GEMM
            let z_all =
                par_matmul(&h_prev, &self.gates.recurrent_kernel) + xw_t + &self.gates.bias;

            // Gates use the recurrent activation (sigmoid)
            let i_t = apply_sigmoid(z_all.slice(s![.., 0..u]).to_owned());
            let f_t = apply_sigmoid(z_all.slice(s![.., u..2 * u]).to_owned());
            let g_t = act
                .forward(&z_all.slice(s![.., 2 * u..3 * u]).to_owned().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();
            let o_t = apply_sigmoid(z_all.slice(s![.., 3 * u..4 * u]).to_owned());

            // Update cell state
            let c_t = &f_t * &c_prev + &i_t * &g_t;

            // Apply the configurable activation to the cell state
            let c_t_activated = act
                .forward(&c_t.clone().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update hidden state
            let h_t = &o_t * &c_t_activated;

            // Cache values
            i_vals.push(i_t);
            f_vals.push(f_t);
            g_vals.push(g_t);
            o_vals.push(o_t);
            cs.push(c_t.clone());
            cs_activated.push(c_t_activated);
            hs.push(h_t.clone());

            h_prev = h_t;
            c_prev = c_t;
        }

        // Store caches
        self.hidden_cache = Some(hs);
        self.cell_cache = Some(cs);
        self.cell_activated_cache = Some(cs_activated);
        self.i_cache = Some(i_vals);
        self.f_cache = Some(f_vals);
        self.g_cache = Some(g_vals);
        self.o_cache = Some(o_vals);

        Ok(h_prev.into_dyn())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);

        let u = self.units;
        let mut h_prev = Array2::<f32>::zeros((batch, u));
        let mut c_prev = Array2::<f32>::zeros((batch, u));

        // Configurable activation, applied to the candidate and the cell state
        let act = self.activation;

        // Batched fused input projection for all 4 gates
        let xw = project_input(&self.gates.kernel, &x3);

        for t in 0..timesteps {
            let xw_t = xw.index_axis(Axis(1), t); // [batch, 4*units]

            // All 4 gate pre-activations in one fused recurrent GEMM
            let z_all =
                par_matmul(&h_prev, &self.gates.recurrent_kernel) + xw_t + &self.gates.bias;

            // Gates use the recurrent activation (sigmoid); the candidate uses `act`
            let i_t = apply_sigmoid(z_all.slice(s![.., 0..u]).to_owned());
            let f_t = apply_sigmoid(z_all.slice(s![.., u..2 * u]).to_owned());
            let g_t = act
                .forward(&z_all.slice(s![.., 2 * u..3 * u]).to_owned().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();
            let o_t = apply_sigmoid(z_all.slice(s![.., 3 * u..4 * u]).to_owned());

            // Update cell state
            let c_t = &f_t * &c_prev + &i_t * &g_t;

            // Apply the configurable activation to the cell state
            let c_t_activated = act
                .forward(&c_t.clone().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update hidden state
            let h_t = &o_t * &c_t_activated;

            h_prev = h_t;
            c_prev = c_t;
        }

        Ok(h_prev.into_dyn())
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
        let hs = take_cache(&mut self.hidden_cache, "LSTM")?;
        let cs = take_cache(&mut self.cell_cache, "LSTM")?;
        let cs_activated = take_cache(&mut self.cell_activated_cache, "LSTM")?;
        let i_vals = take_cache(&mut self.i_cache, "LSTM")?;
        let f_vals = take_cache(&mut self.f_cache, "LSTM")?;
        let g_vals = take_cache(&mut self.g_cache, "LSTM")?;
        let o_vals = take_cache(&mut self.o_cache, "LSTM")?;

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

            // Gradient w.r.t. the previous hidden state: one fused GEMM instead of four
            grad_h = par_matmul(&dz_t, &self.gates.recurrent_kernel.t());

            dz3.index_axis_mut(Axis(1), t).assign(&dz_t);

            // Gradient with respect to previous cell state
            grad_c = grad_c_prev;
        }

        // Batched reductions over all timesteps, one fused GEMM each (were four apiece)
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

        let grad_kernel = par_matmul(&x_flat.t(), &dz_flat);
        let grad_recurrent = par_matmul(&h_prev_flat.t(), &dz_flat);
        let grad_bias = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_x3 = crate::neural_network::layers::recurrent::gate::reshape_2d_to_3d(
            par_matmul(&dz_flat, &self.gates.kernel.t()),
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
            kernel: &self.gates.kernel,
            recurrent_kernel: &self.gates.recurrent_kernel,
            bias: &self.gates.bias,
        })
    }
}
