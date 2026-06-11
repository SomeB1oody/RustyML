//! Long Short-Term Memory (LSTM) recurrent layer with input, forget, cell, and output gates

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::layer_weight::{LSTMGateWeight, LSTMLayerWeight, LayerWeight};
use crate::neural_network::layers::recurrent::apply_sigmoid;
use crate::neural_network::layers::recurrent::gate::{
    Gate, gate_value_from_projection, project_gate_input, store_gate_gradients, take_cache,
};
use crate::neural_network::layers::recurrent::validation::{
    validate_input_3d, validate_recurrent_dimensions,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array2, Array3, Axis, Ix2, Ix3};

/// Threshold for switching LSTM computation to parallel execution
///
/// When `batch_size * units` is below this value, execution is sequential; at or above it,
/// execution is parallel. Chosen from empirical benchmarks where rayon's thread-pool overhead
/// is amortized by the gains from parallelization
const LSTM_PARALLEL_THRESHOLD: usize = 1024;

/// Long Short-Term Memory (LSTM) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses input, forget, cell, and
/// output gates to control memory flow and mitigate vanishing gradients
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

    /// Gate controlling what new information to store in cell state
    input_gate: Gate,
    /// Gate controlling what information to discard from cell state
    forget_gate: Gate,
    /// Gate proposing new candidate values for cell state
    cell_gate: Gate,
    /// Gate controlling what to output from cell state
    output_gate: Gate,

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

        // One RNG threaded through all four gates so the layer shares a single
        // reproducible initialization stream
        let mut rng = crate::random::make_rng(random_state);

        Ok(Self {
            input_dim,
            units,
            input_gate: Gate::new(input_dim, units, 0.0, &mut rng)?,
            forget_gate: Gate::new(input_dim, units, 1.0, &mut rng)?, // forget gate bias = 1.0
            cell_gate: Gate::new(input_dim, units, 0.0, &mut rng)?,
            output_gate: Gate::new(input_dim, units, 0.0, &mut rng)?,
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

    /// Sets the weights for all four gates in this LSTM layer
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
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the expected shape
    #[allow(clippy::too_many_arguments)] // four gates x (kernel, recurrent_kernel, bias)
    pub fn set_weights(
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
        validate_weight_shape(
            "input_kernel",
            self.input_gate.kernel.shape(),
            input_kernel.shape(),
        )?;
        validate_weight_shape(
            "input_recurrent_kernel",
            self.input_gate.recurrent_kernel.shape(),
            input_recurrent_kernel.shape(),
        )?;
        validate_weight_shape(
            "input_bias",
            self.input_gate.bias.shape(),
            input_bias.shape(),
        )?;
        validate_weight_shape(
            "forget_kernel",
            self.forget_gate.kernel.shape(),
            forget_kernel.shape(),
        )?;
        validate_weight_shape(
            "forget_recurrent_kernel",
            self.forget_gate.recurrent_kernel.shape(),
            forget_recurrent_kernel.shape(),
        )?;
        validate_weight_shape(
            "forget_bias",
            self.forget_gate.bias.shape(),
            forget_bias.shape(),
        )?;
        validate_weight_shape(
            "cell_kernel",
            self.cell_gate.kernel.shape(),
            cell_kernel.shape(),
        )?;
        validate_weight_shape(
            "cell_recurrent_kernel",
            self.cell_gate.recurrent_kernel.shape(),
            cell_recurrent_kernel.shape(),
        )?;
        validate_weight_shape("cell_bias", self.cell_gate.bias.shape(), cell_bias.shape())?;
        validate_weight_shape(
            "output_kernel",
            self.output_gate.kernel.shape(),
            output_kernel.shape(),
        )?;
        validate_weight_shape(
            "output_recurrent_kernel",
            self.output_gate.recurrent_kernel.shape(),
            output_recurrent_kernel.shape(),
        )?;
        validate_weight_shape(
            "output_bias",
            self.output_gate.bias.shape(),
            output_bias.shape(),
        )?;

        self.input_gate.kernel = input_kernel;
        self.input_gate.recurrent_kernel = input_recurrent_kernel;
        self.input_gate.bias = input_bias;

        self.forget_gate.kernel = forget_kernel;
        self.forget_gate.recurrent_kernel = forget_recurrent_kernel;
        self.forget_gate.bias = forget_bias;

        self.cell_gate.kernel = cell_kernel;
        self.cell_gate.recurrent_kernel = cell_recurrent_kernel;
        self.cell_gate.bias = cell_bias;

        self.output_gate.kernel = output_kernel;
        self.output_gate.recurrent_kernel = output_recurrent_kernel;
        self.output_gate.bias = output_bias;

        Ok(())
    }
}

impl Layer for LSTM {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        self.input_cache = Some(x3.to_owned());

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut c_prev = Array2::<f32>::zeros((batch, self.units));

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

        // Parallelize once the computational load clears the threshold
        let use_parallel = batch * self.units >= LSTM_PARALLEL_THRESHOLD;

        // Configurable activation, applied to the candidate and the cell state
        let act = self.activation;

        // Batched input projection for all 4 gates
        let xw_i = project_gate_input(&self.input_gate, &x3);
        let xw_f = project_gate_input(&self.forget_gate, &x3);
        let xw_g = project_gate_input(&self.cell_gate, &x3);
        let xw_o = project_gate_input(&self.output_gate, &x3);

        for t in 0..timesteps {
            let xw_i_t = xw_i.index_axis(Axis(1), t);
            let xw_f_t = xw_f.index_axis(Axis(1), t);
            let xw_g_t = xw_g.index_axis(Axis(1), t);
            let xw_o_t = xw_o.index_axis(Axis(1), t);

            // Compute all 4 gate values (parallel or sequential)
            let (i_raw, f_raw, g_raw, o_raw) = if use_parallel {
                let ((i_raw, f_raw), (g_raw, o_raw)) = rayon::join(
                    || {
                        rayon::join(
                            || gate_value_from_projection(&self.input_gate, &xw_i_t, &h_prev),
                            || gate_value_from_projection(&self.forget_gate, &xw_f_t, &h_prev),
                        )
                    },
                    || {
                        rayon::join(
                            || gate_value_from_projection(&self.cell_gate, &xw_g_t, &h_prev),
                            || gate_value_from_projection(&self.output_gate, &xw_o_t, &h_prev),
                        )
                    },
                );
                (i_raw, f_raw, g_raw, o_raw)
            } else {
                (
                    gate_value_from_projection(&self.input_gate, &xw_i_t, &h_prev),
                    gate_value_from_projection(&self.forget_gate, &xw_f_t, &h_prev),
                    gate_value_from_projection(&self.cell_gate, &xw_g_t, &h_prev),
                    gate_value_from_projection(&self.output_gate, &xw_o_t, &h_prev),
                )
            };

            // Gates use the recurrent activation (sigmoid)
            let i_t = apply_sigmoid(i_raw);
            let f_t = apply_sigmoid(f_raw);
            let o_t = apply_sigmoid(o_raw);
            let g_t = act
                .forward(&g_raw.into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update cell state: c_t = f_t * c_prev + i_t * g_t
            let c_t = &f_t * &c_prev + &i_t * &g_t;

            // Apply the configurable activation to the cell state
            let c_t_activated = act
                .forward(&c_t.clone().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update hidden state: h_t = o_t * activation(c_t)
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

        // The hidden state already passed through the configurable activation at each timestep (Keras-style)
        Ok(h_prev.into_dyn())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;

        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();

        // Input shape: (batch, timesteps, input_dim)
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        let mut c_prev = Array2::<f32>::zeros((batch, self.units));

        // Parallelize once the computational load clears the threshold
        let use_parallel = batch * self.units >= LSTM_PARALLEL_THRESHOLD;

        // Configurable activation, applied to the candidate and the cell state
        let act = self.activation;

        // Batched input projection for all 4 gates
        let xw_i = project_gate_input(&self.input_gate, &x3);
        let xw_f = project_gate_input(&self.forget_gate, &x3);
        let xw_g = project_gate_input(&self.cell_gate, &x3);
        let xw_o = project_gate_input(&self.output_gate, &x3);

        for t in 0..timesteps {
            let xw_i_t = xw_i.index_axis(Axis(1), t);
            let xw_f_t = xw_f.index_axis(Axis(1), t);
            let xw_g_t = xw_g.index_axis(Axis(1), t);
            let xw_o_t = xw_o.index_axis(Axis(1), t);

            // Compute all 4 gate values (parallel or sequential)
            let (i_raw, f_raw, g_raw, o_raw) = if use_parallel {
                let ((i_raw, f_raw), (g_raw, o_raw)) = rayon::join(
                    || {
                        rayon::join(
                            || gate_value_from_projection(&self.input_gate, &xw_i_t, &h_prev),
                            || gate_value_from_projection(&self.forget_gate, &xw_f_t, &h_prev),
                        )
                    },
                    || {
                        rayon::join(
                            || gate_value_from_projection(&self.cell_gate, &xw_g_t, &h_prev),
                            || gate_value_from_projection(&self.output_gate, &xw_o_t, &h_prev),
                        )
                    },
                );
                (i_raw, f_raw, g_raw, o_raw)
            } else {
                (
                    gate_value_from_projection(&self.input_gate, &xw_i_t, &h_prev),
                    gate_value_from_projection(&self.forget_gate, &xw_f_t, &h_prev),
                    gate_value_from_projection(&self.cell_gate, &xw_g_t, &h_prev),
                    gate_value_from_projection(&self.output_gate, &xw_o_t, &h_prev),
                )
            };

            // Gates use the recurrent activation (sigmoid)
            let i_t = apply_sigmoid(i_raw);
            let f_t = apply_sigmoid(f_raw);
            let o_t = apply_sigmoid(o_raw);
            let g_t = act
                .forward(&g_raw.into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update cell state: c_t = f_t * c_prev + i_t * g_t
            let c_t = &f_t * &c_prev + &i_t * &g_t;

            // Apply the configurable activation to the cell state
            let c_t_activated = act
                .forward(&c_t.clone().into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Update hidden state: h_t = o_t * activation(c_t)
            let h_t = &o_t * &c_t_activated;

            h_prev = h_t;
            c_prev = c_t;
        }

        // The hidden state already passed through the configurable activation at each timestep (Keras-style)
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

        // Per-gate pre-activation gradients for every timestep
        let mut dz_i = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut dz_f = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut dz_g = Array3::<f32>::zeros((batch, timesteps, self.units));
        let mut dz_o = Array3::<f32>::zeros((batch, timesteps, self.units));

        let mut grad_h = grad_h_t;
        let mut grad_c = Array2::<f32>::zeros((batch, self.units));

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

            // Gradient w.r.t. the previous hidden state
            grad_h = grad_o_raw.dot(&self.output_gate.recurrent_kernel.t())
                + grad_f_raw.dot(&self.forget_gate.recurrent_kernel.t())
                + grad_i_raw.dot(&self.input_gate.recurrent_kernel.t())
                + grad_g_raw.dot(&self.cell_gate.recurrent_kernel.t());

            dz_o.index_axis_mut(Axis(1), t).assign(&grad_o_raw);
            dz_f.index_axis_mut(Axis(1), t).assign(&grad_f_raw);
            dz_i.index_axis_mut(Axis(1), t).assign(&grad_i_raw);
            dz_g.index_axis_mut(Axis(1), t).assign(&grad_g_raw);

            // Gradient with respect to previous cell state
            grad_c = grad_c_prev;
        }

        // Batched reductions over all timesteps
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

        let flat = |dz: &Array3<f32>| {
            dz.to_shape((batch * timesteps, self.units))
                .expect("contiguous DZ reshape")
                .to_owned()
        };
        let dz_i = flat(&dz_i);
        let dz_f = flat(&dz_f);
        let dz_g = flat(&dz_g);
        let dz_o = flat(&dz_o);

        let gate_grads = |dz_flat: &Array2<f32>| {
            (
                x_flat.t().dot(dz_flat),
                h_prev_flat.t().dot(dz_flat),
                dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0)),
            )
        };
        let (grad_i_kernel, grad_i_recurrent, grad_i_bias) = gate_grads(&dz_i);
        let (grad_f_kernel, grad_f_recurrent, grad_f_bias) = gate_grads(&dz_f);
        let (grad_g_kernel, grad_g_recurrent, grad_g_bias) = gate_grads(&dz_g);
        let (grad_o_kernel, grad_o_recurrent, grad_o_bias) = gate_grads(&dz_o);

        let grad_x3 = (dz_i.dot(&self.input_gate.kernel.t())
            + dz_f.dot(&self.forget_gate.kernel.t())
            + dz_g.dot(&self.cell_gate.kernel.t())
            + dz_o.dot(&self.output_gate.kernel.t()))
        .into_shape_with_order((batch, timesteps, feat))
        .expect("reshape grad_x to [batch, timesteps, feat]");

        store_gate_gradients(
            &mut self.input_gate,
            grad_i_kernel,
            grad_i_recurrent,
            grad_i_bias,
        );
        store_gate_gradients(
            &mut self.forget_gate,
            grad_f_kernel,
            grad_f_recurrent,
            grad_f_bias,
        );
        store_gate_gradients(
            &mut self.cell_gate,
            grad_g_kernel,
            grad_g_recurrent,
            grad_g_bias,
        );
        store_gate_gradients(
            &mut self.output_gate,
            grad_o_kernel,
            grad_o_recurrent,
            grad_o_bias,
        );

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
        let mut params = self.input_gate.parameters();
        params.extend(self.forget_gate.parameters());
        params.extend(self.cell_gate.parameters());
        params.extend(self.output_gate.parameters());
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::LSTM(LSTMLayerWeight {
            input: LSTMGateWeight {
                kernel: &self.input_gate.kernel,
                recurrent_kernel: &self.input_gate.recurrent_kernel,
                bias: &self.input_gate.bias,
            },
            forget: LSTMGateWeight {
                kernel: &self.forget_gate.kernel,
                recurrent_kernel: &self.forget_gate.recurrent_kernel,
                bias: &self.forget_gate.bias,
            },
            cell: LSTMGateWeight {
                kernel: &self.cell_gate.kernel,
                recurrent_kernel: &self.cell_gate.recurrent_kernel,
                bias: &self.cell_gate.bias,
            },
            output: LSTMGateWeight {
                kernel: &self.output_gate.kernel,
                recurrent_kernel: &self.output_gate.recurrent_kernel,
                bias: &self.output_gate.bias,
            },
        })
    }
}
