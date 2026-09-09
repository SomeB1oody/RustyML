//! Long Short-Term Memory (LSTM) recurrent layer with input, forget, cell, and output gates

use crate::error::Error;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::recurrent::apply_sigmoid;
use crate::neural_network::layers::recurrent::cell::{RecurrentGroup, RnnCell};
use crate::neural_network::layers::recurrent::gate::FusedGates;
use crate::neural_network::layers::recurrent::rnn::{Rnn, recurrent_layer_traits};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Bias, Parallelism};
use ndarray::{Array2, ArrayView2, Axis, Ix2, concatenate, s};

/// The arithmetic of 1 timestep of an [`LSTM`]
///
/// The cell carries 2 states. Slot 0 is the hidden state, and slot 1 is the cell state. All 4
/// gate pre-activations come from 1 fused matrix product, whose accumulator starts as the
/// projected `x_t @ kernel` slice and whose epilogue adds the bias.
///
/// The 3 gates use the logistic sigmoid. The candidate and the cell state use the activation
/// that the layer was given.
///
/// # Notes
///
/// The record of 1 step holds 5 arrays, in this order: the activated cell state, then the input,
/// forget, candidate and output gate activations. The cell state that enters a step is not a
/// record, because [`Rnn`] already carries it as state slot 1.
#[derive(Debug)]
pub(crate) struct LstmCell {
    /// Activation applied to the candidate and to the cell state each timestep
    activation: Activation,
}

/// Record slot of the activated cell state
const CELL_ACTIVATED: usize = 0;
/// Record slot of the input-gate activation
const INPUT_GATE: usize = 1;
/// Record slot of the forget-gate activation
const FORGET_GATE: usize = 2;
/// Record slot of the candidate activation
const CANDIDATE: usize = 3;
/// Record slot of the output-gate activation
const OUTPUT_GATE: usize = 4;

impl RnnCell for LstmCell {
    const CELL_TYPE: &'static str = "LSTM";
    const GATE_BIASES: &'static [f32] = &[0.0, 1.0, 0.0, 0.0];
    const STATE_COUNT: usize = 2;
    const RECORD_SLOTS: usize = 5;
    const RECURRENT_GROUPS: &'static [RecurrentGroup] = &[RecurrentGroup {
        first: 0,
        count: 4,
        operand: None,
    }];

    fn new(activation: Activation) -> Self {
        Self { activation }
    }

    fn step(
        &self,
        gates: &FusedGates,
        xw_t: ArrayView2<'_, f32>,
        state: &mut [Array2<f32>],
        record: Option<&mut Vec<Array2<f32>>>,
    ) -> Result<(), Error> {
        let u = gates.units();
        let act = self.activation;
        let bias = gates
            .bias
            .as_slice()
            .expect("fused bias must be contiguous");

        // Recurrent product accumulates onto the `x_t @ kernel` slice already in z_all
        let mut z_all = xw_t.to_owned();
        gemmkit_ndarray::gemm_fused(
            1.0,
            &state[0],
            &gates.recurrent_kernel,
            1.0,
            &mut z_all,
            Some(Bias::PerCol(bias)),
            None,
            Parallelism::Rayon(0),
        );

        // Gates use the recurrent activation (sigmoid). The candidate uses `act`.
        let i_t = apply_sigmoid(z_all.slice(s![.., 0..u]).to_owned());
        let f_t = apply_sigmoid(z_all.slice(s![.., u..2 * u]).to_owned());
        let g_t = act
            .forward(&z_all.slice(s![.., 2 * u..3 * u]).to_owned().into_dyn())?
            .into_dimensionality::<Ix2>()
            .unwrap();
        let o_t = apply_sigmoid(z_all.slice(s![.., 3 * u..4 * u]).to_owned());

        // Update cell state, then apply the configurable activation to it
        let c_t = &f_t * &state[1] + &i_t * &g_t;
        let c_t_activated = act
            .forward(&c_t.clone().into_dyn())?
            .into_dimensionality::<Ix2>()
            .unwrap();

        let h_t = &o_t * &c_t_activated;

        if let Some(record) = record {
            record.push(c_t_activated);
            record.push(i_t);
            record.push(f_t);
            record.push(g_t);
            record.push(o_t);
        }
        state[0] = h_t;
        state[1] = c_t;
        Ok(())
    }

    fn step_backward(
        &self,
        gates: &FusedGates,
        state_prev: &[Array2<f32>],
        _state_next: &[Array2<f32>],
        record: &[Array2<f32>],
        grad_state: &mut [Array2<f32>],
    ) -> Result<Array2<f32>, Error> {
        let u = gates.units();
        let act = self.activation;
        let c_prev = &state_prev[1];
        let c_t_activated = &record[CELL_ACTIVATED];
        let i_t = &record[INPUT_GATE];
        let f_t = &record[FORGET_GATE];
        let g_t = &record[CANDIDATE];
        let o_t = &record[OUTPUT_GATE];
        let batch = o_t.shape()[0];

        // Gradient through h_t = o_t * activation(c_t)
        let grad_o_t = &grad_state[0] * c_t_activated;
        // dL/dc_t += activation'(c_t) * (grad_h * o_t), via the activation backward. The cell
        // state carries its own gradient across the steps, so this accumulates
        let grad_cell_act = act
            .backward(
                &c_t_activated.clone().into_dyn(),
                &(&grad_state[0] * o_t).into_dyn(),
            )?
            .into_dimensionality::<Ix2>()
            .unwrap();
        grad_state[1] += &grad_cell_act;

        // Gradient through c_t = f_t * c_prev + i_t * g_t. Every term reads the accumulated
        // cell gradient, so the write of the carried gradient comes last
        let grad_f_t = &grad_state[1] * c_prev;
        let grad_i_t = &grad_state[1] * g_t;
        let grad_g_t = &grad_state[1] * i_t;
        let grad_c_prev = &grad_state[1] * f_t;

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

        // Gradient with respect to the previous hidden state
        grad_state[0] = dot(&dz_t, &gates.recurrent_kernel.t());
        // Gradient with respect to the previous cell state
        grad_state[1] = grad_c_prev;
        Ok(dz_t)
    }
}

/// Long Short-Term Memory (LSTM) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses input, forget, cell, and
/// output gates to control memory flow and reduce vanishing gradients.
///
/// All 4 gates are stored fused. The kernels are packed side by side into single matrices.
/// Column blocks follow the order `[input | forget | cell | output]` (`[i | f | g | o]`).
///
/// [`LSTM::with_return_sequences`] makes the layer return every timestep's hidden state, with
/// shape (batch_size, timesteps, units). [`LSTM::with_go_backwards`] processes the input
/// timesteps from last to first.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(LSTM::new(3, Activation::Tanh).unwrap())
///     .build(&Shape::known(input.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input).unwrap();
/// println!("LSTM output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
pub struct LSTM(Rnn<LstmCell>);

impl LSTM {
    /// Creates an LSTM layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `units` - Number of output units in the layer
    /// - `activation` - Activation from the activation module (any [`Activation`] variant, or
    ///   any standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new LSTM layer instance
    ///
    /// # Notes
    ///
    /// The constructor draws nothing. [`UnaryLayer::build`] reads the feature count from the input
    /// shape and draws the gates then. The draw takes the global seed or entropy by default.
    /// For reproducible initialization, set a seed with [`LSTM::with_random_state`].
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `units` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(units: usize, activation: impl Into<Activation>) -> Result<Self, Error> {
        Ok(Self(Rnn::new(units, activation)?))
    }

    /// Sets the seed used to initialize the gate weights and re-initializes them deterministically
    ///
    /// By default the draw takes the global seed or entropy (see [`crate::random`]). An unbuilt
    /// layer holds no gate weight, so this records the seed and draws nothing. A layer that is
    /// already built draws its gates again from the new seed, so the order of the 2 calls does
    /// not matter.
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.0 = self.0.with_random_state(random_state);
        self
    }

    /// Sets whether the layer returns every timestep's hidden state
    ///
    /// The default is false, which returns only the last hidden state, with shape
    /// (batch_size, units). With true, the layer returns all hidden states, with shape
    /// (batch_size, timesteps, units). Slot `k` of the time axis holds the state after
    /// processing step `k`. The backward pass then expects a gradient of the same rank-3 shape.
    ///
    /// # Parameters
    ///
    /// - `return_sequences` - True to return every timestep's hidden state
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    ///
    /// # Notes
    ///
    /// The last slot of the returned sequence always equals the output of the same layer with
    /// `return_sequences` set to false. The cell state stays internal in both cases.
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.0.return_sequences = return_sequences;
        self
    }

    /// Sets whether the layer processes the input timesteps from last to first
    ///
    /// The default is false. With true, processing step 0 consumes input timestep
    /// `timesteps` - 1, and the output stays in processing order. The layer does not reverse the
    /// output back to input order, so slot 0 of a returned sequence holds the state that came
    /// from the last input timestep. This flag changes no shape.
    ///
    /// # Parameters
    ///
    /// - `go_backwards` - True to process the input timesteps from last to first
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_go_backwards(mut self, go_backwards: bool) -> Self {
        self.0.go_backwards = go_backwards;
        self
    }

    /// Sets the fused weights for this LSTM layer
    ///
    /// # Parameters
    ///
    /// - `kernel` - Fused input kernel with shape (input_dim, 4 * units), gate column blocks in
    ///   the order `[i | f | g | o]`
    /// - `recurrent_kernel` - Fused recurrent kernel with shape (units, 4 * units), same block
    ///   order
    /// - `bias` - Fused bias with shape (1, 4 * units), same block order
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when every array matches the shape the layer holds
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer is not built
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   expected fused shape
    pub fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        self.0.set_weights(kernel, recurrent_kernel, bias)
    }

    /// Sets the weights gate by gate, packing them into the fused `[i | f | g | o]` layout
    ///
    /// Convenience wrapper over [`LSTM::set_weights`] for callers that hold per-gate matrices.
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
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when every per-gate weight matches the expected shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   expected per-gate shape
    #[allow(clippy::too_many_arguments)] // 4 gates * (kernel, recurrent_kernel, bias)
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
        let per_gate_kernel = [self.0.input_dim, self.0.units];
        let per_gate_recurrent = [self.0.units, self.0.units];
        let per_gate_bias = [1, self.0.units];
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

recurrent_layer_traits!(LSTM);
