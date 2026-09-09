//! Gated Recurrent Unit (GRU) recurrent layer with update, reset, and candidate gates

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

/// The arithmetic of 1 timestep of a [`GRU`]
///
/// The cell carries 1 state, the hidden state. The update and the reset gate both project the
/// previous hidden state, so their pre-activations come from 1 fused matrix product. The
/// candidate instead projects `r_t .* h_prev`, so it needs a second product. That is why the
/// column blocks of the recurrent kernel fall into 2 groups rather than 1.
///
/// # Notes
///
/// The record of 1 step holds 4 arrays, in this order: the reset gate, the update gate, the
/// candidate, and `r_t .* h_prev`. The recurrent-kernel gradient of the candidate block needs
/// `r_t .* h_prev` as its operand, so the cell records it.
#[derive(Debug)]
pub(crate) struct GruCell {
    /// Activation applied to the candidate hidden state each timestep
    activation: Activation,
}

/// Record slot of the reset-gate activation
const RESET_GATE: usize = 0;
/// Record slot of the update-gate activation
const UPDATE_GATE: usize = 1;
/// Record slot of the candidate hidden state
const CANDIDATE: usize = 2;
/// Record slot of `r_t .* h_prev`, which the recurrent kernel's candidate block projects
const RESET_HIDDEN: usize = 3;

impl RnnCell for GruCell {
    const CELL_TYPE: &'static str = "GRU";
    const GATE_BIASES: &'static [f32] = &[0.0, 0.0, 0.0];
    const STATE_COUNT: usize = 1;
    const RECORD_SLOTS: usize = 4;
    const RECURRENT_GROUPS: &'static [RecurrentGroup] = &[
        RecurrentGroup {
            first: 0,
            count: 2,
            operand: None,
        },
        RecurrentGroup {
            first: 2,
            count: 1,
            operand: Some(RESET_HIDDEN),
        },
    ];

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
        // Bias blocks `[z | r]` and `[h]`, each folded into its own product's epilogue
        let (bias_rz, bias_h) = gates
            .bias
            .as_slice()
            .expect("fused bias must be contiguous")
            .split_at(2 * u);
        let h_prev = &state[0];

        // Reset and update share h_prev, so their recurrent projections fuse into 1 product
        let mut rz = xw_t.slice(s![.., 0..2 * u]).to_owned();
        gemmkit_ndarray::gemm_fused(
            1.0,
            h_prev,
            &gates.recurrent_kernel.slice(s![.., 0..2 * u]),
            1.0,
            &mut rz,
            Some(Bias::PerCol(bias_rz)),
            None,
            Parallelism::Rayon(0),
        );
        let rz = apply_sigmoid(rz);
        let z_t = rz.slice(s![.., 0..u]).to_owned();
        let r_t = rz.slice(s![.., u..2 * u]).to_owned();

        // r_t .* h_{t-1}, then the candidate hidden state
        let r_h = &r_t * h_prev;
        let mut h_candidate = xw_t.slice(s![.., 2 * u..]).to_owned();
        let rk_h = gates.recurrent_kernel.slice(s![.., 2 * u..]);
        gemmkit_ndarray::gemm_fused(
            1.0,
            &r_h,
            &rk_h,
            1.0,
            &mut h_candidate,
            Some(Bias::PerCol(bias_h)),
            None,
            Parallelism::Rayon(0),
        );
        let h_candidate = act
            .forward(&h_candidate.into_dyn())?
            .into_dimensionality::<Ix2>()
            .unwrap();

        let h_t = &z_t * h_prev + &(1.0 - &z_t) * &h_candidate;

        if let Some(record) = record {
            record.push(r_t);
            record.push(z_t);
            record.push(h_candidate);
            record.push(r_h);
        }
        state[0] = h_t;
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
        let h_prev = &state_prev[0];
        let r_t = &record[RESET_GATE];
        let z_t = &record[UPDATE_GATE];
        let h_candidate = &record[CANDIDATE];
        let batch = h_prev.shape()[0];

        // Gradient through h_t = z_t .* h_{t-1} + (1 - z_t) .* h_candidate
        let grad_z_t = &grad_state[0] * (h_prev - h_candidate);
        let grad_h_candidate = &grad_state[0] * &(1.0 - z_t);
        let grad_h_prev_from_update = &grad_state[0] * z_t;

        // Gradient through h_candidate = activation(...), via the activation backward
        let grad_h_candidate_raw = act
            .backward(
                &h_candidate.clone().into_dyn(),
                &grad_h_candidate.into_dyn(),
            )?
            .into_dimensionality::<Ix2>()
            .unwrap();

        // Gradient through r_h = r_t .* h_{t-1} (1 recurrent product shared by both terms)
        let grad_rh = dot(
            &grad_h_candidate_raw,
            &gates.recurrent_kernel.slice(s![.., 2 * u..]).t(),
        );
        let grad_r_t = &grad_rh * h_prev;
        let grad_h_prev_from_reset = &grad_rh * r_t;

        // Gate pre-activation gradients (sigmoid derivative)
        let grad_z_raw = &grad_z_t * z_t * &(1.0 - z_t);
        let grad_r_raw = &grad_r_t * r_t * &(1.0 - r_t);

        // Assemble the fused update+reset dz for this timestep, in kernel block order
        let mut dz_rz_t = Array2::<f32>::zeros((batch, 2 * u));
        dz_rz_t.slice_mut(s![.., 0..u]).assign(&grad_z_raw);
        dz_rz_t.slice_mut(s![.., u..2 * u]).assign(&grad_r_raw);

        // Gradient with respect to the previous hidden state. The candidate block never enters
        // this product, because its contribution already went through `grad_rh`
        grad_state[0] = dot(
            &dz_rz_t,
            &gates.recurrent_kernel.slice(s![.., 0..2 * u]).t(),
        ) + &grad_h_prev_from_reset
            + &grad_h_prev_from_update;

        let mut dz_t = Array2::<f32>::zeros((batch, 3 * u));
        dz_t.slice_mut(s![.., 0..2 * u]).assign(&dz_rz_t);
        dz_t.slice_mut(s![.., 2 * u..])
            .assign(&grad_h_candidate_raw);
        Ok(dz_t)
    }
}

/// Gated Recurrent Unit (GRU) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses reset, update, and candidate
/// gates to control information flow and reduce vanishing gradients.
///
/// All 3 gates are stored fused. The kernels are packed side by side into single matrices.
/// Column blocks follow the order `[update | reset | candidate]` (`[z | r | h]`).
///
/// Per timestep, the reset and update recurrent projections fuse into 1 GEMM. Only the
/// candidate's recurrent projection stays separate, because its input `r_t .* h_{t-1}`
/// depends on the freshly computed reset gate.
///
/// [`GRU::with_return_sequences`] makes the layer return every timestep's hidden state, with
/// shape (batch_size, timesteps, units). [`GRU::with_go_backwards`] processes the input
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
/// // Create GRU layer with 4 input features, 3 units, Tanh activation
/// let mut model = SequentialBuilder::new()
///     .add(GRU::new(3, Activation::Tanh).unwrap())
///     .build(&Shape::known(input.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Train the model
/// model.fit(&input, &target, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&input).unwrap();
/// println!("GRU output shape: {:?}", predictions.shape());
/// // Output: [2, 3] (batch_size, units)
/// ```
pub struct GRU(Rnn<GruCell>);

impl GRU {
    /// Creates a GRU layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `units` - Number of output units in the layer
    /// - `activation` - Activation from the activation module (any [`Activation`] variant, or
    ///   any standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new GRU layer instance
    ///
    /// # Notes
    ///
    /// The constructor draws nothing. [`UnaryLayer::build`] reads the feature count from the input
    /// shape and draws the gates then. The draw takes the global seed or entropy by default.
    /// For reproducible initialization, set a seed with [`GRU::with_random_state`].
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
    /// `return_sequences` set to false.
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

    /// Sets the fused weights for this GRU layer
    ///
    /// # Parameters
    ///
    /// - `kernel` - Fused input kernel with shape (input_dim, 3 * units), gate column blocks in
    ///   the order `[z | r | h]` (update, reset, candidate)
    /// - `recurrent_kernel` - Fused recurrent kernel with shape (units, 3 * units), same block
    ///   order
    /// - `bias` - Fused bias with shape (1, 3 * units), same block order
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

    /// Sets the weights gate by gate, packing them into the fused `[z | r | h]` layout
    ///
    /// Convenience wrapper over [`GRU::set_weights`] for callers that hold per-gate matrices.
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
    /// - `candidate_recurrent_kernel` - Recurrent kernel for the candidate gate with shape
    ///   (units, units)
    /// - `candidate_bias` - Bias for the candidate gate with shape (1, units)
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when every per-gate weight matches the expected shape
    ///
    /// # Notes
    ///
    /// The arguments stay in reset, update, candidate order, even though the fused kernel packs
    /// update first. Each argument is an `Array2<f32>`, so reordering them to match the packing
    /// would compile at every call site and silently swap 2 gates.
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any provided weight does not match the
    ///   expected per-gate shape
    #[allow(clippy::too_many_arguments)] // 3 gates * (kernel, recurrent_kernel, bias)
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
        let per_gate_kernel = [self.0.input_dim, self.0.units];
        let per_gate_recurrent = [self.0.units, self.0.units];
        let per_gate_bias = [1, self.0.units];
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
                update_kernel.view(),
                reset_kernel.view(),
                candidate_kernel.view(),
            ],
        )
        .expect("per-gate kernels share [input_dim, units]");
        let recurrent_kernel = concatenate(
            Axis(1),
            &[
                update_recurrent_kernel.view(),
                reset_recurrent_kernel.view(),
                candidate_recurrent_kernel.view(),
            ],
        )
        .expect("per-gate recurrent kernels share [units, units]");
        let bias = concatenate(
            Axis(1),
            &[update_bias.view(), reset_bias.view(), candidate_bias.view()],
        )
        .expect("per-gate biases share [1, units]");

        self.set_weights(kernel, recurrent_kernel, bias)
    }
}

recurrent_layer_traits!(GRU);
