//! Gated Recurrent Unit (GRU) layer with reset, update, and candidate gates

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::recurrent::gate::{FusedGates, project_input, take_cache};
use crate::neural_network::layers::recurrent::validation::{
    split_grad_output, validate_dimension_greater_than_zero, validate_input_3d,
    validate_recurrent_dimensions,
};
use crate::neural_network::layers::recurrent::{apply_sigmoid, input_step};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::layers::{build_config_function, build_on_forward};
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::neural_network::{Shape, Tensor};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Bias, Parallelism};
use ndarray::{Array2, Array3, ArrayView3, Axis, concatenate, s};

/// Gated Recurrent Unit (GRU) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses reset, update, and candidate
/// gates to control information flow and reduce vanishing gradients.
///
/// All 3 gates are stored fused. The kernels are packed side by side into single matrices.
/// Column blocks follow the order `[update | reset | candidate]` (`[z | r | h]`), matching
/// Keras. This lets the input projection run as 1 GEMM instead of 3.
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
#[derive(Debug)]
pub struct GRU {
    /// Feature count per timestep, which [`Layer::build`] reads from the input shape
    input_dim: usize,
    /// Shape the gates depend on, which is `(None, None, input_dim)`. `None` before the build
    built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Number of GRU units (neurons) in the layer
    units: usize,

    /// Fused gate weights, column blocks in the order `[z | r | h]`
    gates: FusedGates,

    /// Cached input tensor for backward propagation
    input_cache: Option<Array3<f32>>,
    /// Per-timestep forward values recorded by `forward` for the backward pass. This is `None`
    /// until the first training forward. `predict` never sets it.
    caches: Option<GruCaches>,

    /// Activation applied to the candidate hidden state each timestep (Keras-style)
    activation: Activation,
    /// Returns the full sequence of hidden states when true, or only the last one when false
    return_sequences: bool,
    /// Processes the input timesteps from last to first when true
    go_backwards: bool,
}

/// Per-timestep forward values a [`GRU`] records so the backward pass can recompute the gate
/// gradients without re-running the forward recurrence
#[derive(Debug)]
struct GruCaches {
    /// Hidden states `h_t`, with `h_0 = 0` prepended (length `timesteps + 1`)
    hs: Vec<Array2<f32>>,
    /// Reset-gate activations (sigmoid) per timestep
    r: Vec<Array2<f32>>,
    /// Update-gate activations (sigmoid) per timestep
    z: Vec<Array2<f32>>,
    /// Candidate hidden states (activation applied) per timestep
    h_candidate: Vec<Array2<f32>>,
    /// `r_t .* h_{t-1}` per timestep (the candidate's recurrent input)
    rh: Vec<Array2<f32>>,
}

impl GRU {
    /// Creates a GRU layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `units` - Number of GRU units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation from the activation module (any [`Activation`] variant, or
    ///   any standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new GRU layer instance
    ///
    /// # Notes
    ///
    /// The constructor draws nothing. [`Layer::build`] reads the feature count from the input
    /// shape and draws the gates then. The draw takes the global seed or entropy by default.
    /// For reproducible initialization, set a seed with [`GRU::with_random_state`].
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `units` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(units: usize, activation: impl Into<Activation>) -> Result<Self, Error> {
        validate_dimension_greater_than_zero(units, "units")?;
        let activation = activation.into();
        activation.validate()?;

        Ok(Self {
            input_dim: 0,
            built: None,
            random_state: None,
            units,
            gates: FusedGates::empty(),
            input_cache: None,
            caches: None,
            activation,
            return_sequences: false,
            go_backwards: false,
        })
    }

    /// Sets the seed used to initialize the gate weights and re-initializes them deterministically.
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
        self.random_state = Some(random_state);
        if self.built.is_some() {
            self.draw_parameters();
        }
        self
    }

    /// Draws the fused gates at the extents the build settled
    ///
    /// 1 generator threads the fused input kernel and then 1 orthogonal block per gate, in gate
    /// order. See [`FusedGates::new`]
    fn draw_parameters(&mut self) {
        // The build validated both dimensions, so the draw cannot fail
        self.gates = Self::init_gates(self.input_dim, self.units, self.random_state)
            .expect("the build validated both dimensions");
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
        self.return_sequences = return_sequences;
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
        self.go_backwards = go_backwards;
        self
    }

    /// Initializes the fused `[z | r | h]` gate blocks from the given seed.
    ///
    /// 1 RNG is threaded through all 3 gate blocks. All biases start at 0.0.
    fn init_gates(
        input_dim: usize,
        units: usize,
        random_state: Option<u64>,
    ) -> Result<FusedGates, Error> {
        let mut rng = crate::random::make_rng(random_state);
        FusedGates::new(input_dim, units, &[0.0, 0.0, 0.0], &mut rng)
    }

    /// Sets the fused weights for this GRU layer
    ///
    /// # Parameters
    ///
    /// - `kernel` - Fused input kernel with shape (input_dim, 3 * units), gate column blocks in
    ///   the order `[z | r | h]` (update, reset, candidate), matching Keras
    /// - `recurrent_kernel` - Fused recurrent kernel with shape (units, 3 * units), same block
    ///   order
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
        if self.built.is_none() {
            return Err(Error::not_built("GRU"));
        }
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

    /// Sets the weights gate by gate, packing them into the fused `[z | r | h]` layout
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
    /// - `candidate_recurrent_kernel` - Recurrent kernel for the candidate gate with shape
    ///   (units, units)
    /// - `candidate_bias` - Bias for the candidate gate with shape (1, units)
    ///
    /// # Notes
    ///
    /// The arguments stay in reset, update, candidate order, even though the fused kernel packs
    /// update first. Each argument is an `Array2<f32>`, so reordering them to match the packing
    /// would compile at every call site and silently swap 2 gates.
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

    /// Runs the recurrence and returns the layer output. This is the shared numeric body of
    /// [`Layer::forward`] and [`Layer::predict`].
    ///
    /// The output is the last hidden state, with shape (batch_size, units). With
    /// `return_sequences` set, it is instead every hidden state in processing order, with shape
    /// (batch_size, timesteps, units).
    ///
    /// When `caches` is `Some`, the pass records every per-timestep value the backward pass
    /// needs. This includes the hidden states, the reset and update gate activations, the
    /// candidate, and `r_t .* h_{t-1}`. `predict` passes `None` and skips the recording. Every
    /// record stays in processing order.
    ///
    /// Each timestep computes the reset and update gates with 1 fused GEMM, then the candidate
    /// with a second GEMM whose input is `r_t .* h_{t-1}`.
    ///
    /// The GEMM calls use gemmkit's automatic parallelism, so gemmkit picks serial or parallel
    /// execution based on its own work-size gate.
    fn run(
        &self,
        x3: &ArrayView3<f32>,
        mut caches: Option<&mut GruCaches>,
    ) -> Result<Tensor, Error> {
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        let u = self.units;
        let act = self.activation;
        // Bias blocks `[z | r]` and `[h]`, each folded into its own product's epilogue
        let (bias_rz, bias_h) = self
            .gates
            .bias
            .as_slice()
            .expect("fused bias must be contiguous")
            .split_at(2 * u);

        let mut sequence = if self.return_sequences {
            Some(Array3::<f32>::zeros((batch, timesteps, u)))
        } else {
            None
        };

        let mut h_prev = Array2::<f32>::zeros((batch, u));
        if let Some(c) = caches.as_deref_mut() {
            c.hs.push(h_prev.clone());
        }

        // Batched fused input projection for all 3 gates
        let xw = project_input(&self.gates.kernel, x3);

        for k in 0..timesteps {
            let t = input_step(k, timesteps, self.go_backwards);
            let xw_t = xw.index_axis(Axis(1), t); // [batch, 3*units]

            // Reset and update share h_prev, so their recurrent projections fuse into 1 GEMM
            let mut rz = xw_t.slice(s![.., 0..2 * u]).to_owned();
            gemmkit_ndarray::gemm_fused(
                1.0,
                &h_prev,
                &self.gates.recurrent_kernel.slice(s![.., 0..2 * u]),
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
            let r_h = &r_t * &h_prev;
            let mut h_candidate = xw_t.slice(s![.., 2 * u..]).to_owned();
            let rk_h = self.gates.recurrent_kernel.slice(s![.., 2 * u..]);
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
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Hidden state update
            let h_t = &z_t * &h_prev + &(1.0 - &z_t) * &h_candidate;

            if let Some(c) = caches.as_deref_mut() {
                c.r.push(r_t);
                c.z.push(z_t);
                c.h_candidate.push(h_candidate);
                c.rh.push(r_h);
                c.hs.push(h_t.clone());
            }
            if let Some(seq) = sequence.as_mut() {
                seq.index_axis_mut(Axis(1), k).assign(&h_t);
            }

            h_prev = h_t;
        }

        Ok(match sequence {
            Some(seq) => seq.into_dyn(),
            None => h_prev.into_dyn(),
        })
    }
}

impl Layer for GRU {
    /// Reads the feature count from the last axis, and draws the fused gates
    ///
    /// The gates depend on the feature count and on the unit count, and on no other extent. The
    /// build shape therefore fixes the last axis alone, and the layer takes a batch of any size
    /// and a sequence of any length
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        input.check_rank("GRU", 3)?;
        let Some(input_dim) = input.axes()[2] else {
            return Err(Error::invalid_input(format!(
                "GRU needs a fixed feature count on axis 2, and the shape {input} leaves \
                 that axis free"
            )));
        };
        validate_recurrent_dimensions(input_dim, self.units)?;
        let canonical = Shape::new(vec![None, None, Some(input_dim)]);
        let Some(built) = start_build(&self.built, "GRU", &canonical)? else {
            return Ok(());
        };
        self.input_dim = input_dim;
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;
        build_on_forward!(self, input);
        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let timesteps = x3.shape()[1];
        self.input_cache = Some(x3.to_owned());

        let mut caches = GruCaches {
            hs: Vec::with_capacity(timesteps + 1),
            r: Vec::with_capacity(timesteps),
            z: Vec::with_capacity(timesteps),
            h_candidate: Vec::with_capacity(timesteps),
            rh: Vec::with_capacity(timesteps),
        };
        let output = self.run(&x3, Some(&mut caches))?;
        self.caches = Some(caches);
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("GRU"));
        }
        validate_input_3d(input)?;
        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        self.run(&x3, None)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Configurable activation (Copy) used for the candidate derivative
        let act = self.activation;

        let x3 = take_cache(&mut self.input_cache, "GRU")?;
        let GruCaches {
            hs,
            r: r_vals,
            z: z_vals,
            h_candidate: h_candidate_vals,
            rh: rh_vals,
        } = take_cache(&mut self.caches, "GRU")?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];
        let u = self.units;

        // With `return_sequences`, every step also takes a direct contribution from `grad_seq`
        let (mut grad_h, grad_seq) = split_grad_output(
            grad_output,
            "GRU",
            self.return_sequences,
            batch,
            timesteps,
            u,
        )?;

        // Fused pre-activation gradients for every timestep, gate blocks [z | r | h]
        let mut dz3 = Array3::<f32>::zeros((batch, timesteps, 3 * u));

        // Backpropagation through time
        for k in (0..timesteps).rev() {
            // The direct contribution accumulates onto the carried gradient. It must land before
            // the update-gate gradient below, which consumes the total gradient of this step
            if let Some(seq) = grad_seq.as_ref() {
                grad_h += &seq.index_axis(Axis(1), k);
            }

            let h_prev = &hs[k];
            let r_t = &r_vals[k];
            let z_t = &z_vals[k];
            let h_candidate = &h_candidate_vals[k];

            // Gradient through h_t = z_t .* h_{t-1} + (1 - z_t) .* h_candidate
            let grad_z_t = &grad_h * (h_prev - h_candidate);
            let grad_h_candidate = &grad_h * &(1.0 - z_t);
            let grad_h_prev_from_update = &grad_h * z_t;

            // Gradient through h_candidate = activation(...), via the activation backward
            let grad_h_candidate_raw = act
                .backward(
                    &h_candidate.clone().into_dyn(),
                    &grad_h_candidate.into_dyn(),
                )?
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            // Gradient through r_h = r_t .* h_{t-1} (1 recurrent matmul shared by both terms)
            let grad_rh = dot(
                &grad_h_candidate_raw,
                &self.gates.recurrent_kernel.slice(s![.., 2 * u..]).t(),
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

            // Gradient with respect to the previous hidden state
            grad_h = dot(
                &dz_rz_t,
                &self.gates.recurrent_kernel.slice(s![.., 0..2 * u]).t(),
            ) + &grad_h_prev_from_reset
                + &grad_h_prev_from_update;

            // The reductions below pair this step's gate gradients with the input row they came
            // from, so the scatter uses the input timestep, not the processing step
            let mut dz_t3 =
                dz3.index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards));
            dz_t3.slice_mut(s![.., 0..2 * u]).assign(&dz_rz_t);
            dz_t3
                .slice_mut(s![.., 2 * u..])
                .assign(&grad_h_candidate_raw);
        }

        // Batched reductions over all timesteps
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");
        // `hs[k]` and `rh_vals[k]` belong to processing step `k`, and they pair with that step's
        // gate gradients, which now sit at the step's input timestep
        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, u));
        let mut rh3 = Array3::<f32>::zeros((batch, timesteps, u));
        for k in 0..timesteps {
            let t = input_step(k, timesteps, self.go_backwards);
            h_prev3.index_axis_mut(Axis(1), t).assign(&hs[k]);
            rh3.index_axis_mut(Axis(1), t).assign(&rh_vals[k]);
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

        // Input-kernel gradient for all 3 gates in 1 GEMM
        let grad_kernel = dot(&x_flat.t(), &dz_flat);
        let grad_bias = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Recurrent gradient: each product is written straight into its column block (`beta = 0`),
        // so neither needs a temporary of its own
        let mut grad_recurrent = Array2::<f32>::zeros((u, 3 * u));
        gemmkit_ndarray::gemm(
            1.0,
            &h_prev_flat.t(),
            &dz_flat.slice(s![.., 0..2 * u]),
            0.0,
            &mut grad_recurrent.slice_mut(s![.., 0..2 * u]),
            Parallelism::Rayon(0),
        );
        gemmkit_ndarray::gemm(
            1.0,
            &rh_flat.t(),
            &dz_flat.slice(s![.., 2 * u..]),
            0.0,
            &mut grad_recurrent.slice_mut(s![.., 2 * u..]),
            Parallelism::Rayon(0),
        );

        // Input gradient for all 3 gates in 1 GEMM
        let grad_x3 = crate::neural_network::layers::recurrent::gate::reshape_2d_to_3d(
            dot(&dz_flat, &self.gates.kernel.t()),
            (batch, timesteps, feat),
        );

        self.gates
            .store_gradients(grad_kernel, grad_recurrent, grad_bias);

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "GRU"
    }

    build_config_function!();

    fn known_input_shape(&self) -> Option<Shape> {
        // The layer keeps no input shape. It knows the feature count of 1 timestep, and it
        // serves every batch size and every sequence length, so both of those axes are free
        self.built.clone()
    }

    /// A returned sequence keeps the time axis, and a returned final state drops it
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("GRU", 3)?;
        let axes = input.axes();
        // The unit count settles the answer, so an unbuilt layer gives it. A built layer holds
        // a kernel of a fixed width, and it refuses a feature count that the kernel cannot take
        if self.built.is_some()
            && let Some(features) = axes[2]
            && features != self.input_dim
        {
            return Err(Error::invalid_input(format!(
                "GRU expects {} features per timestep, got {features}",
                self.input_dim
            )));
        }
        Ok(if self.return_sequences {
            Shape::new(vec![axes[0], axes[1], Some(self.units)])
        } else {
            Shape::new(vec![axes[0], Some(self.units)])
        })
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so a change to
        // the roster corrects the count with no second formula to keep in step
        ParamCounts::trainable(
            self.gates.kernel.len() + self.gates.recurrent_kernel.len() + self.gates.bias.len(),
        )
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        self.gates.parameters()
    }

    named_weight_layer_functions!(
        trainable "kernel" => gates.kernel,
        trainable "recurrent_kernel" => gates.recurrent_kernel,
        trainable "bias" => gates.bias,
    );
}
