//! Long Short-Term Memory (LSTM) recurrent layer with input, forget, cell, and output gates

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::recurrent::gate::{FusedGates, project_input};
use crate::neural_network::layers::recurrent::validation::{
    split_grad_output, validate_dimension_greater_than_zero, validate_input_3d,
    validate_recurrent_dimensions,
};
use crate::neural_network::layers::recurrent::{apply_sigmoid, input_step};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Bias, Parallelism};
use ndarray::{Array2, Array3, ArrayView3, Axis, Ix2, Ix3, concatenate, s};

/// Long Short-Term Memory (LSTM) neural network layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). Uses input, forget, cell, and
/// output gates to control memory flow and reduce vanishing gradients.
///
/// All 4 gates are stored fused. The kernels are packed side by side into single matrices.
/// Column blocks follow the order `[input | forget | cell | output]` (`[i | f | g | o]`),
/// matching Keras. Each projection runs as 1 GEMM instead of 4.
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
#[derive(Debug)]
pub struct LSTM {
    /// Feature count per timestep, which [`UnaryLayer::build`] reads from the input shape
    input_dim: usize,
    /// Shape the gates depend on, which is `(None, None, input_dim)`. `None` before the build
    built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Number of LSTM units (neurons) in the layer
    units: usize,

    /// Fused gate weights, column blocks in the order `[i | f | g | o]`
    gates: FusedGates,

    /// Activation applied to the candidate and to the cell state each timestep (Keras-style)
    activation: Activation,
    /// Returns the full sequence of hidden states when true, or only the last one when false
    return_sequences: bool,
    /// Processes the input timesteps from last to first when true
    go_backwards: bool,
}

/// What the forward pass of [`LSTM`] parks for its backward pass
///
/// The per-timestep values let the backward pass recompute the gate gradients without a second
/// run of the forward recurrence
#[derive(Debug)]
struct LstmCaches {
    /// The input of the pass, with shape (batch_size, timesteps, input_dim)
    input: Array3<f32>,
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
    /// - `units` - Number of LSTM units/neurons in the layer (determines output dimensionality)
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
        validate_dimension_greater_than_zero(units, "units")?;
        let activation = activation.into();
        activation.validate()?;

        Ok(Self {
            input_dim: 0,
            built: None,
            random_state: None,
            units,
            gates: FusedGates::empty(),
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
    /// `return_sequences` set to false. The cell state stays internal in both cases.
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

    /// Initializes the fused `[i | f | g | o]` gate blocks from the given seed.
    ///
    /// 1 RNG is threaded through all 4 gate blocks. The forget gate bias starts at 1.0.
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
    ///   the order `[i | f | g | o]`
    /// - `recurrent_kernel` - Fused recurrent kernel with shape (units, 4 * units), same block
    ///   order
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
        if self.built.is_none() {
            return Err(Error::not_built("LSTM"));
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

    /// Runs the recurrence and returns the layer output. This is the numeric body of
    /// [`UnaryLayer::forward`].
    ///
    /// The output is the last hidden state, with shape (batch_size, units). With
    /// `return_sequences` set, it is instead every hidden state in processing order, with shape
    /// (batch_size, timesteps, units).
    ///
    /// When `caches` is `Some`, the pass records every per-timestep value the backward pass
    /// needs: the hidden and cell states, `activation(c_t)`, and the 4 gate activations. An
    /// inference pass passes `None` and skips the recording. Every record stays in processing
    /// order.
    ///
    /// Each timestep computes all 4 gate pre-activations with 1 fused GEMM. The recurrent
    /// product accumulates onto the pre-projected `x_t @ kernel` slice, and the epilogue adds
    /// the bias.
    ///
    /// The GEMM calls use gemmkit's automatic parallelism, so gemmkit picks serial or parallel
    /// execution based on its own work-size gate.
    fn run(
        &self,
        x3: &ArrayView3<f32>,
        mut caches: Option<&mut LstmCaches>,
    ) -> Result<Tensor, Error> {
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        let u = self.units;
        let act = self.activation;
        let bias = self
            .gates
            .bias
            .as_slice()
            .expect("fused bias must be contiguous");

        let mut sequence = if self.return_sequences {
            Some(Array3::<f32>::zeros((batch, timesteps, u)))
        } else {
            None
        };

        let mut h_prev = Array2::<f32>::zeros((batch, u));
        let mut c_prev = Array2::<f32>::zeros((batch, u));
        if let Some(c) = caches.as_deref_mut() {
            c.hs.push(h_prev.clone());
            c.cs.push(c_prev.clone());
        }

        // Batched fused input projection for all 4 gates
        let xw = project_input(&self.gates.kernel, x3);

        for k in 0..timesteps {
            let t = input_step(k, timesteps, self.go_backwards);
            // All 4 gate pre-activations in 1 fused recurrent GEMM, accumulated on top of the
            // pre-projected `x_t @ kernel` slice
            let mut z_all = xw.index_axis(Axis(1), t).to_owned(); // [batch, 4*units]

            gemmkit_ndarray::gemm_fused(
                1.0,
                &h_prev,
                &self.gates.recurrent_kernel,
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
            if let Some(seq) = sequence.as_mut() {
                seq.index_axis_mut(Axis(1), k).assign(&h_t);
            }

            h_prev = h_t;
            c_prev = c_t;
        }

        Ok(match sequence {
            Some(seq) => seq.into_dyn(),
            None => h_prev.into_dyn(),
        })
    }
}

impl LayerBase for LSTM {
    fn layer_type(&self) -> &str {
        "LSTM"
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so a change to
        // the roster corrects the count with no second formula to keep in step
        ParamCounts::trainable(
            self.gates.kernel.len() + self.gates.recurrent_kernel.len() + self.gates.bias.len(),
        )
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        self.gates.parameters_mut()
    }

    // The layer keeps no input shape. It knows the feature count of 1 timestep, and it
    // serves every batch size and every sequence length, so both of those axes are free
    built_layer_shape_functions!();

    named_weight_layer_functions!(
        trainable "kernel" => gates.kernel,
        trainable "recurrent_kernel" => gates.recurrent_kernel,
        trainable "bias" => gates.bias,
    );
}

impl UnaryLayer for LSTM {
    /// Reads the feature count from the last axis, and draws the fused gates
    ///
    /// The gates depend on the feature count and on the unit count, and on no other extent. The
    /// build shape therefore fixes the last axis alone, and the layer takes a batch of any size
    /// and a sequence of any length
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        input.check_rank("LSTM", 3)?;
        let Some(input_dim) = input.axes()[2] else {
            return Err(Error::invalid_input(format!(
                "LSTM needs a fixed feature count on axis 2, and the shape {input} leaves \
                 that axis free"
            )));
        };
        validate_recurrent_dimensions(input_dim, self.units)?;
        let canonical = Shape::new(vec![None, None, Some(input_dim)]);
        let Some(built) = start_build(&self.built, "LSTM", &canonical)? else {
            return Ok(());
        };
        self.input_dim = input_dim;
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    /// An inference pass records no per-timestep value at all, and it parks no cache
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.is_built() {
            return Err(Error::not_built("LSTM"));
        }
        validate_input_3d(input)?;
        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();

        if !ctx.is_training() {
            return self.run(&x3, None);
        }

        let timesteps = x3.shape()[1];
        let mut caches = LstmCaches {
            input: x3.to_owned(),
            hs: Vec::with_capacity(timesteps + 1),
            cs: Vec::with_capacity(timesteps + 1),
            cs_activated: Vec::with_capacity(timesteps),
            i: Vec::with_capacity(timesteps),
            f: Vec::with_capacity(timesteps),
            g: Vec::with_capacity(timesteps),
            o: Vec::with_capacity(timesteps),
        };
        let output = self.run(&x3, Some(&mut caches))?;
        ctx.push_cache(caches);
        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        // Configurable activation, used for the candidate and cell-state derivatives
        let act = self.activation;

        let LstmCaches {
            input: x3,
            hs,
            cs,
            cs_activated,
            i: i_vals,
            f: f_vals,
            g: g_vals,
            o: o_vals,
        } = ctx.pop_cache("LSTM")?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];
        let u = self.units;

        // With `return_sequences`, every step also takes a direct contribution from `grad_seq`.
        // The cell state is reachable only through the hidden state, so `grad_c` takes none
        let (mut grad_h, grad_seq) = split_grad_output(
            grad_output,
            "LSTM",
            self.return_sequences,
            batch,
            timesteps,
            u,
        )?;

        // Fused pre-activation gradients for every timestep, gate blocks [i | f | g | o]
        let mut dz3 = Array3::<f32>::zeros((batch, timesteps, 4 * u));

        let mut grad_c = Array2::<f32>::zeros((batch, u));

        // Backpropagation through time
        for k in (0..timesteps).rev() {
            // The direct contribution accumulates onto the carried gradient, before any gate
            // backward reads it. `cs[k]` is the cell state that enters processing step `k`,
            // while `cs_activated[k]` is the activated cell state that leaves it
            if let Some(seq) = grad_seq.as_ref() {
                grad_h += &seq.index_axis(Axis(1), k);
            }

            let c_prev = &cs[k];
            let c_t_activated = &cs_activated[k];
            let i_t = &i_vals[k];
            let f_t = &f_vals[k];
            let g_t = &g_vals[k];
            let o_t = &o_vals[k];

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

            // Gradient with respect to the previous hidden state: 1 fused GEMM instead of 4
            grad_h = dot(&dz_t, &self.gates.recurrent_kernel.t());

            // The reductions below pair `dz_t` with the input row it came from, so the scatter
            // uses the input timestep, not the processing step
            dz3.index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards))
                .assign(&dz_t);

            // Gradient with respect to previous cell state
            grad_c = grad_c_prev;
        }

        // Batched reductions over all timesteps, 1 fused GEMM each
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");
        // `hs[k]` is the state that enters processing step `k`, and it pairs with the `dz_t` of
        // that same step, which now sits at the step's input timestep
        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, u));
        for (k, h) in hs.iter().take(timesteps).enumerate() {
            h_prev3
                .index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards))
                .assign(h);
        }
        let h_prev_flat = h_prev3
            .to_shape((batch * timesteps, u))
            .expect("contiguous H_prev reshape");
        let dz_flat = dz3
            .to_shape((batch * timesteps, 4 * u))
            .expect("contiguous DZ reshape");

        let grad_kernel = dot(&x_flat.t(), &dz_flat);
        let grad_recurrent = dot(&h_prev_flat.t(), &dz_flat);
        let grad_bias = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_x3 = crate::neural_network::layers::recurrent::gate::reshape_2d_to_3d(
            dot(&dz_flat, &self.gates.kernel.t()),
            (batch, timesteps, feat),
        );

        ctx.add_grad(
            "kernel",
            grad_kernel.as_standard_layout().to_owned().into_dyn(),
        )?;
        ctx.add_grad(
            "recurrent_kernel",
            grad_recurrent.as_standard_layout().to_owned().into_dyn(),
        )?;
        ctx.add_grad("bias", grad_bias.as_standard_layout().to_owned().into_dyn())?;

        Ok(grad_x3.into_dyn())
    }

    /// A returned sequence keeps the time axis, and a returned final state drops it
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("LSTM", 3)?;
        let axes = input.axes();
        // The unit count settles the answer, so an unbuilt layer gives it. A built layer holds
        // a kernel of a fixed width, and it refuses a feature count that the kernel cannot take
        if self.built.is_some()
            && let Some(features) = axes[2]
            && features != self.input_dim
        {
            return Err(Error::invalid_input(format!(
                "LSTM expects {} features per timestep, got {features}",
                self.input_dim
            )));
        }
        Ok(if self.return_sequences {
            Shape::new(vec![axes[0], axes[1], Some(self.units)])
        } else {
            Shape::new(vec![axes[0], Some(self.units)])
        })
    }
}
