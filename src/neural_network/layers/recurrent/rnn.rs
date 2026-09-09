//! The 1 recurrent layer of the crate, and the driver of its cell
//!
//! Every recurrent layer of the crate is an [`Rnn`] over 1 [`RnnCell`]. The public names
//! `SimpleRNN`, `LSTM` and `GRU` each hold 1 of these, and they forward every method to it. This
//! module holds the parts that do not depend on which cell runs: the build, batched input
//! projection, and the walk along the time axis. It also holds the cache, the backpropagation
//! through time, and the reduction of the per-step gate gradients into 1 gradient per array.

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::recurrent::cell::RnnCell;
use crate::neural_network::layers::recurrent::gate::{FusedGates, project_input, reshape_2d_to_3d};
use crate::neural_network::layers::recurrent::input_step;
use crate::neural_network::layers::recurrent::validation::{
    split_grad_output, validate_dimension_greater_than_zero, validate_input_3d,
    validate_recurrent_dimensions,
};
use crate::neural_network::layers::validation::{start_build, validate_weight_shape};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};
use gemmkit_ndarray::{Parallelism, dot};
use ndarray::{Array2, Array3, Axis, Ix3, s};
use std::marker::PhantomData;

/// A recurrent layer over 1 cell
///
/// The layer takes an input of shape `[batch, timesteps, features]`. It returns the last
/// hidden state, with shape `[batch, units]`, or every hidden state in processing order,
/// with shape `[batch, timesteps, units]`.
///
/// The gates are fused. All the input kernels sit side by side in 1 matrix, and so do all the
/// recurrent kernels and all the biases. The column-block order is a property of the cell. This
/// lets the input projection of every timestep and every gate run as 1 matrix product.
pub(crate) struct Rnn<C: RnnCell> {
    /// Feature count per timestep, which the build reads from the input shape
    pub(super) input_dim: usize,
    /// Shape that the gates depend on, which is `(None, None, input_dim)`. `None` before the
    /// build
    pub(super) built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    pub(super) random_state: Option<u64>,
    /// Number of output units per gate
    pub(super) units: usize,
    /// The fused gate weights
    pub(super) gates: FusedGates,
    /// The arithmetic of 1 timestep
    pub(super) cell: C,
    /// Returns every timestep's hidden state when true, or only the last one when false
    pub(super) return_sequences: bool,
    /// Processes the input timesteps from last to first when true
    pub(super) go_backwards: bool,
}

/// What the forward pass parks for its backward pass
///
/// The 2 record vectors are flat. The state of processing step `k` sits at
/// `k * STATE_COUNT`, and the values that step `k` parked start at `k * RECORD_SLOTS`.
///
/// The cell type is part of the cache type, so a cache of 1 recurrent layer cannot decode as the
/// cache of another. The layer name that [`Ctx::push_cache`] records is the second guard.
#[derive(Debug)]
struct RnnCache<C: RnnCell> {
    /// The input of the pass, with shape `[batch, timesteps, features]`
    input: Array3<f32>,
    /// The state entering each processing step, and the state leaving the last step
    states: Vec<Array2<f32>>,
    /// Every value that the cell parked, step after step
    records: Vec<Array2<f32>>,
    /// Ties the cache to 1 cell type
    cell: PhantomData<C>,
}

/// Reports the layer under the name of its cell, and not under the name of the base
///
/// A user holds a `SimpleRNN`, an `LSTM` or a `GRU`, and never holds the base. The name of the
/// cell is therefore the name to print
impl<C: RnnCell> std::fmt::Debug for Rnn<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(C::CELL_TYPE)
            .field("input_dim", &self.input_dim)
            .field("built", &self.built)
            .field("random_state", &self.random_state)
            .field("units", &self.units)
            .field("gates", &self.gates)
            .field("cell", &self.cell)
            .field("return_sequences", &self.return_sequences)
            .field("go_backwards", &self.go_backwards)
            .finish()
    }
}

impl<C: RnnCell> Rnn<C> {
    /// Creates a recurrent layer with the given unit count and activation
    ///
    /// # Parameters
    ///
    /// - `units` - Number of output units
    /// - `activation` - Activation of the cell
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The layer
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidParameter`] - If `units` is 0
    /// - [`Error::InvalidParameter`] - If the activation carries an unusable parameter
    pub(super) fn new(units: usize, activation: impl Into<Activation>) -> Result<Self, Error> {
        validate_dimension_greater_than_zero(units, "units")?;
        let activation = activation.into();
        activation.validate()?;
        Ok(Self {
            input_dim: 0,
            built: None,
            random_state: None,
            units,
            gates: FusedGates::empty(),
            cell: C::new(activation),
            return_sequences: false,
            go_backwards: false,
        })
    }

    /// Sets the seed of the weight draw, and redraws when the layer is already built
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed of the draw
    ///
    /// # Returns
    ///
    /// - `Self` - The layer
    pub(super) fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        if self.built.is_some() {
            self.draw_parameters();
        }
        self
    }

    /// Draws the fused gates from the seed of the layer
    ///
    /// The draw order is part of the contract of the layer. The fused input kernel comes first,
    /// and then 1 orthogonal block per gate in column-block order, all against 1 generator.
    fn draw_parameters(&mut self) {
        let mut rng = crate::random::make_rng(self.random_state);
        self.gates = FusedGates::new(self.input_dim, self.units, C::GATE_BIASES, &mut rng)
            .expect("the build validates both dimensions before the draw");
    }

    /// Replaces the 3 fused arrays of the layer
    ///
    /// Each array takes the standard layout, because [`LayerBase::parameters_mut`] hands an
    /// optimizer a flat slice of every array.
    ///
    /// # Parameters
    ///
    /// - `kernel` - Fused input kernel, with shape `[features, gate count * units]`
    /// - `recurrent_kernel` - Fused recurrent kernel, with shape `[units, gate count * units]`
    /// - `bias` - Fused bias, with shape `[1, gate count * units]`
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when every array matches the shape that the layer holds
    ///
    /// # Errors
    ///
    /// - [`Error::NeuralNetwork`] - If the layer is not built
    /// - [`Error::NeuralNetwork`] - If any array does not match the shape that the layer holds
    pub(super) fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built(C::CELL_TYPE));
        }
        validate_weight_shape("kernel", self.gates.kernel.shape(), kernel.shape())?;
        validate_weight_shape(
            "recurrent_kernel",
            self.gates.recurrent_kernel.shape(),
            recurrent_kernel.shape(),
        )?;
        validate_weight_shape("bias", self.gates.bias.shape(), bias.shape())?;
        self.gates.kernel = kernel.as_standard_layout().into_owned();
        self.gates.recurrent_kernel = recurrent_kernel.as_standard_layout().into_owned();
        self.gates.bias = bias.as_standard_layout().into_owned();
        Ok(())
    }
}

impl<C: RnnCell> LayerBase for Rnn<C> {
    fn layer_type(&self) -> &str {
        C::CELL_TYPE
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

impl<C: RnnCell> UnaryLayer for Rnn<C> {
    /// Reads the feature count from the last axis, and draws the fused gates
    ///
    /// The gates depend on the feature count and on the unit count, and on no other extent. The
    /// build shape therefore fixes the last axis alone, and the layer takes a batch of any size
    /// and a sequence of any length
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        debug_assert!(
            groups_partition_the_gates::<C>(),
            "{} leaves a gate block out of its recurrent groups, or covers one twice",
            C::CELL_TYPE
        );
        input.check_rank(C::CELL_TYPE, 3)?;
        let Some(input_dim) = input.axes()[2] else {
            return Err(Error::invalid_input(format!(
                "{} needs a fixed feature count on axis 2, and the shape {input} leaves \
                 that axis free",
                C::CELL_TYPE
            )));
        };
        validate_recurrent_dimensions(input_dim, self.units)?;
        let canonical = Shape::new(vec![None, None, Some(input_dim)]);
        let Some(built) = start_build(&self.built, C::CELL_TYPE, &canonical)? else {
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
            return Err(Error::not_built(C::CELL_TYPE));
        }
        validate_input_3d(input)?;
        let x3 = input.view().into_dimensionality::<Ix3>().unwrap();
        let (batch, timesteps) = (x3.shape()[0], x3.shape()[1]);
        let training = ctx.is_training();

        // The input projection does not depend on the recurrence, so every timestep and every
        // gate goes through 1 matrix product
        let xw = project_input(&self.gates.kernel, &x3);

        let mut sequence = if self.return_sequences {
            Some(Array3::<f32>::zeros((batch, timesteps, self.units)))
        } else {
            None
        };

        let mut state: Vec<Array2<f32>> = (0..C::STATE_COUNT)
            .map(|_| Array2::<f32>::zeros((batch, self.units)))
            .collect();
        let mut states = Vec::with_capacity(if training {
            (timesteps + 1) * C::STATE_COUNT
        } else {
            0
        });
        let mut records = Vec::with_capacity(if training {
            timesteps * C::RECORD_SLOTS
        } else {
            0
        });
        if training {
            states.extend_from_slice(&state);
        }

        // A recurrence runs 1 timestep after another, so this loop cannot spread over threads
        for k in 0..timesteps {
            let t = input_step(k, timesteps, self.go_backwards);
            let record = if training { Some(&mut records) } else { None };
            self.cell
                .step(&self.gates, xw.index_axis(Axis(1), t), &mut state, record)?;
            debug_assert!(
                !training || records.len() == (k + 1) * C::RECORD_SLOTS,
                "{} parked a number of values that RECORD_SLOTS does not name",
                C::CELL_TYPE
            );
            if let Some(seq) = sequence.as_mut() {
                seq.index_axis_mut(Axis(1), k).assign(&state[0]);
            }
            if training {
                states.extend_from_slice(&state);
            }
        }

        // The cache goes in once, after the last step. A step that fails therefore leaves the
        // context as it found it
        if training {
            ctx.push_cache(
                C::CELL_TYPE,
                RnnCache::<C> {
                    input: x3.to_owned(),
                    states,
                    records,
                    cell: PhantomData,
                },
            );
        }

        Ok(match sequence {
            Some(seq) => seq.into_dyn(),
            None => state
                .into_iter()
                .next()
                .expect("every cell carries a hidden state at slot 0")
                .into_dyn(),
        })
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let RnnCache::<C> {
            input: x3,
            states,
            records,
            ..
        } = ctx.pop_cache(C::CELL_TYPE)?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];
        let units = self.units;
        let width = C::GATE_BIASES.len() * units;

        // With `return_sequences`, every step also takes a direct contribution from `grad_seq`.
        // Only the hidden state takes it. Every other state is reachable through the hidden
        // state alone
        let (grad_hidden, grad_seq) = split_grad_output(
            grad_output,
            C::CELL_TYPE,
            self.return_sequences,
            batch,
            timesteps,
            units,
        )?;
        let mut grad_state: Vec<Array2<f32>> = std::iter::once(grad_hidden)
            .chain((1..C::STATE_COUNT).map(|_| Array2::<f32>::zeros((batch, units))))
            .collect();

        // Fused pre-activation gradients of every timestep, in column-block order
        let mut dz3 = Array3::<f32>::zeros((batch, timesteps, width));

        // Backpropagation through time
        for k in (0..timesteps).rev() {
            // The direct contribution accumulates onto the carried gradient. It must land before
            // the cell reads that gradient, which consumes the total gradient of this step
            if let Some(seq) = grad_seq.as_ref() {
                grad_state[0] += &seq.index_axis(Axis(1), k);
            }
            let dz_t = self.cell.step_backward(
                &self.gates,
                &states[k * C::STATE_COUNT..(k + 1) * C::STATE_COUNT],
                &states[(k + 1) * C::STATE_COUNT..(k + 2) * C::STATE_COUNT],
                &records[k * C::RECORD_SLOTS..(k + 1) * C::RECORD_SLOTS],
                &mut grad_state,
            )?;
            // The reductions below pair this step's gate gradients with the input row they came
            // from, so the scatter uses the input timestep, not the processing step
            dz3.index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards))
                .assign(&dz_t);
        }

        // Batched reductions over every timestep, 1 matrix product each
        let dz_flat = dz3
            .to_shape((batch * timesteps, width))
            .expect("contiguous DZ reshape");
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");

        let grad_kernel = dot(&x_flat.t(), &dz_flat);
        let grad_bias = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_x3 = reshape_2d_to_3d(
            dot(&dz_flat, &self.gates.kernel.t()),
            (batch, timesteps, feat),
        );

        // Each group of gate blocks projected 1 array, so each group needs its own product. The
        // product goes straight into the group's column block, and no group needs a buffer of
        // its own
        let mut grad_recurrent = Array2::<f32>::zeros((units, width));
        for group in C::RECURRENT_GROUPS {
            let columns = group.first * units..(group.first + group.count) * units;
            let mut operand3 = Array3::<f32>::zeros((batch, timesteps, units));
            for k in 0..timesteps {
                let operand = match group.operand {
                    None => &states[k * C::STATE_COUNT],
                    Some(slot) => &records[k * C::RECORD_SLOTS + slot],
                };
                operand3
                    .index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards))
                    .assign(operand);
            }
            let operand_flat = operand3
                .to_shape((batch * timesteps, units))
                .expect("contiguous operand reshape");
            gemmkit_ndarray::gemm(
                1.0,
                &operand_flat.t(),
                &dz_flat.slice(s![.., columns.clone()]),
                0.0,
                &mut grad_recurrent.slice_mut(s![.., columns]),
                Parallelism::Rayon(0),
            );
        }

        ctx.add_grad(
            "kernel",
            grad_kernel.as_standard_layout().to_owned().into_dyn(),
        )?;
        ctx.add_grad("recurrent_kernel", grad_recurrent.into_dyn())?;
        ctx.add_grad("bias", grad_bias.as_standard_layout().to_owned().into_dyn())?;

        Ok(grad_x3.into_dyn())
    }

    /// A returned sequence keeps the time axis, and a returned final state drops it
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank(C::CELL_TYPE, 3)?;
        let axes = input.axes();
        // The unit count settles the answer, so an unbuilt layer gives it. A built layer holds
        // a kernel of a fixed width, and it refuses a feature count that the kernel cannot take
        if self.built.is_some()
            && let Some(features) = axes[2]
            && features != self.input_dim
        {
            return Err(Error::invalid_input(format!(
                "{} expects {} features per timestep, got {features}",
                C::CELL_TYPE,
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

/// Reports whether the recurrent groups of a cell cover every gate block exactly once
///
/// A gap leaves a column block of the recurrent gradient at 0, and an overlap makes 1 product
/// overwrite another. Neither reports an error of its own, so a debug build checks the map at
/// the build of the layer
fn groups_partition_the_gates<C: RnnCell>() -> bool {
    let mut covered = vec![0_usize; C::GATE_BIASES.len()];
    for group in C::RECURRENT_GROUPS {
        for block in group.first..group.first + group.count {
            match covered.get_mut(block) {
                Some(count) => *count += 1,
                None => return false,
            }
        }
    }
    covered.iter().all(|count| *count == 1)
}

/// Generates the 2 layer traits of a public recurrent layer, each method forwarding to the
/// [`Rnn`] that the layer holds
///
/// The public layers are newtypes over [`Rnn`]. Each one keeps its own documentation page,
/// its own name in a compiler message, and its own `Debug` output. Only these 13 methods are
/// mechanical, so only these 13 come from a macro. Every method that a user calls directly is
/// written out in the file of its layer, with the prose of that layer
///
/// The macro also gives the layer a `Debug` that prints what the base prints, so the layer
/// reports its own name
///
/// The file that calls this macro must have [`LayerBase`] and [`UnaryLayer`] in scope
macro_rules! recurrent_layer_traits {
    ($layer:ident) => {
        impl std::fmt::Debug for $layer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl LayerBase for $layer {
            fn layer_type(&self) -> &str {
                self.0.layer_type()
            }

            fn param_count(&self) -> $crate::neural_network::layers::ParamCounts {
                self.0.param_count()
            }

            fn parameters_mut(&mut self) -> Vec<$crate::neural_network::traits::ParamRef<'_>> {
                self.0.parameters_mut()
            }

            fn weights(&self) -> Vec<$crate::neural_network::traits::WeightRef<'_>> {
                self.0.weights()
            }

            fn weights_mut(&mut self) -> Vec<$crate::neural_network::traits::WeightMut<'_>> {
                self.0.weights_mut()
            }

            fn known_input_shapes(&self) -> Option<Vec<$crate::neural_network::Shape>> {
                self.0.known_input_shapes()
            }

            fn is_built(&self) -> bool {
                self.0.is_built()
            }

            fn build_config(
                &self,
            ) -> Option<$crate::neural_network::layers::checkpoint::BuildConfig> {
                self.0.build_config()
            }
        }

        impl UnaryLayer for $layer {
            fn build(
                &mut self,
                input: &$crate::neural_network::Shape,
            ) -> Result<(), $crate::error::Error> {
                self.0.build(input)
            }

            fn forward(
                &self,
                input: &$crate::neural_network::Tensor,
                ctx: &mut $crate::neural_network::Ctx,
            ) -> Result<$crate::neural_network::Tensor, $crate::error::Error> {
                self.0.forward(input, ctx)
            }

            fn backward(
                &self,
                grad_output: &$crate::neural_network::Tensor,
                ctx: &mut $crate::neural_network::Ctx,
            ) -> Result<$crate::neural_network::Tensor, $crate::error::Error> {
                self.0.backward(grad_output, ctx)
            }

            fn compute_output_shape(
                &self,
                input: &$crate::neural_network::Shape,
            ) -> Result<$crate::neural_network::Shape, $crate::error::Error> {
                self.0.compute_output_shape(input)
            }
        }
    };
}
pub(super) use recurrent_layer_traits;
