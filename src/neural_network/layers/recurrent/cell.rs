//! The protocol that 1 recurrent cell follows
//!
//! A recurrent layer is 2 separable parts. The first part is the scaffolding. It reads the
//! feature count from the input shape and draws the gates. It projects the whole input in 1
//! matrix product, walks the time axis, and keeps what the backward pass needs. The scaffolding
//! also reduces the per-step gate gradients into 1 gradient per array. That part is the same
//! for every cell, and [`Rnn`](super::rnn::Rnn) holds the only copy of it.
//!
//! The second part is the arithmetic of 1 timestep. That part is what makes a SimpleRNN
//! different from an LSTM. [`RnnCell`] is the protocol for it.
//!
//! The protocol is internal to the crate and it is sealed. The 3 cells of the crate are the only
//! implementations, so the shape of the protocol can change with no effect on any public name.

use crate::error::Error;
use crate::neural_network::layers::activation::{Activation, ActivationCache, ActivationInput};
use crate::neural_network::layers::recurrent::gate::FusedGates;
use ndarray::{Array2, ArrayView2, Ix2};
use std::borrow::Cow;

/// 1 group of gate blocks, and the array that their recurrent projection consumed
///
/// The recurrent kernel packs 1 column block per gate. The gradient of that kernel is the
/// product of whatever entered the recurrent projection against the pre-activation gradient of
/// the gate. Most cells project the previous hidden state into every gate, so 1 product covers
/// the whole kernel. A GRU projects the previous hidden state into its update and reset blocks,
/// and `r_t .* h_prev` into its candidate block. A GRU therefore needs 2 products.
///
/// The groups of a cell must cover every gate block exactly once. A gap leaves a column block of
/// the gradient at 0, which stops 1 gate from learning and reports no error at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecurrentGroup {
    /// Index of the first gate block of the group
    pub first: usize,
    /// Number of gate blocks in the group
    pub count: usize,
    /// The array that the group projected. `None` names the previous hidden state, and
    /// `Some(slot)` names the record that the cell parked at that slot
    pub operand: Option<usize>,
}

/// The arithmetic of 1 timestep of 1 recurrent layer
///
/// [`Rnn`](super::rnn::Rnn) holds a cell and drives it. The cell holds no weight and no state of
/// its own beyond its activation. The weights arrive as a [`FusedGates`] argument, and the
/// recurrent state arrives as a slice that the cell reads and then overwrites.
///
/// # Notes
///
/// Every array of the protocol has the shape `[batch, units]`, except the projected input, which
/// is `[batch, gate count * units]`. The cell reads the unit count from the gates with
/// [`FusedGates::units`], so no cell holds a second copy of that number.
pub(crate) trait RnnCell: std::fmt::Debug + Send + Sync + 'static {
    /// The name that the layer reports, such as `"LSTM"`
    ///
    /// A checkpoint records this name per node, and a strict load compares it. The name also
    /// keys the cache of the pass and it names the layer in every error message.
    const CELL_TYPE: &'static str;

    /// The initial bias of each gate block, in column-block order
    ///
    /// The length of this slice is the gate count of the cell. An LSTM gives its forget gate a
    /// bias of 1.0 and every other gate a bias of 0.0.
    const GATE_BIASES: &'static [f32];

    /// Number of recurrent states that the cell carries from 1 timestep to the next
    ///
    /// Slot 0 is always the hidden state, because the output of the layer reads it. An LSTM
    /// carries the cell state at slot 1.
    const STATE_COUNT: usize;

    /// The map from a group of gate blocks to the array that the group projected
    ///
    /// The groups must partition `0..GATE_BIASES.len()`. [`Rnn::build`](super::rnn::Rnn) asserts
    /// that they do.
    const RECURRENT_GROUPS: &'static [RecurrentGroup];

    /// Creates the cell from the activation that the layer was given
    ///
    /// # Parameters
    ///
    /// - `activation` - Activation of the layer, already validated by the layer
    ///
    /// # Returns
    ///
    /// - `Self` - The cell
    fn new(activation: Activation) -> Self;

    /// Number of arrays that the cell parks per timestep for its own backward pass
    ///
    /// A value that the base already keeps does not belong here. The state of every step is 1
    /// such value, and [`RnnCell::step_backward`] receives it directly. The count can depend on
    /// the activation, because [`Activation::saves`] names the tensor that its backward pass
    /// reads.
    ///
    /// # Returns
    ///
    /// - `usize` - Number of arrays per timestep
    fn record_slots(&self) -> usize;

    /// Runs 1 timestep
    ///
    /// The cell reads `state`, computes the state that leaves the step, and writes it back into
    /// `state`. A training pass also gives a `record` buffer, and the cell appends exactly
    /// [`RnnCell::record_slots`] arrays to it. An inference pass gives `None` and the cell parks
    /// nothing.
    ///
    /// # Parameters
    ///
    /// - `gates` - The fused weights of the layer
    /// - `xw_t` - The projected input of this step, with shape `[batch, gate count * units]`
    /// - `state` - The state entering the step, which the cell overwrites with the state leaving
    ///   it
    /// - `record` - Buffer for the values that only the backward pass reads, or `None`
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when the step ran
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] - If an activation of the cell refuses its input
    fn step(
        &self,
        gates: &FusedGates,
        xw_t: ArrayView2<'_, f32>,
        state: &mut [Array2<f32>],
        record: Option<&mut Vec<Array2<f32>>>,
    ) -> Result<(), Error>;

    /// Runs the backward pass of 1 timestep
    ///
    /// On entry `grad_state` holds the gradient of the state that LEAVES the step. The base has
    /// already added the direct contribution of a returned sequence to slot 0. The cell
    /// overwrites `grad_state` with the gradient of the state that ENTERS the step.
    ///
    /// # Parameters
    ///
    /// - `gates` - The fused weights of the layer
    /// - `state_prev` - The state entering the step
    /// - `state_next` - The state leaving the step
    /// - `record` - The [`RnnCell::record_slots`] arrays that this step parked
    /// - `grad_state` - The gradient of the state, which the cell carries back 1 step
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f32>, Error>` - The pre-activation gradient of every gate of this step,
    ///   with shape `[batch, gate count * units]`, in column-block order
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] - If an activation of the cell refuses its input
    fn step_backward(
        &self,
        gates: &FusedGates,
        state_prev: &[Array2<f32>],
        state_next: &[Array2<f32>],
        record: &[Array2<f32>],
        grad_state: &mut [Array2<f32>],
    ) -> Result<Array2<f32>, Error>;
}

/// The result of 1 activation inside a cell step
///
/// `output` is the activated tensor that the step uses. [`Activated::into_record`] gives the
/// tensor that the backward pass of the activation reads.
#[derive(Debug)]
pub(super) struct Activated {
    /// The activated tensor
    pub output: Array2<f32>,
    /// The pre-activation, which only a training pass of a
    /// [`ActivationInput::PreActivation`] activation keeps
    pub pre_activation: Option<Array2<f32>>,
}

impl Activated {
    /// Gives the tensor that the backward pass of the activation reads
    ///
    /// # Returns
    ///
    /// - `Array2<f32>` - The pre-activation when the step kept it, otherwise the output
    pub fn into_record(self) -> Array2<f32> {
        self.pre_activation.unwrap_or(self.output)
    }
}

/// Applies the activation of a cell to a pre-activation
///
/// A training pass of an [`ActivationInput::PreActivation`] activation keeps `z`. Every other
/// pass drops `z`. No pass copies the output.
///
/// # Parameters
///
/// - `activation` - Activation of the cell
/// - `z` - The pre-activation
/// - `training` - True when the step parks a record
///
/// # Returns
///
/// - `Result<Activated, Error>` - The output, and the pre-activation when the step keeps it
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If the activation refuses its input
pub(super) fn activate(
    activation: &Activation,
    z: Array2<f32>,
    training: bool,
) -> Result<Activated, Error> {
    if training && activation.saves() == ActivationInput::PreActivation {
        let (output, cache) = activation.forward_train(z.into_dyn())?;
        return Ok(Activated {
            output: into_2d(output),
            pre_activation: Some(into_2d(cache.into_tensor())),
        });
    }
    Ok(Activated {
        output: into_2d(activation.forward(&z.into_dyn())?),
        pre_activation: None,
    })
}

/// Builds the activation cache from 1 record
///
/// # Parameters
///
/// - `activation` - Activation of the cell
/// - `record` - The tensor that [`Activated::into_record`] gave
///
/// # Returns
///
/// - `ActivationCache` - A cache that holds a copy of `record`
pub(super) fn record_cache(activation: &Activation, record: &Array2<f32>) -> ActivationCache {
    ActivationCache::new(activation.saves(), record.clone().into_dyn())
}

/// Gives the activated output that 1 record stands for
///
/// An [`ActivationInput::Output`] record is the output, and the function borrows it. An
/// [`ActivationInput::PreActivation`] record is `z`, and the function runs the forward pass
/// again.
///
/// # Parameters
///
/// - `activation` - Activation of the cell
/// - `cache` - The cache that [`record_cache`] built from `record`
/// - `record` - The tensor that [`Activated::into_record`] gave
///
/// # Returns
///
/// - `Result<Cow<'a, Array2<f32>>, Error>` - The activated output
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If the activation refuses the cache
pub(super) fn recorded_output<'a>(
    activation: &Activation,
    cache: &ActivationCache,
    record: &'a Array2<f32>,
) -> Result<Cow<'a, Array2<f32>>, Error> {
    match activation.saves() {
        ActivationInput::Output => Ok(Cow::Borrowed(record)),
        ActivationInput::PreActivation => Ok(Cow::Owned(into_2d(activation.output_of(cache)?))),
    }
}

/// Runs the backward pass of the activation of a cell
///
/// # Parameters
///
/// - `activation` - Activation of the cell
/// - `cache` - The cache that [`record_cache`] built
/// - `grad_output` - The gradient of the activated output
///
/// # Returns
///
/// - `Result<Array2<f32>, Error>` - The gradient of the pre-activation
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If the activation refuses the cache
/// - [`Error::ShapeMismatch`] - If `grad_output` and the cache differ in shape
pub(super) fn activation_backward(
    activation: &Activation,
    cache: &ActivationCache,
    grad_output: Array2<f32>,
) -> Result<Array2<f32>, Error> {
    Ok(into_2d(
        activation.backward(cache, &grad_output.into_dyn())?,
    ))
}

/// Converts a tensor that an activation gave back to 2 dimensions
///
/// An activation keeps the shape of its input, and every cell gives it a 2D input
pub(super) fn into_2d(tensor: crate::neural_network::Tensor) -> Array2<f32> {
    tensor
        .into_dimensionality::<Ix2>()
        .expect("an activation keeps the 2D shape of its input")
}
