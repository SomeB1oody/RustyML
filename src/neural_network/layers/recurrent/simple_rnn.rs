//! SimpleRNN layer: a basic recurrent layer that returns the last hidden state, or every
//! timestep's hidden state

use crate::error::Error;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::recurrent::cell::{RecurrentGroup, RnnCell};
use crate::neural_network::layers::recurrent::gate::FusedGates;
use crate::neural_network::layers::recurrent::rnn::{Rnn, recurrent_layer_traits};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Activation as FusedActivation, Bias, Parallelism};
use ndarray::{Array2, ArrayView2, Ix2};

/// The arithmetic of 1 timestep of a [`SimpleRNN`]
///
/// The cell holds 1 gate. A timestep needs 1 matrix product and at most 1 activation sweep. The
/// timestep buffer starts as the projected `x_t @ kernel` slice, and the recurrent product and
/// the bias both accumulate into it. `ReLU` folds into the same fused epilogue, and `Linear`
/// needs no further pass. Every other activation runs a separate pass through
/// [`Activation::forward`].
///
/// A fused `f32` epilogue matches the unfused product plus the scalar activation bit for bit,
/// with 1 exception. The fused `Relu` maps `NaN` to `0`, while the scalar closure of this crate
/// propagates `NaN` instead.
///
/// The cell parks nothing of its own. The hidden state that leaves a step is the only value that
/// its backward pass reads, and [`Rnn`] hands that state to every cell.
#[derive(Debug)]
pub(crate) struct SimpleRnnCell {
    /// Activation applied at each timestep of the recurrence
    activation: Activation,
}

impl RnnCell for SimpleRnnCell {
    const CELL_TYPE: &'static str = "SimpleRNN";
    const GATE_BIASES: &'static [f32] = &[0.0];
    const STATE_COUNT: usize = 1;
    const RECORD_SLOTS: usize = 0;
    const RECURRENT_GROUPS: &'static [RecurrentGroup] = &[RecurrentGroup {
        first: 0,
        count: 1,
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
        _record: Option<&mut Vec<Array2<f32>>>,
    ) -> Result<(), Error> {
        let bias = gates.bias.as_slice().expect("bias must be contiguous");
        // z = x_t @ W + h_{t-1} @ U + b, with `x_t @ W` prefilled as the accumulator
        let mut z = xw_t.to_owned();
        let fused_act = match self.activation {
            Activation::ReLU => Some(FusedActivation::Relu),
            _ => None,
        };
        gemmkit_ndarray::gemm_fused(
            1.0,
            &state[0],
            &gates.recurrent_kernel,
            1.0,
            &mut z,
            Some(Bias::PerCol(bias)),
            fused_act,
            Parallelism::Rayon(0),
        );
        state[0] = match self.activation {
            Activation::Linear | Activation::ReLU => z,
            _ => self
                .activation
                .forward(&z.into_dyn())?
                .into_dimensionality::<Ix2>()
                .unwrap(),
        };
        Ok(())
    }

    fn step_backward(
        &self,
        gates: &FusedGates,
        _state_prev: &[Array2<f32>],
        state_next: &[Array2<f32>],
        _record: &[Array2<f32>],
        grad_state: &mut [Array2<f32>],
    ) -> Result<Array2<f32>, Error> {
        // The activation backward reads the state that leaves the step, which is what the
        // activation produced. It also reads the total gradient of that same state.
        let d_z = {
            let h_t = state_next[0].clone().into_dyn();
            let grad_h = grad_state[0].clone().into_dyn();
            self.activation
                .backward(&h_t, &grad_h)?
                .into_dimensionality::<Ix2>()
                .unwrap()
        };
        grad_state[0] = dot(&d_z, &gates.recurrent_kernel.t());
        Ok(d_z)
    }
}

/// Simple Recurrent Neural Network (SimpleRNN) layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). It applies an activation from the
/// activation module at each timestep.
///
/// [`SimpleRNN::with_return_sequences`] makes the layer return every timestep's hidden state,
/// with shape (batch_size, timesteps, units). [`SimpleRNN::with_go_backwards`] processes the
/// input timesteps from last to first.
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
/// // Create input with batch_size=2, timesteps=5, input_dim=4,
/// // and target with batch_size=2, units=3 (same dimension as the last hidden state)
/// let x = Array::ones((2, 5, 4)).into_dyn();
/// let y = Array::ones((2, 3)).into_dyn();
///
/// // Build model: 1 SimpleRNN layer with Tanh activation
/// let mut model = SequentialBuilder::new()
///     .add(SimpleRNN::new(3, Activation::Tanh).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Print structure
/// model.summary();
///
/// // Train for 1 epoch
/// model.fit(&x, &y, 1).unwrap();
///
/// // Predict
/// let pred = model.predict(&x);
/// println!("SimpleRnn prediction:\n{:#?}\n", pred);
/// ```
pub struct SimpleRNN(Rnn<SimpleRnnCell>);

impl SimpleRNN {
    /// Creates a SimpleRNN layer with the specified unit count and activation
    ///
    /// # Parameters
    ///
    /// - `units` - Number of output units in the layer
    /// - `activation` - Activation from the activation module (any [`Activation`] variant, or
    ///   any standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SimpleRNN layer instance
    ///
    /// # Notes
    ///
    /// The constructor draws nothing. [`UnaryLayer::build`] reads the feature count from the
    /// input shape and draws the weights then. The draw takes the global seed or entropy by
    /// default. For reproducible initialization, set a seed with [`SimpleRNN::with_random_state`].
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `units` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(units: usize, activation: impl Into<Activation>) -> Result<Self, Error> {
        Ok(Self(Rnn::new(units, activation)?))
    }

    /// Sets the seed used to initialize the weights and re-initializes them deterministically
    ///
    /// By default the draw takes the global seed or entropy (see [`crate::random`]). An unbuilt
    /// layer holds no weight, so this records the seed and draws nothing. A layer that is already
    /// built redraws its kernel (Xavier/Glorot) and recurrent kernel (orthogonal) from the new
    /// seed, so the order of the 2 calls does not matter. The bias stays zero-initialized.
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

    /// Sets the weights for this layer
    ///
    /// Each matrix takes the standard layout, because an optimizer reads every array of a layer
    /// as a flat slice.
    ///
    /// # Parameters
    ///
    /// - `kernel` - Weight matrix connecting inputs to the layer with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape
    ///   (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when every matrix matches the layer's existing shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer is not built
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any supplied matrix does not match the
    ///   layer's existing shape
    pub fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        self.0.set_weights(kernel, recurrent_kernel, bias)
    }
}

recurrent_layer_traits!(SimpleRNN);
