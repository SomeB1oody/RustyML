//! Golden fixtures for the sequence family.
//!
//! Covers the 3 recurrent layers (SimpleRNN, LSTM, GRU), the Embedding lookup table, and the 4
//! per-sample normalization layers (LayerNormalization, InstanceNormalization,
//! GroupNormalization, UnitNormalization). The data file is `golden/data/sequence.golden`.
//!
//! See the module doc comment of the parent module for the contract, and see `misc.rs` for a
//! worked family.
//!
//! # What the configurations exercise
//!
//! Every recurrent case runs 4 timesteps over a batch of 2. A change to the time loop, to the
//! order of the hidden-state update, or to the backpropagation-through-time sweep therefore
//! moves a recorded value. The 3 layers keep `input_dim` different from `units`, so a swap of
//! the input kernel and the recurrent kernel cannot pass a shape check.
//!
//! The LSTM and the GRU pack every gate into 1 fused matrix. The column blocks follow a fixed
//! order, `[i | f | g | o]` for the LSTM and `[z | r | h]` for the GRU. Each layer records the
//! gradient of the whole fused matrix, never a per-gate slice, because the fused matrix is what
//! `LayerBase::parameters_mut` hands to an optimizer. Each layer also records 1 case built through its
//! per-gate `set_gate_weights` wrapper. That case pins the map from the argument order to the
//! column-block order. The map is not the identity for the GRU, whose arguments stay in reset,
//! update, candidate order while the columns pack update first.
//!
//! Each of the 3 layers also records all 4 combinations of `return_sequences` and
//! `go_backwards`. `return_sequences` changes the output rank, so it changes the shape of the
//! gradient that the backward pass reads. `go_backwards` keeps every shape and reverses the
//! order that the time loop reads the input in. The 4 timesteps make both flags visible: a
//! reversed loop over 4 steps gives a different last state, and a returned sequence over 4
//! steps holds the states in processing order, not in input order. See [`SEQUENCE_FLAGS`].
//!
//! The normalization layers cover more than 1 axis configuration each. LayerNormalization has 3
//! internal layout paths, and the cases reach all 3. InstanceNormalization and
//! GroupNormalization share 1 core, so the group count is the axis knob, and the cases run it
//! from 1 group to 1 group per channel. UnitNormalization takes a fused row path when the
//! normalized axes form the trailing block of the shape, and a strided path otherwise. The
//! cases reach both.
//!
//! LayerNormalization, InstanceNormalization, and GroupNormalization pass the gradient through
//! unchanged in inference mode, and produce no parameter gradient there. Each of the 3 records 1
//! inference case that pins this.
//!
//! # Why the Embedding cases wrap the layer
//!
//! An Embedding layer reads whole-number row indices, and it rejects any value at or below
//! -1.0. The harness input formula starts at -2.0, so the harness input is not a usable index
//! table. The harness offers no hook for a per-case input, so the 2 Embedding cases wrap the
//! layer in [`IndexedEmbedding`]. The wrapper maps the harness input to an index with a pure
//! formula, and delegates every other method to the layer. See [`IndexedEmbedding::indices`] for
//! the formula.

use super::{GoldenCase, LayerFixture, golden_weights, golden_weights_from};
use ndarray::{Array2, ArrayD, Ix2};
use rustyml::error::Error;
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::embedding::Embedding;
use rustyml::neural_network::layers::recurrent::{GRU, LSTM, SimpleRNN};
use rustyml::neural_network::layers::regularization::normalization::{
    GroupNormalization, InstanceNormalization, LayerNormalization, LayerNormalizationAxis,
    UnitNormalization, UnitNormalizationAxis,
};
use rustyml::neural_network::traits::{
    Layer, LayerBase, ParamRef, UnaryLayer, WeightMut, WeightRef,
};
use rustyml::neural_network::{Ctx, Shape, Tensor};

/// Every layer type of the sequence family, in the order the data file records them.
fn fixtures() -> Vec<LayerFixture> {
    vec![
        LayerFixture::new("SimpleRNN", simple_rnn_cases),
        LayerFixture::new("LSTM", lstm_cases),
        LayerFixture::new("GRU", gru_cases),
        LayerFixture::new("Embedding", embedding_cases),
        LayerFixture::new("LayerNormalization", layer_normalization_cases),
        LayerFixture::new("InstanceNormalization", instance_normalization_cases),
        LayerFixture::new("GroupNormalization", group_normalization_cases),
        LayerFixture::new("UnitNormalization", unit_normalization_cases),
    ]
}

/// Replays the sequence family against `golden/data/sequence.golden`.
#[test]
fn golden_sequence_family() {
    super::run_family("sequence", &fixtures());
}

// ---------------------------------------------------------------------------------------
// Shared shape constants and helpers
// ---------------------------------------------------------------------------------------

/// Input features per timestep for every recurrent case.
const RNN_INPUT_DIM: usize = 4;

/// Hidden units for every recurrent case. It differs from [`RNN_INPUT_DIM`] on purpose, so a
/// swap of the input kernel and the recurrent kernel fails its shape check.
const RNN_UNITS: usize = 3;

/// Input shape of every recurrent case: batch 2, 4 timesteps, and [`RNN_INPUT_DIM`] features.
const RNN_SHAPE: [usize; 3] = [2, 4, RNN_INPUT_DIM];

/// The epsilon that every normalization case uses.
const EPSILON: f32 = 1e-5;

/// Turns a rank-2 weight tensor into the `Array2` that a recurrent or embedding setter takes.
fn as_2d(weights: ArrayD<f32>) -> Array2<f32> {
    weights
        .into_dimensionality::<Ix2>()
        .expect("the weight tensor is rank 2")
}

/// Builds the `(kernel, recurrent_kernel, bias)` triple of 1 gate from the weight formula.
///
/// The 3 tensors of 1 gate follow each other in the formula, and the next gate starts after
/// them. No 2 gates therefore hold the same numbers, so a swap of 2 gate blocks moves a
/// recorded value.
///
/// # Parameters
///
/// - `first` - Flat index that this gate's kernel starts from
///
/// # Returns
///
/// - `(Array2<f32>, Array2<f32>, Array2<f32>)` - The kernel, the recurrent kernel, and the bias
fn gate_block(first: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let kernel_size = RNN_INPUT_DIM * RNN_UNITS;
    let recurrent_size = RNN_UNITS * RNN_UNITS;
    (
        as_2d(golden_weights_from(&[RNN_INPUT_DIM, RNN_UNITS], first)),
        as_2d(golden_weights_from(
            &[RNN_UNITS, RNN_UNITS],
            first + kernel_size,
        )),
        as_2d(golden_weights_from(
            &[1, RNN_UNITS],
            first + kernel_size + recurrent_size,
        )),
    )
}

/// Flat-index stride between 2 gate blocks, which is the size of 1 gate block.
const GATE_STRIDE: usize = RNN_INPUT_DIM * RNN_UNITS + RNN_UNITS * RNN_UNITS + RNN_UNITS;

/// The gamma and beta pair for a normalization layer with `size` parameters per tensor.
///
/// Beta continues the weight formula after gamma, so the 2 tensors never hold the same numbers.
fn scale_and_shift(size: usize) -> (Tensor, Tensor) {
    (golden_weights(&[size]), golden_weights_from(&[size], size))
}

/// The parameter gradient names that every recurrent layer reports.
const RECURRENT_PARAMS: [&str; 3] = ["kernel", "recurrent_kernel", "bias"];

/// The 4 combinations of `return_sequences` and `go_backwards`, with the label of each one.
///
/// The 2 flags are independent, and each 1 of the 4 combinations takes a different path through
/// the time loop and through the backpropagation-through-time sweep. All 3 recurrent layers
/// record all 4.
const SEQUENCE_FLAGS: [(&str, bool, bool); 4] = [
    ("seq_false_backwards_false", false, false),
    ("seq_true_backwards_false", true, false),
    ("seq_false_backwards_true", false, true),
    ("seq_true_backwards_true", true, true),
];

/// Builds the 4 [`SEQUENCE_FLAGS`] cases of 1 recurrent layer type.
///
/// Every case runs over [`RNN_SHAPE`], which holds 4 timesteps. A change to the order of the
/// time loop therefore moves a recorded value.
///
/// # Parameters
///
/// - `build` - Builds the layer with fixed weights, `return_sequences`, and `go_backwards`
///
/// # Returns
///
/// - `Vec<GoldenCase>` - 1 case per combination, in [`SEQUENCE_FLAGS`] order
fn sequence_flag_cases(build: fn(bool, bool) -> Box<dyn Layer>) -> Vec<GoldenCase> {
    SEQUENCE_FLAGS
        .iter()
        .map(|&(label, return_sequences, go_backwards)| {
            GoldenCase::new(label, &RNN_SHAPE, move || {
                build(return_sequences, go_backwards)
            })
            .with_parameter_grads(&RECURRENT_PARAMS)
        })
        .collect()
}

/// The parameter gradient names that every normalization layer with weights reports.
const NORMALIZATION_PARAMS: [&str; 2] = ["gamma", "beta"];

// ---------------------------------------------------------------------------------------
// The recurrent layers
// ---------------------------------------------------------------------------------------

/// SimpleRNN cases: 4 features into 3 units, over the 3 forward paths of the timestep loop and
/// the 4 combinations of the 2 sequence flags.
///
/// The activation selects the path. `Tanh` runs a separate activation pass over the fused
/// product, `ReLU` runs the fused epilogue activation of the backend, and `Linear` runs the
/// fused product with no epilogue at all.
///
/// The 4 flag cases keep `Tanh` and move the 2 sequence flags. See [`sequence_flag_cases`].
fn simple_rnn_cases() -> Vec<GoldenCase> {
    /// Builds a SimpleRNN layer with fixed weights, the given activation, and the 2 flags.
    fn build(activation: Activation, return_sequences: bool, go_backwards: bool) -> Box<dyn Layer> {
        let mut layer = SimpleRNN::new(RNN_UNITS, activation)
            .expect("3 units")
            .with_return_sequences(return_sequences)
            .with_go_backwards(go_backwards);
        layer
            .build(&Shape::known(&RNN_SHAPE))
            .expect("the layer accepts the shape of the case");
        let (kernel, recurrent_kernel, bias) = gate_block(0);
        layer
            .set_weights(kernel, recurrent_kernel, bias)
            .expect("every shape matches the layer");
        Box::new(layer)
    }

    /// Builds the Tanh layer for 1 combination of the 2 sequence flags.
    fn build_flags(return_sequences: bool, go_backwards: bool) -> Box<dyn Layer> {
        build(Activation::Tanh, return_sequences, go_backwards)
    }

    let mut cases = vec![
        GoldenCase::new("units_3_tanh", &RNN_SHAPE, || {
            build(Activation::Tanh, false, false)
        })
        .with_parameter_grads(&RECURRENT_PARAMS),
        GoldenCase::new("units_3_relu", &RNN_SHAPE, || {
            build(Activation::ReLU, false, false)
        })
        .with_parameter_grads(&RECURRENT_PARAMS),
        GoldenCase::new("units_3_linear", &RNN_SHAPE, || {
            build(Activation::Linear, false, false)
        })
        .with_parameter_grads(&RECURRENT_PARAMS),
    ];
    cases.extend(sequence_flag_cases(build_flags));
    cases
}

/// LSTM cases: 1 fused-weight case, 1 per-gate case, and the 4 sequence-flag cases.
///
/// The fused case sets the whole `[i | f | g | o]` matrix at 1 time. The per-gate case builds
/// the same layer through `set_gate_weights`, which packs 4 separate triples into that order.
/// The 2 cases together pin both the column-block order and the argument order that feeds it.
///
/// The 4 flag cases take the fused builder and move the 2 sequence flags. See
/// [`sequence_flag_cases`].
fn lstm_cases() -> Vec<GoldenCase> {
    /// Number of gates the LSTM packs side by side.
    const GATES: usize = 4;

    /// Builds an LSTM layer whose fused matrices come straight from the weight formula.
    fn build_fused(return_sequences: bool, go_backwards: bool) -> Box<dyn Layer> {
        let mut layer = LSTM::new(RNN_UNITS, Activation::Tanh)
            .expect("3 units")
            .with_return_sequences(return_sequences)
            .with_go_backwards(go_backwards);
        layer
            .build(&Shape::known(&RNN_SHAPE))
            .expect("the layer accepts the shape of the case");
        let width = GATES * RNN_UNITS;
        let kernel_size = RNN_INPUT_DIM * width;
        let recurrent_size = RNN_UNITS * width;
        layer
            .set_weights(
                as_2d(golden_weights(&[RNN_INPUT_DIM, width])),
                as_2d(golden_weights_from(&[RNN_UNITS, width], kernel_size)),
                as_2d(golden_weights_from(
                    &[1, width],
                    kernel_size + recurrent_size,
                )),
            )
            .expect("every fused shape matches the layer");
        Box::new(layer)
    }

    /// Builds an LSTM layer gate by gate, in input, forget, cell, output argument order.
    fn build_per_gate() -> Box<dyn Layer> {
        let mut layer = LSTM::new(RNN_UNITS, Activation::Tanh).expect("3 units");
        layer
            .build(&Shape::known(&RNN_SHAPE))
            .expect("the layer accepts the shape of the case");
        let (input_kernel, input_recurrent, input_bias) = gate_block(0);
        let (forget_kernel, forget_recurrent, forget_bias) = gate_block(GATE_STRIDE);
        let (cell_kernel, cell_recurrent, cell_bias) = gate_block(2 * GATE_STRIDE);
        let (output_kernel, output_recurrent, output_bias) = gate_block(3 * GATE_STRIDE);
        layer
            .set_gate_weights(
                input_kernel,
                input_recurrent,
                input_bias,
                forget_kernel,
                forget_recurrent,
                forget_bias,
                cell_kernel,
                cell_recurrent,
                cell_bias,
                output_kernel,
                output_recurrent,
                output_bias,
            )
            .expect("every per-gate shape matches the layer");
        Box::new(layer)
    }

    let mut cases = vec![
        GoldenCase::new("units_3_fused", &RNN_SHAPE, || build_fused(false, false))
            .with_parameter_grads(&RECURRENT_PARAMS),
        GoldenCase::new("units_3_per_gate", &RNN_SHAPE, build_per_gate)
            .with_parameter_grads(&RECURRENT_PARAMS),
    ];
    cases.extend(sequence_flag_cases(build_fused));
    cases
}

/// GRU cases: 1 fused-weight case, 1 per-gate case, and the 4 sequence-flag cases.
///
/// The per-gate case matters more here than for the LSTM. The arguments of
/// `set_gate_weights` stay in reset, update, candidate order, and the fused columns pack
/// update first. The case pins that map, which a careless reordering would break silently.
///
/// The 4 flag cases take the fused builder and move the 2 sequence flags. See
/// [`sequence_flag_cases`].
fn gru_cases() -> Vec<GoldenCase> {
    /// Number of gates the GRU packs side by side.
    const GATES: usize = 3;

    /// Builds a GRU layer whose fused matrices come straight from the weight formula.
    fn build_fused(return_sequences: bool, go_backwards: bool) -> Box<dyn Layer> {
        let mut layer = GRU::new(RNN_UNITS, Activation::Tanh)
            .expect("3 units")
            .with_return_sequences(return_sequences)
            .with_go_backwards(go_backwards);
        layer
            .build(&Shape::known(&RNN_SHAPE))
            .expect("the layer accepts the shape of the case");
        let width = GATES * RNN_UNITS;
        let kernel_size = RNN_INPUT_DIM * width;
        let recurrent_size = RNN_UNITS * width;
        layer
            .set_weights(
                as_2d(golden_weights(&[RNN_INPUT_DIM, width])),
                as_2d(golden_weights_from(&[RNN_UNITS, width], kernel_size)),
                as_2d(golden_weights_from(
                    &[1, width],
                    kernel_size + recurrent_size,
                )),
            )
            .expect("every fused shape matches the layer");
        Box::new(layer)
    }

    /// Builds a GRU layer gate by gate, in reset, update, candidate argument order.
    fn build_per_gate() -> Box<dyn Layer> {
        let mut layer = GRU::new(RNN_UNITS, Activation::Tanh).expect("3 units");
        layer
            .build(&Shape::known(&RNN_SHAPE))
            .expect("the layer accepts the shape of the case");
        let (reset_kernel, reset_recurrent, reset_bias) = gate_block(0);
        let (update_kernel, update_recurrent, update_bias) = gate_block(GATE_STRIDE);
        let (candidate_kernel, candidate_recurrent, candidate_bias) = gate_block(2 * GATE_STRIDE);
        layer
            .set_gate_weights(
                reset_kernel,
                reset_recurrent,
                reset_bias,
                update_kernel,
                update_recurrent,
                update_bias,
                candidate_kernel,
                candidate_recurrent,
                candidate_bias,
            )
            .expect("every per-gate shape matches the layer");
        Box::new(layer)
    }

    let mut cases = vec![
        GoldenCase::new("units_3_fused", &RNN_SHAPE, || build_fused(false, false))
            .with_parameter_grads(&RECURRENT_PARAMS),
        GoldenCase::new("units_3_per_gate", &RNN_SHAPE, build_per_gate)
            .with_parameter_grads(&RECURRENT_PARAMS),
    ];
    cases.extend(sequence_flag_cases(build_fused));
    cases
}

// ---------------------------------------------------------------------------------------
// The embedding layer
// ---------------------------------------------------------------------------------------

/// Rows in the fixture lookup table, which is the vocabulary size.
const EMBEDDING_ROWS: usize = 6;

/// Width of 1 fixture embedding vector.
const EMBEDDING_WIDTH: usize = 3;

/// Input shape of every Embedding case: batch 2 and 4 indices per sample.
const EMBEDDING_SHAPE: [usize; 2] = [2, 4];

/// An [`Embedding`] layer behind a pure map from the harness input to a row index.
///
/// The layer reads whole-number indices and rejects any value at or below -1.0. The harness
/// input formula starts at -2.0, and the harness offers no hook for a per-case input, so the
/// fixture maps the input itself. Every other method delegates to the layer, and `layer_type`
/// therefore still reports `"Embedding"`.
struct IndexedEmbedding {
    /// The layer under record
    inner: Embedding,
    /// Fractional part added to each whole index. See [`IndexedEmbedding::indices`]
    fraction: f32,
}

impl IndexedEmbedding {
    /// Builds the wrapper around a table that comes from the weight formula.
    ///
    /// # Parameters
    ///
    /// - `fraction` - Fractional part that the index map adds. Use 0.0 for whole indices
    ///
    /// # Returns
    ///
    /// - `Self` - A wrapper whose table holds `EMBEDDING_ROWS` rows of `EMBEDDING_WIDTH` values
    fn new(fraction: f32) -> Self {
        let mut inner =
            Embedding::new(EMBEDDING_ROWS, EMBEDDING_WIDTH).expect("6 rows of 3 values each");
        inner
            .build(&Shape::known(&EMBEDDING_SHAPE))
            .expect("the layer accepts the shape of the case");
        inner
            .set_weights(as_2d(golden_weights(&[EMBEDDING_ROWS, EMBEDDING_WIDTH])))
            .expect("the table shape matches the layer");
        Self { inner, fraction }
    }

    /// Maps the harness input to the index tensor that the layer reads.
    ///
    /// The harness input value at flat index `i` is `((i * 37) % 101 - 50) / 25`, so
    /// `round((value + 2) * 25)` recovers the whole number `(i * 37) % 101` exactly. The row is
    /// that number modulo [`EMBEDDING_ROWS`]. Over the 8 elements of a case the rows are
    /// `0, 1, 2, 4, 5, 0, 2, 3`. Row 0 and row 2 each appear 2 times, so the backward pass
    /// accumulates 2 gradients into 1 row, and every row of the table is reached.
    ///
    /// `fraction` then selects how the index is written. A `fraction` of 0.0 writes the whole
    /// row number. A positive `fraction` writes `row + fraction` for a row above 0, and
    /// `-fraction` for row 0. The layer truncates toward 0, so both forms select the same rows,
    /// and the 2 cases must record the same output. The negative value also pins the rule that
    /// a value above -1.0 folds into row 0.
    ///
    /// # Parameters
    ///
    /// - `input` - The tensor that the harness generated for the case
    ///
    /// # Returns
    ///
    /// - `Tensor` - Index values of the same shape as `input`
    fn indices(&self, input: &Tensor) -> Tensor {
        input.mapv(|value| {
            let whole = ((value + 2.0) * 25.0).round() as usize;
            let row = whole % EMBEDDING_ROWS;
            if row == 0 && self.fraction > 0.0 {
                -self.fraction
            } else {
                row as f32 + self.fraction
            }
        })
    }
}

impl LayerBase for IndexedEmbedding {
    fn layer_type(&self) -> &str {
        self.inner.layer_type()
    }

    fn param_count(&self) -> ParamCounts {
        self.inner.param_count()
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        self.inner.parameters_mut()
    }

    fn weights(&self) -> Vec<WeightRef<'_>> {
        self.inner.weights()
    }

    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        self.inner.weights_mut()
    }

    /// Gives the shapes of the layer under record, so the wrapper reports the output shape of
    /// that layer
    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        self.inner.known_input_shapes()
    }

    fn is_built(&self) -> bool {
        self.inner.is_built()
    }
}

impl UnaryLayer for IndexedEmbedding {
    /// The index map keeps the shape of the tensor, so the layer under record builds for the
    /// shape that the wrapper received
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        self.inner.build(input)
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        self.inner.forward(&self.indices(input), ctx)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        self.inner.backward(grad_output, ctx)
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        self.inner.compute_output_shape(input)
    }
}

/// Embedding cases: 1 with whole indices, and 1 with indices that need truncation.
///
/// Both cases read the same 8 rows, `0, 1, 2, 4, 5, 0, 2, 3`. The rows repeat, so the backward
/// pass accumulates into 1 row more than 1 time, and row 0 is among them. The 2 cases must
/// record the same forward output, because the layer truncates an index toward 0.
fn embedding_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase::new("indices_whole", &EMBEDDING_SHAPE, || {
            Box::new(IndexedEmbedding::new(0.0))
        })
        .with_parameter_grads(&["embeddings"]),
        GoldenCase::new("indices_truncated", &EMBEDDING_SHAPE, || {
            Box::new(IndexedEmbedding::new(0.9))
        })
        .with_parameter_grads(&["embeddings"]),
    ]
}

// ---------------------------------------------------------------------------------------
// The normalization layers
// ---------------------------------------------------------------------------------------

/// LayerNormalization cases: 5 axis configurations and 1 inference case.
///
/// The axis choice selects 1 of the 3 internal layout paths. The default axis and a trailing
/// `Multiple` list take the fused row path with no layout transform. A `Multiple` list that
/// needs a real permutation takes the merged-row path, which transposes in and back out. A
/// `Custom` axis that is not last takes the strided path.
///
/// The inference case pins 2 things. The backward pass returns the gradient unchanged, and the
/// layer reports no parameter gradient at all.
fn layer_normalization_cases() -> Vec<GoldenCase> {
    /// Builds a LayerNormalization layer with fixed weights over `size` normalized elements.
    fn build(
        input_shape: Vec<usize>,
        axis: Option<LayerNormalizationAxis>,
        size: usize,
    ) -> Box<dyn Layer> {
        let base = LayerNormalization::new(EPSILON).expect("a positive epsilon");
        let mut layer = match axis {
            Some(axis) => base.with_normalized_axis(axis).expect("a usable axis list"),
            None => base,
        };
        layer
            .build(&Shape::known(&input_shape))
            .expect("the layer accepts the shape of the case");
        let (gamma, beta) = scale_and_shift(size);
        layer
            .set_weights(gamma, beta)
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        // The last axis of a rank-2 input: the fused row path over 8 elements per row
        GoldenCase::new("axis_default_rank_2", &[2, 8], || {
            build(vec![2, 8], None, 8)
        })
        .with_parameter_grads(&NORMALIZATION_PARAMS),
        // The last axis of a rank-3 input: the same row path over 4 elements per row
        GoldenCase::new("axis_default_rank_3", &[2, 3, 4], || {
            build(vec![2, 3, 4], None, 4)
        })
        .with_parameter_grads(&NORMALIZATION_PARAMS),
        // A middle axis: the strided path, which reduces the lanes in place
        GoldenCase::new("axis_custom_1", &[2, 3, 4], || {
            build(vec![2, 3, 4], Some(LayerNormalizationAxis::Custom(1)), 3)
        })
        .with_parameter_grads(&NORMALIZATION_PARAMS),
        // Trailing axes: the merge is pure reshape, so the row path runs with no transpose
        GoldenCase::new("axes_multiple_2_3", &[2, 2, 3, 2], || {
            build(
                vec![2, 2, 3, 2],
                Some(LayerNormalizationAxis::Multiple(vec![2, 3])),
                6,
            )
        })
        .with_parameter_grads(&NORMALIZATION_PARAMS),
        // Non-trailing axes: the merged-row path, with a transpose in and a transpose back out
        GoldenCase::new("axes_multiple_1_3", &[2, 2, 3, 2], || {
            build(
                vec![2, 2, 3, 2],
                Some(LayerNormalizationAxis::Multiple(vec![1, 3])),
                4,
            )
        })
        .with_parameter_grads(&NORMALIZATION_PARAMS),
        // Inference mode: the backward pass returns the gradient unchanged and stores no
        // parameter gradient, so the case declares none
        GoldenCase::new("inference_rank_2", &[2, 8], || build(vec![2, 8], None, 8))
            .in_inference_mode(),
    ]
}

/// InstanceNormalization cases: a rank-3 input, a rank-4 input, and 1 inference case.
///
/// The layer is group normalization with 1 group per channel, so the channel count is the whole
/// configuration. The rank-4 case folds 6 spatial positions into each channel statistic instead
/// of 4, which moves every recorded value.
fn instance_normalization_cases() -> Vec<GoldenCase> {
    /// Builds an InstanceNormalization layer with fixed weights over `channels` channels.
    fn build(input_shape: Vec<usize>, channels: usize) -> Box<dyn Layer> {
        let mut layer = InstanceNormalization::new(EPSILON).expect("a positive epsilon");
        layer
            .build(&Shape::known(&input_shape))
            .expect("the layer accepts the shape of the case");
        let (gamma, beta) = scale_and_shift(channels);
        layer
            .set_weights(gamma, beta)
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("rank_3", &[2, 4, 3], || build(vec![2, 4, 3], 3))
            .with_parameter_grads(&NORMALIZATION_PARAMS),
        GoldenCase::new("rank_4", &[2, 2, 3, 4], || build(vec![2, 2, 3, 4], 4))
            .with_parameter_grads(&NORMALIZATION_PARAMS),
        GoldenCase::new("inference_rank_3", &[2, 4, 3], || build(vec![2, 4, 3], 3))
            .in_inference_mode(),
    ]
}

/// GroupNormalization cases: 3 group counts over 6 channels, 1 rank-4 case, and 1 inference case.
///
/// The group count is the axis knob of this layer. 1 group normalizes the whole channel vector
/// of a sample together. 6 groups over 6 channels is instance normalization. 2 groups sits
/// between them and is the only count whose groups hold more than 1 channel and are more than
/// 1 group.
fn group_normalization_cases() -> Vec<GoldenCase> {
    /// Builds a GroupNormalization layer with fixed weights over `channels` channels.
    fn build(input_shape: Vec<usize>, groups: usize, channels: usize) -> Box<dyn Layer> {
        let mut layer =
            GroupNormalization::new(groups, EPSILON).expect("a positive group count and epsilon");
        layer
            .build(&Shape::known(&input_shape))
            .expect("the layer accepts the shape of the case");
        let (gamma, beta) = scale_and_shift(channels);
        layer
            .set_weights(gamma, beta)
            .expect("both shapes match the layer");
        Box::new(layer)
    }

    vec![
        GoldenCase::new("groups_1", &[2, 4, 6], || build(vec![2, 4, 6], 1, 6))
            .with_parameter_grads(&NORMALIZATION_PARAMS),
        GoldenCase::new("groups_2", &[2, 4, 6], || build(vec![2, 4, 6], 2, 6))
            .with_parameter_grads(&NORMALIZATION_PARAMS),
        GoldenCase::new("groups_6", &[2, 4, 6], || build(vec![2, 4, 6], 6, 6))
            .with_parameter_grads(&NORMALIZATION_PARAMS),
        GoldenCase::new("rank_4_groups_2", &[2, 2, 3, 4], || {
            build(vec![2, 2, 3, 4], 2, 4)
        })
        .with_parameter_grads(&NORMALIZATION_PARAMS),
        GoldenCase::new("inference_groups_2", &[2, 4, 6], || {
            build(vec![2, 4, 6], 2, 6)
        })
        .in_inference_mode(),
    ]
}

/// UnitNormalization cases: 3 configurations on the fused row path and 2 on the strided path.
///
/// The layer takes the row path when the normalized axes form the trailing block of the shape,
/// which the default axis always does. Any other choice leaves the groups as strided lanes and
/// takes the second path. The layer holds no weight and does not depend on the mode, so every
/// case declares no parameter gradient and stays in the default mode.
fn unit_normalization_cases() -> Vec<GoldenCase> {
    /// Builds a UnitNormalization layer for 1 axis configuration.
    fn build(axis: UnitNormalizationAxis) -> Box<dyn Layer> {
        Box::new(UnitNormalization::new(axis).expect("a usable axis list"))
    }

    vec![
        // The last axis of a rank-2 input: 1 contiguous group of 6 per row
        GoldenCase::new("axis_default_rank_2", &[2, 6], || {
            build(UnitNormalizationAxis::Default)
        }),
        // The last axis of a rank-3 input: 6 contiguous groups of 4
        GoldenCase::new("axis_default_rank_3", &[2, 3, 4], || {
            build(UnitNormalizationAxis::Default)
        }),
        // A middle axis: the groups are strided lanes, so the strided path runs
        GoldenCase::new("axis_custom_1", &[2, 3, 4], || {
            build(UnitNormalizationAxis::Custom(1))
        }),
        // Trailing axes: the group is the whole 12-element tail, so the row path runs
        GoldenCase::new("axes_multiple_1_2", &[2, 3, 4], || {
            build(UnitNormalizationAxis::Multiple(vec![1, 2]))
        }),
        // Non-trailing axes: the strided path again, over 2 axes instead of 1
        GoldenCase::new("axes_multiple_1_3", &[2, 2, 3, 2], || {
            build(UnitNormalizationAxis::Multiple(vec![1, 3]))
        }),
    ]
}
