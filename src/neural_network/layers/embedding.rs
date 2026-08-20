//! Embedding layer: a trainable lookup table that turns whole-number indices into dense vectors

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::{EmbeddingLayerWeight, LayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::parallel_gates::cheap_map_parallel_threshold;
use ndarray::{Array, Array2, IxDyn};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rayon::prelude::*;
use std::borrow::Cow;

/// Half-width of the uniform range that initializes the lookup table
///
/// The table starts from `Uniform(-0.05, 0.05)`, which is this layer's `"uniform"` initializer.
/// A dense layer starts from Xavier/Glorot instead, because its fan-in is the
/// whole input vector. An embedding reads exactly 1 row per index, so its fan-in is 1, and a
/// fan-based rule has nothing to scale
const INIT_LIMIT: f32 = 0.05;

/// Target element count for 1 parallel gather task
///
/// The gather copies whole rows, so a task holds `max(1, TASK_ELEMENTS / output_dim)` rows. This
/// keeps each task large enough to cover the scheduling cost at any vector width
const TASK_ELEMENTS: usize = 16_384;

/// Trainable lookup table that maps whole-number indices to dense vectors
///
/// The layer holds an `(input_dim, output_dim)` table. The forward pass reads row `i` for every
/// index `i` in the input and stacks the rows into a new trailing axis. An input of shape
/// `[d0, d1, ..., dk]` gives an output of shape `[d0, d1, ..., dk, output_dim]`, so the output
/// rank is always 1 more than the input rank
///
/// This is the entry point for text and for any other categorical sequence. A word index enters,
/// and a learned vector leaves. The layer is the same function as a `Dense` layer without a
/// bias, applied to a one-hot input. It reads 1 row instead of multiplying by a mostly-zero
/// matrix
///
/// # Notes
///
/// A [`Tensor`] holds `f32`, so the indices arrive as floating-point values. The layer truncates
/// each value toward zero and then range-checks it. `2.0` and `2.9` both select row 2, and
/// `-0.5` selects row 0.
///
/// An index outside `0..input_dim` after truncation is an `Error::InvalidInput`, and so is a
/// non-finite value. An unchecked backend can turn an out-of-range index into `NaN`, or
/// silently wrap a negative index to the end of the table. Neither result is a useful contract,
/// so this layer reports the bad index instead
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::{Activation, Dense, Embedding, Flatten};
/// use rustyml::neural_network::optimizers::SGD;
/// use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
/// use rustyml::neural_network::traits::Layer;
///
/// // A batch of 2 sequences of 3 word indices each, drawn from a vocabulary of 10
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 4.0, 2.0, 7.0, 0.0, 4.0])
///     .unwrap()
///     .into_dyn();
/// let y = Array2::zeros((2, 1)).into_dyn();
///
/// // The layer alone turns each of the 6 indices into a 5-element vector
/// let mut lookup = Embedding::new(10, 5).unwrap();
/// let vectors = lookup.predict(&x).unwrap();
/// assert_eq!(vectors.shape(), &[2, 3, 5]);
///
/// // A flatten step then feeds the vectors to a dense head
/// let mut model = Sequential::new();
/// model
///     .add(Embedding::new(10, 5).unwrap())
///     .add(Flatten::new(vec![2, 3, 5]).unwrap())
///     .add(Dense::new(15, 1, Activation::Linear).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// model.summary();
/// model.fit(&x, &y, 2).unwrap();
/// assert_eq!(model.predict(&x).unwrap().shape(), &[2, 1]);
/// ```
///
/// # Performance
///
/// The forward gather runs in parallel once the output element count reaches the shared
/// cheap-map gate. Override that gate through
/// [`crate::tuning::elementwise::set_cheap_map_f32`]. A gather is a pure copy, so the gate never
/// changes a value
///
/// The backward pass builds a dense gradient over the whole table, even when a batch selects
/// few rows. The optimizer reads that whole gradient, so 1 training step costs
/// `input_dim * output_dim` regardless of the batch. Keep `input_dim` to the vocabulary that the
/// data really uses
///
/// A momentum-based or weight-decaying optimizer moves a row whose gradient is 0, because both
/// rules act on the parameter and not only on the gradient. This matches the update path of a
/// `Dense` layer
#[derive(Debug)]
pub struct Embedding {
    /// Number of rows in the table, which is the vocabulary size
    input_dim: usize,
    /// Width of 1 embedding vector
    output_dim: usize,
    /// Lookup table with shape (input_dim, output_dim)
    embeddings: Array2<f32>,
    /// Row indices the forward pass read, in output order. The backward pass scatters into them
    index_cache: Option<Vec<usize>>,
    /// Shape of the most recent forward input, used to check and to shape the gradient
    input_shape: Option<Vec<usize>>,
    /// Stored table gradients, kept allocated across steps and refilled with 0 on each backward
    grad_embeddings: Option<Array2<f32>>,
}

impl Embedding {
    /// Creates a new embedding layer with a randomly initialized lookup table
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Size of the vocabulary, which is the largest usable index plus 1
    /// - `output_dim` - Width of 1 embedding vector
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Embedding` layer instance with an initialized table
    ///
    /// # Notes
    ///
    /// The layer seeds the table from the global seed or from entropy by default. For a
    /// reproducible table, set a seed with [`Embedding::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `input_dim` or `output_dim` is 0
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self, Error> {
        if input_dim == 0 {
            return Err(Error::invalid_parameter(
                "input_dim",
                "is 0, and the table must hold at least 1 row",
            ));
        }
        if output_dim == 0 {
            return Err(Error::invalid_parameter(
                "output_dim",
                "is 0, and an embedding vector must hold at least 1 element",
            ));
        }

        Ok(Self {
            input_dim,
            output_dim,
            embeddings: Self::init_table(input_dim, output_dim, None),
            index_cache: None,
            input_shape: None,
            grad_embeddings: None,
        })
    }

    /// Sets the seed used to initialize the table and re-initializes it deterministically
    ///
    /// By default the layer seeds the table from the global seed or from entropy (see
    /// [`crate::random`]). This re-runs the uniform initialization with `random_state`, so call
    /// it before assigning custom weights or training
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for table initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.embeddings = Self::init_table(self.input_dim, self.output_dim, Some(random_state));
        self
    }

    /// Uniform table initialization over `[-INIT_LIMIT, INIT_LIMIT]` for the given seed
    fn init_table(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f32> {
        let mut rng = crate::random::make_rng(random_state);
        Array::random_using(
            (input_dim, output_dim),
            Uniform::new(-INIT_LIMIT, INIT_LIMIT).unwrap(),
            &mut rng,
        )
    }

    /// Sets the lookup table for this layer
    ///
    /// # Parameters
    ///
    /// - `embeddings` - Table with shape (input_dim, output_dim)
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - Ok when `embeddings` matches the layer's configured shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `embeddings` does not match the
    ///   layer's configured shape
    pub fn set_weights(&mut self, embeddings: Array2<f32>) -> Result<(), Error> {
        validate_weight_shape("embeddings", self.embeddings.shape(), embeddings.shape())?;

        self.embeddings = embeddings.as_standard_layout().into_owned();
        Ok(())
    }

    /// Turns the floating-point input into checked row indices, in output order
    ///
    /// Truncates each value toward zero, as a cast to a whole number does, and then checks it
    /// against the table height
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If the input holds no element
    /// - `Error::InvalidInput` - If the input has rank 0, or holds a non-finite value, or holds
    ///   an index outside `0..input_dim` after truncation
    fn to_indices(&self, input: &Tensor) -> Result<Vec<usize>, Error> {
        if input.ndim() == 0 {
            return Err(Error::invalid_input(
                "Embedding layer expects an input of rank 1 or more, got a scalar tensor",
            ));
        }
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let mut indices = Vec::with_capacity(input.len());
        for &value in input.iter() {
            // The cast saturates. It folds every value at or below 0, and every `NaN`, into row
            // 0. It folds `inf` and every huge value into `usize::MAX`. The first 2 tests
            // reject what the low end of that fold would hide. The third test rejects the rest.
            // Comparing after the cast keeps the upper bound exact, which a comparison against
            // `input_dim as f32` would not be for a table of over 2^24 rows
            let index = value as usize;
            if value.is_nan() || value <= -1.0 || index >= self.input_dim {
                return Err(Error::invalid_input(format!(
                    "Embedding layer received the index {}, and every index must truncate into \
                     0..{}",
                    value, self.input_dim
                )));
            }
            indices.push(index);
        }
        Ok(indices)
    }

    /// Stacks 1 table row per index into a new trailing axis
    ///
    /// A gather is a pure copy, so the 2 paths give the same bits. The gate is only a
    /// performance knob
    fn gather(&self, indices: &[usize], input_shape: &[usize]) -> Tensor {
        let width = self.output_dim;
        let elements = indices.len() * width;
        let table = self
            .embeddings
            .as_slice()
            .expect("the table is kept in C order");

        let data = if elements >= cheap_map_parallel_threshold() {
            // A `vec!` of 0 asks the allocator for pages that are already 0, so no element is
            // written twice. Each worker then faults in the pages of its own block, and that
            // parallel first touch is most of what the gate buys
            let mut data = vec![0.0f32; elements];
            let rows_per_task = (TASK_ELEMENTS / width).max(1);
            data.par_chunks_mut(rows_per_task * width)
                .enumerate()
                .for_each(|(task, block)| {
                    let first_row = task * rows_per_task;
                    for (row, destination) in block.chunks_mut(width).enumerate() {
                        let index = indices[first_row + row];
                        destination.copy_from_slice(&table[index * width..(index + 1) * width]);
                    }
                });
            data
        } else {
            let mut data = Vec::with_capacity(elements);
            for &index in indices {
                data.extend_from_slice(&table[index * width..(index + 1) * width]);
            }
            data
        };

        let mut shape = input_shape.to_vec();
        shape.push(width);
        Tensor::from_shape_vec(IxDyn(&shape), data).expect("the shape matches the data")
    }
}

impl Layer for Embedding {
    /// Training forward: caches the row indices and the input shape for the backward pass
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let indices = self.to_indices(input)?;
        let output = self.gather(&indices, input.shape());
        self.index_cache = Some(indices);
        self.input_shape = Some(input.shape().to_vec());
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        let indices = self.to_indices(input)?;
        Ok(self.gather(&indices, input.shape()))
    }

    /// Scatters the upstream gradient back into the table rows the forward pass read
    ///
    /// A row that several indices selected receives the sum of their gradients. The sum runs in
    /// output order, 1 index at a time, so the result reproduces bit for bit across runs
    ///
    /// The returned tensor is 0 everywhere. An index carries no derivative, so no gradient flows
    /// back through the input. The shape still matches the input, so a layer before this one
    /// still receives a well-formed tensor
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let Self {
            input_dim,
            output_dim,
            index_cache,
            input_shape,
            grad_embeddings,
            ..
        } = self;

        let (Some(indices), Some(input_shape)) = (index_cache.as_ref(), input_shape.as_ref())
        else {
            return Err(Error::forward_pass_not_run("Embedding"));
        };

        let mut expected = input_shape.clone();
        expected.push(*output_dim);
        if grad_output.shape() != expected.as_slice() {
            return Err(Error::shape_mismatch(expected, grad_output.shape()));
        }

        // The buffer survives between steps, so a large table allocates once instead of per batch
        let grad = grad_embeddings.get_or_insert_with(|| Array2::zeros((*input_dim, *output_dim)));
        grad.fill(0.0);
        let table = grad
            .as_slice_mut()
            .expect("the gradient buffer is kept in C order");

        // A gradient from a layer such as Permute can arrive strided, so read it in C order
        let source = grad_output.as_standard_layout();
        let source = source
            .as_slice()
            .expect("as_standard_layout gives a C-order array");

        let width = *output_dim;
        for (row, &index) in source.chunks_exact(width).zip(indices) {
            let slot = &mut table[index * width..(index + 1) * width];
            for (accumulator, &value) in slot.iter_mut().zip(row) {
                *accumulator += value;
            }
        }

        Ok(Tensor::zeros(IxDyn(input_shape)))
    }

    fn layer_type(&self) -> &str {
        "Embedding"
    }

    fn output_shape(&self) -> String {
        match &self.input_shape {
            // Element 0 is the batch axis, which `summary()` prints as "None"
            Some(shape) => {
                let mut axes: Vec<String> = shape[1..].iter().map(|e| e.to_string()).collect();
                axes.push(self.output_dim.to_string());
                format!("(None, {})", axes.join(", "))
            }
            None => "Unknown".to_string(),
        }
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.input_dim * self.output_dim)
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            embeddings,
            grad_embeddings,
            ..
        } = self;
        let mut params = Vec::new();
        if let Some(grad) = grad_embeddings.as_ref() {
            params.push(ParamGrad::weight(
                embeddings
                    .as_slice_mut()
                    .expect("the table is kept in C order"),
                grad.as_slice()
                    .expect("the gradient buffer is kept in C order"),
            ));
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Embedding(EmbeddingLayerWeight {
            embeddings: Cow::Borrowed(&self.embeddings),
        })
    }
}
