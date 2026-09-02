//! Sequential model that stacks layers into a feedforward network
//!
//! Supports training, prediction, summary, and binary save/load
//!
//! # 2 types, and only 1 of them trains
//!
//! [`SequentialBuilder`](crate::neural_network::sequential::SequentialBuilder) collects
//! layers. [`Sequential`](crate::neural_network::sequential::Sequential) is a built model.
//! [`SequentialBuilder::build`](crate::neural_network::sequential::SequentialBuilder::build) is
//! the only way to reach a built model, and it takes the
//! shape of the input. It walks the stack once, gives every layer the shape that reaches it,
//! and threads each output shape into the next layer
//!
//! `fit`, `train_batch`, `evaluate`, `predict`, `save_to_path`, and `load_from_path` are
//! methods of the built model alone. Training a model that was never built is therefore a
//! compile error, and no run-time state says whether a model is ready

use super::traits::{Layer, Loss, Optimizer};
use crate::error::{Error, IoError};
use crate::math::reduction::det_reduce;
use crate::neural_network::NnError;
use crate::neural_network::Shape;
use crate::neural_network::Tensor;
use crate::neural_network::layers::checkpoint::{
    LoadReport, MODEL_FORMAT_VERSION, MODEL_MAGIC, ModelCheckpoint, apply, apply_partial, capture,
    weight_path,
};
use crate::parallel_gates::sq_sum_f32_parallel_min_elems;
use ndarray::{ArrayViewD, Axis};
use ndarray_rand::rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

/// A sequential neural network model for building and training feedforward networks
///
/// Build a network by stacking layers in a linear fashion. Each layer feeds its output to
/// the next layer in sequence. The model fits most feedforward architectures where data
/// flows from input to output through a series of transformations
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::{
///     Shape,
///     sequential::SequentialBuilder,
///     layers::{Activation, Dense},
///     optimizers::Adam,
///     losses::CategoricalCrossEntropy,
/// };
/// use ndarray::Array;
///
/// // Create training data
/// let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features
/// let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
///
/// // Build a neural network
/// let mut model = SequentialBuilder::new()
///     .add(Dense::new(128, Activation::ReLU).unwrap())
///     .add(Dense::new(64, Activation::ReLU).unwrap())
///     .add(Dense::new(10, Activation::Softmax { axis: -1 }).unwrap())
///     .build(&Shape::known(&[32, 784]))
///     .unwrap();
/// model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
///
/// // Display model structure
/// model.summary();
///
/// // Train the model. History holds 1 loss value per epoch, each measured while that
/// // epoch ran, rather than after it
/// let history = model.fit(&x, &y, 10).unwrap();
/// println!("Per-epoch loss: {:?}", history.loss());
///
/// // Score the weights the model is holding now: an inference-mode pass that updates nothing
/// println!("Loss after training: {}", model.evaluate(&x, &y).unwrap());
///
/// // Save model weights to file
/// model.save_to_path("model.bin").unwrap();
///
/// // Create a new model with the same architecture, built for the same input shape
/// let mut new_model = SequentialBuilder::new()
///     .add(Dense::new(128, Activation::ReLU).unwrap())
///     .add(Dense::new(64, Activation::ReLU).unwrap())
///     .add(Dense::new(10, Activation::Softmax { axis: -1 }).unwrap())
///     .build(&Shape::known(&[32, 784]))
///     .unwrap();
///
/// // Load weights from file
/// new_model.load_from_path("model.bin").unwrap();
///
/// // Compile before using (required for training, optional for prediction)
/// new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
///
/// // Make predictions with loaded model
/// let predictions = new_model.predict(&x).unwrap();
/// println!("Predictions shape: {:?}", predictions.shape());
///
/// // Clean up: remove the created file
/// std::fs::remove_file("model.bin").unwrap();
/// ```
pub struct Sequential {
    /// All layers in the model
    layers: Vec<Box<dyn Layer>>,
    /// The shape that reaches each layer, in layer order
    ///
    /// [`SequentialBuilder::build`] fills it, so entry `i` is the shape that layer `i` was
    /// built for. [`summary`](Sequential::summary) reads it, and it is why a model that has
    /// only ever run [`predict`](Sequential::predict) still prints a real output shape for
    /// every position
    input_shapes: Vec<Shape>,
    /// Optimizer used for updating parameters during training
    optimizer: Option<Box<dyn Optimizer>>,
    /// Loss function used to compute training loss
    loss: Option<Box<dyn Loss>>,
    /// Optional seed governing the fit-time batch shuffle. Falls back to the global seed or
    /// entropy. See crate::random
    seed: Option<u64>,
}

/// Collects the layers of a model, and builds them against 1 input shape
///
/// This is the only way to reach a [`Sequential`]. [`add`](SequentialBuilder::add) never fails
/// and chains, so a whole stack reads as 1 expression.
/// [`build`](SequentialBuilder::build) takes the shape of the input, gives every layer the
/// shape that reaches it, and threads each output shape into the next layer
///
/// A layer allocates its arrays in [`Layer::build`], so a model that was never built holds no
/// weight at all. Splitting the 2 types is what makes that impossible to use by mistake:
/// `fit`, `train_batch`, `evaluate`, and `predict` are not methods of this type
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::{Activation, Dense};
///
/// let model = SequentialBuilder::new()
///     .add(Dense::new(8, Activation::ReLU).unwrap())
///     .add(Dense::new(1, Activation::Linear).unwrap())
///     .build(&Shape::known(&[4, 3]))
///     .unwrap();
///
/// // Every layer now holds its arrays
/// assert_eq!(model.weight_paths(), vec!["0.kernel", "0.bias", "1.kernel", "1.bias"]);
/// ```
///
/// A stack whose shapes do not agree is refused, and the message names the position of the
/// layer and its type:
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::{Activation, Dense, MaxPooling2D};
///
/// let refused = SequentialBuilder::new()
///     .add(Dense::new(8, Activation::ReLU).unwrap())
///     .add(MaxPooling2D::new((2, 2)))
///     .build(&Shape::known(&[4, 3]));
///
/// let message = match refused {
///     Ok(_) => panic!("the stack does not agree"),
///     Err(error) => error.to_string(),
/// };
/// assert!(message.contains("layer 1"), "{message}");
/// assert!(message.contains("MaxPooling2D"), "{message}");
/// ```
#[derive(Default)]
pub struct SequentialBuilder {
    /// The layers collected so far, from the input
    layers: Vec<Box<dyn Layer>>,
    /// Optional seed governing the fit-time batch shuffle of the built model
    seed: Option<u64>,
}

impl SequentialBuilder {
    /// Creates a builder that holds no layer
    ///
    /// # Returns
    ///
    /// - `SequentialBuilder` - An empty builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            seed: None,
        }
    }

    /// Creates a builder that holds no layer, with the fit-time shuffle seed preset
    ///
    /// The seed only governs the per-epoch batch shuffle that
    /// [`Sequential::fit_with_batches`] uses. It reinitializes nothing. See [`crate::random`]
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the reproducible fit-time shuffle
    ///
    /// # Returns
    ///
    /// - `SequentialBuilder` - An empty builder with the shuffle seed set
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            layers: Vec::new(),
            seed: Some(seed),
        }
    }

    /// Adds a layer to the end of the stack
    ///
    /// The method never fails. Nothing about a layer can disagree with the stack until a shape
    /// runs through it, and [`build`](SequentialBuilder::build) is where that happens
    ///
    /// # Parameters
    ///
    /// - `layer` - The layer to add
    ///
    /// # Returns
    ///
    /// - `Self` - The builder, for chaining
    // The name is the Keras name of this operation, and the signature is what a chained
    // builder needs. `std::ops::Add` takes 2 values of 1 type and this takes a layer, so the 2
    // have nothing in common but the word
    #[allow(clippy::should_implement_trait)]
    pub fn add<L: 'static + Layer>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Builds every layer against `input_shape`, and gives back the model
    ///
    /// The walk runs from the input. Each layer is built for the shape that reaches it, and
    /// [`Layer::compute_output_shape`] gives the shape that reaches the next one. A layer that
    /// refuses the shape stops the walk, and the message names the position of the layer and
    /// its type. Nothing is allocated past that position
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the tensor that enters the model, batch axis first. The batch
    ///   extent may be any value, and a layer never checks it
    ///
    /// # Returns
    ///
    /// - `Result<Sequential, Error>` - The built model
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the builder holds no layer
    /// - `Error::InvalidInput` - If a layer refuses the shape that reaches it. The message
    ///   names the position of the layer and its type
    pub fn build(mut self, input_shape: &Shape) -> Result<Sequential, Error> {
        if self.layers.is_empty() {
            return Err(Error::NeuralNetwork(NnError::EmptyModel));
        }

        let mut input_shapes = Vec::with_capacity(self.layers.len());
        let mut shape = input_shape.clone();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let layer_type = layer.layer_type().to_string();
            layer
                .build(&shape)
                .map_err(|source| build_refusal(index, &layer_type, &shape, source))?;
            let output = layer
                .compute_output_shape(&shape)
                .map_err(|source| build_refusal(index, &layer_type, &shape, source))?;
            input_shapes.push(shape);
            shape = output;
        }

        Ok(Sequential {
            layers: self.layers,
            input_shapes,
            optimizer: None,
            loss: None,
            seed: self.seed,
        })
    }
}

/// Names the layer that refused a shape during a model build
///
/// A shape error used to appear in the middle of a forward pass, deep inside a model, with no
/// layer named. The position and the type are what a caller needs to find the layer
#[cold]
fn build_refusal(index: usize, layer_type: &str, input: &Shape, source: Error) -> Error {
    Error::invalid_input(format!(
        "layer {index} (`{layer_type}`) refused the input shape {input}: {source}"
    ))
}

/// Global L2 norm of every gradient currently stored across `layers`, for clip-by-global-norm
///
/// Squared terms accumulate in f64 to limit round-off when summing across many parameters.
/// Each tensor folds as deterministic blocks. The rayon path at or above the square-sum gate
/// is a performance switch, and it gives the same result as the serial path. The per-tensor
/// totals merge in the fixed (layer, parameter) order, so rerunning on the same machine gives
/// the same result. Layers without gradients contribute nothing. With no gradients at all, the
/// norm is 0.0
///
/// The walk is forward, from the input. That is the canonical order of the model, and the
/// parameter-update walk in [`Sequential::train_batch`] uses the same one. The sum itself does
/// not depend on the order, but the model has 1 order and both walks follow it
fn global_grad_norm(layers: &mut [Box<dyn Layer>]) -> f32 {
    let mut sum_sq = 0.0_f64;
    for layer in layers.iter_mut() {
        for pg in layer.parameters() {
            sum_sq += det_reduce(
                pg.grad,
                pg.grad.len() >= sq_sum_f32_parallel_min_elems(),
                |block| block.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>(),
                |a, b| a + b,
                0.0,
            );
        }
    }
    sum_sq.sqrt() as f32
}

/// The per-epoch training loss that [`Sequential::fit`] and [`Sequential::fit_with_batches`]
/// record
///
/// This is Keras' `History`, in the shape this crate can fill today. It holds 1 entry per
/// epoch, in epoch order, so `loss()[e]` is epoch `e`'s loss. `loss().len()` is the number of
/// epochs that actually ran. Training for 0 epochs yields an empty slice
///
/// # What the number means
///
/// Each entry is the mean per-sample loss **measured during** the epoch, not after it. Every
/// batch contributes the loss from the forward pass that preceded that batch's own weight
/// update. The value therefore describes the weights the model held while the epoch ran, never
/// the weights the epoch ends with. Read the final entry as the trained model's loss, and you
/// will be wrong in either direction. It reads **above** the truth while training converges,
/// because the epoch's own updates improved on the weights it measured. It reads **below** the
/// truth once the step size starts overshooting and those updates make things worse.
/// [`evaluate`](Sequential::evaluate) is the call that scores the weights the model currently
/// holds. This matches Keras' convention for `History`
///
/// Batches contribute in proportion to their sample count. So the short trailing batch that
/// [`fit_with_batches`](Sequential::fit_with_batches) produces, when `batch_size` does not
/// divide the dataset, pulls the epoch mean less than a full batch does. That makes the entry
/// exactly the dataset-wide mean per-sample loss, matching Keras. Keras' loss metric accumulates
/// each batch with `sample_weight = batch_size`, rather than taking a plain mean over batches
#[derive(Debug, Clone, PartialEq)]
pub struct History {
    /// 1 loss value per epoch, in epoch order
    loss: Vec<f32>,
}

impl History {
    /// The per-epoch loss, in epoch order
    ///
    /// # Returns
    ///
    /// - `&[f32]` - 1 entry per epoch that ran (empty if `epochs` was `0`)
    pub fn loss(&self) -> &[f32] {
        &self.loss
    }
}

impl Sequential {
    /// Sets the seed governing the fit-time batch shuffle
    ///
    /// Controls only the data shuffling order used by `fit_with_batches`. It does not
    /// reinitialize or otherwise touch the model's layers. A fixed seed makes the per-epoch
    /// shuffle reproducible
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the reproducible fit-time shuffle. See crate::random
    ///
    /// # Returns
    ///
    /// - `&mut Self` - Mutable reference to self for method chaining
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the learning rate on the compiled optimizer
    ///
    /// The entry point for external learning-rate scheduling (step decay, warmup, ...) between
    /// epochs or batches. Does nothing if the model has not been compiled yet. The optimizer
    /// keeps all of its accumulated state (momentum buffers, Adam moments, ...) across the change
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - The new learning rate for subsequent parameter updates
    ///
    /// # Returns
    ///
    /// - `&mut Self` - Mutable reference to self for method chaining
    pub fn set_learning_rate(&mut self, learning_rate: f32) -> &mut Self {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.set_learning_rate(learning_rate);
        }
        self
    }

    /// The compiled optimizer's current learning rate
    ///
    /// The read half of [`set_learning_rate`](Self::set_learning_rate), so a schedule can derive
    /// the next step size from the current one instead of tracking its own copy
    ///
    /// # Returns
    ///
    /// - `Option<f32>` - The current learning rate, or `None` if the model has not been compiled
    pub fn learning_rate(&self) -> Option<f32> {
        self.optimizer.as_ref().map(|opt| opt.learning_rate())
    }

    /// Configures the optimizer and loss function for the model
    ///
    /// # Parameters
    ///
    /// - `optimizer` - The optimizer to use for training
    /// - `loss` - The loss function to use for training
    ///
    /// # Returns
    ///
    /// - `&mut Self` - Mutable reference to self for method chaining
    pub fn compile<O, LFunc>(&mut self, optimizer: O, loss: LFunc) -> &mut Self
    where
        O: 'static + Optimizer,
        LFunc: 'static + Loss,
    {
        self.optimizer = Some(Box::new(optimizer));
        self.loss = Some(Box::new(loss));
        self
    }

    /// Validates the model state and input data for a training step
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor containing training data
    /// - `y` - Target tensor containing expected outputs
    ///
    /// # Returns
    ///
    /// - `Ok(())` - If validation passes
    /// - `Err(Error)` - If validation fails
    fn validate_training_inputs(&self, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        if self.optimizer.is_none() {
            return Err(Error::NeuralNetwork(NnError::NotCompiled("optimizer")));
        }

        self.validate_evaluation_inputs(x, y)
    }

    /// Validates the model state and input data for computing a loss
    ///
    /// Everything [`validate_training_inputs`](Self::validate_training_inputs) checks except the
    /// optimizer, which only a parameter update needs. [`evaluate`](Self::evaluate) runs on a
    /// model that has a loss, but it never has to step
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor
    /// - `y` - Target tensor
    ///
    /// # Returns
    ///
    /// - `Ok(())` - If validation passes
    /// - `Err(Error)` - If validation fails
    fn validate_evaluation_inputs(&self, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        if self.loss.is_none() {
            return Err(Error::NeuralNetwork(NnError::NotCompiled("loss function")));
        }

        if self.layers.is_empty() {
            return Err(Error::NeuralNetwork(NnError::EmptyModel));
        }

        // A rank-0 tensor holds 1 element, so `is_empty` is false and the batch-axis index
        // below would panic. Reject it before then
        if x.ndim() == 0 || y.ndim() == 0 {
            return Err(Error::invalid_input(
                "input tensors must have a leading batch axis, but a rank-0 tensor was supplied",
            ));
        }

        // Input shape validation
        if x.is_empty() || y.is_empty() {
            return Err(Error::empty_input("input tensors"));
        }

        // Verify batch size match
        if x.shape()[0] != y.shape()[0] {
            return Err(Error::dimension_mismatch(x.shape()[0], y.shape()[0]));
        }

        Ok(())
    }

    /// Trains on a single batch: 1 forward pass, 1 gradient step
    ///
    /// [`fit`](Self::fit) and [`fit_with_batches`](Self::fit_with_batches) build on this unit.
    /// It is public so a custom loop can own the epoch structure: curriculum ordering, a
    /// per-step schedule, or an early-stopping probe between steps. This avoids reimplementing
    /// the forward, loss, backward, clip, and update sequencing. Keras calls this
    /// `train_on_batch`
    ///
    /// The whole of `x` is the batch. Nothing is split or shuffled. Mode-dependent layers run in
    /// **training** mode, so dropout samples a fresh mask and batch normalization updates its
    /// running statistics. On such a model, the returned loss is not comparable with
    /// [`evaluate`](Self::evaluate)'s
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor for the batch
    /// - `y` - Target tensor for the batch
    ///
    /// # Returns
    ///
    /// - `Ok(f32)` - The batch's loss, measured on the forward pass **before** this call's own
    ///   parameter update, as Keras' `train_on_batch` reports it
    /// - `Err(Error)` - If validation or training fails
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the optimizer or loss function is
    ///   not specified
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::EmptyInput` / `Error::InvalidInput` / `Error::DimensionMismatch` - If the
    ///   tensors are empty, rank-0, or disagree on the batch size
    /// - `Error::Computation` - If a layer fails during forward or backward pass
    pub fn train_batch(&mut self, x: &Tensor, y: &Tensor) -> Result<f32, Error> {
        // The unwraps below rest on this: it rejects a missing optimizer, a missing loss and an
        // empty layer stack before anything is touched
        self.validate_training_inputs(x, y)?;

        // Forward pass: first layer takes an input reference, later layers take owned tensors
        let mut layers_iter = self.layers.iter_mut();
        let first_layer = layers_iter
            .next()
            .ok_or_else(|| Error::NeuralNetwork(NnError::EmptyModel))?;
        first_layer.set_training_if_mode_dependent(true);
        let mut output = first_layer.forward(x)?;

        for layer in layers_iter {
            layer.set_training_if_mode_dependent(true);
            output = layer.forward(&output)?;
        }

        // Calculate loss
        let loss_value = self.loss.as_ref().unwrap().compute_loss(y, &output)?;

        // Calculate gradient of loss with respect to output
        let mut grad = self.loss.as_ref().unwrap().compute_grad(y, &output)?;

        // Advance the optimizer's global step once per batch, before the per-layer updates
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.step();
        }

        // Run every layer's backward so each stashes its gradients
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }

        // Clip-by-global-norm
        let global_clipnorm = self
            .optimizer
            .as_ref()
            .and_then(|opt| opt.global_clipnorm());
        let grad_scale = match global_clipnorm {
            Some(max_norm) => {
                let norm = global_grad_norm(&mut self.layers);
                if norm.is_finite() && norm > max_norm {
                    max_norm / norm
                } else {
                    1.0
                }
            }
            None => 1.0,
        };

        // Parameter updates. The walk is forward, from the input, and it is the canonical order
        // of the model: `global_grad_norm` above uses the same one. The index is the layer half
        // of the parameter address that the optimizer keys its state on, so it must count from
        // the input and never from the output
        if let Some(ref mut optimizer) = self.optimizer {
            for (scope, layer) in self.layers.iter_mut().enumerate() {
                optimizer.update(scope, &mut **layer, grad_scale);
            }
        }

        Ok(loss_value)
    }

    /// Trains the model on the provided data
    ///
    /// Executes the forward pass, loss calculation, backward pass, and parameter updates
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor containing training data
    /// - `y` - Target tensor containing expected outputs
    /// - `epochs` - Number of training epochs to perform
    ///
    /// # Returns
    ///
    /// - `Result<History, Error>` - 1 loss value per epoch, in epoch order, or an error
    ///
    /// # Notes
    ///
    /// Each epoch trains on the entire dataset as a single full-batch gradient step. There is
    /// only 1 batch, so no shuffling happens and the fit-time seed is unused. For mini-batch
    /// training that splits the data into fixed-size batches and reshuffles every epoch, use
    /// [`fit_with_batches`](Self::fit_with_batches)
    ///
    /// Each epoch's loss is measured before that epoch's own update. So the last entry
    /// describes the weights going *into* the final step, not the trained model. See
    /// [`History`]
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the optimizer or loss function is
    ///   not specified
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::EmptyInput` / `Error::InvalidInput` / `Error::DimensionMismatch` - If inputs
    ///   are empty, rank-0, or batch sizes disagree
    /// - `Error::Computation` - If a layer fails during forward or backward pass
    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: u32) -> Result<History, Error> {
        // Validate up front so a broken model or mismatched data fails before any epoch runs.
        // With `epochs == 0`, the per-batch validation inside `train_batch` never happens
        self.validate_training_inputs(x, y)?;

        // Create progress bar for training epochs
        #[cfg(feature = "show_progress")]
        let progress_bar = crate::create_progress_bar(
            epochs as u64,
            "[{elapsed_precise}] {bar:40} {pos}/{len} | Loss: {msg}",
        );

        let mut loss = Vec::new();

        for _ in 0..epochs {
            // Train on the entire dataset as 1 batch
            let epoch_loss = self.train_batch(x, y)?;
            loss.push(epoch_loss);

            // Update progress bar with current loss
            #[cfg(feature = "show_progress")]
            {
                progress_bar.set_message(format!("{:.6}", epoch_loss));
                progress_bar.inc(1);
            }
        }

        // Finish progress bar
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Training completed");

        Ok(History { loss })
    }

    /// Trains the model using mini-batch processing
    ///
    /// Splits data into batches of the specified size and trains on each in turn. With the
    /// `show_progress` feature, a progress bar reports the running average loss per epoch
    ///
    /// # Parameters
    ///
    /// - `x` - Input training data tensor
    /// - `y` - Target output data tensor
    /// - `epochs` - Number of training epochs
    /// - `batch_size` - Size of each training batch
    ///
    /// # Returns
    ///
    /// - `Result<History, Error>` - 1 loss value per epoch, in epoch order, or an error
    ///
    /// # Notes
    ///
    /// The sample order is reshuffled at the start of every epoch. Seed it via
    /// [`set_seed`](Self::set_seed) or [`SequentialBuilder::new_with_seed`] for a reproducible
    /// shuffle. To train on the whole dataset as a single full-batch gradient step per epoch
    /// instead (no batching, no shuffling), use [`fit`](Self::fit)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the optimizer or loss function is
    ///   not specified
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::EmptyInput` / `Error::InvalidInput` / `Error::DimensionMismatch` - If inputs
    ///   are empty, rank-0, or batch sizes disagree
    /// - `Error::InvalidParameter` - If `batch_size` is 0 or larger than the dataset
    /// - `Error::Computation` - If a layer fails during forward or backward pass, or a batch
    ///   tensor cannot be built
    pub fn fit_with_batches(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: u32,
        batch_size: usize,
    ) -> Result<History, Error> {
        // Validate inputs
        self.validate_training_inputs(x, y)?;

        let n_samples = x.shape()[0];

        // Validate batch size
        if batch_size == 0 {
            return Err(Error::invalid_parameter(
                "batch_size",
                "must be greater than 0",
            ));
        }

        if batch_size > n_samples {
            return Err(Error::invalid_parameter(
                "batch_size",
                format!(
                    "({}) cannot be larger than dataset size ({})",
                    batch_size, n_samples
                ),
            ));
        }

        // Creates batch tensors by gathering the selected rows along axis 0
        let create_batch_tensors =
            |x: &Tensor, y: &Tensor, indices: &[usize]| -> Result<(Tensor, Tensor), Error> {
                Ok((x.select(Axis(0), indices), y.select(Axis(0), indices)))
            };

        // Create sample indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Seed the per-epoch shuffle once. `None` consults the thread-local global seed
        let mut shuffle_rng = crate::random::make_rng(self.seed);

        #[cfg(feature = "show_progress")]
        let total_batches = n_samples.div_ceil(batch_size);
        #[cfg(feature = "show_progress")]
        let total_iterations = epochs as u64 * total_batches as u64;

        // Create progress bar for batch training
        #[cfg(feature = "show_progress")]
        let progress_bar = crate::create_progress_bar(
            total_iterations,
            "[{elapsed_precise}] {bar:40} {pos}/{len} | Epoch {msg}",
        );

        let mut loss = Vec::new();

        // Shuffle each epoch, then process fixed-size batches. Only the progress-bar bookkeeping
        // is gated on `show_progress`. The shuffle, chunk, and train logic is shared
        for epoch in 0..epochs {
            indices.shuffle(&mut shuffle_rng);

            // Each batch counts for as many samples as it holds, rather than 1 vote each. So the
            // short trailing batch left when `batch_size` does not divide the dataset pulls the
            // epoch figure less than a full batch does. That makes it exactly the dataset-wide
            // mean per-sample loss, which is what Keras' loss metric does
            // (`sample_weight = batch_size`).
            // The running sum is f64 because an epoch can hold many thousands of batches
            let (mut weighted_loss, mut samples_seen) = (0.0_f64, 0_usize);

            for batch_indices in indices.chunks(batch_size) {
                let (batch_x, batch_y) = create_batch_tensors(x, y, batch_indices)?;
                let batch_loss = self.train_batch(&batch_x, &batch_y)?;

                weighted_loss += batch_loss as f64 * batch_indices.len() as f64;
                samples_seen += batch_indices.len();

                #[cfg(feature = "show_progress")]
                {
                    progress_bar.set_message(format!(
                        "{}/{} | Avg Loss: {:.6}",
                        epoch + 1,
                        epochs,
                        weighted_loss / samples_seen as f64
                    ));
                    progress_bar.inc(1);
                }
            }

            // `samples_seen` is `n_samples`, which validation proved non-zero
            loss.push((weighted_loss / samples_seen as f64) as f32);

            #[cfg(not(feature = "show_progress"))]
            let _ = epoch;
        }

        // Finish progress bar
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Training completed");

        Ok(History { loss })
    }

    /// Computes the loss on data without training on it
    ///
    /// Keras' `evaluate`: 1 inference-mode forward pass over the whole of `x`, scored with the
    /// compiled loss. Nothing is updated: no gradients, no parameters, and no batch-normalization
    /// running statistics. This is what a validation pass, an early-stopping test, or a
    /// checkpoint-selection rule needs. It borrows `&self`, so it can score a model between
    /// training steps without disturbing it. It also draws from no RNG, so it cannot perturb
    /// the shuffle stream that [`fit_with_batches`](Self::fit_with_batches) depends on
    ///
    /// Layers behave exactly as in [`predict`](Self::predict): dropout and noise layers are the
    /// identity, and batch normalization reads its running statistics. On a model that contains
    /// one of these layers, this number is therefore *not* the number [`fit`](Self::fit) records
    /// for the same data. Training-mode dropout inflates that number. `evaluate`'s number is the
    /// more accurate of the 2
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor to score
    /// - `y` - Target tensor
    ///
    /// # Returns
    ///
    /// - `Result<f32, Error>` - The loss over `x`, computed as a single full-batch pass
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled("loss function"))` - If the model has no loss.
    ///   The optimizer is not consulted, since nothing is updated
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::EmptyInput` / `Error::InvalidInput` / `Error::DimensionMismatch` - If inputs are
    ///   empty, rank-0, or batch sizes disagree
    /// - `Error::Computation` - If a layer fails during the forward pass
    pub fn evaluate(&self, x: &Tensor, y: &Tensor) -> Result<f32, Error> {
        self.validate_evaluation_inputs(x, y)?;

        let predictions = self.predict(x)?;

        // Validation above established that the loss is present
        self.loss.as_ref().unwrap().compute_loss(y, &predictions)
    }

    /// Generates predictions for the input data
    ///
    /// Runs only a forward pass, without any training. To score those predictions against
    /// targets with the compiled loss, use [`evaluate`](Self::evaluate)
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor containing data to predict on
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - Tensor containing the model's predictions or an error
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `x` is empty
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::Computation` - If any layer fails during forward pass
    pub fn predict(&self, x: &Tensor) -> Result<Tensor, Error> {
        // Input validation
        if x.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        // Inference path: each layer's `predict` runs in eval mode and writes no caches
        let mut layers_iter = self.layers.iter();
        let first_layer = layers_iter
            .next()
            .ok_or_else(|| Error::NeuralNetwork(NnError::EmptyModel))?;
        let mut output = first_layer.predict(x)?;

        for layer in layers_iter {
            output = layer.predict(&output)?;
        }
        Ok(output)
    }

    /// Prints a summary of the model's structure
    ///
    /// Displays each layer's information and parameter statistics in a tabular format to stdout
    pub fn summary(&self) {
        let col1_width = 33;
        let col2_width = 24;
        let col3_width = 15;

        let mut output = String::new();

        output.push_str("Model: \"sequential\"\n");
        output.push_str(&format!(
            "┏{}┳{}┳{}┓\n",
            "━".repeat(col1_width),
            "━".repeat(col2_width),
            "━".repeat(col3_width)
        ));
        output.push_str(&format!(
            "┃ {:<31} ┃ {:<22} ┃ {:>13} ┃\n",
            "Layer (type)", "Output Shape", "Param #"
        ));
        output.push_str(&format!(
            "┡{}╇{}╇{}┩\n",
            "━".repeat(col1_width),
            "━".repeat(col2_width),
            "━".repeat(col3_width)
        ));

        let mut total_params: usize = 0;
        let mut trainable_param_count: usize = 0;
        let mut non_trainable_param_count: usize = 0;

        // Per-type counter for Keras-style names: "dense", "dense_1", "conv2d", ...
        let mut type_counts: HashMap<&str, usize> = HashMap::new();

        for (index, layer) in self.layers.iter().enumerate() {
            let layer_type = layer.layer_type();

            // Generate name from the layer type with a per-type index
            let count = type_counts.entry(layer_type).or_insert(0);
            let layer_name = if *count == 0 {
                layer_type.to_lowercase()
            } else {
                format!("{}_{}", layer_type.to_lowercase(), count)
            };
            *count += 1;

            // The shape table comes from the build, so a model that has only ever run
            // `predict` still prints a real shape for every position
            let out_shape = match layer.compute_output_shape(&self.input_shapes[index]) {
                Ok(shape) => shape.to_string(),
                Err(_) => "Unknown".to_string(),
            };

            // Both counts are added, so a layer that holds non-trainable state (the running
            // statistics of batch normalization) reaches the total and the third column
            let counts = layer.param_count();
            trainable_param_count += counts.trainable;
            non_trainable_param_count += counts.non_trainable;
            total_params += counts.total();
            let param_count_num = counts.total();

            output.push_str(&format!(
                "│ {:<31} │ {:<22} │ {:>13} │\n",
                format!("{} ({})", layer_name, layer_type),
                out_shape,
                param_count_num
            ));
        }

        output.push_str(&format!(
            "└{}┴{}┴{}┘\n",
            "─".repeat(col1_width),
            "─".repeat(col2_width),
            "─".repeat(col3_width)
        ));
        output.push_str(&format!(
            " Total params: {} ({} B)\n",
            total_params,
            total_params * 4
        )); // f32 stores each parameter in 4 bytes
        output.push_str(&format!(
            " Trainable params: {} ({} B)\n",
            trainable_param_count,
            trainable_param_count * 4
        ));
        output.push_str(&format!(
            " Non-trainable params: {} ({} B)",
            non_trainable_param_count,
            non_trainable_param_count * 4
        ));

        println!("{}", output);
    }

    /// The shape the model was built for
    ///
    /// It is the shape that [`SequentialBuilder::build`] received, batch axis included
    ///
    /// # Returns
    ///
    /// - `&Shape` - The input shape of the model
    pub fn input_shape(&self) -> &Shape {
        // `build` refuses an empty stack, so the table always holds at least 1 entry
        &self.input_shapes[0]
    }

    /// The shape the model gives back for the shape it was built for
    ///
    /// # Returns
    ///
    /// - `Result<Shape, Error>` - Shape of the output of the last layer
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the last layer refuses the shape that reaches it. A build
    ///   already ruled that out, so this cannot happen on a model this crate built
    pub fn output_shape(&self) -> Result<Shape, Error> {
        let last = self.layers.len() - 1;
        self.layers[last].compute_output_shape(&self.input_shapes[last])
    }

    /// Every checkpoint path of the model, in order
    ///
    /// A path is `<scope>.<name>`: the position of the layer, counted from the input, and the
    /// name that the layer gives the array. The list holds the trainable arrays and the
    /// non-trainable state alike, which is exactly what a saved file holds
    ///
    /// # Returns
    ///
    /// - `Vec<String>` - 1 path per array of the model
    pub fn weight_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        for (scope, layer) in self.layers.iter().enumerate() {
            paths.extend(
                layer
                    .weights()
                    .iter()
                    .map(|entry| weight_path(scope, entry.name)),
            );
        }
        paths
    }

    /// 1 array of the model, by its checkpoint path
    ///
    /// The view borrows the live array, so nothing is copied
    ///
    /// # Parameters
    ///
    /// - `path` - The dotted path of the array, such as `"0.kernel"`. See
    ///   [`weight_paths`](Sequential::weight_paths)
    ///
    /// # Returns
    ///
    /// - `Option<ArrayViewD<'_, f32>>` - A read view of the array, or `None` when the model
    ///   holds no array at that path
    pub fn weight(&self, path: &str) -> Option<ArrayViewD<'_, f32>> {
        let (scope, name) = path.split_once('.')?;
        let layer = self.layers.get(scope.parse::<usize>().ok()?)?;
        layer.weight(name)
    }

    /// Writes a named checkpoint of every array of the model to a binary file
    ///
    /// The file holds the layer type of each position, and every array that the layer at that
    /// position holds, under the name that the layer gives it. It holds the non-trainable
    /// state next to the trainable parameters, so the running statistics of
    /// [`BatchNormalization`](crate::neural_network::layers::BatchNormalization) go in it as
    /// well. postcard writes the bytes. See
    /// [`checkpoint`](crate::neural_network::layers::checkpoint)
    ///
    /// The file carries no architecture and no layer configuration. A load therefore needs a
    /// model that is already built, and it compares that model against the file. The file
    /// holds no optimizer and no loss function either. Call `compile` again after a load
    ///
    /// # Parameters
    ///
    /// - `path` - File path where the model will be saved (e.g., "stored_model.bin"). Accepts
    ///   anything convertible to a `Path` (`&str`, `String`, `Path`, `PathBuf`, ...)
    ///
    /// # Returns
    ///
    /// - `crate::error::RustymlResult<()>` - Ok if the model is saved, or an IO/serialization error
    ///
    /// # Errors
    ///
    /// - `Error::Io(IoError::Std)` - File creation or write operation failed
    /// - `Error::Io(IoError::Serialization)` - Serialization failed
    pub fn save_to_path(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::RustymlResult<()> {
        // `capture` borrows the live arrays, so nothing is copied before postcard reads them
        let bytes = postcard::to_allocvec(&capture(&self.layers))?;

        // Create or overwrite the file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write the serialized bytes to file
        writer.write_all(&bytes)?;

        // Make sure all data is written to disk
        writer.flush()?;

        Ok(())
    }

    /// Loads model weights from a binary file and applies them to the current model
    ///
    /// Reads a checkpoint that [`save_to_path`](Sequential::save_to_path) wrote and applies
    /// every array of it to the layer that holds the matching path. The load is strict: the
    /// file and the model must agree on the layer count, on the layer type of every position,
    /// and on the name, the kind, and the shape of every array. A disagreement is an error
    /// that names the checkpoint path, and no lenient rule replaces it. Use
    /// [`load_partial_from_path`](Sequential::load_partial_from_path) to load what matches and
    /// read a report of the rest
    ///
    /// The model is already built, because [`SequentialBuilder::build`] is the only way to
    /// reach this type. Every array therefore exists before the load writes into it. The file
    /// carries the shape each layer was built for, and the load refuses a file whose build
    /// shape differs from the model, so a checkpoint never lands in a model shaped for another
    /// input. After loading, call `compile()` to set the optimizer and loss function
    ///
    /// # Parameters
    ///
    /// - `path` - File path from which to load the weights (e.g., "stored_model.bin"). Accepts
    ///   anything convertible to a `Path` (`&str`, `String`, `Path`, `PathBuf`, ...)
    ///
    /// # Returns
    ///
    /// - `crate::error::RustymlResult<()>` - Ok if weights are loaded, or an
    ///   IO/deserialization error
    ///
    /// # Errors
    ///
    /// - `Error::Io(IoError::Std)` - File not found or read operation failed
    /// - `Error::Io(IoError::UnsupportedModelFormat)` - The file is not a RustyML model, or a
    ///   release whose on-disk format version differs from this one wrote it
    /// - `Error::Io(IoError::Serialization)` - Deserialization failed
    /// - `Error::Io(IoError::ModelStructureMismatch)` - The file and the model disagree. The
    ///   message names the checkpoint path, or the layer position for a whole-layer
    ///   disagreement
    pub fn load_from_path(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::RustymlResult<()> {
        let file = read_checkpoint(path)?;
        apply(&mut self.layers, &file)?;
        Ok(())
    }

    /// Loads what the file and the model agree on, and reports the rest
    ///
    /// This is the lenient load, and a caller asks for it by name. It writes an array when the
    /// position holds the same layer type and the file holds the same name, the same kind, and
    /// the same shape. Every other path of the model, and every path of the file that reached
    /// no array, goes into the report. Nothing about the layer roster fails
    ///
    /// The header is still checked, because a file of another format version carries bytes
    /// that mean something else. Such a file is an error here as well
    ///
    /// # Parameters
    ///
    /// - `path` - File path from which to load the weights
    ///
    /// # Returns
    ///
    /// - `crate::error::RustymlResult<LoadReport>` - The paths that took a value, the paths of
    ///   the model that got none, and the paths of the file that reached no array
    ///
    /// # Errors
    ///
    /// - `Error::Io(IoError::Std)` - File not found or read operation failed
    /// - `Error::Io(IoError::UnsupportedModelFormat)` - The file is not a RustyML model, or a
    ///   release whose on-disk format version differs from this one wrote it
    /// - `Error::Io(IoError::Serialization)` - Deserialization failed
    pub fn load_partial_from_path(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::RustymlResult<LoadReport> {
        let file = read_checkpoint(path)?;
        Ok(apply_partial(&mut self.layers, &file))
    }
}

/// Reads 1 checkpoint file, and validates its header before its body
///
/// postcard is sequential and carries no field names, so a file from an incompatible release
/// otherwise runs off the end of some array. It then reports an opaque deserialization failure
/// instead of naming the real problem. Where the extents happen to coincide, it does not fail
/// at all
///
/// # Parameters
///
/// - `path` - File path to read
///
/// # Returns
///
/// - `crate::error::RustymlResult<ModelCheckpoint<'static>>` - The whole file, with owned
///   arrays
///
/// # Errors
///
/// - `Error::Io(IoError::Std)` - File not found or read operation failed
/// - `Error::Io(IoError::UnsupportedModelFormat)` - The file is too short for a header, does
///   not carry the magic tag, or carries another format version
/// - `Error::Io(IoError::Serialization)` - Deserialization failed
fn read_checkpoint(
    path: impl AsRef<std::path::Path>,
) -> crate::error::RustymlResult<ModelCheckpoint<'static>> {
    let bytes = std::fs::read(path)?;

    let (magic, format_version): (u32, u32) = postcard::take_from_bytes(&bytes)
        .map(|(header, _rest)| header)
        .map_err(|_| {
            Error::Io(IoError::UnsupportedModelFormat(
                "file is too short to contain a model header".to_string(),
            ))
        })?;
    if magic != MODEL_MAGIC {
        return Err(Error::Io(IoError::UnsupportedModelFormat(format!(
            "not a RustyML model file: expected magic {MODEL_MAGIC:#010x}, found {magic:#010x} \
             (a model saved before the format carried a header must be re-saved)"
        ))));
    }
    if format_version != MODEL_FORMAT_VERSION {
        return Err(Error::Io(IoError::UnsupportedModelFormat(format!(
            "model file is format version {format_version}, and this build reads version \
             {MODEL_FORMAT_VERSION}; re-save the model with this version of RustyML"
        ))));
    }

    Ok(postcard::from_bytes(&bytes)?)
}
