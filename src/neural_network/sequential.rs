//! Sequential model that stacks layers into a feedforward network
//!
//! Supports training, prediction, summary, and binary save/load

use super::traits::{Layer, Loss, Optimizer};
use crate::error::{Error, IoError, NnError};
use crate::math::reduction::det_reduce;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::serialize_model::{
    LayerInfo, SerializableLayer, SerializableSequential, apply_weights_to_layer,
};
use crate::parallel_gates::SQ_SUM_F32_PARALLEL_MIN_ELEMS;
use ndarray::Axis;
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
///     sequential::Sequential,
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
/// let mut model = Sequential::new();
/// model
///     .add(Dense::new(784, 128, Activation::ReLU).unwrap())
///     .add(Dense::new(128, 64, Activation::ReLU).unwrap())
///     .add(Dense::new(64, 10, Activation::Softmax).unwrap())
///     .compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
///
/// // Display model structure
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 10).unwrap();
///
/// // Save model weights to file
/// model.save_to_path("model.bin").unwrap();
///
/// // Create a new model with the same architecture
/// let mut new_model = Sequential::new();
/// new_model
///     .add(Dense::new(784, 128, Activation::ReLU).unwrap())
///     .add(Dense::new(128, 64, Activation::ReLU).unwrap())
///     .add(Dense::new(64, 10, Activation::Softmax).unwrap());
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
    /// Optimizer used for updating parameters during training
    optimizer: Option<Box<dyn Optimizer>>,
    /// Loss function used to compute training loss
    loss: Option<Box<dyn Loss>>,
    /// Optional seed governing the fit-time batch shuffle; falls back to the global seed or entropy. See crate::random
    seed: Option<u64>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

/// Global L2 norm of every gradient currently stored across `layers`, for clip-by-global-norm
///
/// Squared terms accumulate in f64 to limit round-off when summing across many parameters;
/// each tensor folds as deterministic blocks (on rayon at or above the square-sum gate, a
/// performance switch that is bitwise identical either way), and the per-tensor totals merge
/// in the fixed (layer, parameter) order. Layers without gradients contribute nothing; with
/// no gradients at all the norm is 0.0
fn global_grad_norm(layers: &mut [Box<dyn Layer>]) -> f32 {
    let mut sum_sq = 0.0_f64;
    for layer in layers.iter_mut() {
        for pg in layer.parameters() {
            sum_sq += det_reduce(
                pg.grad,
                pg.grad.len() >= SQ_SUM_F32_PARALLEL_MIN_ELEMS,
                |block| block.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>(),
                |a, b| a + b,
                0.0,
            );
        }
    }
    sum_sq.sqrt() as f32
}

impl Sequential {
    /// Creates a new empty Sequential model
    ///
    /// # Returns
    ///
    /// - `Sequential` - An empty Sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            loss: None,
            seed: None,
        }
    }

    /// Sets the seed governing the fit-time batch shuffle
    ///
    /// Controls only the data shuffling order used by `fit_with_batches`; it does not
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
    pub fn set_learning_rate(&mut self, learning_rate: f32) -> &mut Self {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.set_learning_rate(learning_rate);
        }
        self
    }

    /// Creates a new empty Sequential model with the fit-time shuffle seed preset
    ///
    /// Equivalent to `Sequential::new()` followed by `set_seed(seed)`. The seed only governs
    /// the per-epoch batch shuffle used by `fit_with_batches`; see crate::random
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the reproducible fit-time shuffle. See crate::random
    ///
    /// # Returns
    ///
    /// - `Sequential` - An empty Sequential model with the shuffle seed set
    pub fn new_with_seed(seed: u64) -> Self {
        let mut model = Self::new();
        model.seed = Some(seed);
        model
    }

    /// Adds a layer to the model
    ///
    /// Supports method chaining
    ///
    /// # Parameters
    ///
    /// - `layer` - The layer to add to the model
    ///
    /// # Returns
    ///
    /// - `&mut Self` - Mutable reference to self for method chaining
    pub fn add<L: 'static + Layer>(&mut self, layer: L) -> &mut Self {
        self.layers.push(Box::new(layer));
        self
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

    /// Validates the model state and input data
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

        if self.loss.is_none() {
            return Err(Error::NeuralNetwork(NnError::NotCompiled("loss function")));
        }

        if self.layers.is_empty() {
            return Err(Error::NeuralNetwork(NnError::EmptyModel));
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

    /// Performs training on a single batch of data
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor for the batch
    /// - `y` - Target tensor for the batch
    ///
    /// # Returns
    ///
    /// - `Ok(f32)` - The loss value for this batch
    /// - `Err(Error)` - If training fails
    fn train_batch(&mut self, x: &Tensor, y: &Tensor) -> Result<f32, Error> {
        // Forward pass: first layer takes an input reference, later layers take owned tensors
        let mut layers_iter = self.layers.iter_mut();
        let first_layer = layers_iter
            .next()
            .ok_or_else(|| Error::computation("no layers in model"))?;
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

        // run every layer's backward so each stashes its gradients
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }

        // Clip-by-global-norm
        let clip_norm = self.optimizer.as_ref().and_then(|opt| opt.clip_norm());
        let grad_scale = match clip_norm {
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

        // Parameter updates
        if let Some(ref mut optimizer) = self.optimizer {
            for layer in self.layers.iter_mut().rev() {
                optimizer.update(&mut **layer, grad_scale);
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
    /// - `Result<&mut Self, Error>` - Mutable reference to self after training or an error
    ///
    /// # Notes
    ///
    /// Each epoch trains on the entire dataset as a single full-batch gradient step (there is only
    /// one batch, so no shuffling happens and the fit-time seed is unused). For mini-batch training
    /// that splits the data into fixed-size batches and reshuffles every epoch, use
    /// [`fit_with_batches`](Self::fit_with_batches)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the optimizer or loss function is not specified
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - If inputs are empty or batch sizes disagree
    /// - `Error::Computation` - If a layer fails during forward or backward pass
    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: u32) -> Result<&mut Self, Error> {
        // Validate inputs
        self.validate_training_inputs(x, y)?;

        // Create progress bar for training epochs
        #[cfg(feature = "show_progress")]
        let progress_bar = crate::create_progress_bar(
            epochs as u64,
            "[{elapsed_precise}] {bar:40} {pos}/{len} | Loss: {msg}",
        );

        for _ in 0..epochs {
            // Train on the entire dataset as one batch
            #[cfg(feature = "show_progress")]
            let loss_value = self.train_batch(x, y)?;
            #[cfg(not(feature = "show_progress"))]
            let _ = self.train_batch(x, y)?;

            // Update progress bar with current loss
            #[cfg(feature = "show_progress")]
            {
                progress_bar.set_message(format!("{:.6}", loss_value));
                progress_bar.inc(1);
            }
        }

        // Finish progress bar
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Training completed");

        Ok(self)
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
    /// - `Result<&mut Self, Error>` - Mutable reference to trained model or error
    ///
    /// # Notes
    ///
    /// The sample order is reshuffled at the start of every epoch; seed it via
    /// [`set_seed`](Self::set_seed) / [`new_with_seed`](Self::new_with_seed) for a reproducible
    /// shuffle. To train on the whole dataset as a single full-batch gradient step per epoch
    /// instead (no batching, no shuffling), use [`fit`](Self::fit)
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotCompiled)` - If the optimizer or loss function is not specified
    /// - `Error::NeuralNetwork(NnError::EmptyModel)` - If the model has no layers
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - If inputs are empty or batch sizes disagree
    /// - `Error::InvalidParameter` - If `batch_size` is zero or larger than the dataset
    /// - `Error::Computation` - If a layer fails during forward or backward pass, or a batch tensor cannot be built
    pub fn fit_with_batches(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: u32,
        batch_size: usize,
    ) -> Result<&mut Self, Error> {
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

        // Seed the per-epoch shuffle once; `None` consults the thread-local global seed
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

        // Shuffle each epoch, then process fixed-size batches; only the progress-bar bookkeeping
        // is gated on `show_progress`, the shuffle/chunk/train logic is shared
        for epoch in 0..epochs {
            indices.shuffle(&mut shuffle_rng);

            #[cfg(feature = "show_progress")]
            let (mut epoch_loss, mut batch_count) = (0.0_f32, 0_usize);

            for batch_indices in indices.chunks(batch_size) {
                let (batch_x, batch_y) = create_batch_tensors(x, y, batch_indices)?;
                let batch_loss = self.train_batch(&batch_x, &batch_y)?;

                #[cfg(feature = "show_progress")]
                {
                    batch_count += 1;
                    epoch_loss += batch_loss;
                    progress_bar.set_message(format!(
                        "{}/{} | Avg Loss: {:.6}",
                        epoch + 1,
                        epochs,
                        epoch_loss / batch_count as f32
                    ));
                    progress_bar.inc(1);
                }
                #[cfg(not(feature = "show_progress"))]
                let _ = batch_loss;
            }

            #[cfg(not(feature = "show_progress"))]
            let _ = epoch;
        }

        // Finish progress bar
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Training completed");

        Ok(self)
    }

    /// Generates predictions for the input data
    ///
    /// Only performs a forward pass without any training
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

        for layer in self.layers.iter() {
            let layer_type = layer.layer_type();

            // Generate name from the layer type with a per-type index
            let count = type_counts.entry(layer_type).or_insert(0);
            let layer_name = if *count == 0 {
                layer_type.to_lowercase()
            } else {
                format!("{}_{}", layer_type.to_lowercase(), count)
            };
            *count += 1;

            let out_shape = layer.output_shape();

            // Exhaustive match so adding a TrainingParameters variant forces a compile error
            // instead of silently being counted as zero
            let param_count_num = match layer.param_count() {
                TrainingParameters::Trainable(count) => {
                    trainable_param_count += count;
                    total_params += count;
                    count
                }
                TrainingParameters::NonTrainable(count) => {
                    non_trainable_param_count += count;
                    total_params += count;
                    count
                }
                TrainingParameters::NoTrainable => 0,
            };

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
        )); // Using f32, each parameter is 4 bytes
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

    /// Returns all the weights from each layer in the model
    ///
    /// Collects the weights from all layers in the sequential model and returns them
    /// as a vector of `LayerWeight` enums. Each `LayerWeight` borrows the weight matrices and
    /// bias vectors of its corresponding layer (via `Cow`), so no weights are cloned
    ///
    /// # Returns
    ///
    /// - `Vec<LayerWeight<'_>>` - A vector borrowing each layer's weights
    pub fn get_weights(&self) -> Vec<LayerWeight<'_>> {
        let mut weights = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            weights.push(layer.get_weights());
        }
        weights
    }

    /// Saves the model architecture and weights to a binary file at the specified path
    ///
    /// Serializes the model structure including layer types, configurations,
    /// and all trainable parameters (weights and biases) to a compact binary format using
    /// postcard. Note that the optimizer and loss function are not saved and must be
    /// reconfigured after loading
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
        // Convert layers to serializable format
        let serializable_layers = self
            .layers
            .iter()
            .map(|layer| {
                let layer_info = LayerInfo {
                    layer_type: layer.layer_type().to_string(),
                    output_shape: layer.output_shape(),
                };

                // `get_weights` already borrows the live arrays via `Cow`, so this is clone-free
                SerializableLayer {
                    info: layer_info,
                    weights: layer.get_weights(),
                }
            })
            .collect();

        let serializable_model = SerializableSequential {
            layers: serializable_layers,
        };

        // Serialize the model to the compact postcard binary format
        let bytes = postcard::to_allocvec(&serializable_model)?;

        // Create or overwrite the file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write the serialized bytes to file
        writer.write_all(&bytes)?;

        // Ensure all data is written to disk
        writer.flush()?;

        Ok(())
    }

    /// Loads model weights from a binary file and applies them to the current model
    ///
    /// Deserializes weights from a previously saved model file and applies them
    /// to the current model's layers. The current model must have the same architecture
    /// (same number and types of layers) as the saved model
    ///
    /// Build the model structure first, then call this method to load weights. After loading,
    /// call `compile()` to set the optimizer and loss function
    ///
    /// # Parameters
    ///
    /// - `path` - File path from which to load the weights (e.g., "stored_model.bin"). Accepts
    ///   anything convertible to a `Path` (`&str`, `String`, `Path`, `PathBuf`, ...)
    ///
    /// # Returns
    ///
    /// - `crate::error::RustymlResult<()>` - Ok if weights are loaded, or an IO/deserialization error
    ///
    /// # Errors
    ///
    /// - `Error::Io(IoError::Std)` - File not found or read operation failed
    /// - `Error::Io(IoError::Serialization)` - Deserialization failed
    /// - `Error::Io(IoError::ModelStructureMismatch)` - The current model's structure (layer count, a layer
    ///   type at some position, or a weight shape) does not match the saved model
    pub fn load_from_path(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::RustymlResult<()> {
        // Read the whole file into memory
        let bytes = std::fs::read(path)?;

        // Deserialize the model from the postcard binary format
        let serializable_model: SerializableSequential<'static> = postcard::from_bytes(&bytes)?;

        // Verify layer count matches
        if serializable_model.layers.len() != self.layers.len() {
            return Err(Error::Io(IoError::ModelStructureMismatch(format!(
                "layer count mismatch: model has {} layers, file has {} layers",
                self.layers.len(),
                serializable_model.layers.len()
            ))));
        }

        // Apply weights to each layer
        for (i, serializable_layer) in serializable_model.layers.iter().enumerate() {
            let expected_type = self.layers[i].layer_type();
            let saved_type = serializable_layer.info.layer_type.as_str();
            if expected_type != saved_type {
                return Err(Error::Io(IoError::ModelStructureMismatch(format!(
                    "layer {} type mismatch: model has `{}`, file has `{}`",
                    i, expected_type, saved_type
                ))));
            }

            apply_weights_to_layer(
                &mut *self.layers[i],
                &serializable_layer.weights,
                saved_type,
            )?;
        }

        Ok(())
    }
}
