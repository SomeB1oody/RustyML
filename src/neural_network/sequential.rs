use super::traits::{Layer, Loss, Optimizer};
use crate::error::{Context, Error, IoError, NnError};
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::serialize_weight::{
    LayerInfo, SerializableLayer, SerializableLayerWeight, SerializableSequential,
    apply_weights_to_layer,
};
use ndarray::{Array, Axis, IxDyn};
use ndarray_rand::rand::seq::SliceRandom;
use serde_json::{from_reader, to_writer_pretty};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

/// A Sequential neural network model for building and training feedforward networks.
///
/// The Sequential model allows you to build neural networks by stacking layers in a linear fashion.
/// Each layer feeds its output to the next layer in sequence. This model is suitable for
/// most feedforward neural network architectures where data flows from input to output
/// through a series of transformations.
///
/// # Fields
///
/// - `layers` - A vector containing all layers in the model
/// - `optimizer` - Optimizer used for updating parameters during training
/// - `loss` - Loss function used to compute training loss
/// - `seed` - Optional seed governing the fit-time batch shuffle; falls back to the global seed or entropy. See crate::random.
///
/// # Examples
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
///     .add(Dense::new(784, 128, Activation::ReLU, None).unwrap())
///     .add(Dense::new(128, 64, Activation::ReLU, None).unwrap())
///     .add(Dense::new(64, 10, Activation::Softmax, None).unwrap())
///     .compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), CategoricalCrossEntropy::new());
///
/// // Display model structure
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 10).unwrap();
///
/// // Save model weights to file
/// model.save_to_path("model.json").unwrap();
///
/// // Create a new model with the same architecture
/// let mut new_model = Sequential::new();
/// new_model
///     .add(Dense::new(784, 128, Activation::ReLU, None).unwrap())
///     .add(Dense::new(128, 64, Activation::ReLU, None).unwrap())
///     .add(Dense::new(64, 10, Activation::Softmax, None).unwrap());
///
/// // Load weights from file
/// new_model.load_from_path("model.json").unwrap();
///
/// // Compile before using (required for training, optional for prediction)
/// new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), CategoricalCrossEntropy::new());
///
/// // Make predictions with loaded model
/// let predictions = new_model.predict(&x).unwrap();
/// println!("Predictions shape: {:?}", predictions.shape());
///
/// // Clean up: remove the created file
/// std::fs::remove_file("model.json").unwrap();
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    loss: Option<Box<dyn Loss>>,
    seed: Option<u64>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    /// Creates a new empty Sequential model.
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

    /// Sets the seed governing the fit-time batch shuffle.
    ///
    /// This is a pure setter that only controls the data shuffling order used by
    /// `fit_with_batches`; it does not reinitialize or otherwise touch the model's layers.
    /// Setting a fixed seed makes the per-epoch shuffle reproducible.
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the reproducible fit-time shuffle. See crate::random.
    ///
    /// # Returns
    ///
    /// - `&mut Self` - Mutable reference to self for method chaining
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Creates a new empty Sequential model with the fit-time shuffle seed preset.
    ///
    /// Equivalent to `Sequential::new()` followed by `set_seed(seed)`. The seed only governs
    /// the per-epoch batch shuffle used by `fit_with_batches`; see crate::random.
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the reproducible fit-time shuffle. See crate::random.
    ///
    /// # Returns
    ///
    /// - `Sequential` - An empty Sequential model with the shuffle seed set
    pub fn new_with_seed(seed: u64) -> Self {
        let mut model = Self::new();
        model.seed = Some(seed);
        model
    }

    /// Adds a layer to the model.
    ///
    /// Supports method chaining pattern.
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

    /// Configures the optimizer and loss function for the model.
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

        // Verify batch size match. `dimension_mismatch(expected, found)` takes the input batch
        // as the reference (`expected`) and the target batch as `found` — the same `(x, y)`
        // ordering used crate-wide (e.g. knn, lda, machine_learning::validation).
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
        // Forward pass - first layer takes input reference, subsequent layers take owned tensors
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

        // Advance the optimizer's global step once per batch, before the per-layer updates, so
        // step-dependent optimizers (Adam) use a single consistent timestep across all layers.
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.step();
        }

        // Backward pass and parameter updates (iterate through layers in reverse)
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
            if let Some(ref mut optimizer) = self.optimizer {
                optimizer.update(&mut **layer);
            }
        }

        Ok(loss_value)
    }

    /// Trains the model on the provided data.
    ///
    /// Executes the forward pass, loss calculation, backward pass, and parameter updates.
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

    /// Trains the model using batch processing.
    ///
    /// Splits data into batches of specified size for training, automatically prints
    /// detailed information for each batch by default.
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

        // Creates batch tensors from the full dataset
        let create_batch_tensors =
            |x: &Tensor, y: &Tensor, indices: &[usize]| -> Result<(Tensor, Tensor), Error> {
                let batch_size = indices.len();

                // Get shapes for batch tensors
                let mut x_batch_shape = x.shape().to_vec();
                x_batch_shape[0] = batch_size;

                let mut y_batch_shape = y.shape().to_vec();
                y_batch_shape[0] = batch_size;

                // Create batch arrays
                let mut x_batch_data = Vec::new();
                let mut y_batch_data = Vec::new();

                // Extract data for selected indices. Index along axis 0 so this works for any
                // input rank (Dense=2D, Conv1D/RNN=3D, Conv2D=4D, Conv3D=5D), not just 2D.
                for &idx in indices {
                    // Extract sample from x
                    let x_sample = x.index_axis(Axis(0), idx);
                    x_batch_data.extend(x_sample.iter().cloned());

                    // Extract sample from y
                    let y_sample = y.index_axis(Axis(0), idx);
                    y_batch_data.extend(y_sample.iter().cloned());
                }

                // Create batch tensors
                let x_batch = Array::from_shape_vec(IxDyn(&x_batch_shape), x_batch_data)
                    .context("create batch tensor for x")?;

                let y_batch = Array::from_shape_vec(IxDyn(&y_batch_shape), y_batch_data)
                    .context("create batch tensor for y")?;

                Ok((x_batch, y_batch))
            };

        // Create sample indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Seed the per-epoch shuffle once; `None` consults the thread-local global seed.
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

        // Training loop
        #[cfg(feature = "show_progress")]
        for epoch in 0..epochs {
            // Shuffle data at the beginning of each epoch
            indices.shuffle(&mut shuffle_rng);

            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Process data in batches
            for batch_indices in indices.chunks(batch_size) {
                batch_count += 1;

                // Create batch tensors
                let (batch_x, batch_y) = create_batch_tensors(x, y, batch_indices)?;

                // Train on this batch
                let batch_loss = self.train_batch(&batch_x, &batch_y)?;
                epoch_loss += batch_loss;

                // Update progress bar
                let avg_loss = epoch_loss / batch_count as f32;
                progress_bar.set_message(format!(
                    "{}/{} | Avg Loss: {:.6}",
                    epoch + 1,
                    epochs,
                    avg_loss
                ));
                progress_bar.inc(1);
            }
        }

        #[cfg(not(feature = "show_progress"))]
        for _ in 0..epochs {
            // Shuffle data at the beginning of each epoch
            indices.shuffle(&mut shuffle_rng);

            // Process data in batches
            for batch_indices in indices.chunks(batch_size) {
                // Create batch tensors
                let (batch_x, batch_y) = create_batch_tensors(x, y, batch_indices)?;

                // Train on this batch
                let _ = self.train_batch(&batch_x, &batch_y)?;
            }
        }

        // Finish progress bar
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Training completed");

        Ok(self)
    }

    /// Generates predictions for the input data.
    ///
    /// Only performs forward pass without any training.
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

        // Inference path: each layer's `predict` runs in eval mode and writes no caches, so this
        // only needs `&self` (the model can be shared across threads for concurrent inference).
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

    /// Prints a summary of the model's structure.
    ///
    /// Displays each layer's information and parameter statistics in a tabular format to stdout.
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

            // Generate name from the layer type with a per-type index.
            let count = type_counts.entry(layer_type).or_insert(0);
            let layer_name = if *count == 0 {
                layer_type.to_lowercase()
            } else {
                format!("{}_{}", layer_type.to_lowercase(), count)
            };
            *count += 1;

            let out_shape = layer.output_shape();

            // Exhaustive match so adding a TrainingParameters variant forces a compile error
            // instead of silently being counted as zero.
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

    /// Returns all the weights from each layer in the model.
    ///
    /// This method collects the weights from all layers in the sequential model and returns them
    /// as a vector of `LayerWeight` enums. Each `LayerWeight` contains references to the weight
    /// matrices and bias vectors of its corresponding layer.
    ///
    /// # Returns
    ///
    /// - `Vec<LayerWeight<'_>>` - A vector containing weight references for each layer in the model
    pub fn get_weights(&self) -> Vec<LayerWeight<'_>> {
        let mut weights = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            weights.push(layer.get_weights());
        }
        weights
    }

    /// Saves the model architecture and weights to a JSON file at the specified path.
    ///
    /// This method serializes the model structure including layer types, configurations,
    /// and all trainable parameters (weights and biases) to JSON format. Note that the
    /// optimizer and loss function are not saved and must be reconfigured after loading.
    ///
    /// # Parameters
    ///
    /// - `path` - File path where the model will be saved (e.g., "stored_model.json"). Accepts
    ///   anything convertible to a `Path` (`&str`, `String`, `Path`, `PathBuf`, ...).
    ///
    /// # Returns
    ///
    /// - `crate::error::RustymlResult<()>` - Ok if the model is saved, or an IO/serialization error
    ///
    /// # Errors
    ///
    /// - `Error::Io(IoError::Std)` - File creation or write operation failed
    /// - `Error::Io(IoError::Json)` - Serialization to JSON failed
    pub fn save_to_path(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::RustymlResult<()> {
        // Convert layers to serializable format
        let serializable_layers = self
            .layers
            .iter()
            .map(|layer| {
                let weights = layer.get_weights();
                let layer_info = LayerInfo {
                    layer_type: layer.layer_type().to_string(),
                    output_shape: layer.output_shape(),
                };

                SerializableLayer {
                    info: layer_info,
                    weights: SerializableLayerWeight::from_layer_weight(&weights),
                }
            })
            .collect();

        let serializable_model = SerializableSequential {
            layers: serializable_layers,
        };

        // Create or overwrite the file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Serialize the model to JSON and write to file
        to_writer_pretty(&mut writer, &serializable_model)?;

        // Ensure all data is written to disk
        writer.flush()?;

        Ok(())
    }

    /// Loads model weights from a JSON file and applies them to the current model.
    ///
    /// This method deserializes weights from a previously saved model file and applies them
    /// to the current model's layers. The current model must have the same architecture
    /// (same number and types of layers) as the saved model.
    ///
    /// Note: You must build the model structure first and then call this method to load weights.
    /// After loading, you should call `compile()` to set the optimizer and loss function.
    ///
    /// # Parameters
    ///
    /// - `path` - File path from which to load the weights (e.g., "stored_model.json"). Accepts
    ///   anything convertible to a `Path` (`&str`, `String`, `Path`, `PathBuf`, ...).
    ///
    /// # Returns
    ///
    /// - `crate::error::RustymlResult<()>` - Ok if weights are loaded, or an IO/deserialization error
    ///
    /// # Errors
    ///
    /// - `Error::Io(IoError::Std)` - File not found or read operation failed
    /// - `Error::Io(IoError::Json)` - Deserialization from JSON failed
    /// - `Error::Io(IoError::ModelStructureMismatch)` - The current model's structure (layer count, a layer
    ///   type at some position, or a weight shape) does not match the saved model
    pub fn load_from_path(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::RustymlResult<()> {
        // Open and buffer the file for reading
        let reader = IoError::load_in_buf_reader(path)?;

        // Deserialize the model from JSON
        let serializable_model: SerializableSequential = from_reader(reader)?;

        // Verify layer count matches
        if serializable_model.layers.len() != self.layers.len() {
            return Err(Error::Io(IoError::ModelStructureMismatch(format!(
                "layer count mismatch: model has {} layers, file has {} layers",
                self.layers.len(),
                serializable_model.layers.len()
            ))));
        }

        // Apply weights to each layer, verifying the layer type at each position first.
        // This catches architecture drift (e.g. a pooling layer where a Dense was saved, or
        // two parameter-less layers swapped) that weight application alone cannot detect.
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
