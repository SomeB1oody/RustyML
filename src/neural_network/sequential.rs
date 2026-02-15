use super::neural_network_trait::{Layer, LossFunction, Optimizer};
use crate::error::{IoError, ModelError};
use crate::neural_network::Tensor;
use crate::neural_network::layer::TrainingParameters;
use crate::neural_network::layer::layer_weight::LayerWeight;
use crate::neural_network::layer::serialize_weight::{
    LayerInfo, SerializableLayer, SerializableLayerWeight, SerializableSequential,
    apply_weights_to_layer,
};
use ndarray::{Array, IxDyn, s};
use ndarray_rand::rand::{rng, seq::SliceRandom};
use serde_json::{from_reader, to_writer_pretty};
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
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::{
///     sequential::Sequential,
///     layer::{Dense, ReLU, Softmax},
///     optimizer::Adam,
///     loss_function::CategoricalCrossEntropy,
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
///     .add(Dense::new(784, 128, ReLU::new()).unwrap())
///     .add(Dense::new(128, 64, ReLU::new()).unwrap())
///     .add(Dense::new(64, 10, Softmax::new()).unwrap())
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
///     .add(Dense::new(784, 128, ReLU::new()).unwrap())
///     .add(Dense::new(128, 64, ReLU::new()).unwrap())
///     .add(Dense::new(64, 10, Softmax::new()).unwrap());
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
    loss: Option<Box<dyn LossFunction>>,
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
        }
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
        LFunc: 'static + LossFunction,
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
    /// - `Err(ModelError)` - If validation fails
    fn validate_training_inputs(&self, x: &Tensor, y: &Tensor) -> Result<(), ModelError> {
        if self.optimizer.is_none() {
            return Err(ModelError::InputValidationError(
                "Optimizer not specified".to_string(),
            ));
        }

        if self.loss.is_none() {
            return Err(ModelError::InputValidationError(
                "Loss function not specified".to_string(),
            ));
        }

        if self.layers.is_empty() {
            return Err(ModelError::InputValidationError(
                "Layers not specified".to_string(),
            ));
        }

        // Input shape validation
        if x.is_empty() || y.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input tensors cannot be empty".to_string(),
            ));
        }

        // Verify batch size match
        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::InputValidationError(format!(
                "Batch size mismatch: input has {} samples, target has {} samples",
                x.shape()[0],
                y.shape()[0]
            )));
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
    /// - `Err(ModelError)` - If training fails
    fn train_batch(&mut self, x: &Tensor, y: &Tensor) -> Result<f32, ModelError> {
        // Forward pass - first layer takes input reference, subsequent layers take owned tensors
        let mut layers_iter = self.layers.iter_mut();
        let first_layer = layers_iter
            .next()
            .ok_or_else(|| ModelError::InputValidationError("No layers in model".to_string()))?;
        first_layer.set_training_if_mode_dependent(true);
        let mut output = first_layer.forward(x)?;

        for layer in layers_iter {
            layer.set_training_if_mode_dependent(true);
            output = layer.forward(&output)?;
        }

        // Calculate loss
        let loss_value = self.loss.as_ref().unwrap().compute_loss(y, &output);

        // Calculate gradient of loss with respect to output
        let mut grad = self.loss.as_ref().unwrap().compute_grad(y, &output);

        // Backward pass and parameter updates (iterate through layers in reverse)
        for layer in self.layers.iter_mut().rev() {
            grad = match layer.backward(&grad) {
                Ok(grad) => grad,
                Err(e) => return Err(e),
            };
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
    /// - `Result<&mut Self, ModelError>` - Mutable reference to self after training or an error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the model is not compiled, has no layers, or inputs are invalid
    /// - `ModelError::ProcessingError` - If a layer fails during forward or backward pass
    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: u32) -> Result<&mut Self, ModelError> {
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
    /// - `Result<&mut Self, ModelError>` - Mutable reference to trained model or error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the model is not compiled, has no layers, inputs are invalid, or batch size is invalid
    /// - `ModelError::ProcessingError` - If a layer fails during forward or backward pass
    pub fn fit_with_batches(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: u32,
        batch_size: usize,
    ) -> Result<&mut Self, ModelError> {
        // Validate inputs
        self.validate_training_inputs(x, y)?;

        let n_samples = x.shape()[0];

        // Validate batch size
        if batch_size == 0 {
            return Err(ModelError::InputValidationError(
                "Batch size must be greater than 0".to_string(),
            ));
        }

        if batch_size > n_samples {
            return Err(ModelError::InputValidationError(format!(
                "Batch size ({}) cannot be larger than dataset size ({})",
                batch_size, n_samples
            )));
        }

        // Creates batch tensors from the full dataset
        let create_batch_tensors =
            |x: &Tensor, y: &Tensor, indices: &[usize]| -> Result<(Tensor, Tensor), ModelError> {
                let batch_size = indices.len();

                // Get shapes for batch tensors
                let mut x_batch_shape = x.shape().to_vec();
                x_batch_shape[0] = batch_size;

                let mut y_batch_shape = y.shape().to_vec();
                y_batch_shape[0] = batch_size;

                // Create batch arrays
                let mut x_batch_data = Vec::new();
                let mut y_batch_data = Vec::new();

                // Extract data for selected indices
                for &idx in indices {
                    // Extract sample from x
                    let x_sample = x.slice(s![idx, ..]);
                    x_batch_data.extend(x_sample.iter().cloned());

                    // Extract sample from y
                    let y_sample = y.slice(s![idx, ..]);
                    y_batch_data.extend(y_sample.iter().cloned());
                }

                // Create batch tensors
                let x_batch =
                    Array::from_shape_vec(IxDyn(&x_batch_shape), x_batch_data).map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to create batch tensor for x: {}",
                            e
                        ))
                    })?;

                let y_batch =
                    Array::from_shape_vec(IxDyn(&y_batch_shape), y_batch_data).map_err(|e| {
                        ModelError::ProcessingError(format!(
                            "Failed to create batch tensor for y: {}",
                            e
                        ))
                    })?;

                Ok((x_batch, y_batch))
            };

        // Create sample indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();

        #[cfg(feature = "show_progress")]
        let total_batches = (n_samples + batch_size - 1) / batch_size;
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
            indices.shuffle(&mut rng());

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
            indices.shuffle(&mut rng());

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
    /// - `Result<Tensor, ModelError>` - Tensor containing the model's predictions or an error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `x` is empty or the model has no layers
    /// - `ModelError::ProcessingError` - If any layer fails during forward pass
    pub fn predict(&mut self, x: &Tensor) -> Result<Tensor, ModelError> {
        // Input validation
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input tensor cannot be empty".to_string(),
            ));
        }

        let mut layers_iter = self.layers.iter_mut();
        let first_layer = layers_iter
            .next()
            .ok_or_else(|| ModelError::InputValidationError("Model has no layers".to_string()))?;
        first_layer.set_training_if_mode_dependent(false);
        let mut output = first_layer.forward(x)?;

        for layer in layers_iter {
            layer.set_training_if_mode_dependent(false);
            output = layer.forward(&output)?;
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

        for (i, layer) in self.layers.iter().enumerate() {
            // Generate name for each layer: first layer is named "Layer", then "Layer_1", "Layer_2", etc.
            let layer_name = if i == 0 {
                "Layer".to_string()
            } else {
                format!("Layer_{}", i)
            };
            let out_shape = layer.output_shape();
            let param_count = layer.param_count();
            let param_count_num: usize;

            match param_count {
                TrainingParameters::Trainable(count) => {
                    trainable_param_count += count;
                    total_params += count;
                    param_count_num = count;
                }
                TrainingParameters::NonTrainable(count) => {
                    non_trainable_param_count += count;
                    total_params += count;
                    param_count_num = count;
                }
                _ => {
                    param_count_num = 0;
                }
            };

            output.push_str(&format!(
                "│ {:<31} │ {:<22} │ {:>13} │\n",
                format!("{} ({})", layer_name, layer.layer_type()),
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
    /// - `path` - File path where the model will be saved (e.g., "stored_model.json")
    ///
    /// # Returns
    ///
    /// - `Result<(), IoError>` - Ok if the model is saved, or an IO/serialization error
    ///
    /// # Errors
    ///
    /// - `IoError::StdIoError` - File creation or write operation failed
    /// - `IoError::JsonError` - Serialization to JSON failed
    pub fn save_to_path(&self, path: &str) -> Result<(), IoError> {
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
        let file = File::create(path).map_err(IoError::StdIoError)?;
        let mut writer = BufWriter::new(file);

        // Serialize the model to JSON and write to file
        to_writer_pretty(&mut writer, &serializable_model).map_err(IoError::JsonError)?;

        // Ensure all data is written to disk
        writer.flush().map_err(IoError::StdIoError)?;

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
    /// - `path` - File path from which to load the weights (e.g., "stored_model.json")
    ///
    /// # Returns
    ///
    /// - `Result<(), IoError>` - Ok if weights are loaded, or an IO/deserialization error
    ///
    /// # Errors
    ///
    /// - `IoError::StdIoError` - File not found or read operation failed
    /// - `IoError::JsonError` - Deserialization from JSON failed
    /// - `IoError::ModelStructureMismatch` - Model structure doesn't match saved weights
    pub fn load_from_path(&mut self, path: &str) -> Result<(), IoError> {
        // Open and buffer the file for reading
        let reader = IoError::load_in_buf_reader(path)?;

        // Deserialize the model from JSON
        let serializable_model: SerializableSequential =
            from_reader(reader).map_err(IoError::JsonError)?;

        // Verify layer count matches
        if serializable_model.layers.len() != self.layers.len() {
            return Err(IoError::StdIoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Layer count mismatch: model has {} layers, file has {} layers",
                    self.layers.len(),
                    serializable_model.layers.len()
                ),
            )));
        }

        // Apply weights to each layer
        for (i, serializable_layer) in serializable_model.layers.iter().enumerate() {
            apply_weights_to_layer(
                &mut *self.layers[i],
                &serializable_layer.weights,
                &serializable_layer.info.layer_type,
            )?;
        }

        Ok(())
    }
}
