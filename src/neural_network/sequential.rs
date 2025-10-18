use super::*;
use crate::error::IoError;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
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
/// - `layers` - A vector containing all the layers in the model. Each layer implements
///   the `Layer` trait and is stored as a boxed dynamic trait object.
///
/// - `optimizer` - An optional optimizer used for updating model parameters during training.
///   Common optimizers include SGD, Adam, and RMSprop.
///
/// - `loss` - An optional loss function used to compute the training loss. Examples include
///   mean squared error for regression and categorical crossentropy for classification.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array;
///
/// // Create training data
/// let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features
/// let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
///
/// // Build a neural network
/// let mut model = Sequential::new();
/// model
///     .add(Dense::new(784, 128, ReLU::new()))
///     .add(Dense::new(128, 64, ReLU::new()))
///     .add(Dense::new(64, 10, Softmax::new()))
///     .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), CategoricalCrossEntropy::new());
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
///     .add(Dense::new(784, 128, ReLU::new()))
///     .add(Dense::new(128, 64, ReLU::new()))
///     .add(Dense::new(64, 10, Softmax::new()));
///
/// // Load weights from file
/// new_model.load_from_path("model.json").unwrap();
///
/// // Compile before using (required for training, optional for prediction)
/// new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8), CategoricalCrossEntropy::new());
///
/// // Make predictions with loaded model
/// let predictions = new_model.predict(&x);
/// println!("Predictions shape: {:?}", predictions.shape());
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    loss: Option<Box<dyn LossFunction>>,
}

impl Sequential {
    /// Creates a new empty Sequential model
    ///
    /// # Returns
    ///
    /// * `Sequential` - an empty Sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            loss: None,
        }
    }

    /// Adds a layer to the model
    ///
    /// Supports method chaining pattern
    ///
    /// # Parameters
    ///
    /// * `layer` - The layer to add to the model
    ///
    /// # Returns
    ///
    /// * `&mut Sequential` - Mutable reference to self for method chaining
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
    /// * `&mut Sequential` - Mutable reference to self for method chaining
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
        // Forward pass
        let mut output = x.clone();
        for layer in &mut self.layers {
            layer.set_training_if_mode_dependent(true);

            output = match layer.forward(&output) {
                Ok(output) => output,
                Err(e) => return Err(e),
            };
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

    /// Trains the model on the provided data
    ///
    /// Executes the forward pass, loss calculation, backward pass, and parameter updates
    ///
    /// # Parameters
    ///
    /// - `x` - Input tensor containing training data
    /// - `y` - Target tensor containing expected outputs
    /// - `epochs` - Number of training epochs to perform
    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: u32) -> Result<&mut Self, ModelError> {
        // Validate inputs
        self.validate_training_inputs(x, y)?;

        let n_samples = x.shape()[0];

        // Create progress bar for training epochs
        let progress_bar = ProgressBar::new(epochs as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Loss: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );

        for _ in 0..epochs {
            // Train on the entire dataset as one batch
            let loss_value = self.train_batch(x, y)?;

            // Update progress bar with current loss
            progress_bar.set_message(format!("{:.6}", loss_value));
            progress_bar.inc(1);
        }

        // Finish progress bar
        progress_bar.finish_with_message("Training completed");

        println!(
            "\nNeural network training completed: {} samples, {} epochs",
            n_samples, epochs
        );

        Ok(self)
    }

    /// Trains the model using batch processing
    ///
    /// Splits data into batches of specified size for training, automatically prints
    /// detailed information for each batch by default
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
    /// * `Result<&mut Self, ModelError>` - Mutable reference to trained model or error
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

        let total_batches = (n_samples + batch_size - 1) / batch_size;
        let total_iterations = epochs as u64 * total_batches as u64;

        // Create progress bar for batch training
        let progress_bar = ProgressBar::new(total_iterations);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Epoch {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );

        // Training loop
        for epoch in 0..epochs {
            // Shuffle data at the beginning of each epoch
            indices.shuffle(&mut rand::rng());

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

        // Finish progress bar
        progress_bar.finish_with_message("Training completed");

        println!(
            "\nNeural network batch training completed: {} samples, {} batch size, {} epochs",
            n_samples, batch_size, epochs
        );

        Ok(self)
    }

    /// Generates predictions for the input data
    ///
    /// Only performs forward pass without any training
    ///
    /// # Parameters
    ///
    /// * `x` - Input tensor containing data to predict on
    ///
    /// # Returns
    ///
    /// * `Tensor` - Tensor containing the model's predictions
    pub fn predict(&mut self, x: &Tensor) -> Tensor {
        // Input validation
        if x.is_empty() {
            panic!("Input tensor cannot be empty");
        }

        let mut output = x.clone();
        for layer in &mut self.layers {
            layer.set_training_if_mode_dependent(false);

            output = match layer.forward(&output) {
                Ok(output) => output,
                Err(e) => panic!("Failed to forward pass: {}", e),
            };
        }
        output
    }

    /// Prints a summary of the model's structure
    ///
    /// Displays each layer's information and parameter statistics in a tabular format
    pub fn summary(&self) {
        let col1_width = 33;
        let col2_width = 24;
        let col3_width = 15;
        println!("Model: \"sequential\"");
        println!(
            "┏{}┳{}┳{}┓",
            "━".repeat(col1_width),
            "━".repeat(col2_width),
            "━".repeat(col3_width)
        );
        println!(
            "┃ {:<31} ┃ {:<22} ┃ {:>13} ┃",
            "Layer (type)", "Output Shape", "Param #"
        );
        println!(
            "┡{}╇{}╇{}┩",
            "━".repeat(col1_width),
            "━".repeat(col2_width),
            "━".repeat(col3_width)
        );
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

            println!(
                "│ {:<31} │ {:<22} │ {:>13} │",
                format!("{} ({})", layer_name, layer.layer_type()),
                out_shape,
                param_count_num
            );
        }
        println!(
            "└{}┴{}┴{}┘",
            "─".repeat(col1_width),
            "─".repeat(col2_width),
            "─".repeat(col3_width)
        );
        println!(" Total params: {} ({} B)", total_params, total_params * 4); // Using f32, each parameter is 4 bytes
        println!(
            " Trainable params: {} ({} B)",
            trainable_param_count,
            trainable_param_count * 4
        );
        println!(
            " Non-trainable params: {} ({} B)",
            non_trainable_param_count,
            non_trainable_param_count * 4
        );
    }

    /// Returns all the weights from each layer in the model.
    ///
    /// This method collects the weights from all layers in the sequential model and returns them
    /// as a vector of `LayerWeight` enums. Each `LayerWeight` contains references to the weight
    /// matrices and bias vectors of its corresponding layer.
    ///
    /// # Returns
    ///
    /// * `Vec<LayerWeight>` - A vector containing weight references for each layer in the model.
    ///   The type of each `LayerWeight` depends on the layer type:
    ///   - `LayerWeight::Dense` for Dense layers with weight and bias
    ///   - `LayerWeight::SimpleRNN` for SimpleRNN layers with kernel, recurrent_kernel, and bias
    ///   - `LayerWeight::LSTM` for LSTM layers with weights for input, forget, cell, and output gates
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
    /// * `path` - File path where the model will be saved (e.g., "stored_model.json")
    ///
    /// # Returns
    ///
    /// - `Ok(())` - Model successfully saved to file
    /// - `Err(IoError::StdIoError)` - File creation or write operation failed
    /// - `Err(IoError::JsonError)` - Serialization to JSON failed
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
    /// * `path` - File path from which to load the weights (e.g., "stored_model.json")
    ///
    /// # Returns
    ///
    /// - `Ok(())` - Successfully loaded weights into the model
    /// - `Err(IoError::StdIoError)` - File not found or read operation failed
    /// - `Err(IoError::JsonError)` - Deserialization from JSON failed
    /// - `Err(IoError::ModelStructureMismatch)` - Model structure doesn't match saved weights
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
