use super::*;
use ndarray::{Array, IxDyn};
use rand::seq::SliceRandom;

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
///     .add(Dense::new(784, 128, Activation::ReLU))
///     .add(Dense::new(128, 64, Activation::ReLU))
///     .add(Dense::new(64, 10, Activation::Softmax))
///     .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), CategoricalCrossEntropy::new());
///
/// // Display model structure
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 10).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&x);
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
            output = layer.forward(&output);
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

        for epoch in 0..epochs {
            println!("Epoch {}", epoch + 1);

            // Train on the entire dataset as one batch
            let loss_value = self.train_batch(x, y)?;
            println!("Loss: {}", loss_value);
        }

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
                    let x_sample = x.slice(ndarray::s![idx, ..]);
                    x_batch_data.extend(x_sample.iter().cloned());

                    // Extract sample from y
                    let y_sample = y.slice(ndarray::s![idx, ..]);
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

        // Training loop
        for epoch in 0..epochs {
            println!("Epoch {}/{}", epoch + 1, epochs);

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

                println!(
                    "  Batch {}/{}: Loss = {:.6}",
                    batch_count,
                    (n_samples + batch_size - 1) / batch_size, // Ceiling division for total batches
                    batch_loss
                );
            }

            // Print epoch summary
            let avg_loss = epoch_loss / batch_count as f32;
            println!("  Average Loss: {}", avg_loss);
            println!();
        }

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
            output = layer.forward(&output);
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
        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            // Generate name for each layer: first layer is named "Layer", then "Layer_1", "Layer_2", etc.
            let layer_name = if i == 0 {
                "Layer".to_string()
            } else {
                format!("Layer_{}", i)
            };
            let out_shape = layer.output_shape();
            let param_count = layer.param_count();
            total_params += param_count;
            println!(
                "│ {:<31} │ {:<22} │ {:>13} │",
                format!("{} ({})", layer_name, layer.layer_type()),
                out_shape,
                param_count
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
            total_params,
            total_params * 4
        );
        println!(" Non-trainable params: 0 (0 B)");
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
}
