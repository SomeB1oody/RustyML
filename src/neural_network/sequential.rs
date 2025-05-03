use super::*;

/// Sequential model: Mimics the Keras API style, supporting chained calls to add, compile, and fit methods
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    loss: Option<Box<dyn LossFunction>>,
}

impl Sequential {
    /// Creates a new empty Sequential model
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
    /// * `&mut Self` - Mutable reference to self for method chaining
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
    /// * `&mut Self` - Mutable reference to self for method chaining
    pub fn compile<O, LFunc>(&mut self, optimizer: O, loss: LFunc) -> &mut Self
    where
        O: 'static + Optimizer,
        LFunc: 'static + LossFunction,
    {
        self.optimizer = Some(Box::new(optimizer));
        self.loss = Some(Box::new(loss));
        self
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
        if self.optimizer.is_none() {
            return Err(ModelError::InputValidationError(
                "Optimizer not specified".to_string(),
            ));
        }

        if self.layers.is_empty() {
            return Err(ModelError::InputValidationError(
                "Layers not specified".to_string(),
            ));
        }

        for epoch in 0..epochs {
            println!("Epoch {}", epoch + 1);
            // Forward pass
            let mut output = x.clone();
            for layer in &mut self.layers {
                output = layer.forward(&output);
            }
            // Calculate loss
            let loss_value = self.loss.as_ref().unwrap().compute_loss(y, &output);
            println!("Loss: {}", loss_value);
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
            // Generate name for each layer: first layer is named "dense", then "dense_1", "dense_2", etc.
            let layer_name = if i == 0 {
                "dense".to_string()
            } else {
                format!("dense_{}", i)
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
}
