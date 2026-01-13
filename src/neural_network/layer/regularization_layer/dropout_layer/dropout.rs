use super::*;

/// Threshold for using parallel computation in Dropout layer.
/// When the total number of elements >= this threshold, parallel computation is used.
const DROPOUT_PARALLEL_THRESHOLD: usize = 10000;

/// Dropout layer for neural networks, which randomly sets a fraction of input units to 0
/// at each update during training time, which helps prevent overfitting.
///
/// Dropout is a regularization technique that randomly drops (sets to zero) a proportion
/// of neurons during training. This prevents the network from becoming too dependent on
/// any particular neuron and improves generalization.
///
/// # Fields
///
/// - `rate` - Dropout rate: fraction of the input units to drop (between 0 and 1).
/// - `input_shape` - Shape of the input tensor.
/// - `mask` - Binary mask used during training to determine which neurons to drop.
/// - `training` - Whether the layer is in training mode or inference mode.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a Dropout layer with 50% dropout rate
/// let mut dropout = Dropout::new(0.5, vec![32, 128]);
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, approximately 50% of values will be set to 0
/// let output = dropout.forward(&input).unwrap();
/// ```
pub struct Dropout {
    rate: f32,
    input_shape: Vec<usize>,
    mask: Option<Tensor>,
    training: bool,
}

impl Dropout {
    /// Creates a new Dropout layer.
    ///
    /// # Parameters
    ///
    /// - `rate` - Dropout rate: fraction of the input units to drop. Must be between 0 and 1.
    /// - `input_shape` - Shape of the input tensor.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the Dropout layer.
    ///
    /// # Panics
    ///
    /// Panics if rate is not between 0 and 1.
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Self {
        Dropout {
            rate,
            input_shape,
            mask: None,
            training: true,
        }
    }

    mode_dependent_layer_set_training!();
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        if self.rate < 0.0 || self.rate > 1.0 {
            return Err(ModelError::InputValidationError(
                "Dropout rate cannot be less than 0 or greater than 1".to_string(),
            ));
        }

        // Validate input shape matches expected shape
        if !self.input_shape.is_empty() && input.shape() != self.input_shape.as_slice() {
            return Err(ModelError::InputValidationError(format!(
                "Input shape mismatch: expected {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )));
        }

        if !self.training {
            // During inference, pass input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            eprintln!("Dropout rate is 0.0, so this layer has no effect on the output.");
            return Ok(input.clone());
        }

        if self.rate == 1.0 {
            eprintln!("Dropout rate is 1.0, so this layer will return all zeros.");
            // If dropout rate is 1.0, return zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        // Generate random mask: 1 with probability (1 - rate), 0 with probability rate
        let mut mask = Tensor::random(input.raw_dim(), Uniform::new(0.0, 1.0).unwrap());

        // Apply threshold to create binary mask with parallel or sequential computation
        if input.len() >= DROPOUT_PARALLEL_THRESHOLD {
            mask.par_mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        } else {
            mask.mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        }

        // Apply mask and scale by (1 - rate) to maintain expected value
        // This is "inverted dropout" technique
        let scale = 1.0 / (1.0 - self.rate);
        let output = input * &mask * scale;

        // Store mask for backpropagation
        self.mask = Some(mask);

        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        dropout_backward(grad_output, &self.mask, self.training, self.rate)
    }

    fn layer_type(&self) -> &str {
        "Dropout"
    }

    fn output_shape(&self) -> String {
        dropout_output_shape(&self.input_shape)
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
