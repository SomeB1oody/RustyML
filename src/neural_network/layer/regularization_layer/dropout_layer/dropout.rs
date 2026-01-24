use super::*;

/// Threshold for using parallel computation in Dropout layer.
/// When the total number of elements >= this threshold, parallel computation is used.
const DROPOUT_PARALLEL_THRESHOLD: usize = 10000;

/// Dropout layer for neural networks.
///
/// Randomly drops a fraction of input units during training to reduce overfitting and
/// improve generalization.
///
/// # Fields
///
/// - `rate` - Dropout rate, fraction of input units to drop (between 0 and 1)
/// - `input_shape` - Expected shape of the input tensor
/// - `mask` - Binary mask used during training to determine which units to drop
/// - `training` - Whether the layer is in training mode or inference mode
///
/// # Examples
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::Array2;
///
/// // Create a Dropout layer with 50% dropout rate
/// let mut dropout = Dropout::new(0.5, vec![32, 128]).unwrap();
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
    /// - `rate` - Dropout rate, fraction of the input units to drop (between 0 and 1)
    /// - `input_shape` - Shape of the input tensor
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - New Dropout layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `rate` is not between 0 and 1
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, ModelError> {
        validate_rate(rate, "Dropout rate")?;

        Ok(Dropout {
            rate,
            input_shape,
            mask: None,
            training: true,
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, ModelError> {
        validate_rate(self.rate, "Dropout rate")?;
        validate_input_shape(input.shape(), &self.input_shape)?;

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
