//! Dropout layer that randomly zeroes input units during training to reduce overfitting

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::dropout::{
    dropout_backward, dropout_output_shape,
};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    validate_input_shape, validate_rate,
};
use crate::neural_network::traits::Layer;
use crate::parallel_gates::CHEAP_MAP_PARALLEL_THRESHOLD;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Dropout layer for neural networks
///
/// Randomly drops a fraction of input units during training to reduce overfitting and
/// improve generalization
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a Dropout layer with 50% dropout rate
/// let mut dropout = Dropout::new(0.5, vec![32, 128], None).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, approximately 50% of values will be set to 0
/// let output = dropout.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct Dropout {
    /// Fraction of input units to drop (between 0 and 1)
    rate: f32,
    /// Expected shape of the input tensor
    input_shape: Vec<usize>,
    /// Binary mask from the last training forward pass, reused in backward
    mask: Option<Tensor>,
    /// Whether the layer is in training mode (true) or inference mode (false)
    training: bool,
    /// Random number generator used to sample the dropout mask
    rng: StdRng,
}

impl Dropout {
    /// Creates a new Dropout layer
    ///
    /// # Parameters
    ///
    /// - `rate` - Fraction of the input units to drop (between 0 and 1)
    /// - `input_shape` - Shape of the input tensor
    /// - `random_state` - Optional seed for reproducible masks; falls back to the global seed or entropy. See `crate::random`
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New Dropout layer instance, or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not between 0 and 1
    pub fn new(
        rate: f32,
        input_shape: Vec<usize>,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(Dropout {
            rate,
            input_shape,
            mask: None,
            training: true,
            rng: crate::random::make_rng(random_state),
        })
    }

    mode_dependent_layer_set_training!();
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` was validated in `new()`
        validate_input_shape(input.shape(), &self.input_shape)?;

        if !self.training {
            // Inference passes the input through unchanged
            return Ok(input.clone());
        }

        if self.rate == 0.0 {
            return Ok(input.clone());
        }

        if self.rate == 1.0 {
            // Dropping every unit yields all zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        // Sample a uniform value per element
        let mut mask = Tensor::random_using(
            input.raw_dim(),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut self.rng,
        );

        // Threshold into a binary mask, in parallel for large inputs
        if input.len() >= CHEAP_MAP_PARALLEL_THRESHOLD {
            mask.par_mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        } else {
            mask.mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        }

        // Inverted dropout: scale kept units by 1 / (1 - rate) to preserve the expected value
        let scale = 1.0 / (1.0 - self.rate);
        let output = input * &mask * scale;

        // Cache the mask for backpropagation
        self.mask = Some(mask);

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // `rate` was validated in `new()`
        validate_input_shape(input.shape(), &self.input_shape)?;

        // Inverted dropout passes the input through unchanged during inference
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        dropout_backward(grad_output, &self.mask, self.training, self.rate, "Dropout")
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
