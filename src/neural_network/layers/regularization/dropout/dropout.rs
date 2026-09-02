//! Dropout layer that randomly zeroes input units during training to reduce overfitting

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::layers::regularization::dropout::{
    broadcast_dropout_scale, dropout_backward,
};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::validation::{
    shape_preserving_output, validate_input_shape, validate_rate,
};
use crate::neural_network::traits::Layer;
use crate::neural_network::{Shape, Tensor};
use crate::parallel_gates::cheap_map_parallel_threshold;
use ndarray::IxDyn;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Dropout layer for neural networks
///
/// Randomly drops a fraction of input units during training to reduce overfitting and
/// improve generalization
///
/// By default the layer draws 1 independent value per input element. [`Dropout::with_noise_shape`]
/// changes the shape of the drawn mask, so an axis can carry 1 shared draw instead of 1 draw per
/// position
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a Dropout layer with 50% dropout rate
/// let mut dropout = Dropout::new(0.5, vec![32, 128]).unwrap();
///
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, about 50% of values are set to 0
/// let output = dropout.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct Dropout {
    /// Fraction of input units to drop (between 0 and 1)
    rate: f32,
    /// Expected shape of the input tensor
    input_shape: Vec<usize>,
    /// Shape of the random mask, or `None` for 1 independent draw per input element
    ///
    /// An entry of `None` takes the extent of the input on that axis, and an entry of 1 makes
    /// the axis share 1 draw. A shorter vector lines up against the last axes of the input
    noise_shape: Option<Vec<Option<usize>>>,
    /// Binary mask from the last training forward pass, reused in backward
    ///
    /// The mask is at the resolved noise shape, which is the input shape unless
    /// [`Dropout::with_noise_shape`] set a smaller one. The layer never builds a full-size copy
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
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New Dropout layer instance, or a validation error
    ///
    /// # Notes
    ///
    /// By default, `new` seeds the mask RNG from the global seed or entropy. For reproducible
    /// masks, set a seed with [`Dropout::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `rate` is not between 0 and 1
    pub fn new(rate: f32, input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_rate(rate, "Dropout rate")?;

        Ok(Dropout {
            rate,
            input_shape,
            noise_shape: None,
            mask: None,
            training: true,
            rng: crate::random::make_rng(None),
        })
    }

    /// Sets the shape of the random mask, which then broadcasts up to the input
    ///
    /// By default the layer draws 1 independent value per input element. A `noise_shape` makes
    /// the draw coarser. An entry of 1 gives the whole axis 1 shared draw, so the same units
    /// drop at every position of that axis. An entry of `None` takes the extent of the input on
    /// that axis, which keeps the draws on that axis independent. Any other entry must equal the
    /// extent of the input on that axis
    ///
    /// A vector that is shorter than the rank of the input lines up against the last axes,
    /// which is the usual right-aligned broadcast rule. The axes that the vector leaves out
    /// therefore share 1 draw. A caller that omits the batch axis shares 1 mask across the whole
    /// batch
    ///
    /// The layer samples the mask at this shape, keeps it at this shape for the backward pass,
    /// and broadcasts it in both directions. It never builds a full-size copy
    ///
    /// # Parameters
    ///
    /// - `noise_shape` - Shape of the random mask, 1 entry per axis
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `noise_shape` is empty
    /// - `Error::InvalidParameter` - If any entry is `Some(0)`
    pub fn with_noise_shape(mut self, noise_shape: Vec<Option<usize>>) -> Result<Self, Error> {
        if noise_shape.is_empty() {
            return Err(Error::empty_input("noise_shape"));
        }
        if let Some(axis) = noise_shape.iter().position(|entry| *entry == Some(0)) {
            return Err(Error::invalid_parameter(
                "noise_shape",
                format!(
                    "every entry must be positive or None, got 0 at axis {}",
                    axis
                ),
            ));
        }

        self.noise_shape = Some(noise_shape);
        Ok(self)
    }

    /// Sets the seed for reproducible mask sampling
    ///
    /// By default, `new` seeds the RNG from the global seed or entropy (see [`crate::random`]).
    /// This method re-seeds it deterministically from `random_state`
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for the layer's random number generator
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.rng = crate::random::make_rng(Some(random_state));
        self
    }

    /// Resolves the configured noise shape against the shape of an actual input
    ///
    /// Without a `noise_shape` the answer is the input shape itself, which gives 1 draw per
    /// input element. With one, the vector lines up against the last axes of the input, and
    /// each entry becomes the input extent (`None`), 1, or the input extent again
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If the noise shape has a higher rank than the input, or if
    ///   an entry is neither 1 nor the extent of the input on its axis
    fn resolve_noise_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, Error> {
        let Some(noise_shape) = &self.noise_shape else {
            return Ok(input_shape.to_vec());
        };

        if noise_shape.len() > input_shape.len() {
            return Err(Error::invalid_parameter(
                "noise_shape",
                format!(
                    "has rank {}, which is more than the input rank {}",
                    noise_shape.len(),
                    input_shape.len()
                ),
            ));
        }

        // Right-aligned against the input, so the leading axes that the caller left out keep an
        // extent of 1 and share 1 draw
        let leading = input_shape.len() - noise_shape.len();
        let mut resolved = Vec::with_capacity(noise_shape.len());
        for (index, entry) in noise_shape.iter().enumerate() {
            let extent = input_shape[leading + index];
            match *entry {
                None => resolved.push(extent),
                Some(1) => resolved.push(1),
                Some(size) if size == extent => resolved.push(size),
                Some(size) => {
                    return Err(Error::invalid_parameter(
                        "noise_shape",
                        format!(
                            "entry {} at axis {} of the input must be 1, {}, or None",
                            size,
                            leading + index,
                            extent
                        ),
                    ));
                }
            }
        }
        Ok(resolved)
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

        // The mask shape, which is the input shape unless a noise shape makes it coarser
        let noise_shape = self.resolve_noise_shape(input.shape())?;

        if self.rate == 1.0 {
            // Dropping every unit yields all zeros
            return Ok(Tensor::zeros(input.raw_dim()));
        }

        // Sample a uniform value per draw. With no noise shape this is 1 value per input
        // element. With one, the sampler runs at the smaller shape, so an axis of extent 1
        // takes 1 draw that every position of that axis then shares
        let mut mask = Tensor::random_using(
            IxDyn(&noise_shape),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut self.rng,
        );

        // Threshold into a binary mask, in parallel for large masks
        if mask.len() >= cheap_map_parallel_threshold() {
            mask.par_mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        } else {
            mask.mapv_inplace(|x| if x >= self.rate { 1.0 } else { 0.0 });
        }

        // Inverted dropout: scale kept units by 1 / (1 - rate) to preserve the expected value
        let output = broadcast_dropout_scale(input, &mask, self.rate)?;

        // Cache the mask, at its own shape, for backpropagation
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

    fn known_input_shape(&self) -> Option<Shape> {
        (!self.input_shape.is_empty()).then(|| Shape::known(&self.input_shape))
    }

    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        shape_preserving_output(input, &self.input_shape, "Dropout")
    }

    no_trainable_parameters_layer_functions!();

    mode_dependent_layer_trait!();
}
