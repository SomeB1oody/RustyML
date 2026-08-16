//! Identity layer that passes its input through unchanged, and caches the input shape for
//! backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;

/// Passes its input through unchanged
///
/// The output equals the input, at every rank and every shape. The backward pass returns the
/// gradient it receives. The layer holds no parameter and reads no configuration
///
/// Its use is a placeholder. A model built by a function that chooses among several layers
/// needs something to return when the choice is "no operation". A stack whose depth is a
/// runtime value needs a filler that leaves the activations alone. Both are cleaner with a
/// layer that does nothing than with an `Option` at every position
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array2;
///
/// // A rank-2 input: 2 samples of 3 features each
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(Identity::new())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let same = model.predict(&x).unwrap();
///
/// // Every value comes back untouched
/// assert_eq!(same, x);
/// ```
///
/// # Performance
///
/// The layer copies. It cannot borrow, because a layer returns an owned tensor. The copy is 1
/// linear pass, so it runs at memory speed. When a model is finished, remove the layer instead
/// of keeping it
#[derive(Debug, Default)]
pub struct Identity {
    /// Shape of the most recent forward input. The backward pass needs it to check the gradient
    input_shape: Option<Vec<usize>>,
}

impl Identity {
    /// Creates a new Identity layer
    ///
    /// # Returns
    ///
    /// - `Self` - New `Identity` layer instance
    pub fn new() -> Self {
        Identity::default()
    }

    /// Checks the rank and the element count of a tensor entering the layer
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the input has no batch axis
    /// - `Error::EmptyInput` - If any axis has an extent of 0
    fn validate(input: &Tensor) -> Result<(), Error> {
        if input.ndim() == 0 {
            return Err(Error::invalid_input(
                "Identity layer expects an input with a batch axis, got a 0D tensor",
            ));
        }
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }
        Ok(())
    }
}

/// Copies `input` into a tensor that is in C order
///
/// Every layer here emits C order, so a consumer can read any layer output as 1 contiguous
/// slice. An input that arrives in another layout therefore cannot pass straight through
fn copy_in_c_order(input: &Tensor) -> Tensor {
    input.as_standard_layout().into_owned()
}

impl Layer for Identity {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        Self::validate(input)?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(copy_in_c_order(input))
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        Self::validate(input)?;
        Ok(copy_in_c_order(input))
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let Some(input_shape) = &self.input_shape else {
            return Err(Error::forward_pass_not_run("Identity"));
        };

        if grad_output.shape() != input_shape.as_slice() {
            return Err(Error::shape_mismatch(
                input_shape.clone(),
                grad_output.shape(),
            ));
        }

        Ok(copy_in_c_order(grad_output))
    }

    fn layer_type(&self) -> &str {
        "Identity"
    }

    fn output_shape(&self) -> String {
        match &self.input_shape {
            // Element 0 is the batch axis, which `summary()` prints as "None"
            Some(shape) => {
                let axes: Vec<String> = shape[1..].iter().map(|e| e.to_string()).collect();
                if axes.is_empty() {
                    "(None,)".to_string()
                } else {
                    format!("(None, {})", axes.join(", "))
                }
            }
            None => "Unknown".to_string(),
        }
    }

    no_trainable_parameters_layer_functions!();
}
