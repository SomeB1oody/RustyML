//! RepeatVector layer that repeats a feature vector into a sequence, and caches the input shape
//! for backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;
use ndarray::{Axis, IxDyn};

/// Repeats each feature vector `n` times along a new step axis
///
/// The input shape is `[batch_size, features]` and the output shape is
/// `[batch_size, n, features]`. Every one of the `n` steps holds the same vector. The batch axis
/// passes through unchanged, so 1 layer instance serves every batch size
///
/// The layer holds no parameter. Its use is the decoder side of an encoder-decoder model. A
/// recurrent layer in this crate returns only its last hidden state, a rank-2 tensor, and a
/// recurrent layer needs a rank-3 input. This layer bridges the 2, so an `LSTM` can feed another
/// `LSTM`
///
/// This layer emits the same vector at every step, which is the standard way to seed a decoder
/// with a fixed context
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
/// // A rank-2 input: 2 samples of 4 features each
/// let x = Array2::ones((2, 4)).into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(RepeatVector::new(3).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let sequence = model.predict(&x).unwrap();
///
/// // 1 vector of 4 features becomes 3 steps of the same 4 features
/// assert_eq!(sequence.shape(), &[2, 3, 4]);
/// ```
#[derive(Debug)]
pub struct RepeatVector {
    /// Number of steps the output holds
    n: usize,
    /// Shape of the most recent forward input. The backward pass needs it to check the gradient
    input_shape: Option<Vec<usize>>,
}

impl RepeatVector {
    /// Creates a new RepeatVector layer
    ///
    /// # Parameters
    ///
    /// - `n` - Number of steps the output holds. Must be greater than 0
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `RepeatVector` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `n` is 0
    pub fn new(n: usize) -> Result<Self, Error> {
        if n == 0 {
            return Err(Error::invalid_parameter(
                "n",
                "is 0, and the output must hold at least 1 step",
            ));
        }

        Ok(RepeatVector {
            n,
            input_shape: None,
        })
    }

    /// Checks the rank and the element count of a tensor entering the layer
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the rank is not 2
    /// - `Error::EmptyInput` - If either axis has an extent of 0
    fn validate(&self, input: &Tensor) -> Result<(), Error> {
        if input.ndim() != 2 {
            return Err(Error::invalid_input(format!(
                "RepeatVector layer expects a 2D input [batch_size, features], got a {}D tensor",
                input.ndim()
            )));
        }
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }
        Ok(())
    }

    /// Writes `n` copies of every feature vector into a new tensor
    fn repeat(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let mut output = Tensor::zeros(IxDyn(&[shape[0], self.n, shape[1]]));
        // The `[batch, 1, features]` view broadcasts across the new step axis
        output.assign(&input.view().insert_axis(Axis(1)));
        output
    }
}

impl Layer for RepeatVector {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate(input)?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(self.repeat(input))
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate(input)?;
        Ok(self.repeat(input))
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let Some(input_shape) = &self.input_shape else {
            return Err(Error::forward_pass_not_run("RepeatVector"));
        };

        let expected = [input_shape[0], self.n, input_shape[1]];
        if grad_output.shape() != expected {
            return Err(Error::shape_mismatch(expected, grad_output.shape()));
        }

        // Every step reads the same input vector, so the input gradient is the sum over the
        // step axis
        Ok(grad_output.sum_axis(Axis(1)))
    }

    fn layer_type(&self) -> &str {
        "RepeatVector"
    }

    fn output_shape(&self) -> String {
        match &self.input_shape {
            // Element 0 is the batch axis, which `summary()` prints as "None"
            Some(shape) => format!("(None, {}, {})", self.n, shape[1]),
            None => "Unknown".to_string(),
        }
    }

    no_trainable_parameters_layer_functions!();
}
