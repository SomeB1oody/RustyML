//! Permute layer that reorders the axes after the batch axis, and caches the input shape for
//! backpropagation

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;
use ndarray::IxDyn;

/// Reorders the axes after the batch axis
///
/// The batch axis stays at axis 0 and never moves, so 1 layer instance serves every batch size.
/// `dims` names the new order of the remaining axes, counting from 1 as Keras does.
/// `Permute::new(vec![2, 1])` on a `[batch, height, width]` input gives `[batch, width, height]`
///
/// The input rank must be `dims.len() + 1`. A rank the layer does not serve is an
/// `Error::InvalidInput` at the forward pass, because the constructor sees no tensor
///
/// Unlike [`Reshape`](crate::neural_network::layers::reshape::Reshape), this layer moves data.
/// A reshape reads the same buffer in the same order under a new shape. A permute reads it in a
/// new order, so the output holds the same values at different positions. Use a permute when the
/// axis meaning must follow the values, such as before a layer that reads the last axis as the
/// channel axis
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array3;
///
/// // A rank-3 input: 2 samples, 5 steps, 3 features
/// let x = Array3::ones((2, 5, 3)).into_dyn();
///
/// // Swap the step axis and the feature axis
/// let mut model = Sequential::new();
/// model
///     .add(Permute::new(vec![2, 1]).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let swapped = model.predict(&x).unwrap();
///
/// // The batch axis stays first, and the other 2 axes trade places
/// assert_eq!(swapped.shape(), &[2, 3, 5]);
/// ```
///
/// # Performance
///
/// A permute reads the input in an order its memory layout does not follow, so it costs more
/// than a plain copy. The cost depends on which axis moves. A permute that leaves the last axis
/// last keeps whole rows contiguous, and it stays close to copy speed. A permute that moves the
/// last axis cuts the contiguous run to 1 element, and it costs several times more. Put a
/// permute where the tensor is small when the model allows a choice
#[derive(Debug)]
pub struct Permute {
    /// Axis order the forward pass applies, batch axis included and counted from 0
    ///
    /// Element 0 is always 0. The rest is the `dims` argument, which counts from 1
    forward_axes: Vec<usize>,
    /// Axis order the backward pass applies, the inverse of `forward_axes`
    backward_axes: Vec<usize>,
    /// Shape of the most recent forward input. The backward pass needs it to check the gradient
    input_shape: Option<Vec<usize>>,
}

impl Permute {
    /// Creates a new Permute layer
    ///
    /// # Parameters
    ///
    /// - `dims` - New order of the axes after the batch axis, counting from 1. Do not include
    ///   the batch axis. The entries must be a permutation of `1..=dims.len()`, so each axis
    ///   appears exactly once
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Permute` layer instance
    ///
    /// # Notes
    ///
    /// Keras accepts an empty `dims` and then serves a rank-1 input. This constructor rejects
    /// it, because such a layer reorders nothing and every other layer here needs a batch axis
    /// and at least 1 more axis
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `dims` is empty, or if it is not a permutation of
    ///   `1..=dims.len()`
    pub fn new(dims: Vec<usize>) -> Result<Self, Error> {
        if dims.is_empty() {
            return Err(Error::invalid_parameter(
                "dims",
                "is empty, and a permutation must name at least 1 axis",
            ));
        }

        let mut seen = vec![false; dims.len()];
        for &axis in &dims {
            match axis.checked_sub(1).and_then(|index| seen.get_mut(index)) {
                Some(slot) if !*slot => *slot = true,
                Some(_) => {
                    return Err(Error::invalid_parameter(
                        "dims",
                        format!("names axis {axis} more than once"),
                    ));
                }
                None => {
                    return Err(Error::invalid_parameter(
                        "dims",
                        format!(
                            "holds {}, and every entry must be between 1 and {}",
                            axis,
                            dims.len()
                        ),
                    ));
                }
            }
        }

        // The batch axis is axis 0 and never moves, so a `dims` entry is already the 0-based
        // index of the axis it names
        let forward_axes: Vec<usize> = std::iter::once(0).chain(dims).collect();
        let mut backward_axes = vec![0; forward_axes.len()];
        for (output_axis, &input_axis) in forward_axes.iter().enumerate() {
            backward_axes[input_axis] = output_axis;
        }

        Ok(Permute {
            forward_axes,
            backward_axes,
            input_shape: None,
        })
    }

    /// Shape this layer produces from `input_shape`
    fn permuted_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        self.forward_axes.iter().map(|&a| input_shape[a]).collect()
    }

    /// Checks the rank and the element count of a tensor entering the layer
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the rank is not `dims.len() + 1`
    /// - `Error::EmptyInput` - If any axis has an extent of 0
    fn validate(&self, input: &Tensor) -> Result<(), Error> {
        if input.ndim() != self.forward_axes.len() {
            return Err(Error::invalid_input(format!(
                "Permute layer expects a {}D input, got a {}D tensor",
                self.forward_axes.len(),
                input.ndim()
            )));
        }
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }
        Ok(())
    }
}

/// Copies `input` under a new axis order, into a tensor that is in C order
///
/// `permuted_axes` reorders the strides and shares the buffer, so it moves no data. `to_owned`
/// on such a view keeps those reordered strides, which leaves the result out of C order. Every
/// layer here emits C order, so a consumer can read any layer output as 1 contiguous slice.
/// This layer pays for a real copy to hold that contract
fn permute_into(input: &Tensor, axes: &[usize]) -> Tensor {
    let view = input.view().permuted_axes(IxDyn(axes));
    let mut output = Tensor::zeros(view.raw_dim());
    output.assign(&view);
    output
}

impl Layer for Permute {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate(input)?;
        self.input_shape = Some(input.shape().to_vec());
        Ok(permute_into(input, &self.forward_axes))
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate(input)?;
        Ok(permute_into(input, &self.forward_axes))
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let Some(input_shape) = &self.input_shape else {
            return Err(Error::forward_pass_not_run("Permute"));
        };

        let expected = self.permuted_shape(input_shape);
        if grad_output.shape() != expected.as_slice() {
            return Err(Error::shape_mismatch(expected, grad_output.shape()));
        }

        // A permute moves each value to 1 new position, so the gradient runs back through the
        // inverse order
        Ok(permute_into(grad_output, &self.backward_axes))
    }

    fn layer_type(&self) -> &str {
        "Permute"
    }

    fn output_shape(&self) -> String {
        match &self.input_shape {
            // Element 0 is the batch axis, which `summary()` prints as "None"
            Some(shape) => {
                let axes: Vec<String> = self.permuted_shape(shape)[1..]
                    .iter()
                    .map(|e| e.to_string())
                    .collect();
                format!("(None, {})", axes.join(", "))
            }
            None => "Unknown".to_string(),
        }
    }

    no_trainable_parameters_layer_functions!();
}
