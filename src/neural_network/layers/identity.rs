//! Identity layer that passes its input through unchanged, and parks the input shape for
//! backpropagation

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    built_layer_shape_functions, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Passes its input through unchanged
///
/// The output equals the input, at every rank and every shape. The backward pass returns the
/// gradient it receives. The layer holds no parameter and reads no configuration
///
/// Its use is a placeholder. A function that chooses among several layers to build a model
/// needs something to return when the choice is "no operation". A stack whose depth is a
/// runtime value needs a filler that leaves the activations alone. Both are cleaner with a
/// layer that does nothing than with an `Option` at every position
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
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
/// let mut model = SequentialBuilder::new()
///     .add(Identity::new())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
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
/// linear pass, so it runs at memory speed. When you finish a model, remove the layer instead
/// of keeping it
#[derive(Debug, Default)]
pub struct Identity {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
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

impl LayerBase for Identity {
    fn layer_type(&self) -> &str {
        "Identity"
    }

    built_layer_shape_functions!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Identity {
    /// Records the shape the layer passes through. The layer holds no array, so nothing is
    /// allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Identity", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        Self::validate(input)?;

        if ctx.is_training() {
            ctx.push_cache("Identity", input.shape().to_vec());
        }

        Ok(copy_in_c_order(input))
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("Identity")?;

        if grad_output.shape() != input_shape.as_slice() {
            return Err(Error::shape_mismatch(input_shape, grad_output.shape()));
        }

        Ok(copy_in_c_order(grad_output))
    }
}
