//! Flatten layer that reshapes a 3D, 4D, or 5D tensor into a 2D tensor for dense layers

use crate::error::{Context, Error};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::{start_build, validate_built_input};
use crate::neural_network::layers::{
    build_config_function, build_on_forward, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::Layer;
use crate::neural_network::{Shape, Tensor};
use ndarray::IxDyn;

/// Flattens a 3D, 4D, or 5D tensor into a 2D tensor
///
/// Reshapes inputs from feature extraction layers into a format suitable for dense layers
///
/// The reshape itself is layout-agnostic. It collapses every axis after the batch axis in C
/// order, without regard to which feature lands at which output index. Under the crate's
/// channels-last layout, the channel axis is innermost. The flattened vector then runs position
/// by position, with all channels of one position adjacent, rather than plane by plane. A
/// `Dense` layer trained against the other ordering then reads its inputs permuted, even though
/// its weight shape stays the same. This is why saved models carry a format version (see
/// [`MODEL_FORMAT_VERSION`](crate::neural_network::layers::checkpoint::MODEL_FORMAT_VERSION))
/// instead of relying on a shape check to catch the mismatch
///
/// Input shapes are `[batch_size, length, features]`, `[batch_size, height, width, channels]`,
/// or `[batch_size, depth, height, width, channels]`. The output shape is always
/// `[batch_size, flattened_features]`, where `flattened_features` is the product of all
/// dimensions except the batch size
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use rustyml::neural_network::Shape;
/// use ndarray::Array4;
///
/// // Create a 4D input tensor: [batch_size, height, width, channels]
/// // Batch size=2, 4x4 pixels, 3 channels
/// let x = Array4::ones((2, 4, 4, 3)).into_dyn();
///
/// // Build a model containing a Flatten layer
/// let mut model = SequentialBuilder::new()
///     .add(Flatten::new())
///     .build(&Shape::known(&[2, 4, 4, 3]))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let flattened = model.predict(&x).unwrap();
///
/// // The output shape should be [2, 48]
/// assert_eq!(flattened.shape(), &[2, 48]);
/// ```
#[derive(Debug, Default)]
pub struct Flatten {
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
    /// Shape of the most recent forward input. The backward pass restores it
    ///
    /// A flatten moves no data, so the backward pass needs the shape alone
    input_shape: Option<Vec<usize>>,
}

impl Flatten {
    /// Creates a new Flatten layer
    ///
    /// The layer takes no input shape. [`Layer::build`] gives it one, and a forward pass on a
    /// layer that a caller drives by hand builds it from the tensor that arrives
    ///
    /// # Returns
    ///
    /// - `Flatten` - A new `Flatten` layer
    pub fn new() -> Self {
        Self {
            built: None,
            input_shape: None,
        }
    }
}

impl Layer for Flatten {
    /// Records the shape the layer folds. The layer holds no array, so nothing is allocated
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Flatten", input)? else {
            return Ok(());
        };
        if !(3..=5).contains(&built.rank()) {
            return Err(Error::invalid_input(format!(
                "Flatten layer expects 3D, 4D, or 5D input, got {}D tensor",
                built.rank()
            )));
        }
        built.split_batch("Flatten")?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        build_on_forward!(self, input);
        validate_built_input(&self.built, "Flatten", input.shape())?;
        let input_shape = input.shape();

        self.input_shape = Some(input_shape.to_vec());

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_built_input(&self.built, "Flatten", input.shape())?;
        let input_shape = input.shape();

        let batch_size = input_shape[0];
        let flattened_features: usize = input_shape[1..].iter().product();

        Ok(input
            .to_shape(IxDyn(&[batch_size, flattened_features]))
            .unwrap()
            .to_owned())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if let Some(input_shape) = &self.input_shape {
            let expected_grad_shape = [input_shape[0], input_shape[1..].iter().product()];
            if grad_output.shape() != expected_grad_shape {
                return Err(Error::shape_mismatch(
                    expected_grad_shape,
                    grad_output.shape(),
                ));
            }

            // Reshape gradient back to input shape
            let reshaped_grad = grad_output
                .to_shape(IxDyn(input_shape))
                .context("reshape gradient")?
                .to_owned();

            Ok(reshaped_grad)
        } else {
            Err(Error::forward_pass_not_run("Flatten"))
        }
    }

    fn layer_type(&self) -> &str {
        "Flatten"
    }

    /// The batch axis is free, because the fold serves every batch size
    fn known_input_shape(&self) -> Option<Shape> {
        self.built.as_ref().map(Shape::free_batch)
    }

    build_config_function!();

    /// Every axis after the batch axis folds into 1 feature axis
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_min_rank("Flatten", 2)?;
        let (batch, tail) = input.split_batch("Flatten")?;
        Ok(Shape::from_batch(batch, &[tail.iter().product()]))
    }

    no_trainable_parameters_layer_functions!();
}
