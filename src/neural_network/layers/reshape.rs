//! Reshape layer that rewrites the axes after the batch axis into a target shape, and parks
//! the input shape for backpropagation

use crate::error::{Context, Error};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::{
    build_config_function, no_trainable_parameters_layer_functions,
};
use crate::neural_network::traits::{LayerBase, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};
use ndarray::IxDyn;

/// Rewrites the axes after the batch axis into a target shape
///
/// The target shape never names the batch axis. Axis 0 passes through untouched, so 1 layer
/// serves every batch size. The element count of the remaining axes must not change, since a
/// reshape moves no data.
///
/// Exactly 1 entry of the target shape may be `-1`. That axis takes whatever extent makes the
/// element count match. `Reshape::new(vec![-1, 2])` on a `[batch, 4]` input gives
/// `[batch, 2, 2]`.
///
/// The reshape reads and writes in C order, which is the layout [`Tensor`] already uses. The
/// last axis therefore varies fastest. Under the crate's channels-last layout the channel axis
/// is innermost, so a reshape that splits or merges the trailing axes regroups channels before
/// positions. See [`Flatten`](crate::neural_network::layers::flatten::Flatten) for the same
/// caution about ordering.
///
/// Unlike `Flatten`, this layer derives the output shape from the tensor it receives. A target
/// that names every extent therefore fixes the element count an input must carry. The layer
/// then reports its own output shape before the build
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
/// // A rank-2 input: 2 samples of 12 features each
/// let x = Array2::ones((2, 12)).into_dyn();
///
/// // Fold the 12 features into a 2x3x2 volume. The -1 axis takes the leftover extent
/// let mut model = SequentialBuilder::new()
///     .add(Reshape::new(vec![-1, 3, 2]).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// // Forward propagation
/// let folded = model.predict(&x).unwrap();
///
/// // The batch axis passes through, and -1 resolves to 2
/// assert_eq!(folded.shape(), &[2, 2, 3, 2]);
/// ```
#[derive(Debug)]
pub struct Reshape {
    /// Target extent of every axis after the batch axis. At most 1 entry is `-1`
    target_shape: Vec<isize>,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl Reshape {
    /// Creates a new Reshape layer
    ///
    /// # Parameters
    ///
    /// - `target_shape` - Extent of every axis after the batch axis. Do not include the batch
    ///   axis. At most 1 entry may be `-1`, which takes whatever extent makes the element count
    ///   match. An empty vector reshapes to the rank-1 shape `[batch]`
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Reshape` layer instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If more than 1 entry is `-1`, or if any entry is 0 or
    ///   below `-1`
    pub fn new(target_shape: Vec<isize>) -> Result<Self, Error> {
        let mut inferred_axes = 0;
        for (axis, &extent) in target_shape.iter().enumerate() {
            match extent {
                -1 => inferred_axes += 1,
                e if e < -1 => {
                    return Err(Error::invalid_parameter(
                        "target_shape",
                        format!("axis {axis} is {e}, which must be -1 or greater than 0"),
                    ));
                }
                0 => {
                    return Err(Error::invalid_parameter(
                        "target_shape",
                        format!("axis {axis} is 0, which no non-empty input can match"),
                    ));
                }
                _ => {}
            }
        }

        if inferred_axes > 1 {
            return Err(Error::invalid_parameter(
                "target_shape",
                format!("holds {inferred_axes} entries of -1, and at most 1 may be inferred"),
            ));
        }

        Ok(Reshape {
            target_shape,
            built: None,
        })
    }

    /// Resolves the target shape against a concrete input shape
    ///
    /// Returns the full output shape, batch axis first. The batch axis passes through, and a
    /// `-1` entry takes the extent that makes the element count match
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the tensor entering the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<Vec<usize>, Error>` - Output shape, batch axis first
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If `input_shape` has no batch axis
    /// - `Error::ShapeMismatch` - If the element count cannot match the target
    fn resolve(&self, input_shape: &[usize]) -> Result<Vec<usize>, Error> {
        if input_shape.is_empty() {
            return Err(Error::invalid_input(
                "Reshape layer expects an input with a batch axis, got a 0D tensor",
            ));
        }

        let batch = input_shape[0];
        let elements: usize = input_shape[1..].iter().product();

        // The product of every named axis. A `-1` axis contributes nothing to it
        let named: usize = self
            .target_shape
            .iter()
            .filter(|&&e| e != -1)
            .map(|&e| e as usize)
            .product();
        let has_inferred = self.target_shape.contains(&-1);

        let mut output_shape = Vec::with_capacity(self.target_shape.len() + 1);
        output_shape.push(batch);

        if has_inferred {
            if !elements.is_multiple_of(named) {
                return Err(Error::shape_mismatch(
                    vec![batch, named],
                    input_shape.to_vec(),
                ));
            }
            let inferred = elements / named;
            output_shape.extend(
                self.target_shape
                    .iter()
                    .map(|&e| if e == -1 { inferred } else { e as usize }),
            );
        } else {
            if named != elements {
                let mut expected = vec![batch];
                expected.extend(self.target_shape.iter().map(|&e| e as usize));
                return Err(Error::shape_mismatch(expected, input_shape.to_vec()));
            }
            output_shape.extend(self.target_shape.iter().map(|&e| e as usize));
        }

        Ok(output_shape)
    }
}

impl LayerBase for Reshape {
    fn layer_type(&self) -> &str {
        "Reshape"
    }

    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        match &self.built {
            Some(shape) => Some(vec![shape.free_batch()]),
            // A target that holds a -1 fixes no element count, so the layer knows nothing yet
            None if self.target_shape.contains(&-1) => None,
            // A target with no -1 fixes the element count an input must carry, so the layer
            // describes its own output before any tensor arrives
            None => Some(vec![Shape::new(vec![
                None,
                Some(self.target_shape.iter().map(|&e| e as usize).product()),
            ])]),
        }
    }

    build_config_function!();

    no_trainable_parameters_layer_functions!();
}

impl UnaryLayer for Reshape {
    /// Records the shape the layer rewrites. The layer holds no array, so nothing is allocated.
    /// The shape algebra checks the element count against the target
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "Reshape", input)? else {
            return Ok(());
        };
        self.compute_output_shape(&built)?;
        self.built = Some(built);
        Ok(())
    }

    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }

        let output_shape = self.resolve(input.shape())?;

        if ctx.is_training() {
            ctx.push_cache("Reshape", input.shape().to_vec());
        }

        Ok(input
            .to_shape(IxDyn(&output_shape))
            .context("reshape input")?
            .to_owned())
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let input_shape: Vec<usize> = ctx.pop_cache("Reshape")?;

        let expected = self.resolve(&input_shape)?;
        if grad_output.shape() != expected.as_slice() {
            return Err(Error::shape_mismatch(expected, grad_output.shape()));
        }

        // A reshape moves no data, so the gradient goes back through the inverse reshape
        Ok(grad_output
            .to_shape(IxDyn(input_shape.as_slice()))
            .context("reshape gradient")?
            .to_owned())
    }

    /// The batch axis passes through, and the target rewrites every later axis
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        let (batch, tail) = input.split_batch("Reshape")?;
        // `resolve` expects a list whose first entry is the batch size. A free batch uses a
        // placeholder of 1 there, instead of 0. A 0 would read as a real extent of 0, since
        // `resolve` reports any mismatch using the entry as given
        let mut dims = vec![batch.unwrap_or(1)];
        dims.extend(tail);
        Ok(Shape::from_batch(batch, &self.resolve(&dims)?[1..]))
    }
}
