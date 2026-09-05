//! The merge family: 7 layers that take several inputs and give 1 output
//!
//! [`Add`], [`Subtract`], [`Multiply`], [`Average`], [`Maximum`], and [`Minimum`] reduce their
//! inputs element by element. [`Concatenate`] joins its inputs along 1 axis. No layer of the
//! family holds a trainable array, and every one of them implements
//! [`Layer`](crate::neural_network::traits::Layer) by hand.
//! [`UnaryLayer`](crate::neural_network::traits::UnaryLayer) takes 1 input alone, so no layer
//! of this family can implement it
//!
//! Each layer records `built: Option<Vec<Shape>>`, which holds 1 shape per input. The layer
//! reports every one of those shapes, and a checkpoint records them all
//!
//! # The shape rule of the 6 elementwise layers
//!
//! The rule has 2 clauses, and the 2 clauses differ:
//!
//! 1. The batch axis does not broadcast. Axis 0 holds the same extent on every input, or it
//!    stays free on 1 side. A shape `(2, 3)` with a shape `(1, 3)` is an error, and a shape
//!    `(2, 3)` with a shape `(2, 1)` is not.
//! 2. Every axis after the batch axis does broadcast. An extent of 1 takes the extent of the
//!    other side, and a free axis leaves that axis of the output free.
//!
//! Rank alignment inserts every extra axis AFTER the batch axis. The rule splits the batch
//! axis off each input, right-aligns the 2 tails, broadcasts them, and joins the batch axis
//! back. A shape `(2, 3, 4)` with a shape `(2, 4)` therefore behaves as `(2, 3, 4)` with
//! `(2, 1, 4)`, and never as `(2, 3, 4)` with `(1, 2, 4)`
//!
//! [`Concatenate`] shares none of that rule. It normalizes its axis against the full rank, it
//! takes the same rank on every input, and it adds the extents on 1 axis alone. Its own module
//! holds that rule
//!
//! # What the family shares
//!
//! The private items of this module are the shared half, and every layer file reaches them
//! through `use super::..`:
//!
//! - `elementwise_output_shape` applies the 2 clauses above and gives the output shape.
//! - `merged_dims` runs the same rule over the live tensors of a forward pass, and
//!   `broadcast_input` lifts 1 of those tensors to the output extents.
//! - `reduce_to` sums a gradient back to the shape of the input that broadcast to make it.
//!   Every backward pass of the family ends with 1 call of it per input.
//! - `merge_layer_base_functions` emits the build reports and the empty array roster of any
//!   merge layer. `elementwise_merge_layer_functions` emits the arity, the build, and the
//!   shape algebra of the 6 elementwise layers.
//! - [`Concatenate`] takes the first macro alone, and it calls
//!   `validation::start_build_many` from its own `build_many`

use crate::error::{Context, Error};
use crate::neural_network::layers::validation::start_build_many;
use crate::neural_network::traits::Arity;
use crate::neural_network::{Shape, Tensor};
use ndarray::Axis;

/// A layer that adds its inputs element by element
pub mod add;
/// A layer that averages its inputs element by element
pub mod average;
/// A layer that joins its inputs along 1 axis
pub mod concatenate;
/// A layer that takes the larger value of its inputs element by element
pub mod maximum;
/// A layer that takes the smaller value of its inputs element by element
pub mod minimum;
/// A layer that multiplies its inputs element by element
pub mod multiply;
/// A layer that subtracts its second input from its first
pub mod subtract;

pub use add::Add;
pub use average::Average;
pub use concatenate::Concatenate;
pub use maximum::Maximum;
pub use minimum::Minimum;
pub use multiply::Multiply;
pub use subtract::Subtract;

/// The output shape of the 6 elementwise merge layers, for inputs of the given shapes
///
/// The 2 clauses of the [module rule](self) run here, and they run over every input in order.
/// The batch axis takes the 1 fixed extent that the inputs agree on, and it stays free while
/// every input leaves it free. Every axis after the batch axis broadcasts
///
/// # Parameters
///
/// - `layer` - Layer name, which every message names
/// - `inputs` - 1 shape per input of the layer, batch axis first
///
/// # Returns
///
/// - `Result<Shape, Error>` - The shape of the tensor the layer gives back
///
/// # Errors
///
/// - `Error::InvalidInput` - If `inputs` is empty, if a shape holds no batch axis, if 2 batch
///   extents differ, or if 2 extents of another axis differ and neither is 1. Each message
///   names the 2 shapes that disagree
fn elementwise_output_shape(layer: &str, inputs: &[Shape]) -> Result<Shape, Error> {
    let Some((first, rest)) = inputs.split_first() else {
        return Err(Error::invalid_input(format!(
            "{layer} takes 1 input at least, and it received none"
        )));
    };
    first.check_min_rank(layer, 1)?;

    let mut merged = first.clone();
    for input in rest {
        input.check_min_rank(layer, 1)?;

        // Clause 1: the batch axis does not broadcast. A free side takes the fixed extent of
        // the other side, and 2 fixed extents must be equal
        let batch = match (merged.axes()[0], input.axes()[0]) {
            (Some(held), Some(found)) if held != found => {
                return Err(Error::invalid_input(format!(
                    "{layer} cannot merge the input shape {merged} with the input shape \
                     {input}, because the batch axis does not broadcast. Axis 0 holds {held} \
                     on 1 side and {found} on the other"
                )));
            }
            (Some(held), _) => Some(held),
            (None, found) => found,
        };

        // Clause 2: every axis after the batch axis broadcasts. The 2 tails are right-aligned,
        // so every axis that rank alignment inserts lands directly after the batch axis
        let held_tail = &merged.axes()[1..];
        let found_tail = &input.axes()[1..];
        let rank = held_tail.len().max(found_tail.len());
        let mut axes = Vec::with_capacity(rank + 1);
        axes.push(batch);
        for position in 0..rank {
            let extent = match (
                aligned_axis(held_tail, rank, position),
                aligned_axis(found_tail, rank, position),
            ) {
                // A free axis on either side leaves the axis of the output free
                (None, _) | (_, None) => None,
                (Some(1), Some(extent)) | (Some(extent), Some(1)) => Some(extent),
                (Some(held), Some(found)) if held == found => Some(held),
                (Some(held), Some(found)) => {
                    return Err(Error::invalid_input(format!(
                        "{layer} cannot merge the input shape {merged} with the input shape \
                         {input}. Axis {} holds {held} on 1 side and {found} on the other, and \
                         neither extent is 1",
                        position + 1
                    )));
                }
            };
            axes.push(extent);
        }
        merged = Shape::new(axes);
    }
    Ok(merged)
}

/// The axis that 1 input tail holds at 1 position of the aligned tail
///
/// The tails are right-aligned, so a tail that is shorter than `rank` holds no axis at the
/// leading positions. Every such position takes an extent of 1, which is what makes the axis
/// broadcast
///
/// # Parameters
///
/// - `tail` - Every axis of 1 input after its batch axis
/// - `rank` - Length of the aligned tail, which is the longest tail of the inputs
/// - `position` - Position in the aligned tail, counted from the batch axis
///
/// # Returns
///
/// - `Option<usize>` - The extent, or `None` when the input leaves that axis free
fn aligned_axis(tail: &[Option<usize>], rank: usize, position: usize) -> Option<usize> {
    let inserted = rank - tail.len();
    if position < inserted {
        Some(1)
    } else {
        tail[position - inserted]
    }
}

/// The output extents of an elementwise merge layer, for the tensors of 1 forward pass
///
/// The tensors of a pass fix every extent, so the merged shape fixes every extent as well
///
/// # Parameters
///
/// - `layer` - Layer name, which every message names
/// - `inputs` - 1 tensor per input of the layer, in the order the model wired them
///
/// # Returns
///
/// - `Result<Vec<usize>, Error>` - Extent of every axis of the output, batch axis first
///
/// # Errors
///
/// - `Error::InvalidInput` - If the shape rule refuses the shapes of the tensors
/// - `Error::Computation` - If the merged shape leaves an axis free, which no set of live
///   tensors can produce
fn merged_dims(layer: &str, inputs: &[&Tensor]) -> Result<Vec<usize>, Error> {
    let shapes: Vec<Shape> = inputs
        .iter()
        .map(|tensor| Shape::known(tensor.shape()))
        .collect();
    let merged = elementwise_output_shape(layer, &shapes)?;
    merged.dims().ok_or_else(|| {
        Error::computation(format!(
            "{layer} merged tensors of fixed extents into the shape {merged}, which leaves an \
             axis free"
        ))
    })
}

/// Lifts 1 input tensor to the extents of the output, for a forward pass
///
/// The extents of 1 that rank alignment inserts land AFTER the batch axis, exactly as the
/// [shape rule](self) says. A tensor of shape `[2, 4]` therefore reaches the output shape
/// `[2, 3, 4]` as `[2, 1, 4]`, and each of the 3 positions of axis 1 reads the same row
///
/// The result owns its data and is in the standard memory order, which every layer of the
/// crate emits
///
/// # Parameters
///
/// - `layer` - Layer name, which every message names
/// - `input` - 1 tensor of the forward pass
/// - `output` - Extent of every axis of the output, batch axis first
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The input at the extents of the output
///
/// # Errors
///
/// - `Error::InvalidInput` - If the tensor holds no batch axis, if the output holds fewer axes
///   than the tensor, or if an extent of the tensor is neither 1 nor the output extent
/// - `Error::Computation` - If the tensor cannot take the aligned rank
fn broadcast_input(layer: &str, input: &Tensor, output: &[usize]) -> Result<Tensor, Error> {
    let Some((&batch, tail)) = input.shape().split_first() else {
        return Err(Error::invalid_input(format!(
            "{layer} expects an input with a batch axis, got a tensor of rank 0"
        )));
    };
    if output.len() < input.ndim() {
        return Err(Error::invalid_input(format!(
            "{layer} cannot broadcast an input of shape {:?} to the output shape {output:?}, \
             because the output holds fewer axes",
            input.shape()
        )));
    }

    // The batch axis stays in front, every inserted axis follows it, and the tail keeps its
    // own order
    let mut aligned = Vec::with_capacity(output.len());
    aligned.push(batch);
    aligned.resize(output.len() - tail.len(), 1);
    aligned.extend_from_slice(tail);

    let view = input
        .to_shape(aligned)
        .context("Failed to align the rank of a merge input")?;
    let broadcast = view.broadcast(output).ok_or_else(|| {
        Error::invalid_input(format!(
            "{layer} cannot broadcast an input of shape {:?} to the output shape {output:?}",
            input.shape()
        ))
    })?;

    // A broadcast view repeats 1 element over a whole axis, so it never owns its data. The
    // copy goes through `assign`, which writes the standard memory order
    let mut owned = Tensor::zeros(output);
    owned.assign(&broadcast);
    Ok(owned)
}

/// Sums a gradient back to the shape of the input that broadcast to make the output
///
/// An input that broadcast in the forward pass reaches several positions of the output, so its
/// gradient is the sum over every position it reached. The 2 kinds of axis reduce alike:
///
/// 1. An axis that rank alignment inserted after the batch axis goes away, and the sum runs
///    over the whole axis.
/// 2. An axis where the input holds 1 position and the output holds more keeps its 1 position,
///    and the sum runs over the positions of the output.
///
/// An input that did not broadcast at all takes the gradient as it is
///
/// # Parameters
///
/// - `grad` - The gradient at the shape of the output
/// - `target` - Extent of every axis of the input, batch axis first
///
/// # Returns
///
/// - `Tensor` - The gradient at the shape of `target`, in the standard memory order
///
/// # Panics
///
/// - If `grad` holds fewer axes than `target`, which no forward pass of the family produces
fn reduce_to(grad: &Tensor, target: &[usize]) -> Tensor {
    debug_assert!(
        grad.ndim() >= target.len(),
        "a merge gradient holds the rank of the output"
    );
    let inserted = grad.ndim() - target.len();

    // Rank alignment inserts every extra axis directly after the batch axis, so each sum takes
    // axis 1 away and the next inserted axis takes its place
    let mut reduced = if inserted == 0 {
        let mut owned = Tensor::zeros(grad.raw_dim());
        owned.assign(grad);
        owned
    } else {
        let mut summed = grad.sum_axis(Axis(1));
        for _ in 1..inserted {
            summed = summed.sum_axis(Axis(1));
        }
        summed
    };

    for (axis, &extent) in target.iter().enumerate() {
        if extent == 1 && reduced.shape()[axis] != 1 {
            reduced = reduced.sum_axis(Axis(axis)).insert_axis(Axis(axis));
        }
    }
    reduced
}

/// Opens the build of 1 elementwise merge layer, and records the shapes it accepted
///
/// The 3 steps repeat over the 6 elementwise layers: refuse an input count that the arity
/// rejects, refuse a second build for other shapes, and refuse shapes that the shape rule
/// cannot merge. A layer that already holds the build for these shapes keeps it
///
/// # Parameters
///
/// - `built` - The shapes the layer already built for, or `None` before its first build
/// - `layer` - Layer name, which every message names
/// - `arity` - How many inputs the layer takes
/// - `inputs` - 1 shape per input of the layer, batch axis first
///
/// # Returns
///
/// - `Result<(), Error>` - `Ok` when the layer holds a build for these shapes
///
/// # Errors
///
/// - `Error::InvalidInput` - If the layer takes another input count, if it is already built for
///   other shapes, or if the shape rule refuses the shapes
fn build_elementwise(
    built: &mut Option<Vec<Shape>>,
    layer: &str,
    arity: Arity,
    inputs: &[Shape],
) -> Result<(), Error> {
    arity.check(layer, inputs.len())?;
    let Some(shapes) = start_build_many(built, layer, inputs)? else {
        return Ok(());
    };
    // Refuse a set of shapes that no pass could merge, before any tensor arrives
    elementwise_output_shape(layer, &shapes)?;
    *built = Some(shapes);
    Ok(())
}

/// Generates the build reports and the empty array roster of any merge layer
///
/// The layer holds a `built: Option<Vec<Shape>>` field, which holds 1 shape per input. The
/// macro emits [`LayerBase::known_input_shapes`], [`LayerBase::is_built`],
/// [`LayerBase::build_config`], and the 3 methods of a layer without a trainable array. It
/// belongs in the `impl LayerBase` block of the layer, and the layer file must have
/// [`ParamCounts`](crate::neural_network::layers::ParamCounts) in scope
///
/// Every reported shape carries a free batch axis, because 1 layer serves every batch size
///
/// [`Concatenate`] takes this macro as well, because it holds the same field and the same
/// empty roster
///
/// [`LayerBase::known_input_shapes`]: crate::neural_network::traits::LayerBase::known_input_shapes
/// [`LayerBase::is_built`]: crate::neural_network::traits::LayerBase::is_built
/// [`LayerBase::build_config`]: crate::neural_network::traits::LayerBase::build_config
macro_rules! merge_layer_base_functions {
    () => {
        fn known_input_shapes(&self) -> Option<Vec<$crate::neural_network::Shape>> {
            self.built.as_ref().map(|shapes| {
                shapes
                    .iter()
                    .map($crate::neural_network::Shape::free_batch)
                    .collect()
            })
        }

        fn is_built(&self) -> bool {
            self.built.is_some()
        }

        fn build_config(&self) -> Option<$crate::neural_network::layers::checkpoint::BuildConfig> {
            self.built
                .as_ref()
                .map(|shapes| $crate::neural_network::layers::checkpoint::BuildConfig::new(shapes))
        }

        $crate::neural_network::layers::no_trainable_parameters_layer_functions!();
    };
}
pub(in crate::neural_network::layers::merge) use merge_layer_base_functions;

/// Generates the arity, the build, and the shape algebra of 1 elementwise merge layer
///
/// The macro emits [`Layer::arity`], [`Layer::build_many`], and
/// [`Layer::compute_output_shape_many`], and it belongs in the `impl Layer` block of the layer.
/// The 6 elementwise layers differ in their forward and backward passes alone, so those 2
/// methods stay in the layer file
///
/// The first argument is the layer name, which every message names. The second is the
/// [`Arity`] of the layer, which is `Arity::AtLeast(1)` everywhere except [`Subtract`]
///
/// [`Layer::arity`]: crate::neural_network::traits::Layer::arity
/// [`Layer::build_many`]: crate::neural_network::traits::Layer::build_many
/// [`Layer::compute_output_shape_many`]: crate::neural_network::traits::Layer::compute_output_shape_many
macro_rules! elementwise_merge_layer_functions {
    ($layer:literal, $arity:expr) => {
        fn arity(&self) -> $crate::neural_network::traits::Arity {
            $arity
        }

        fn build_many(
            &mut self,
            inputs: &[$crate::neural_network::Shape],
        ) -> Result<(), $crate::error::Error> {
            $crate::neural_network::layers::merge::build_elementwise(
                &mut self.built,
                $layer,
                $arity,
                inputs,
            )
        }

        fn compute_output_shape_many(
            &self,
            inputs: &[$crate::neural_network::Shape],
        ) -> Result<$crate::neural_network::Shape, $crate::error::Error> {
            $crate::neural_network::traits::Arity::check($arity, $layer, inputs.len())?;
            $crate::neural_network::layers::merge::elementwise_output_shape($layer, inputs)
        }
    };
}
pub(in crate::neural_network::layers::merge) use elementwise_merge_layer_functions;

/// Unit tests of the shared shape rule, the forward broadcast, and the gradient reduction
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Every input of 1 shape gives that shape back
    #[test]
    fn equal_shapes_merge_unchanged() {
        let shapes = vec![Shape::known(&[2, 3, 4]), Shape::known(&[2, 3, 4])];
        assert_eq!(
            elementwise_output_shape("Add", &shapes).unwrap(),
            Shape::known(&[2, 3, 4])
        );
    }

    /// 1 input alone gives its own shape back, free axes included
    #[test]
    fn a_single_input_gives_its_own_shape() {
        let shapes = vec![Shape::with_free_batch(&[2, 3])];
        assert_eq!(
            elementwise_output_shape("Add", &shapes).unwrap(),
            Shape::with_free_batch(&[2, 3])
        );
    }

    /// An extent of 1 after the batch axis takes the extent of the other side
    #[test]
    fn an_axis_after_the_batch_axis_broadcasts() {
        let shapes = vec![Shape::known(&[2, 3]), Shape::known(&[2, 1])];
        assert_eq!(
            elementwise_output_shape("Multiply", &shapes).unwrap(),
            Shape::known(&[2, 3])
        );

        // Each side broadcasts into the other, on its own axis
        let shapes = vec![Shape::known(&[2, 1, 4]), Shape::known(&[2, 3, 1])];
        assert_eq!(
            elementwise_output_shape("Multiply", &shapes).unwrap(),
            Shape::known(&[2, 3, 4])
        );
    }

    /// The batch axis does not broadcast, and the message names the 2 shapes
    #[test]
    fn the_batch_axis_does_not_broadcast() {
        let shapes = vec![Shape::known(&[2, 3]), Shape::known(&[1, 3])];
        let message = elementwise_output_shape("Add", &shapes)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Add"), "{message}");
        assert!(message.contains("batch axis"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(1, 3)"), "{message}");
    }

    /// A free batch axis takes the fixed batch extent of the other side
    #[test]
    fn a_free_batch_axis_takes_the_other_extent() {
        let shapes = vec![Shape::with_free_batch(&[9, 3]), Shape::known(&[2, 3])];
        assert_eq!(
            elementwise_output_shape("Add", &shapes).unwrap(),
            Shape::known(&[2, 3])
        );

        // 2 free batch axes leave the batch axis of the output free
        let shapes = vec![
            Shape::with_free_batch(&[9, 3]),
            Shape::with_free_batch(&[1, 3]),
        ];
        assert_eq!(
            elementwise_output_shape("Add", &shapes).unwrap(),
            Shape::with_free_batch(&[9, 3])
        );
    }

    /// Rank alignment inserts every extra axis after the batch axis
    ///
    /// A rule that aligned the ranks from the left would accept the second pair, and the rule
    /// of this family refuses it
    #[test]
    fn rank_alignment_inserts_after_the_batch_axis() {
        let shapes = vec![Shape::known(&[2, 3, 4]), Shape::known(&[2, 4])];
        assert_eq!(
            elementwise_output_shape("Add", &shapes).unwrap(),
            Shape::known(&[2, 3, 4])
        );

        let shapes = vec![Shape::known(&[2, 3, 4]), Shape::known(&[2, 3])];
        assert!(elementwise_output_shape("Add", &shapes).is_err());
    }

    /// A free axis after the batch axis leaves that axis of the output free
    #[test]
    fn a_free_axis_frees_the_output_axis() {
        let free = Shape::new(vec![Some(2), None, Some(4)]);
        let shapes = vec![free.clone(), Shape::known(&[2, 3, 4])];
        assert_eq!(elementwise_output_shape("Add", &shapes).unwrap(), free);
    }

    /// The rule folds over every input, and it names the 2 shapes that disagree
    #[test]
    fn the_rule_folds_over_every_input() {
        let shapes = vec![
            Shape::known(&[2, 1, 4]),
            Shape::known(&[2, 3, 1]),
            Shape::known(&[2, 1, 1]),
        ];
        assert_eq!(
            elementwise_output_shape("Average", &shapes).unwrap(),
            Shape::known(&[2, 3, 4])
        );

        let shapes = vec![
            Shape::known(&[2, 3]),
            Shape::known(&[2, 3]),
            Shape::known(&[2, 5]),
        ];
        let message = elementwise_output_shape("Average", &shapes)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Average"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(2, 5)"), "{message}");
        assert!(message.contains("Axis 1"), "{message}");
    }

    /// A shape that holds no batch axis is refused, and the message names the layer
    #[test]
    fn a_shape_without_a_batch_axis_is_refused() {
        let shapes = vec![Shape::new(Vec::new()), Shape::known(&[2])];
        let message = elementwise_output_shape("Add", &shapes)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Add"), "{message}");
        assert!(message.contains("rank 1 or more"), "{message}");
    }

    /// The forward broadcast inserts the extents of 1 after the batch axis
    #[test]
    fn broadcast_input_inserts_after_the_batch_axis() {
        let input =
            Tensor::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();

        let wide = broadcast_input("Add", &input, &[2, 3, 4]).unwrap();
        assert_eq!(wide.shape(), &[2, 3, 4]);
        for position in 0..3 {
            assert_eq!(wide[[0, position, 0]], 1.0);
            assert_eq!(wide[[1, position, 3]], 8.0);
        }
        assert!(wide.is_standard_layout(), "a layer emits the C order");
    }

    /// A gradient of an input that never broadcast comes back with its own values
    #[test]
    fn reduce_to_keeps_a_gradient_that_never_broadcast() {
        let grad =
            Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let reduced = reduce_to(&grad, &[2, 3]);
        assert_eq!(reduced, grad);
        assert!(reduced.is_standard_layout(), "a layer emits the C order");
    }

    /// An axis of 1 position takes the sum over the positions it reached
    #[test]
    fn reduce_to_sums_an_axis_of_one_position() {
        let grad =
            Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let reduced = reduce_to(&grad, &[2, 1]);
        assert_eq!(
            reduced,
            Tensor::from_shape_vec(IxDyn(&[2, 1]), vec![6.0, 15.0]).unwrap()
        );
    }

    /// The axes that rank alignment inserted after the batch axis go away
    #[test]
    fn reduce_to_drops_the_inserted_axes() {
        let grad = Tensor::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let reduced = reduce_to(&grad, &[2, 3]);
        assert_eq!(
            reduced,
            Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]).unwrap()
        );
        assert!(reduced.is_standard_layout(), "a layer emits the C order");
    }

    /// An inserted axis and an axis of 1 position both reduce in 1 call
    #[test]
    fn reduce_to_handles_both_kinds_of_axis() {
        let grad = Tensor::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let reduced = reduce_to(&grad, &[2, 1]);
        assert_eq!(
            reduced,
            Tensor::from_shape_vec(IxDyn(&[2, 1]), vec![21.0, 57.0]).unwrap()
        );
    }
}
