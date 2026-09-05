//! The [`Add`] merge layer, which sums every input element by element

use super::{
    broadcast_input, elementwise_merge_layer_functions, merge_layer_base_functions, merged_dims,
    reduce_to,
};
use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Adds every input element by element
///
/// The layer takes 1 input or more, and the output is the sum of them all. It holds no
/// trainable array, so no optimizer reaches it and a checkpoint of it records no array. The
/// layer records 1 build shape per input, and it reports all of them
///
/// The [shape rule of the merge family](super) applies. The batch axis does not broadcast,
/// every axis after the batch axis does broadcast, and rank alignment inserts each extra axis
/// after the batch axis. A shape `(2, 3)` and a shape `(2, 1)` therefore give the output
/// shape `(2, 3)`, and a shape `(2, 3)` and a shape `(1, 3)` are refused
///
/// # Notes
///
/// The backward pass gives 1 gradient per input, at the shape of that input. The derivative of
/// a sum is 1 for every term, so each input takes the whole gradient. An input that broadcast
/// in the forward pass reached several positions of the output, and its gradient is the sum
/// over every position it reached
///
/// # Examples
///
/// ```rust
/// use ndarray::IxDyn;
/// use rustyml::neural_network::layers::Add;
/// use rustyml::neural_network::traits::Layer;
/// use rustyml::neural_network::{Ctx, Tensor};
///
/// // 2 samples of 3 features, and 1 offset per sample that broadcasts over the 3 features
/// let features =
///     Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let offset = Tensor::from_shape_vec(IxDyn(&[2, 1]), vec![10.0, 20.0]).unwrap();
///
/// let mut layer = Add::new();
/// let mut ctx = Ctx::training();
/// let sum = layer
///     .forward_many_mut(&[&features, &offset], &mut ctx)
///     .unwrap();
/// assert_eq!(
///     sum,
///     Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]).unwrap()
/// );
///
/// // Each input takes its gradient back at its own shape, and the offset sums over the 3
/// // positions of axis 1 that it reached
/// let grads = layer
///     .backward_many(&Tensor::ones(IxDyn(&[2, 3])), &mut ctx)
///     .unwrap();
/// assert_eq!(grads[0], Tensor::ones(IxDyn(&[2, 3])));
/// assert_eq!(
///     grads[1],
///     Tensor::from_shape_vec(IxDyn(&[2, 1]), vec![3.0, 3.0]).unwrap()
/// );
/// ```
///
/// # Performance
///
/// An input that already holds the extents of the output is added in place, and nothing is
/// copied. Every other input is first lifted to those extents, which costs 1 tensor of the
/// size of the output
#[derive(Debug, Default)]
pub struct Add {
    /// 1 shape per input that the build accepted, and `None` before the build
    built: Option<Vec<Shape>>,
}

impl Add {
    /// Creates a layer that adds its inputs
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Add` layer, which holds no build
    pub fn new() -> Self {
        Self::default()
    }
}

/// What the forward pass of [`Add`] parks for its backward pass
///
/// A sum needs no input value to differentiate, so the cache holds extents alone
struct AddCache {
    /// Extent of every axis of each input, in the order the forward pass took them
    inputs: Vec<Vec<usize>>,
    /// Extent of every axis of the output, which the gradient must match
    output: Vec<usize>,
}

impl LayerBase for Add {
    fn layer_type(&self) -> &str {
        "Add"
    }

    merge_layer_base_functions!();
}

impl Layer for Add {
    elementwise_merge_layer_functions!("Add", Arity::AtLeast(1));

    /// Sums every input, and lifts an input that broadcasts to the extents of the output
    ///
    /// A training pass parks the extents of the inputs and of the output. An inference pass
    /// parks nothing
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::AtLeast(1).check("Add", inputs.len())?;
        let dims = merged_dims("Add", inputs)?;

        let mut output = Tensor::zeros(dims.as_slice());
        for input in inputs {
            if input.shape() == dims.as_slice() {
                // The input already holds the extents of the output, so the sum reads it in
                // place and copies nothing
                output += *input;
            } else {
                output += &broadcast_input("Add", input, &dims)?;
            }
        }

        if ctx.is_training() {
            ctx.push_cache(
                "Add",
                AddCache {
                    inputs: inputs.iter().map(|input| input.shape().to_vec()).collect(),
                    output: dims,
                },
            );
        }
        Ok(output)
    }

    /// Gives the whole gradient to every input, reduced to the shape of that input
    ///
    /// The derivative of a sum is 1 for every term. An input that broadcast in the forward
    /// pass takes the sum over every position of the output that it reached
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let cache: AddCache = ctx.pop_cache("Add")?;
        if grad_output.shape() != cache.output.as_slice() {
            return Err(Error::shape_mismatch(cache.output, grad_output.shape()));
        }
        Ok(cache
            .inputs
            .iter()
            .map(|shape| reduce_to(grad_output, shape))
            .collect())
    }
}

/// Unit tests of the sum, of the gradient of every input, and of the refusals of the layer
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Builds a tensor of the given extents from its values, in the standard memory order
    fn tensor(dims: &[usize], values: &[f32]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values.to_vec()).unwrap()
    }

    /// The output is the sum of every input, over any number of inputs
    #[test]
    fn the_forward_pass_sums_every_input() {
        let first = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let second = tensor(&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let third = tensor(&[2, 3], &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);

        let mut layer = Add::new();
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many_mut(&[&first, &second, &third], &mut ctx)
            .unwrap();

        assert_eq!(
            output,
            tensor(&[2, 3], &[111.0, 222.0, 333.0, 444.0, 555.0, 666.0])
        );
        assert!(output.is_standard_layout(), "a layer emits the C order");
    }

    /// 1 input alone is accepted, and it comes back with its own values
    #[test]
    fn one_input_alone_comes_back_unchanged() {
        let only = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut layer = Add::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&only], &mut ctx).unwrap();
        assert_eq!(output, only);

        let grads = layer.backward_many(&only, &mut ctx).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0], only);
    }

    /// Every input of the shape of the output takes the whole gradient
    #[test]
    fn the_backward_pass_gives_the_gradient_to_every_input() {
        let first = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let second = tensor(&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let third = tensor(&[2, 3], &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);

        let mut layer = Add::new();
        let mut ctx = Ctx::training();
        layer
            .forward_many_mut(&[&first, &second, &third], &mut ctx)
            .unwrap();

        let grad = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads.len(), 3);
        for one in &grads {
            assert_eq!(one, &grad);
            assert!(one.is_standard_layout(), "a layer emits the C order");
        }
        assert_eq!(ctx.pending_caches(), 0, "the backward pass takes the cache");
    }

    /// An axis of 1 position broadcasts, and its gradient sums over the positions it reached
    #[test]
    fn an_extent_of_one_broadcasts_and_its_gradient_reduces() {
        let wide = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let column = tensor(&[2, 1], &[10.0, 20.0]);

        let mut layer = Add::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&wide, &column], &mut ctx).unwrap();
        assert_eq!(
            output,
            tensor(&[2, 3], &[11.0, 12.0, 13.0, 24.0, 25.0, 26.0])
        );

        let grad = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads[0], grad);
        assert_eq!(grads[1], tensor(&[2, 1], &[6.0, 15.0]));
    }

    /// Rank alignment inserts the extra axis after the batch axis, and the gradient drops it
    ///
    /// A shape `(2, 3)` therefore meets a shape `(2, 2, 3)` as `(2, 1, 3)`. A rule that
    /// aligned the 2 ranks from the left would read it as `(1, 2, 3)` and give other values
    #[test]
    fn rank_alignment_inserts_after_the_batch_axis() {
        let deep = tensor(
            &[2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let flat = tensor(&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        let mut layer = Add::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&deep, &flat], &mut ctx).unwrap();

        assert_eq!(
            output,
            tensor(
                &[2, 2, 3],
                &[
                    11.0, 22.0, 33.0, 14.0, 25.0, 36.0, 47.0, 58.0, 69.0, 50.0, 61.0, 72.0,
                ],
            )
        );

        let grad = tensor(
            &[2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads[0], grad);
        assert_eq!(
            grads[1],
            tensor(&[2, 3], &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]),
            "the inserted axis goes away, and the gradient sums over it"
        );
    }

    /// The batch axis does not broadcast, and the refusal leaves the layer unbuilt
    #[test]
    fn the_batch_axis_does_not_broadcast() {
        let mut layer = Add::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[1, 3])])
            .unwrap_err()
            .to_string();

        assert!(message.contains("Add"), "{message}");
        assert!(message.contains("batch axis"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(1, 3)"), "{message}");
        assert!(!layer.is_built(), "a refused build allocates nothing");
    }

    /// 2 extents of another axis that differ, and where neither is 1, are refused
    #[test]
    fn two_extents_that_neither_hold_one_are_refused() {
        let mut layer = Add::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Axis 1"), "{message}");

        // The live tensors meet the same rule, so a hand-driven pass is refused as well
        let left = tensor(&[2, 3], &[1.0; 6]);
        let right = tensor(&[2, 5], &[1.0; 10]);
        let mut ctx = Ctx::training();
        assert!(
            Add::new()
                .forward_many_mut(&[&left, &right], &mut ctx)
                .is_err()
        );
        assert_eq!(ctx.pending_caches(), 0, "a refused pass parks no cache");
    }

    /// A second build for other shapes is refused, and another batch extent is accepted
    #[test]
    fn a_second_build_for_other_shapes_is_refused() {
        let mut layer = Add::new();
        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 1])])
            .unwrap();

        // 1 layer serves every batch size, so the batch extent is not part of the build
        layer
            .build_many(&[Shape::known(&[8, 3]), Shape::known(&[8, 1])])
            .unwrap();

        let message = layer
            .build_many(&[Shape::known(&[2, 5]), Shape::known(&[2, 1])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("already built"), "{message}");
    }

    /// The layer takes 1 input at least, and every entry point says so
    #[test]
    fn the_layer_takes_one_input_at_least() {
        let mut layer = Add::new();
        assert_eq!(layer.arity(), Arity::AtLeast(1));

        let message = layer.build_many(&[]).unwrap_err().to_string();
        assert!(message.contains("Add"), "{message}");
        assert!(message.contains("1 or more"), "{message}");

        assert!(layer.compute_output_shape_many(&[]).is_err());

        let mut ctx = Ctx::training();
        assert!(layer.forward_many(&[], &mut ctx).is_err());
        assert_eq!(ctx.pending_caches(), 0, "a refused pass parks no cache");
    }

    /// A gradient of another shape than the output is refused, and the message names both
    #[test]
    fn a_gradient_of_another_shape_is_refused() {
        let input = tensor(&[2, 3], &[1.0; 6]);

        let mut layer = Add::new();
        let mut ctx = Ctx::training();
        layer.forward_many_mut(&[&input, &input], &mut ctx).unwrap();

        let message = layer
            .backward_many(&tensor(&[2, 4], &[1.0; 8]), &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(message.contains("[2, 3]"), "{message}");
        assert!(message.contains("[2, 4]"), "{message}");
    }

    /// An inference pass parks no cache at all
    #[test]
    fn an_inference_pass_parks_no_cache() {
        let input = tensor(&[2, 3], &[1.0; 6]);

        let mut layer = Add::new();
        let mut ctx = Ctx::inference();
        let output = layer.forward_many_mut(&[&input, &input], &mut ctx).unwrap();

        assert_eq!(output, tensor(&[2, 3], &[2.0; 6]));
        assert_eq!(ctx.pending_caches(), 0);
    }

    /// The layer holds no array, and it reports 1 build shape per input
    #[test]
    fn the_layer_holds_no_array_and_reports_every_build_shape() {
        let mut layer = Add::new();
        assert_eq!(layer.param_count(), ParamCounts::none());
        assert!(layer.weights().is_empty());
        assert!(layer.weights_mut().is_empty());
        assert!(!layer.is_built());
        assert!(layer.build_config().is_none());
        assert_eq!(layer.output_shape(), "Unknown");

        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 1])])
            .unwrap();

        assert!(layer.is_built());
        assert_eq!(
            layer.known_input_shapes(),
            Some(vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 1]),
            ]),
            "every reported shape carries a free batch axis"
        );
        assert_eq!(layer.output_shape(), "(None, 3)");
        assert_eq!(
            layer.build_config().unwrap().input_shapes,
            vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 1]),
            ]
        );
    }
}
