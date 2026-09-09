//! The merge layer that takes the larger value of its inputs, element by element

use super::{
    broadcast_input, elementwise_merge_layer_functions, merge_layer_base_functions, merged_dims,
    reduce_to,
};
use crate::error::{Context, Error};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};
use ndarray::IxDyn;

/// Takes the larger value of its inputs, element by element
///
/// The layer folds its inputs left-associatively and pairwise. Input 0 seeds the running
/// value, and each later input takes over a position where it holds a larger value. The layer
/// accepts 1 input or more, and 1 input alone comes back unchanged. The layer holds no
/// trainable array
///
/// The inputs follow the shape rule of the [merge family](super). The batch axis does not
/// broadcast, and every axis after the batch axis does. Rank alignment inserts every extra
/// axis directly after the batch axis. An input of shape `(2, 4)` therefore meets an input of
/// shape `(2, 3, 4)` as `(2, 1, 4)`
///
/// The backward pass gives 1 gradient per input, at the shape of that input. The winner of
/// each output position takes the whole gradient of that position, and every other input takes
/// 0 there. An input that broadcast in the forward pass then sums its gradient back to its own
/// shape
///
/// # The tie rule
///
/// A tie routes the whole gradient to the first input that holds the winning value. Every
/// other input that ties there takes 0.
/// [`MaxPooling2D`](crate::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D)
/// resolves a tied window position by the same rule. The test
/// `tie_routes_the_gradient_to_the_first_input` pins the rule
///
/// # Notes
///
/// A NaN wins its position and keeps it. The first input that holds a NaN at a position takes
/// the whole gradient there, and no later value displaces it. The max pooling layers of this
/// crate keep a NaN the same way
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array, IxDyn};
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::Maximum;
/// use rustyml::neural_network::traits::Layer;
///
/// // 1 sample of 3 features, from each of 2 branches of a model
/// let left = Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 5.0, 3.0]).unwrap();
/// let right = Array::from_shape_vec(IxDyn(&[1, 3]), vec![4.0, 2.0, 3.0]).unwrap();
///
/// let mut layer = Maximum::new();
/// let mut ctx = Ctx::training();
/// let output = layer.forward_many_mut(&[&left, &right], &mut ctx).unwrap();
/// assert_eq!(
///     output,
///     Array::from_shape_vec(IxDyn(&[1, 3]), vec![4.0, 5.0, 3.0]).unwrap()
/// );
///
/// // The last position holds a tie, and its whole gradient goes to the first input
/// let grads = layer
///     .backward_many(&Array::ones(output.raw_dim()), &mut ctx)
///     .unwrap();
/// assert_eq!(
///     grads[0],
///     Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.0, 1.0, 1.0]).unwrap()
/// );
/// assert_eq!(
///     grads[1],
///     Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 0.0, 0.0]).unwrap()
/// );
/// ```
#[derive(Debug, Default)]
pub struct Maximum {
    /// Shape of every input the layer was built for, batch axis first. `None` before the build
    built: Option<Vec<Shape>>,
}

impl Maximum {
    /// Creates a new Maximum layer
    ///
    /// # Returns
    ///
    /// - `Self` - New `Maximum` layer instance, before its build
    pub fn new() -> Self {
        Self::default()
    }
}

/// What the forward pass of [`Maximum`] parks for its backward pass
///
/// The winner list is the whole answer of the backward pass, so the cache holds no tensor of
/// the forward pass. It costs 1 entry per element of the output, whatever number of inputs the
/// layer took
struct MaximumCache {
    /// Extent of every axis of the output, batch axis first
    output: Vec<usize>,
    /// Extent of every axis of each input, in the order the forward pass took them
    inputs: Vec<Vec<usize>>,
    /// The input that won each output position, in the C order of the output
    ///
    /// A tie names the first input that holds the winning value, which is the
    /// [tie rule](Maximum) of the layer
    winners: Vec<usize>,
}

impl LayerBase for Maximum {
    fn layer_type(&self) -> &str {
        "Maximum"
    }

    merge_layer_base_functions!();
}

impl Layer for Maximum {
    elementwise_merge_layer_functions!("Maximum", Arity::AtLeast(1));

    /// Folds the inputs left-associatively, and parks the winner of every output position
    ///
    /// Every input reaches the extents of the output first, so 1 flat scan per input settles
    /// both the value and the winner. The scan keeps the association of a pairwise fold,
    /// because a position changes hands only for a strictly larger value
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer received no input, or if the shape rule of the
    ///   family refuses the shapes of the tensors
    /// - `Error::Computation` - If an input cannot take the rank of the output
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::AtLeast(1).check("Maximum", inputs.len())?;
        let dims = merged_dims("Maximum", inputs)?;

        let mut folded = broadcast_input("Maximum", inputs[0], &dims)?;
        let mut winners = vec![0_usize; folded.len()];
        for (position, &input) in inputs.iter().enumerate().skip(1) {
            let lifted = broadcast_input("Maximum", input, &dims)?;
            for ((value, &candidate), winner) in
                folded.iter_mut().zip(lifted.iter()).zip(winners.iter_mut())
            {
                // A position that holds a NaN keeps it. No later candidate can replace a NaN
                // value, however large the candidate is.
                let takes = !value.is_nan() && (candidate.is_nan() || candidate > *value);
                if takes {
                    *value = candidate;
                    *winner = position;
                }
            }
        }

        if ctx.is_training() {
            ctx.push_cache(
                "Maximum",
                MaximumCache {
                    output: dims,
                    inputs: inputs
                        .iter()
                        .map(|tensor| tensor.shape().to_vec())
                        .collect(),
                    winners,
                },
            );
        }

        Ok(folded)
    }

    /// Routes the gradient of each output position to the input that won it
    ///
    /// The other inputs take 0 at that position. An input that broadcast in the forward pass
    /// then sums its gradient back to its own shape, so every gradient comes back at the shape
    /// of its own input
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer
    /// - `Error::ShapeMismatch` - If `grad_output` does not carry the shape of the output that
    ///   the forward pass gave
    /// - `Error::Computation` - If a routed buffer cannot take the shape of the output
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let cache: MaximumCache = ctx.pop_cache("Maximum")?;
        if grad_output.shape() != cache.output.as_slice() {
            return Err(Error::shape_mismatch(cache.output, grad_output.shape()));
        }

        // 1 buffer per input, in the C order of the output, filled by 1 scan of the winners
        let mut routed = vec![vec![0.0_f32; cache.winners.len()]; cache.inputs.len()];
        for ((offset, &winner), &grad) in cache.winners.iter().enumerate().zip(grad_output.iter()) {
            routed[winner][offset] = grad;
        }

        let mut grads = Vec::with_capacity(cache.inputs.len());
        for (values, shape) in routed.into_iter().zip(cache.inputs.iter()) {
            let whole = Tensor::from_shape_vec(IxDyn(&cache.output), values)
                .context("Failed to give a Maximum gradient the shape of the output")?;
            grads.push(reduce_to(&whole, shape));
        }
        Ok(grads)
    }
}

/// Unit tests of the fold, the gradient routing, the tie rule, and the refusals
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tensor of the given extents from its values, read in C order
    fn tensor(dims: &[usize], values: &[f32]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values.to_vec()).unwrap()
    }

    /// The output holds the larger value of each position
    #[test]
    fn the_fold_takes_the_larger_value() {
        let left = tensor(&[2, 2], &[1.0, 8.0, 3.0, 4.0]);
        let right = tensor(&[2, 2], &[5.0, 2.0, 7.0, 0.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::inference();
        let output = layer.forward_many(&[&left, &right], &mut ctx).unwrap();

        assert_eq!(output, tensor(&[2, 2], &[5.0, 8.0, 7.0, 4.0]));
        assert!(output.is_standard_layout(), "a layer emits the C order");
        assert_eq!(ctx.pending_caches(), 0, "an inference pass parks nothing");
    }

    /// The fold runs over every input, and the arity accepts 1 input alone
    #[test]
    fn a_single_input_comes_back_unchanged() {
        let only = tensor(&[2, 2], &[1.0, -2.0, 3.0, -4.0]);

        let layer = Maximum::new();
        assert_eq!(layer.arity(), Arity::AtLeast(1));
        let mut ctx = Ctx::training();
        let output = layer.forward_many(&[&only], &mut ctx).unwrap();
        assert_eq!(output, only);

        let grad = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0], grad, "the 1 input takes every position");
    }

    /// The winner of a position takes the whole gradient of that position
    #[test]
    fn the_winner_takes_the_whole_gradient() {
        let left = tensor(&[2, 2], &[1.0, 8.0, 3.0, 4.0]);
        let right = tensor(&[2, 2], &[5.0, 2.0, 7.0, 0.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::training();
        layer.forward_many(&[&left, &right], &mut ctx).unwrap();

        let grads = layer
            .backward_many(&tensor(&[2, 2], &[10.0, 20.0, 30.0, 40.0]), &mut ctx)
            .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0], tensor(&[2, 2], &[0.0, 20.0, 0.0, 40.0]));
        assert_eq!(grads[1], tensor(&[2, 2], &[10.0, 0.0, 30.0, 0.0]));
        assert_eq!(ctx.pending_caches(), 0, "the backward pass takes the cache");
    }

    /// A tie routes the whole gradient to the first input that holds the winning value
    ///
    /// The third input ties on the middle position and takes nothing at all
    #[test]
    fn tie_routes_the_gradient_to_the_first_input() {
        let first = tensor(&[1, 3], &[1.0, 2.0, 3.0]);
        let second = tensor(&[1, 3], &[3.0, 2.0, 1.0]);
        let third = tensor(&[1, 3], &[2.0, 2.0, 2.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many(&[&first, &second, &third], &mut ctx)
            .unwrap();
        assert_eq!(output, tensor(&[1, 3], &[3.0, 2.0, 3.0]));

        let grads = layer
            .backward_many(&tensor(&[1, 3], &[1.0, 1.0, 1.0]), &mut ctx)
            .unwrap();

        assert_eq!(grads[0], tensor(&[1, 3], &[0.0, 1.0, 1.0]));
        assert_eq!(grads[1], tensor(&[1, 3], &[1.0, 0.0, 0.0]));
        assert_eq!(
            grads[2],
            tensor(&[1, 3], &[0.0, 0.0, 0.0]),
            "a later input that ties takes nothing"
        );
    }

    /// A NaN wins its position, and the first input that holds one keeps it
    #[test]
    fn a_nan_wins_its_position() {
        let left = tensor(&[1, 2], &[1.0, f32::NAN]);
        let right = tensor(&[1, 2], &[f32::NAN, 2.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many(&[&left, &right], &mut ctx).unwrap();
        assert!(output[[0, 0]].is_nan(), "a later NaN takes the position");
        assert!(output[[0, 1]].is_nan(), "an earlier NaN keeps the position");

        let grads = layer
            .backward_many(&tensor(&[1, 2], &[1.0, 1.0]), &mut ctx)
            .unwrap();
        assert_eq!(grads[0], tensor(&[1, 2], &[0.0, 1.0]));
        assert_eq!(grads[1], tensor(&[1, 2], &[1.0, 0.0]));
    }

    /// 2 inputs broadcast into each other, and each gradient comes back at its own shape
    #[test]
    fn each_side_broadcasts_into_the_other() {
        let left = tensor(&[2, 1, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = tensor(&[2, 2, 1], &[0.0, 9.0, 7.0, 0.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many(&[&left, &right], &mut ctx).unwrap();
        assert_eq!(
            output,
            tensor(
                &[2, 2, 3],
                &[1.0, 2.0, 3.0, 9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 4.0, 5.0, 6.0]
            )
        );

        let grads = layer
            .backward_many(&Tensor::ones([2, 2, 3].as_slice()), &mut ctx)
            .unwrap();
        assert_eq!(
            grads[0],
            tensor(&[2, 1, 3], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "the gradient keeps the shape of its own input"
        );
        assert_eq!(grads[1], tensor(&[2, 2, 1], &[0.0, 3.0, 3.0, 0.0]));
        assert!(grads[0].is_standard_layout(), "a layer emits the C order");
    }

    /// Rank alignment inserts the extra axis after the batch axis, and the gradient drops it
    #[test]
    fn rank_alignment_inserts_after_the_batch_axis() {
        let wide = tensor(
            &[2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let narrow = tensor(&[2, 3], &[5.0, 5.0, 5.0, 8.0, 8.0, 8.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many(&[&wide, &narrow], &mut ctx).unwrap();
        assert_eq!(
            output,
            tensor(
                &[2, 2, 3],
                &[
                    5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 8.0, 8.0, 9.0, 10.0, 11.0, 12.0
                ]
            )
        );

        let grads = layer
            .backward_many(&Tensor::ones([2, 2, 3].as_slice()), &mut ctx)
            .unwrap();
        assert_eq!(
            grads[0],
            tensor(
                &[2, 2, 3],
                &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
        );
        assert_eq!(
            grads[1],
            tensor(&[2, 3], &[2.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            "the inserted axis sums away"
        );
    }

    /// The build records 1 shape per input, and every report frees the batch axis
    #[test]
    fn the_build_records_every_input_shape() {
        let mut layer = Maximum::new();
        assert!(!layer.is_built());
        assert_eq!(layer.output_shape(), "Unknown");

        layer
            .build_many(&[Shape::known(&[2, 1, 4]), Shape::known(&[2, 3, 1])])
            .unwrap();

        assert!(layer.is_built());
        assert_eq!(
            layer.known_input_shapes(),
            Some(vec![
                Shape::with_free_batch(&[2, 1, 4]),
                Shape::with_free_batch(&[2, 3, 1]),
            ])
        );
        assert_eq!(layer.output_shape(), "(None, 3, 4)");
        assert_eq!(layer.param_count(), ParamCounts::none());
        assert!(layer.weights().is_empty(), "the layer holds no array");
    }

    /// The batch axis does not broadcast, and both the build and a pass refuse it
    #[test]
    fn the_batch_axis_does_not_broadcast() {
        let mut layer = Maximum::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[1, 3])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Maximum"), "{message}");
        assert!(message.contains("batch axis"), "{message}");

        let layer = Maximum::new();
        let mut ctx = Ctx::inference();
        let tall = tensor(&[2, 3], &[0.0; 6]);
        let flat = tensor(&[1, 3], &[0.0; 3]);
        let message = layer
            .forward_many(&[&tall, &flat], &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(message.contains("batch axis"), "{message}");
    }

    /// 2 extents that differ, where neither is 1, are refused with the axis named
    #[test]
    fn an_axis_that_cannot_broadcast_is_refused() {
        let mut layer = Maximum::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Maximum"), "{message}");
        assert!(message.contains("Axis 1"), "{message}");

        assert!(
            Maximum::new()
                .compute_output_shape_many(&[Shape::known(&[2, 3, 4]), Shape::known(&[2, 3])])
                .is_err(),
            "the rank alignment refuses a tail that does not right-align"
        );
    }

    /// A second build for other shapes is refused, and the same shapes keep the build
    #[test]
    fn a_second_build_for_other_shapes_is_refused() {
        let mut layer = Maximum::new();
        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 3])])
            .unwrap();

        // 1 layer serves every batch size, so another batch extent is the same build
        layer
            .build_many(&[Shape::known(&[5, 3]), Shape::known(&[5, 3])])
            .unwrap();

        assert!(
            layer
                .build_many(&[Shape::known(&[2, 4]), Shape::known(&[2, 4])])
                .is_err()
        );
        assert!(layer.build_many(&[Shape::known(&[2, 3])]).is_err());
    }

    /// The arity refuses an empty input list, in the build and in a pass alike
    #[test]
    fn no_input_at_all_is_refused() {
        let mut layer = Maximum::new();
        let message = layer.build_many(&[]).unwrap_err().to_string();
        assert!(message.contains("Maximum"), "{message}");
        assert!(message.contains("1 or more"), "{message}");

        let layer = Maximum::new();
        let mut ctx = Ctx::inference();
        let message = layer.forward_many(&[], &mut ctx).unwrap_err().to_string();
        assert!(message.contains("1 or more"), "{message}");

        assert!(layer.compute_output_shape_many(&[]).is_err());
    }

    /// A gradient of another shape than the output is refused
    #[test]
    fn a_gradient_of_another_shape_is_refused() {
        let left = tensor(&[1, 3], &[1.0, 2.0, 3.0]);
        let right = tensor(&[1, 3], &[3.0, 2.0, 1.0]);

        let layer = Maximum::new();
        let mut ctx = Ctx::training();
        layer.forward_many(&[&left, &right], &mut ctx).unwrap();

        assert!(
            layer
                .backward_many(&tensor(&[1, 2], &[1.0, 1.0]), &mut ctx)
                .is_err()
        );
    }
}
