//! The merge layer that takes the smaller value of its inputs, element by element

use super::{
    broadcast_input, elementwise_merge_layer_functions, merge_layer_base_functions, merged_dims,
    reduce_to,
};
use crate::error::{Context, Error};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};
use ndarray::IxDyn;

/// Takes the smaller value of its inputs, element by element
///
/// The layer folds its inputs from the left, 1 pair at a time. Every position of the output
/// holds the smallest value that the inputs offer there. The layer accepts 1 input or more,
/// and 1 input alone comes back unchanged
///
/// The inputs broadcast under the [rule of the family](super). The batch axis does not
/// broadcast, every later axis does, and rank alignment inserts each extra axis after the
/// batch axis. The backward pass gives 1 gradient per input, at the shape of that input. The
/// gradient of an input that broadcast is summed back to that shape
///
/// The layer holds no trainable array and no configuration. [`Maximum`](super::Maximum) is the
/// same layer with the other comparison
///
/// # The tie rule
///
/// A tie routes the whole gradient to the first input that holds the winning value. Every
/// other input that ties there takes 0. [`Maximum`](super::Maximum) resolves a tie by the same
/// rule. The test `backward_gives_a_tie_to_the_first_input` pins the rule
///
/// # Notes
///
/// A NaN wins its position and keeps it. The first input that holds a NaN at a position takes
/// the whole gradient there, and no later value displaces it
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::Minimum;
/// use rustyml::neural_network::traits::Layer;
///
/// let a = array![[1.0_f32, 5.0], [3.0, 2.0]].into_dyn();
/// let b = array![[4.0_f32, 2.0], [3.0, 6.0]].into_dyn();
///
/// let mut layer = Minimum::new();
/// let mut ctx = Ctx::training();
/// let output = layer.forward_many_mut(&[&a, &b], &mut ctx).unwrap();
/// assert_eq!(output, array![[1.0_f32, 2.0], [3.0, 2.0]].into_dyn());
///
/// // Position (1, 0) is a tie of 3.0, and the first input takes the whole gradient there
/// let seed = array![[1.0_f32, 1.0], [1.0, 1.0]].into_dyn();
/// let grads = layer.backward_many(&seed, &mut ctx).unwrap();
/// assert_eq!(grads[0], array![[1.0_f32, 0.0], [1.0, 1.0]].into_dyn());
/// assert_eq!(grads[1], array![[0.0_f32, 1.0], [0.0, 0.0]].into_dyn());
/// ```
#[derive(Debug, Default)]
pub struct Minimum {
    /// Shape of every input the layer was built for, batch axis first. `None` before the build
    built: Option<Vec<Shape>>,
}

impl Minimum {
    /// Creates a new Minimum layer
    ///
    /// # Returns
    ///
    /// - `Self` - New `Minimum` layer instance, before its build
    pub fn new() -> Self {
        Minimum::default()
    }
}

/// What the forward pass of [`Minimum`] parks for its backward pass
///
/// The winner list is the whole answer of the backward pass, so the cache holds no tensor of
/// the forward pass. It costs 1 entry per element of the output, whatever number of inputs the
/// layer took
struct MinimumCache {
    /// Extent of every axis of the output, batch axis first
    output: Vec<usize>,
    /// Extent of every axis of each input, in the order the forward pass took them
    inputs: Vec<Vec<usize>>,
    /// Position of the input that holds the smallest value, 1 entry per element of the output
    ///
    /// The entries follow the C order of the output. A tie names the first input that
    /// holds the value, which is the [tie rule](Minimum) of the layer
    winners: Vec<usize>,
}

impl LayerBase for Minimum {
    fn layer_type(&self) -> &str {
        "Minimum"
    }

    merge_layer_base_functions!();
}

impl Layer for Minimum {
    elementwise_merge_layer_functions!("Minimum", Arity::AtLeast(1));

    /// Folds the inputs from the left, and keeps the smaller value of each pair
    ///
    /// Each input reaches the extents of the output first, so 1 comparison serves every
    /// position. A training pass parks the winner of each position, because the backward pass
    /// routes the whole gradient of a position to that 1 input
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the layer received no input, or if the shape rule of the
    ///   family refuses the shapes of the tensors
    /// - `Error::Computation` - If an input cannot take the rank of the output
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::AtLeast(1).check("Minimum", inputs.len())?;
        let dims = merged_dims("Minimum", inputs)?;

        let mut folded = broadcast_input("Minimum", inputs[0], &dims)?;
        let mut winners = vec![0_usize; folded.len()];
        for (position, &input) in inputs.iter().enumerate().skip(1) {
            let lifted = broadcast_input("Minimum", input, &dims)?;
            for ((value, &candidate), winner) in
                folded.iter_mut().zip(lifted.iter()).zip(winners.iter_mut())
            {
                // A position that holds a NaN keeps it. No later candidate can replace a NaN
                // value, however small the candidate is.
                let takes = !value.is_nan() && (candidate.is_nan() || candidate < *value);
                if takes {
                    *value = candidate;
                    *winner = position;
                }
            }
        }

        if ctx.is_training() {
            ctx.push_cache(
                "Minimum",
                MinimumCache {
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

    /// Routes the gradient of each position to the 1 input that won that position
    ///
    /// Every other input receives 0 there, because a change of a value that the minimum did
    /// not take changes no output. Each gradient then reduces to the shape of its own input,
    /// which sums over every position that the input reached through a broadcast
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer
    /// - `Error::ShapeMismatch` - If `grad_output` does not carry the shape of the output that
    ///   the forward pass gave
    /// - `Error::Computation` - If a routed buffer cannot take the shape of the output
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let cache: MinimumCache = ctx.pop_cache("Minimum")?;
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
                .context("Failed to give a Minimum gradient the shape of the output")?;
            grads.push(reduce_to(&whole, shape));
        }
        Ok(grads)
    }
}

/// Unit tests of the fold, the tie rule, the broadcast, and the refusals of [`Minimum`]
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tensor from its extents and its values, in the C order
    fn tensor(dims: &[usize], values: &[f32]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values.to_vec()).expect("the values fill the shape")
    }

    /// Runs a forward pass of a fresh layer over the tensors, and gives the layer back
    fn forward(inputs: &[&Tensor], ctx: &mut Ctx) -> (Minimum, Tensor) {
        let mut layer = Minimum::new();
        let output = layer
            .forward_many_mut(inputs, ctx)
            .expect("the shapes merge");
        (layer, output)
    }

    /// Every position of the output holds the smallest value of the inputs
    #[test]
    fn forward_takes_the_smallest_value() {
        let a = tensor(&[2, 2], &[1.0, 5.0, 3.0, 2.0]);
        let b = tensor(&[2, 2], &[4.0, 2.0, 3.0, 6.0]);
        let c = tensor(&[2, 2], &[0.0, 9.0, 7.0, 1.0]);

        let mut ctx = Ctx::inference();
        let (_, output) = forward(&[&a, &b, &c], &mut ctx);
        assert_eq!(output, tensor(&[2, 2], &[0.0, 2.0, 3.0, 1.0]));
        assert!(output.is_standard_layout(), "a layer emits the C order");
    }

    /// The arity accepts 1 input, and that 1 input comes back unchanged
    #[test]
    fn forward_accepts_a_single_input() {
        let a = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut ctx = Ctx::training();
        let (layer, output) = forward(&[&a], &mut ctx);
        assert_eq!(output, a);
        assert_eq!(layer.arity(), Arity::AtLeast(1));

        let grads = layer
            .backward_many(&Tensor::ones(output.raw_dim()), &mut ctx)
            .expect("the cache is there");
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0], Tensor::ones([2, 3].as_slice()));
    }

    /// The winner of a position takes the whole gradient, and every other input takes 0
    #[test]
    fn backward_routes_the_whole_gradient_to_the_winner() {
        let a = tensor(&[2, 2], &[1.0, 5.0, 3.0, 2.0]);
        let b = tensor(&[2, 2], &[4.0, 2.0, 3.0, 6.0]);
        let c = tensor(&[2, 2], &[0.0, 9.0, 7.0, 1.0]);

        let mut ctx = Ctx::training();
        let (layer, _) = forward(&[&a, &b, &c], &mut ctx);
        let grads = layer
            .backward_many(&tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), &mut ctx)
            .expect("the cache is there");

        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0], tensor(&[2, 2], &[0.0, 0.0, 3.0, 0.0]));
        assert_eq!(grads[1], tensor(&[2, 2], &[0.0, 2.0, 0.0, 0.0]));
        assert_eq!(grads[2], tensor(&[2, 2], &[1.0, 0.0, 0.0, 4.0]));
        assert_eq!(ctx.pending_caches(), 0, "the backward pass took the cache");
    }

    /// A tie routes the whole gradient to the first input that holds the winning value
    ///
    /// All 3 inputs tie at 2 of the 3 positions, and the first input takes every position
    #[test]
    fn backward_gives_a_tie_to_the_first_input() {
        let a = tensor(&[1, 3], &[2.0, 2.0, 5.0]);
        let b = tensor(&[1, 3], &[2.0, 7.0, 5.0]);
        let c = tensor(&[1, 3], &[2.0, 7.0, 5.0]);

        let mut ctx = Ctx::training();
        let (layer, output) = forward(&[&a, &b, &c], &mut ctx);
        assert_eq!(output, tensor(&[1, 3], &[2.0, 2.0, 5.0]));

        let grads = layer
            .backward_many(&Tensor::ones(output.raw_dim()), &mut ctx)
            .expect("the cache is there");
        assert_eq!(grads[0], tensor(&[1, 3], &[1.0, 1.0, 1.0]));
        assert_eq!(grads[1], tensor(&[1, 3], &[0.0, 0.0, 0.0]));
        assert_eq!(grads[2], tensor(&[1, 3], &[0.0, 0.0, 0.0]));
    }

    /// A NaN wins its position, and the first input that holds one keeps it
    #[test]
    fn a_nan_wins_its_position() {
        let a = tensor(&[1, 2], &[1.0, f32::NAN]);
        let b = tensor(&[1, 2], &[f32::NAN, 2.0]);

        let layer = Minimum::new();
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many(&[&a, &b], &mut ctx)
            .expect("the shapes merge");
        assert!(output[[0, 0]].is_nan(), "a later NaN takes the position");
        assert!(output[[0, 1]].is_nan(), "an earlier NaN keeps the position");

        let grads = layer
            .backward_many(&tensor(&[1, 2], &[1.0, 1.0]), &mut ctx)
            .expect("the cache is there");
        assert_eq!(grads[0], tensor(&[1, 2], &[0.0, 1.0]));
        assert_eq!(grads[1], tensor(&[1, 2], &[1.0, 0.0]));
    }

    /// An axis of 1 position broadcasts, and its gradient sums back to that 1 position
    #[test]
    fn an_axis_of_one_position_broadcasts() {
        let a = tensor(&[2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let b = tensor(&[2, 1], &[2.0, 5.0]);

        let mut ctx = Ctx::training();
        let (layer, output) = forward(&[&a, &b], &mut ctx);
        assert_eq!(output, tensor(&[2, 3], &[1.0, 2.0, 2.0, 4.0, 2.0, 5.0]));

        let grads = layer
            .backward_many(&Tensor::ones(output.raw_dim()), &mut ctx)
            .expect("the cache is there");
        assert_eq!(grads[0], tensor(&[2, 3], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]));
        assert_eq!(
            grads[1],
            tensor(&[2, 1], &[2.0, 1.0]),
            "the gradient of a broadcast input sums back to its own shape"
        );
    }

    /// Rank alignment inserts the extra axis after the batch axis, and the gradient drops it
    #[test]
    fn a_shorter_input_aligns_after_the_batch_axis() {
        let a = tensor(&[2, 2, 2], &[1.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = tensor(&[2, 2], &[2.0, 7.0, 6.0, 9.0]);

        let mut ctx = Ctx::training();
        let (layer, output) = forward(&[&a, &b], &mut ctx);
        assert_eq!(
            output,
            tensor(&[2, 2, 2], &[1.0, 7.0, 2.0, 4.0, 5.0, 6.0, 6.0, 8.0])
        );

        let grads = layer
            .backward_many(&Tensor::ones(output.raw_dim()), &mut ctx)
            .expect("the cache is there");
        assert_eq!(
            grads[0],
            tensor(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        );
        assert_eq!(
            grads[1],
            tensor(&[2, 2], &[1.0, 1.0, 1.0, 0.0]),
            "the inserted axis goes away, and the sum runs over it"
        );
    }

    /// The batch axis does not broadcast, and the message names both shapes
    #[test]
    fn the_batch_axis_is_refused() {
        let mut layer = Minimum::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[1, 3])])
            .expect_err("the batch axis does not broadcast")
            .to_string();
        assert!(message.contains("Minimum"), "{message}");
        assert!(message.contains("batch axis"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(1, 3)"), "{message}");

        // A forward pass over the live tensors refuses the same pair
        let a = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tensor(&[1, 3], &[1.0, 2.0, 3.0]);
        let mut ctx = Ctx::training();
        let refused = Minimum::new().forward_many_mut(&[&a, &b], &mut ctx);
        assert!(refused.is_err());
    }

    /// 2 extents of another axis that differ, where neither is 1, are refused
    #[test]
    fn an_axis_that_cannot_broadcast_is_refused() {
        let mut layer = Minimum::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])])
            .expect_err("3 and 5 do not broadcast")
            .to_string();
        assert!(message.contains("Minimum"), "{message}");
        assert!(message.contains("Axis 1"), "{message}");
        assert!(!layer.is_built(), "a refused build records nothing");

        // The shape algebra refuses the pair as well, before any tensor arrives
        let refused =
            layer.compute_output_shape_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])]);
        assert!(refused.is_err());
    }

    /// A second build for other shapes is refused, and the message names both shapes
    #[test]
    fn a_second_build_for_other_shapes_is_refused() {
        let mut layer = Minimum::new();
        let shapes = vec![Shape::known(&[2, 3]), Shape::known(&[2, 3])];
        layer.build_many(&shapes).expect("the shapes merge");
        assert_eq!(
            layer.known_input_shapes(),
            Some(vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 3])
            ]),
            "a build report frees the batch axis"
        );
        assert_eq!(layer.output_shape(), "(None, 3)");

        // The same shapes build again and change nothing
        layer.build_many(&shapes).expect("the build is idempotent");

        let message = layer
            .build_many(&[Shape::known(&[2, 4]), Shape::known(&[2, 4])])
            .expect_err("the layer is built for another shape")
            .to_string();
        assert!(message.contains("Minimum"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(2, 4)"), "{message}");
    }

    /// The layer takes 1 input or more, and it refuses an empty input list
    #[test]
    fn no_input_at_all_is_refused() {
        let mut layer = Minimum::new();
        let message = layer
            .build_many(&[])
            .expect_err("the layer takes 1 input at least")
            .to_string();
        assert!(message.contains("Minimum"), "{message}");
        assert!(message.contains("1 or more"), "{message}");

        let mut ctx = Ctx::training();
        assert!(layer.forward_many(&[], &mut ctx).is_err());
        assert!(layer.compute_output_shape_many(&[]).is_err());
        assert_eq!(layer.param_count(), ParamCounts::none());
    }
}
