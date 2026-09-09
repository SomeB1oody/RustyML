//! The merge layer that subtracts its second input from its first

use super::{
    broadcast_input, elementwise_merge_layer_functions, merge_layer_base_functions, merged_dims,
    reduce_to,
};
use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Subtracts the second input of the layer from the first, element by element
///
/// The output is `a - b`, where `a` is the first input and `b` is the second. The layer takes
/// exactly 2 inputs, and it gives 1 output. It is the 1 layer of the merge family with a fixed
/// input count. A difference of 3 or more terms has no single meaning. A caller that offers
/// another count gets a refusal that names the count it offered
///
/// The 2 inputs broadcast under the shape rule of the
/// [merge module](crate::neural_network::layers::merge). The batch axis does not broadcast, and
/// every axis after it does. Rank alignment inserts every extra axis directly after the batch
/// axis, so a shape `(2, 4)` meets a shape `(2, 3, 4)` as `(2, 1, 4)`
///
/// The layer holds no trainable array and no state. It records the 2 shapes it was built for.
/// It then reports an honest output shape, and it refuses a later pair of other shapes
///
/// # Notes
///
/// The backward pass gives 1 gradient per input, at the shape of that input. The first input
/// takes the gradient of the output, and the second input takes the negation of that gradient.
/// An input that broadcast in the forward pass reaches several positions of the output, so its
/// gradient sums over every position it reached
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::Subtract;
/// use rustyml::neural_network::traits::Layer;
///
/// // 2 samples of 3 features each, and 1 value to remove from each sample
/// let a = Array::from_shape_vec((2, 3), vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
///     .unwrap()
///     .into_dyn();
/// let b = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap().into_dyn();
///
/// let mut layer = Subtract::new();
/// let mut ctx = Ctx::training();
/// let output = layer.forward_many_mut(&[&a, &b], &mut ctx).unwrap();
///
/// // The second input broadcasts over axis 1, and the layer subtracts it
/// assert_eq!(output.shape(), &[2, 3]);
/// assert_eq!(output[[0, 0]], 4.0);
/// assert_eq!(output[[1, 2]], 8.0);
///
/// // The 2 gradients come back at the shapes of the 2 inputs
/// let grads = layer
///     .backward_many(&Array::ones(output.raw_dim()), &mut ctx)
///     .unwrap();
/// assert_eq!(grads[0].shape(), &[2, 3]);
/// assert_eq!(grads[1].shape(), &[2, 1]);
///
/// // The second input enters with a minus sign, and its 3 positions sum back into 1
/// assert_eq!(grads[1][[0, 0]], -3.0);
/// ```
#[derive(Debug, Default)]
pub struct Subtract {
    /// The 2 shapes the layer was built for, batch axis first. `None` before the build
    built: Option<Vec<Shape>>,
}

impl Subtract {
    /// Creates a layer that subtracts its second input from its first
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Subtract` layer, which holds no build
    pub fn new() -> Self {
        Self::default()
    }
}

/// What the forward pass of [`Subtract`] parks for its backward pass
///
/// The backward pass reduces each gradient back to the shape of its own input, so it needs the
/// 2 input shapes. It holds the output shape as well, to refuse a gradient of another shape
struct SubtractCache {
    /// Extent of every axis of the first input, batch axis first
    first: Vec<usize>,
    /// Extent of every axis of the second input, batch axis first
    second: Vec<usize>,
    /// Extent of every axis of the output, batch axis first
    output: Vec<usize>,
}

impl LayerBase for Subtract {
    fn layer_type(&self) -> &str {
        "Subtract"
    }

    merge_layer_base_functions!();
}

impl Layer for Subtract {
    elementwise_merge_layer_functions!("Subtract", Arity::Exactly(2));

    /// Lifts the 2 inputs to the merged extents, and subtracts the second from the first
    ///
    /// The merged extents come from the live tensors, so every extent is fixed. A training
    /// pass parks the 2 input shapes and the output shape, which is all the backward pass
    /// needs
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::Exactly(2).check("Subtract", inputs.len())?;
        let dims = merged_dims("Subtract", inputs)?;
        let mut output = broadcast_input("Subtract", inputs[0], &dims)?;
        let second = broadcast_input("Subtract", inputs[1], &dims)?;
        output -= &second;

        if ctx.is_training() {
            ctx.push_cache(
                "Subtract",
                SubtractCache {
                    first: inputs[0].shape().to_vec(),
                    second: inputs[1].shape().to_vec(),
                    output: dims,
                },
            );
        }
        Ok(output)
    }

    /// Gives the gradient to the first input, and the negation of it to the second
    ///
    /// The derivative of `a - b` is 1 for `a` and -1 for `b`, at every position. Each gradient
    /// then reduces back to the shape of its own input, which undoes the forward broadcast
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let cache: SubtractCache = ctx.pop_cache("Subtract")?;
        if grad_output.shape() != cache.output.as_slice() {
            return Err(Error::shape_mismatch(cache.output, grad_output.shape()));
        }

        let negated = -grad_output;
        Ok(vec![
            reduce_to(grad_output, &cache.first),
            reduce_to(&negated, &cache.second),
        ])
    }
}

/// Unit tests of the forward values, the 2 gradients, the broadcast, and the refusals
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Builds a tensor of the given extents from a value list
    fn tensor(dims: &[usize], values: Vec<f32>) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values).expect("the value count fits the extents")
    }

    /// The output is the first input less the second, position by position
    #[test]
    fn forward_subtracts_the_second_input_from_the_first() {
        let first = tensor(&[2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let second = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut layer = Subtract::new();
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap();

        assert_eq!(output, tensor(&[2, 3], vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0]));
        assert!(output.is_standard_layout(), "a layer emits the C order");
        assert_eq!(layer.layer_type(), "Subtract");
        assert_eq!(layer.param_count(), ParamCounts::none());
    }

    /// The order of the 2 inputs decides the sign, so the layer is not commutative
    #[test]
    fn the_order_of_the_two_inputs_decides_the_sign() {
        let first = tensor(&[1, 2], vec![1.0, 2.0]);
        let second = tensor(&[1, 2], vec![4.0, 8.0]);

        let mut ctx = Ctx::inference();
        let forward = Subtract::new()
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap();
        let reversed = Subtract::new()
            .forward_many_mut(&[&second, &first], &mut ctx)
            .unwrap();

        assert_eq!(forward, tensor(&[1, 2], vec![-3.0, -6.0]));
        assert_eq!(reversed, tensor(&[1, 2], vec![3.0, 6.0]));

        // An inference pass parks nothing at all
        assert_eq!(ctx.pending_caches(), 0);
    }

    /// The first input takes the gradient, and the second takes the negation of it
    #[test]
    fn backward_negates_the_gradient_of_the_second_input() {
        let first = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let second = tensor(&[2, 2], vec![4.0, 3.0, 2.0, 1.0]);

        let mut layer = Subtract::new();
        let mut ctx = Ctx::training();
        layer
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap();

        let grad = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads.len(), 2, "1 gradient per input");
        assert_eq!(grads[0], grad);
        assert_eq!(grads[1], tensor(&[2, 2], vec![-1.0, -2.0, -3.0, -4.0]));
        assert_eq!(ctx.pending_caches(), 0, "the backward pass took the cache");
    }

    /// An extent of 1 after the batch axis broadcasts, and its gradient sums back
    #[test]
    fn an_axis_of_one_position_broadcasts_and_sums_back() {
        let first = tensor(&[2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let second = tensor(&[2, 1], vec![1.0, 2.0]);

        let mut layer = Subtract::new();
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap();
        assert_eq!(output, tensor(&[2, 3], vec![4.0, 5.0, 6.0, 6.0, 7.0, 8.0]));

        let grad = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads[0], grad);
        assert_eq!(grads[1], tensor(&[2, 1], vec![-6.0, -15.0]));
    }

    /// Rank alignment inserts the extra axis after the batch axis, and the gradient drops it
    #[test]
    fn rank_alignment_inserts_after_the_batch_axis() {
        let first = tensor(
            &[2, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let second = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut layer = Subtract::new();
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap();

        // Sample 0 removes the row (1, 2, 3) from both of its 2 rows, and sample 1 removes the
        // row (4, 5, 6) from both of its own
        assert_eq!(
            output,
            tensor(
                &[2, 2, 3],
                vec![0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0],
            )
        );

        let grad = Tensor::ones([2, 2, 3].as_slice());
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads[0], grad);
        assert_eq!(
            grads[1],
            tensor(&[2, 3], vec![-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            "the inserted axis of 2 positions sums away"
        );
    }

    /// The layer reports the 2 shapes it built for, with a free batch axis on each
    #[test]
    fn the_layer_reports_the_shapes_it_built_for() {
        let mut layer = Subtract::new();
        assert!(!layer.is_built());
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
            ])
        );
        assert_eq!(layer.output_shape(), "(None, 3)");

        // A second build for the same shapes changes nothing, and other shapes are refused
        layer
            .build_many(&[Shape::known(&[8, 3]), Shape::known(&[8, 1])])
            .unwrap();
        let message = layer
            .build_many(&[Shape::known(&[2, 5]), Shape::known(&[2, 1])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Subtract"), "{message}");
        assert!(message.contains("already built"), "{message}");
    }

    /// The batch axis does not broadcast, and the message names the 2 shapes
    #[test]
    fn the_batch_axis_does_not_broadcast() {
        let mut layer = Subtract::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[1, 3])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Subtract"), "{message}");
        assert!(message.contains("batch axis"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(1, 3)"), "{message}");
        assert!(!layer.is_built(), "a refused build records nothing");

        // The forward pass refuses the same pair of live tensors
        let first = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let second = tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
        let mut ctx = Ctx::training();
        let message = Subtract::new()
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(message.contains("batch axis"), "{message}");
    }

    /// An axis whose 2 extents differ and where neither extent is 1 is refused
    #[test]
    fn an_axis_that_cannot_broadcast_is_refused() {
        let shapes = [Shape::known(&[2, 3]), Shape::known(&[2, 5])];
        let message = Subtract::new()
            .compute_output_shape_many(&shapes)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Subtract"), "{message}");
        assert!(message.contains("Axis 1"), "{message}");
        assert!(message.contains("neither extent is 1"), "{message}");

        // The forward pass refuses the same pair of live tensors
        let first = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let second = tensor(&[2, 5], vec![0.0; 10]);
        let mut ctx = Ctx::training();
        assert!(
            Subtract::new()
                .forward_many_mut(&[&first, &second], &mut ctx)
                .is_err()
        );
    }

    /// The layer takes 2 inputs and no other count, and every refusal names the count
    #[test]
    fn the_layer_takes_exactly_two_inputs() {
        assert_eq!(Subtract::new().arity(), Arity::Exactly(2));

        let message = Subtract::new()
            .compute_output_shape_many(&[Shape::known(&[2, 3])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Subtract"), "{message}");
        assert!(message.contains("exactly 2"), "{message}");
        assert!(message.contains("received 1"), "{message}");

        let mut layer = Subtract::new();
        let message = layer
            .build_many(&[
                Shape::known(&[2, 3]),
                Shape::known(&[2, 3]),
                Shape::known(&[2, 3]),
            ])
            .unwrap_err()
            .to_string();
        assert!(message.contains("exactly 2"), "{message}");
        assert!(message.contains("received 3"), "{message}");

        let single = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut ctx = Ctx::training();
        let message = Subtract::new()
            .forward_many(&[&single], &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(message.contains("received 1"), "{message}");
    }

    /// A gradient of another shape than the output is refused
    #[test]
    fn a_gradient_of_another_shape_is_refused() {
        let first = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let second = tensor(&[2, 2], vec![1.0, 1.0, 1.0, 1.0]);

        let mut layer = Subtract::new();
        let mut ctx = Ctx::training();
        layer
            .forward_many_mut(&[&first, &second], &mut ctx)
            .unwrap();

        let grad = tensor(&[2, 3], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(layer.backward_many(&grad, &mut ctx).is_err());
    }

    /// A backward pass with no forward pass behind it reports the missing pass
    #[test]
    fn a_backward_pass_without_a_forward_pass_is_refused() {
        let mut ctx = Ctx::training();
        let grad = tensor(&[2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let message = Subtract::new()
            .backward_many(&grad, &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("forward pass has not been run"),
            "{message}"
        );
    }
}
