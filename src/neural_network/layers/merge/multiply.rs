//! Multiply layer: the product of every input, element by element

use super::{
    broadcast_input, elementwise_merge_layer_functions, merge_layer_base_functions, merged_dims,
    reduce_to,
};
use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};

/// Multiplies its inputs element by element, and gives 1 output
///
/// The layer takes 1 input or more, and the output is the product of every input. The 2 clauses
/// of the [shape rule of the family](super) apply. The batch axis does not broadcast, and every
/// axis after the batch axis does broadcast. An input that broadcasts reaches several positions
/// of the output, and its gradient comes back at its own extents
///
/// The layer holds no trainable array and no state. The build records the shape of every input,
/// so the layer reports those shapes and refuses a second set
///
/// # The gradient of a product
///
/// Input `i` takes the gradient of the output, multiplied by the product of every OTHER input,
/// and reduced to the shape of input `i`. The backward pass multiplies the other inputs
/// together, and it never divides the whole product by input `i`. A quotient gives `0 / 0`
/// wherever an input holds a 0, and the product of the others gives the correct value there
///
/// # Notes
///
/// A training forward pass parks every input at the extents of the output, because the backward
/// pass reads those values. An inference pass parks nothing at all
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::Multiply;
/// use rustyml::neural_network::traits::Layer;
///
/// // 2 samples of 3 features, and 1 scale per sample
/// let features = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .unwrap()
///     .into_dyn();
/// let scales = Array::from_shape_vec((2, 1), vec![2.0, 10.0]).unwrap().into_dyn();
///
/// let mut layer = Multiply::new();
/// let mut ctx = Ctx::training();
/// let output = layer.forward_many_mut(&[&features, &scales], &mut ctx).unwrap();
///
/// // The scale of each sample reaches all 3 features
/// assert_eq!(output.shape(), &[2, 3]);
/// assert_eq!(output[[0, 1]], 4.0);
/// assert_eq!(output[[1, 2]], 60.0);
///
/// // 1 gradient per input, each at the shape of that input
/// let grads = layer.backward_many(&Array::ones(output.raw_dim()), &mut ctx).unwrap();
/// assert_eq!(grads[0].shape(), &[2, 3]);
/// assert_eq!(grads[1].shape(), &[2, 1]);
///
/// // The features take the scale of their own sample
/// assert_eq!(grads[0][[0, 0]], 2.0);
///
/// // The scale broadcast over 3 features, so its gradient sums those 3 features
/// assert_eq!(grads[1][[0, 0]], 6.0);
/// ```
#[derive(Debug, Default)]
pub struct Multiply {
    /// 1 shape per input the layer built for, batch axis first. `None` before the build
    built: Option<Vec<Shape>>,
}

impl Multiply {
    /// Creates a new Multiply layer
    ///
    /// # Returns
    ///
    /// - `Self` - New `Multiply` layer instance, before any build
    pub fn new() -> Self {
        Multiply::default()
    }
}

/// What the forward pass of [`Multiply`] parks for its backward pass
struct MultiplyCache {
    /// Every input of the pass, lifted to the extents of the output
    ///
    /// The backward pass leaves 1 of these tensors out of a product each time, so it needs the
    /// values themselves and not the product alone
    lifted: Vec<Tensor>,
    /// Extent of every axis of each input, as the forward pass received it
    ///
    /// The gradient of an input comes back at these extents, and the lifted form of that input
    /// holds the extents of the output instead
    shapes: Vec<Vec<usize>>,
}

impl LayerBase for Multiply {
    fn layer_type(&self) -> &str {
        "Multiply"
    }

    merge_layer_base_functions!();
}

impl Layer for Multiply {
    elementwise_merge_layer_functions!("Multiply", Arity::AtLeast(1));

    /// Lifts every input to the extents of the output, and multiplies the lifted tensors
    ///
    /// The lifted tensors are exactly what the backward pass reads, so a training pass parks
    /// them together with the extents of each input
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        self.arity().check("Multiply", inputs.len())?;
        let dims = merged_dims("Multiply", inputs)?;

        let mut lifted = Vec::with_capacity(inputs.len());
        for input in inputs {
            lifted.push(broadcast_input("Multiply", input, &dims)?);
        }

        // Every lifted tensor holds the extents of the output, so the product needs no further
        // broadcast. The first one owns its data and is in the standard memory order
        let mut output = lifted[0].clone();
        for tensor in &lifted[1..] {
            output *= tensor;
        }

        if ctx.is_training() {
            ctx.push_cache(
                "Multiply",
                MultiplyCache {
                    shapes: inputs.iter().map(|input| input.shape().to_vec()).collect(),
                    lifted,
                },
            );
        }

        Ok(output)
    }

    /// Gives input `i` the gradient times the product of every other input, at the shape of
    /// input `i`
    ///
    /// The product of the others never divides the whole product by input `i`. An input that
    /// holds a 0 therefore gives a correct gradient, where a quotient would give `0 / 0`
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let cache: MultiplyCache = ctx.pop_cache("Multiply")?;

        // Every lifted input holds the extents of the output, so the first one describes them
        if let Some(first) = cache.lifted.first()
            && grad_output.shape() != first.shape()
        {
            return Err(Error::shape_mismatch(first.shape(), grad_output.shape()));
        }

        let mut grads = Vec::with_capacity(cache.shapes.len());
        for (position, shape) in cache.shapes.iter().enumerate() {
            let mut product = grad_output.as_standard_layout().into_owned();
            for (other, tensor) in cache.lifted.iter().enumerate() {
                if other != position {
                    product *= tensor;
                }
            }
            // The input reached every position that it broadcast into, so its gradient is the
            // sum over those positions
            grads.push(reduce_to(&product, shape));
        }
        Ok(grads)
    }
}

/// Unit tests of the forward product, the leave-1-out gradient, and the refusals
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Builds a tensor of the given extents from a value list, in the standard memory order
    fn tensor(dims: &[usize], values: &[f32]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values.to_vec()).unwrap()
    }

    /// Runs 1 training forward pass over the given tensors
    fn forward(inputs: &[&Tensor], ctx: &mut Ctx) -> Tensor {
        Multiply::new().forward_many(inputs, ctx).unwrap()
    }

    /// The output is the product of every input, over 3 inputs of 1 shape
    #[test]
    fn the_output_is_the_product_of_every_input() {
        let first = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let second = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let third = tensor(&[2, 2], &[2.0, 2.0, 2.0, 2.0]);

        let mut ctx = Ctx::training();
        let output = forward(&[&first, &second, &third], &mut ctx);
        assert_eq!(output, tensor(&[2, 2], &[10.0, 24.0, 42.0, 64.0]));
        assert!(output.is_standard_layout(), "a layer emits the C order");
    }

    /// 1 input alone passes through, because the arity accepts 1
    #[test]
    fn a_single_input_passes_through() {
        let only = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);

        let mut ctx = Ctx::training();
        let layer = Multiply::new();
        let output = layer.forward_many(&[&only], &mut ctx).unwrap();
        assert_eq!(output, only);

        // The product of every OTHER input is the empty product, which is 1
        let grad = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0], grad);
    }

    /// Each input takes the gradient times the product of every other input
    #[test]
    fn each_input_takes_the_product_of_the_others() {
        let first = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let second = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let third = tensor(&[2, 2], &[2.0, 2.0, 2.0, 2.0]);

        let mut ctx = Ctx::training();
        let layer = Multiply::new();
        layer
            .forward_many(&[&first, &second, &third], &mut ctx)
            .unwrap();

        let grad = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();
        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0], tensor(&[2, 2], &[10.0, 24.0, 42.0, 64.0]));
        assert_eq!(grads[1], tensor(&[2, 2], &[2.0, 8.0, 18.0, 32.0]));
        assert_eq!(grads[2], tensor(&[2, 2], &[5.0, 24.0, 63.0, 128.0]));
        assert!(grads[0].is_standard_layout(), "a layer emits the C order");
    }

    /// An input that holds a 0 gives a correct gradient, which a quotient cannot
    ///
    /// A backward pass that divided the whole product by input `i` would give `0 / 0` at the
    /// position of the 0. The product of every other input gives the value that Keras gives
    #[test]
    fn a_zero_in_an_input_gives_a_finite_gradient() {
        let first = tensor(&[1, 2], &[0.0, 2.0]);
        let second = tensor(&[1, 2], &[3.0, 4.0]);

        let mut ctx = Ctx::training();
        let layer = Multiply::new();
        let output = layer.forward_many(&[&first, &second], &mut ctx).unwrap();
        assert_eq!(output, tensor(&[1, 2], &[0.0, 8.0]));

        let grads = layer
            .backward_many(&tensor(&[1, 2], &[1.0, 1.0]), &mut ctx)
            .unwrap();
        assert_eq!(grads[0], second);
        assert_eq!(grads[1], first);
    }

    /// An extent of 1 after the batch axis broadcasts, and its gradient sums the positions
    #[test]
    fn an_axis_of_one_position_broadcasts() {
        let features = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scales = tensor(&[2, 1], &[10.0, 20.0]);

        let mut ctx = Ctx::training();
        let layer = Multiply::new();
        let output = layer.forward_many(&[&features, &scales], &mut ctx).unwrap();
        assert_eq!(
            output,
            tensor(&[2, 3], &[10.0, 20.0, 30.0, 80.0, 100.0, 120.0])
        );

        let grad = tensor(&[2, 3], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        // Each feature takes the scale of its own sample
        assert_eq!(
            grads[0],
            tensor(&[2, 3], &[10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        );

        // The scale reached all 3 features, so it takes the sum of those 3 features
        assert_eq!(grads[1], tensor(&[2, 1], &[6.0, 15.0]));
        assert!(grads[1].is_standard_layout(), "a layer emits the C order");
    }

    /// Rank alignment inserts the extra axis after the batch axis, and the gradient drops it
    #[test]
    fn rank_alignment_inserts_after_the_batch_axis() {
        let wide = tensor(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let narrow = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);

        let mut ctx = Ctx::training();
        let layer = Multiply::new();
        let output = layer.forward_many(&[&wide, &narrow], &mut ctx).unwrap();

        // The narrow input behaves as a shape (2, 1, 2), and never as a shape (1, 2, 2)
        assert_eq!(
            output,
            tensor(&[2, 2, 2], &[1.0, 4.0, 3.0, 8.0, 15.0, 24.0, 21.0, 32.0])
        );

        let grad = Tensor::from_elem(IxDyn(&[2, 2, 2]), 1.0);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();
        assert_eq!(
            grads[0],
            tensor(&[2, 2, 2], &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0])
        );

        // The inserted axis goes away, and the sum runs over the whole axis
        assert_eq!(grads[1], tensor(&[2, 2], &[4.0, 6.0, 12.0, 14.0]));
    }

    /// An inference pass parks nothing, and a training pass parks 1 cache
    #[test]
    fn a_cache_reaches_the_context_in_training_alone() {
        let first = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let second = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);

        let mut ctx = Ctx::inference();
        forward(&[&first, &second], &mut ctx);
        assert_eq!(ctx.pending_caches(), 0);

        let mut ctx = Ctx::training();
        forward(&[&first, &second], &mut ctx);
        assert_eq!(ctx.pending_caches(), 1);
    }

    /// The batch axis does not broadcast, and the message names the 2 shapes
    #[test]
    fn the_batch_axis_does_not_broadcast() {
        let shapes = vec![Shape::known(&[2, 3]), Shape::known(&[1, 3])];
        let message = Multiply::new().build_many(&shapes).unwrap_err().to_string();
        assert!(message.contains("Multiply"), "{message}");
        assert!(message.contains("batch axis"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(1, 3)"), "{message}");
    }

    /// 2 extents that differ, where neither is 1, are refused at the build and at the pass
    #[test]
    fn two_extents_that_cannot_broadcast_are_refused() {
        let shapes = vec![Shape::known(&[2, 3]), Shape::known(&[2, 5])];
        let message = Multiply::new().build_many(&shapes).unwrap_err().to_string();
        assert!(message.contains("Multiply"), "{message}");
        assert!(message.contains("Axis 1"), "{message}");

        let first = Tensor::zeros(IxDyn(&[2, 3]));
        let second = Tensor::zeros(IxDyn(&[2, 5]));
        let mut ctx = Ctx::training();
        assert!(
            Multiply::new()
                .forward_many(&[&first, &second], &mut ctx)
                .is_err()
        );
    }

    /// A layer that is built for 1 set of shapes refuses a second set
    #[test]
    fn a_second_build_for_other_shapes_is_refused() {
        let mut layer = Multiply::new();
        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 3])])
            .unwrap();

        // The same shapes for another batch size are the same build
        layer
            .build_many(&[Shape::known(&[8, 3]), Shape::known(&[8, 3])])
            .unwrap();
        assert_eq!(
            layer.known_input_shapes(),
            Some(vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 3])
            ])
        );
        assert_eq!(layer.output_shape(), "(None, 3)");

        let message = layer
            .build_many(&[Shape::known(&[2, 4]), Shape::known(&[2, 4])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("already built"), "{message}");

        // A second input count is refused as well
        let message = layer
            .build_many(&[Shape::known(&[2, 3])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("2 inputs"), "{message}");
    }

    /// The layer takes 1 input at least, and every entry point refuses none
    #[test]
    fn no_input_at_all_is_refused() {
        let layer = Multiply::new();
        assert_eq!(layer.arity(), Arity::AtLeast(1));

        let message = Multiply::new().build_many(&[]).unwrap_err().to_string();
        assert!(message.contains("Multiply"), "{message}");
        assert!(message.contains("1 or more"), "{message}");

        assert!(layer.compute_output_shape_many(&[]).is_err());

        let mut ctx = Ctx::training();
        assert!(layer.forward_many(&[], &mut ctx).is_err());
    }

    /// The layer holds no array, so it counts no parameter and offers no checkpoint path
    #[test]
    fn the_layer_holds_no_array() {
        let mut layer = Multiply::new();
        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 1])])
            .unwrap();

        assert_eq!(layer.param_count(), ParamCounts::none());
        assert!(layer.weights().is_empty());
        assert!(layer.weights_mut().is_empty());
        assert!(layer.is_built());

        // The checkpoint records 1 shape per input, each with a free batch axis
        let build = layer.build_config().unwrap();
        assert_eq!(
            build.input_shapes,
            vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 1])
            ]
        );
    }
}
