//! The merge layer that averages its inputs element by element

use super::{
    broadcast_input, elementwise_merge_layer_functions, merge_layer_base_functions, merged_dims,
    reduce_to,
};
use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};

/// A layer that averages its inputs element by element
///
/// The layer adds every input together, and it divides the sum by the input count. The divisor
/// is the number of inputs alone. It is never the number of elements that a broadcast made, so
/// 3 inputs always give a divisor of 3
///
/// The layer takes 1 input or more, and it gives 1 output. It holds no trainable array, so a
/// checkpoint of the layer records no array. The [shape rule of the family](super) applies. The
/// batch axis does not broadcast, and every axis after it does. Rank alignment inserts each
/// extra axis directly after the batch axis
///
/// The backward pass gives 1 gradient per input, at the shape of that input. Each gradient is
/// the incoming gradient divided by the input count. An input that broadcast in the forward
/// pass then takes the sum over every position that it reached
///
/// # Notes
///
/// The sum runs over the inputs in the order the caller gave them, and the division follows the
/// whole sum. A sum of `f32` values is not associative, so that order is part of the answer
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::Average;
/// use rustyml::neural_network::traits::Layer;
///
/// // 3 inputs of 1 sample and 2 features each
/// let a = Array::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap().into_dyn();
/// let b = Array::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap().into_dyn();
/// let c = Array::from_shape_vec((1, 2), vec![8.0, 9.0]).unwrap().into_dyn();
///
/// let mut layer = Average::new();
/// let mut ctx = Ctx::training();
/// let output = layer.forward_many_mut(&[&a, &b, &c], &mut ctx).unwrap();
///
/// // The sum divided by the input count of 3
/// assert_eq!(
///     output,
///     Array::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap().into_dyn()
/// );
///
/// // Every input takes 1 third of the gradient, at its own shape
/// let grads = layer
///     .backward_many(&Array::from_elem((1, 2), 3.0).into_dyn(), &mut ctx)
///     .unwrap();
/// assert_eq!(grads.len(), 3);
/// for grad in &grads {
///     assert_eq!(grad, &Array::from_elem((1, 2), 1.0).into_dyn());
/// }
/// ```
#[derive(Debug, Default)]
pub struct Average {
    /// 1 shape per input of the layer, batch axis first. `None` before the build
    built: Option<Vec<Shape>>,
}

impl Average {
    /// Creates a layer that averages its inputs element by element
    ///
    /// The layer takes no shape here. [`Layer::build_many`] gives it 1 shape per input.
    /// [`Layer::forward_many_mut`] builds the layer as well, when a caller drives it by hand
    /// from the tensors that arrive
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Average` layer, which holds no build
    pub fn new() -> Self {
        Self { built: None }
    }
}

/// What the forward pass of [`Average`] parks for its backward pass
///
/// The backward pass gives every input a gradient at the shape of that input, so it needs the
/// shape of each one. It needs the input count for the divisor, and the list gives that count
struct AverageCache {
    /// Extent of every axis of each input, batch axis first, in the order of the inputs
    inputs: Vec<Vec<usize>>,
    /// Extent of every axis of the output, batch axis first
    output: Vec<usize>,
}

impl LayerBase for Average {
    fn layer_type(&self) -> &str {
        "Average"
    }

    merge_layer_base_functions!();
}

impl Layer for Average {
    elementwise_merge_layer_functions!("Average", Arity::AtLeast(1));

    /// Adds every input at the merged extents, and divides the sum by the input count
    ///
    /// The divisor is the input count. A broadcast changes how many elements the output holds,
    /// and it never changes the divisor. 1 input alone therefore comes back unchanged
    ///
    /// A training pass parks the shape of every input. The backward pass reduces each gradient
    /// to 1 of those shapes
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::AtLeast(1).check("Average", inputs.len())?;
        let output = merged_dims("Average", inputs)?;

        let mut total = broadcast_input("Average", inputs[0], &output)?;
        for input in &inputs[1..] {
            total += &broadcast_input("Average", input, &output)?;
        }
        // The whole sum takes 1 division, so the divisor is the input count and never an
        // element count of the output
        total /= inputs.len() as f32;

        if ctx.is_training() {
            ctx.push_cache(
                "Average",
                AverageCache {
                    inputs: inputs.iter().map(|input| input.shape().to_vec()).collect(),
                    output,
                },
            );
        }
        Ok(total)
    }

    /// Gives every input the incoming gradient divided by the input count
    ///
    /// The forward pass scales every input by the same reciprocal of the input count, so every
    /// gradient carries that same factor. An input that broadcast reached several positions of
    /// the output, and `reduce_to` sums its gradient back over each of them
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `ctx` holds no cache of this
    ///   layer
    /// - `Error::ShapeMismatch` - If `grad_output` has another shape than the output that the
    ///   forward pass gave
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let AverageCache { inputs, output } = ctx.pop_cache::<AverageCache>("Average")?;
        if grad_output.shape() != output.as_slice() {
            return Err(Error::shape_mismatch(output, grad_output.shape()));
        }

        let scaled = grad_output / inputs.len() as f32;
        Ok(inputs
            .iter()
            .map(|shape| reduce_to(&scaled, shape))
            .collect())
    }
}

/// Unit tests of the forward average, of the gradient split, and of every refusal
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Builds a tensor of the given extents from a value list
    fn tensor(dims: &[usize], values: &[f32]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values.to_vec()).unwrap()
    }

    /// The output is the sum of the inputs divided by the input count
    #[test]
    fn the_forward_pass_divides_by_the_input_count() {
        let a = tensor(&[1, 3], &[1.0, 2.0, 3.0]);
        let b = tensor(&[1, 3], &[3.0, 4.0, 5.0]);
        let c = tensor(&[1, 3], &[8.0, 9.0, 10.0]);

        let mut layer = Average::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&a, &b, &c], &mut ctx).unwrap();

        assert_eq!(output, tensor(&[1, 3], &[4.0, 5.0, 6.0]));
        assert!(output.is_standard_layout(), "a layer emits the C order");
    }

    /// The arity accepts 1 input, and the average of 1 input is that input
    #[test]
    fn a_single_input_comes_back_unchanged() {
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);

        let mut layer = Average::new();
        let mut ctx = Ctx::inference();
        let output = layer.forward_many_mut(&[&a], &mut ctx).unwrap();

        assert_eq!(output, a);
        assert_eq!(ctx.pending_caches(), 0, "an inference pass parks nothing");
    }

    /// Every input takes the incoming gradient divided by the input count
    #[test]
    fn the_backward_pass_splits_the_gradient_by_the_input_count() {
        let a = tensor(&[1, 2], &[1.0, 2.0]);
        let b = tensor(&[1, 2], &[3.0, 4.0]);
        let c = tensor(&[1, 2], &[5.0, 6.0]);

        let mut layer = Average::new();
        let mut ctx = Ctx::training();
        layer.forward_many_mut(&[&a, &b, &c], &mut ctx).unwrap();

        let grads = layer
            .backward_many(&tensor(&[1, 2], &[3.0, 6.0]), &mut ctx)
            .unwrap();

        assert_eq!(grads.len(), 3);
        for grad in &grads {
            assert_eq!(grad, &tensor(&[1, 2], &[1.0, 2.0]));
            assert!(grad.is_standard_layout(), "a layer emits the C order");
        }
        assert_eq!(ctx.pending_caches(), 0, "the backward pass takes the cache");
    }

    /// An input of 1 position on an axis reaches every position, and takes the summed gradient
    #[test]
    fn a_broadcast_input_takes_the_summed_gradient() {
        let wide = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let narrow = tensor(&[2, 1], &[1.0, 3.0]);

        let mut layer = Average::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&wide, &narrow], &mut ctx).unwrap();

        // The divisor stays 2, although the broadcast made 6 elements out of 2
        assert_eq!(output, tensor(&[2, 3], &[1.0, 1.5, 2.0, 3.5, 4.0, 4.5]));

        let grads = layer
            .backward_many(&Tensor::ones([2, 3].as_slice()), &mut ctx)
            .unwrap();

        assert_eq!(grads[0], tensor(&[2, 3], &[0.5; 6]));
        // The narrow input reached 3 positions of each row, so it takes 3 halves
        assert_eq!(grads[1], tensor(&[2, 1], &[1.5, 1.5]));
    }

    /// Rank alignment inserts the extra axis after the batch axis, in both directions
    ///
    /// A rule that aligned the ranks from the left would put the extra axis in front of the
    /// batch axis. The gradient of the shorter input would then sum over the wrong axis
    #[test]
    fn rank_alignment_inserts_after_the_batch_axis() {
        let deep = tensor(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let flat = tensor(&[1, 2], &[1.0, 1.0]);

        let mut layer = Average::new();
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&deep, &flat], &mut ctx).unwrap();

        assert_eq!(output, tensor(&[1, 2, 2], &[1.0, 1.5, 2.0, 2.5]));

        let grads = layer
            .backward_many(&Tensor::ones([1, 2, 2].as_slice()), &mut ctx)
            .unwrap();

        assert_eq!(grads[0], tensor(&[1, 2, 2], &[0.5; 4]));
        // The inserted axis holds 2 positions, and each carries a half
        assert_eq!(grads[1], tensor(&[1, 2], &[1.0, 1.0]));
    }

    /// The batch axis does not broadcast, and the build and the pass both refuse it
    #[test]
    fn the_batch_axis_does_not_broadcast() {
        let mut layer = Average::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[1, 3])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Average"), "{message}");
        assert!(message.contains("batch axis"), "{message}");

        let two = Tensor::zeros([2, 3].as_slice());
        let one = Tensor::zeros([1, 3].as_slice());
        let mut layer = Average::new();
        let mut ctx = Ctx::training();
        assert!(layer.forward_many_mut(&[&two, &one], &mut ctx).is_err());
    }

    /// 2 extents that differ, where neither is 1, cannot merge
    #[test]
    fn an_axis_that_cannot_broadcast_is_refused() {
        let mut layer = Average::new();
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Average"), "{message}");
        assert!(message.contains("Axis 1"), "{message}");

        assert!(
            layer
                .compute_output_shape_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])])
                .is_err()
        );
    }

    /// A second build keeps the layer for the same shapes, and refuses other shapes
    #[test]
    fn a_second_build_for_other_shapes_is_refused() {
        let mut layer = Average::new();
        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 3])])
            .unwrap();

        // The batch extent is not part of the build, so another batch keeps the layer
        layer
            .build_many(&[Shape::known(&[5, 3]), Shape::known(&[5, 3])])
            .unwrap();

        let message = layer
            .build_many(&[Shape::known(&[2, 4]), Shape::known(&[2, 4])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("already built"), "{message}");

        // A different input count is refused as well
        assert!(layer.build_many(&[Shape::known(&[2, 3])]).is_err());
    }

    /// The layer takes 1 input at least, and every entry point refuses none
    #[test]
    fn no_input_at_all_is_refused() {
        let mut layer = Average::new();

        let message = layer.build_many(&[]).unwrap_err().to_string();
        assert!(message.contains("Average"), "{message}");
        assert!(message.contains("1 or more"), "{message}");

        let mut ctx = Ctx::training();
        let message = layer.forward_many(&[], &mut ctx).unwrap_err().to_string();
        assert!(message.contains("1 or more"), "{message}");

        assert!(layer.compute_output_shape_many(&[]).is_err());
        assert_eq!(layer.arity(), Arity::AtLeast(1));
    }

    /// The layer reports every input shape it built for, and the output shape they give
    #[test]
    fn the_layer_reports_its_build() {
        let mut layer = Average::new();
        assert_eq!(layer.layer_type(), "Average");
        assert!(!layer.is_built());
        assert_eq!(layer.output_shape(), "Unknown");
        assert!(layer.build_config().is_none());

        layer
            .build_many(&[Shape::known(&[2, 1, 4]), Shape::known(&[2, 3, 1])])
            .unwrap();

        assert!(layer.is_built());
        let reported = vec![
            Shape::new(vec![None, Some(1), Some(4)]),
            Shape::new(vec![None, Some(3), Some(1)]),
        ];
        assert_eq!(layer.known_input_shapes().unwrap(), reported);
        assert_eq!(layer.build_config().unwrap().input_shapes, reported);
        assert_eq!(layer.output_shape(), "(None, 3, 4)");
    }

    /// The layer holds no array at all, so it contributes no path to a checkpoint
    #[test]
    fn the_layer_holds_no_array() {
        let mut layer = Average::new();
        assert_eq!(layer.param_count(), ParamCounts::none());
        assert!(layer.weights().is_empty());
        assert!(layer.weights_mut().is_empty());
        assert!(layer.parameters_mut().is_empty());
    }

    /// A backward pass with no forward pass behind it reports exactly that
    #[test]
    fn a_backward_pass_without_a_forward_pass_is_refused() {
        let layer = Average::new();
        let mut ctx = Ctx::training();

        let message = layer
            .backward_many(&Tensor::ones([2, 3].as_slice()), &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("forward pass has not been run"),
            "{message}"
        );
    }

    /// A gradient of another shape than the output is refused
    #[test]
    fn a_gradient_of_another_shape_is_refused() {
        let a = tensor(&[1, 2], &[1.0, 2.0]);
        let b = tensor(&[1, 2], &[3.0, 4.0]);

        let mut layer = Average::new();
        let mut ctx = Ctx::training();
        layer.forward_many_mut(&[&a, &b], &mut ctx).unwrap();

        assert!(
            layer
                .backward_many(&Tensor::ones([1, 3].as_slice()), &mut ctx)
                .is_err()
        );
    }
}
