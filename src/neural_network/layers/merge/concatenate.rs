//! The [`Concatenate`] merge layer, which joins its inputs along 1 axis

use super::merge_layer_base_functions;
use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::start_build_many;
use crate::neural_network::traits::{Arity, Layer, LayerBase};
use crate::neural_network::{Ctx, Shape, Tensor};
use ndarray::{Axis, Slice};

/// Joins every input along 1 axis
///
/// The layer takes 1 input or more, and the output holds every input, one band after another,
/// along the joined axis. It holds no trainable array, so no optimizer reaches it and a
/// checkpoint of it records no array. The layer records 1 build shape per input, and it
/// reports all of them
///
/// The axis counts against the FULL rank, and the batch axis is part of that rank. The axis 0
/// is therefore legal, and it joins the batches of the inputs. A negative axis counts back from
/// the end, so the axis -1 is the last axis. The layer keeps the axis as given and resolves it
/// against the rank of the inputs on each call
///
/// Every input holds the same rank, and every axis except the joined axis holds the same
/// extent. The joined axis of the output holds the sum of the extents of the inputs
///
/// # Notes
///
/// This layer shares none of the [shape rules of the elementwise merge layers](super). It runs
/// no batch check, and it broadcasts nothing. A shape `(2, 3)` and a shape `(1, 3)` are legal
/// on the axis 0, and they give the output shape `(3, 3)`. A shape `(2, 3)` and a shape
/// `(2, 1)` are refused on the axis 0, because an extent of 1 stays an extent of 1
///
/// A free axis agrees with any extent, and the fixed extent of the other side describes the
/// output. A free extent on the joined axis leaves that axis of the output free, because no
/// sum is available for it
///
/// The backward pass cuts the gradient into 1 band per input, along the joined axis and in the
/// order of the inputs. Each band carries the shape of its own input
///
/// # Examples
///
/// ```rust
/// use ndarray::IxDyn;
/// use rustyml::neural_network::layers::Concatenate;
/// use rustyml::neural_network::traits::Layer;
/// use rustyml::neural_network::{Ctx, Tensor};
///
/// // 2 samples of 2 features, and 1 more feature for each of the same 2 samples
/// let left = Tensor::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let right = Tensor::from_shape_vec(IxDyn(&[2, 1]), vec![5.0, 6.0]).unwrap();
///
/// let mut layer = Concatenate::new(-1);
/// let mut ctx = Ctx::training();
/// let joined = layer.forward_many_mut(&[&left, &right], &mut ctx).unwrap();
/// assert_eq!(
///     joined,
///     Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]).unwrap()
/// );
///
/// // Each input takes the gradient of its own band back, at its own shape
/// let grad =
///     Tensor::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let grads = layer.backward_many(&grad, &mut ctx).unwrap();
/// assert_eq!(
///     grads[0],
///     Tensor::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 4.0, 5.0]).unwrap()
/// );
/// assert_eq!(
///     grads[1],
///     Tensor::from_shape_vec(IxDyn(&[2, 1]), vec![3.0, 6.0]).unwrap()
/// );
///
/// // The axis 0 is legal, and it joins the batches instead of the features
/// let mut batches = Concatenate::new(0);
/// let mut ctx = Ctx::inference();
/// let stacked = batches.forward_many_mut(&[&left, &left], &mut ctx).unwrap();
/// assert_eq!(stacked.shape(), &[4, 2]);
/// ```
///
/// # Performance
///
/// The forward pass writes the output once. Each input is copied into its own band, so the
/// pass costs 1 tensor of the size of the output and no more
#[derive(Debug)]
pub struct Concatenate {
    /// The axis to join on, counted against the full rank. A negative value counts back from
    /// the end
    axis: i32,
    /// 1 shape per input that the build accepted, and `None` before the build
    built: Option<Vec<Shape>>,
}

impl Concatenate {
    /// Creates a layer that joins its inputs along 1 axis
    ///
    /// The layer keeps the axis as given, and resolves it against the rank of the inputs on
    /// each call. A layer that held a resolved index would join the wrong axis as soon as the
    /// rank of the input changed. An axis outside the rank therefore fails the build and the
    /// forward pass, and not this call
    ///
    /// # Parameters
    ///
    /// - `axis` - The axis to join on, counted against the full rank of the inputs. The axis 0
    ///   is the batch axis, and a negative axis counts back from the end
    ///
    /// # Returns
    ///
    /// - `Self` - A new `Concatenate` layer, which holds no build
    pub fn new(axis: i32) -> Self {
        Self { axis, built: None }
    }

    /// Resolves the axis of the layer against the rank of its inputs
    ///
    /// The rank is the full rank, batch axis included, so the axis 0 resolves to the batch
    /// axis
    ///
    /// # Parameters
    ///
    /// - `rank` - Rank that every input of the layer holds
    ///
    /// # Returns
    ///
    /// - `Result<usize, Error>` - The resolved axis, from 0 through `rank - 1`
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If `rank` is 0, or if the resolved axis falls outside the rank
    fn resolve_axis(&self, rank: usize) -> Result<usize, Error> {
        if rank == 0 {
            return Err(Error::invalid_input(
                "Concatenate joins inputs of rank 1 or more, and it received a shape of rank 0",
            ));
        }

        // The widening to i64 keeps the sum in range for every i32 input, `i32::MIN` included
        let resolved = if self.axis < 0 {
            self.axis as i64 + rank as i64
        } else {
            self.axis as i64
        };
        if resolved < 0 || resolved >= rank as i64 {
            return Err(Error::invalid_input(format!(
                "Concatenate joins on the axis {}, which falls outside an input of rank {rank}. \
                 The axis runs from {} through {}",
                self.axis,
                -(rank as i64),
                rank as i64 - 1
            )));
        }
        Ok(resolved as usize)
    }

    /// The shape of the joined output, for inputs of the given shapes
    ///
    /// Every input holds the same rank, and every axis except the joined axis holds 1 extent. A
    /// free axis agrees with any extent, and the fixed extent of the other side describes the
    /// output. The joined axis holds the sum of the input extents, and 1 free extent there
    /// leaves the joined axis of the output free
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 shape per input of the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `Result<Shape, Error>` - The shape of the tensor the layer gives back
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If `inputs` is empty, if the axis falls outside the rank, if 2
    ///   inputs hold different ranks, or if 2 extents of another axis differ. Each message
    ///   names the 2 shapes that disagree
    fn joined_shape(&self, inputs: &[Shape]) -> Result<Shape, Error> {
        let Some((first, rest)) = inputs.split_first() else {
            return Err(Error::invalid_input(
                "Concatenate takes 1 input at least, and it received none",
            ));
        };
        let rank = first.rank();
        let axis = self.resolve_axis(rank)?;

        let mut merged = first.clone();
        for input in rest {
            if input.rank() != rank {
                return Err(Error::invalid_input(format!(
                    "Concatenate joins inputs of 1 rank. The input shape {merged} holds rank \
                     {rank}, and the input shape {input} holds rank {}",
                    input.rank()
                )));
            }

            let mut axes = Vec::with_capacity(rank);
            for (position, (held, found)) in merged.axes().iter().zip(input.axes()).enumerate() {
                let extent = if position == axis {
                    // The joined axis sums, and a free extent on either side frees the sum
                    match (held, found) {
                        (Some(held), Some(found)) => Some(held + found),
                        _ => None,
                    }
                } else {
                    match (held, found) {
                        (Some(held), Some(found)) if held != found => {
                            return Err(Error::invalid_input(format!(
                                "Concatenate joins on axis {axis}, so every other axis holds 1 \
                                 extent. The input shape {merged} holds {held} on axis \
                                 {position}, and the input shape {input} holds {found}"
                            )));
                        }
                        // A free axis agrees with anything, and the fixed extent of the other
                        // side describes the output
                        (Some(held), _) => Some(*held),
                        (None, found) => *found,
                    }
                };
                axes.push(extent);
            }
            merged = Shape::new(axes);
        }
        Ok(merged)
    }
}

/// What the forward pass of [`Concatenate`] parks for its backward pass
///
/// A join needs no input value to differentiate, so the cache holds extents alone
struct ConcatenateCache {
    /// Extent of each input on the joined axis, in the order the forward pass took them
    bands: Vec<usize>,
    /// Extent of every axis of the output, which the gradient must match
    output: Vec<usize>,
}

impl LayerBase for Concatenate {
    fn layer_type(&self) -> &str {
        "Concatenate"
    }

    merge_layer_base_functions!();
}

impl Layer for Concatenate {
    fn arity(&self) -> Arity {
        Arity::AtLeast(1)
    }

    /// Records the shapes the layer joins. The layer holds no array, so nothing is allocated
    ///
    /// A layer that a graph reaches from several nodes builds once, on its first node, and
    /// every later node checks its shapes against that build
    fn build_many(&mut self, inputs: &[Shape]) -> Result<(), Error> {
        Arity::AtLeast(1).check("Concatenate", inputs.len())?;
        let Some(shapes) = start_build_many(&self.built, "Concatenate", inputs)? else {
            return Ok(());
        };
        // Refuse a set of shapes that no pass could join, before any tensor arrives
        self.joined_shape(&shapes)?;
        self.built = Some(shapes);
        Ok(())
    }

    fn compute_output_shape_many(&self, inputs: &[Shape]) -> Result<Shape, Error> {
        Arity::AtLeast(1).check("Concatenate", inputs.len())?;
        self.joined_shape(inputs)
    }

    /// Writes every input into its own band of the output, in the order of the inputs
    ///
    /// A training pass parks the extent of each band and the extents of the output. An
    /// inference pass parks nothing
    fn forward_many(&self, inputs: &[&Tensor], ctx: &mut Ctx) -> Result<Tensor, Error> {
        Arity::AtLeast(1).check("Concatenate", inputs.len())?;
        let shapes: Vec<Shape> = inputs
            .iter()
            .map(|input| Shape::known(input.shape()))
            .collect();
        let joined = self.joined_shape(&shapes)?;
        let Some(dims) = joined.dims() else {
            return Err(Error::computation(format!(
                "Concatenate joined tensors of fixed extents into the shape {joined}, which \
                 leaves an axis free"
            )));
        };
        let axis = self.resolve_axis(dims.len())?;

        // The output owns its data and takes the standard memory order, which every layer of
        // the crate emits. Each band is written once
        let mut output = Tensor::zeros(dims.as_slice());
        let mut bands = Vec::with_capacity(inputs.len());
        let mut start = 0;
        for &input in inputs {
            let extent = input.shape()[axis];
            output
                .slice_axis_mut(Axis(axis), Slice::from(start..start + extent))
                .assign(input);
            start += extent;
            bands.push(extent);
        }

        if ctx.is_training() {
            ctx.push_cache(
                "Concatenate",
                ConcatenateCache {
                    bands,
                    output: dims,
                },
            );
        }
        Ok(output)
    }

    /// Cuts the gradient into 1 band per input, along the joined axis and in the input order
    ///
    /// Each input reaches 1 band of the output and no other position, so it takes the gradient
    /// of that band alone. The band already carries the shape of its own input
    fn backward_many(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Vec<Tensor>, Error> {
        let cache: ConcatenateCache = ctx.pop_cache("Concatenate")?;
        if grad_output.shape() != cache.output.as_slice() {
            return Err(Error::shape_mismatch(cache.output, grad_output.shape()));
        }
        let axis = self.resolve_axis(cache.output.len())?;

        let mut grads = Vec::with_capacity(cache.bands.len());
        let mut start = 0;
        for extent in cache.bands {
            let band = grad_output.slice_axis(Axis(axis), Slice::from(start..start + extent));
            // A band of the gradient is a strided view, so the copy goes through `assign`,
            // which writes the standard memory order
            let mut owned = Tensor::zeros(band.raw_dim());
            owned.assign(&band);
            grads.push(owned);
            start += extent;
        }
        Ok(grads)
    }
}

/// Unit tests of the join, of the gradient of every input, and of the refusals of the layer
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// Builds a tensor of the given extents from its values, in the standard memory order
    fn tensor(dims: &[usize], values: &[f32]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(dims), values.to_vec()).unwrap()
    }

    /// The output holds every input, one band after another, along the last axis
    #[test]
    fn the_forward_pass_joins_along_the_last_axis() {
        let left = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let middle = tensor(&[2, 1], &[5.0, 6.0]);
        let right = tensor(&[2, 2], &[7.0, 8.0, 9.0, 10.0]);

        let mut layer = Concatenate::new(-1);
        let mut ctx = Ctx::training();
        let output = layer
            .forward_many_mut(&[&left, &middle, &right], &mut ctx)
            .unwrap();

        assert_eq!(
            output,
            tensor(
                &[2, 5],
                &[1.0, 2.0, 5.0, 7.0, 8.0, 3.0, 4.0, 6.0, 9.0, 10.0]
            )
        );
        assert!(output.is_standard_layout(), "a layer emits the C order");
    }

    /// The axis 0 is the batch axis, and the layer joins it like any other axis
    #[test]
    fn the_forward_pass_joins_along_the_batch_axis() {
        let left = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = tensor(&[1, 3], &[7.0, 8.0, 9.0]);

        let mut layer = Concatenate::new(0);
        let mut ctx = Ctx::inference();
        let output = layer.forward_many_mut(&[&left, &right], &mut ctx).unwrap();

        assert_eq!(
            output,
            tensor(&[3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        );
        assert_eq!(ctx.pending_caches(), 0, "an inference pass parks no cache");
    }

    /// 1 input alone is accepted, and it comes back with its own values
    #[test]
    fn one_input_alone_comes_back_unchanged() {
        let only = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut layer = Concatenate::new(1);
        let mut ctx = Ctx::training();
        let output = layer.forward_many_mut(&[&only], &mut ctx).unwrap();
        assert_eq!(output, only);

        let grads = layer.backward_many(&only, &mut ctx).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0], only);
    }

    /// Each input takes the gradient of its own band, and the bands follow the input order
    #[test]
    fn the_backward_pass_slices_the_gradient_in_input_order() {
        let left = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let middle = tensor(&[2, 1], &[5.0, 6.0]);
        let right = tensor(&[2, 2], &[7.0, 8.0, 9.0, 10.0]);

        let mut layer = Concatenate::new(-1);
        let mut ctx = Ctx::training();
        layer
            .forward_many_mut(&[&left, &middle, &right], &mut ctx)
            .unwrap();

        let grad = tensor(
            &[2, 5],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        );
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0], tensor(&[2, 2], &[1.0, 2.0, 6.0, 7.0]));
        assert_eq!(grads[1], tensor(&[2, 1], &[3.0, 8.0]));
        assert_eq!(grads[2], tensor(&[2, 2], &[4.0, 5.0, 9.0, 10.0]));
        for one in &grads {
            assert!(one.is_standard_layout(), "a layer emits the C order");
        }
        assert_eq!(ctx.pending_caches(), 0, "the backward pass takes the cache");
    }

    /// A gradient of the batch join comes back at the batch extent of each input
    #[test]
    fn the_backward_pass_slices_the_batch_axis() {
        let left = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let right = tensor(&[1, 2], &[5.0, 6.0]);

        let mut layer = Concatenate::new(0);
        let mut ctx = Ctx::training();
        layer.forward_many_mut(&[&left, &right], &mut ctx).unwrap();

        let grad = tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = layer.backward_many(&grad, &mut ctx).unwrap();

        assert_eq!(grads[0], tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(grads[1], tensor(&[1, 2], &[5.0, 6.0]));
    }

    /// No axis of this layer broadcasts, and the batch axis takes no separate rule
    ///
    /// An elementwise merge layer refuses 2 different batch extents and broadcasts an extent
    /// of 1 after the batch axis. This layer does the opposite of both
    #[test]
    fn no_axis_broadcasts() {
        let layer = Concatenate::new(0);
        assert_eq!(
            layer
                .compute_output_shape_many(&[Shape::known(&[2, 3]), Shape::known(&[1, 3])])
                .unwrap(),
            Shape::known(&[3, 3]),
            "the batch axis joins like any other axis"
        );

        let message = layer
            .compute_output_shape_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 1])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("Concatenate"), "{message}");
        assert!(message.contains("(2, 3)"), "{message}");
        assert!(message.contains("(2, 1)"), "{message}");
        assert!(message.contains("axis 1"), "{message}");

        // The live tensors meet the same rule, so a hand-driven pass is refused as well
        let wide = tensor(&[2, 3], &[1.0; 6]);
        let column = tensor(&[2, 1], &[1.0; 2]);
        let mut ctx = Ctx::training();
        assert!(
            Concatenate::new(0)
                .forward_many_mut(&[&wide, &column], &mut ctx)
                .is_err()
        );
        assert_eq!(ctx.pending_caches(), 0, "a refused pass parks no cache");
    }

    /// Every input holds the same rank, and the message names the 2 shapes
    #[test]
    fn two_ranks_that_differ_are_refused() {
        let mut layer = Concatenate::new(-1);
        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 3, 4])])
            .unwrap_err()
            .to_string();

        assert!(message.contains("Concatenate"), "{message}");
        assert!(message.contains("rank 2"), "{message}");
        assert!(message.contains("(2, 3, 4)"), "{message}");
        assert!(!layer.is_built(), "a refused build records nothing");
    }

    /// An axis outside the rank is refused, and the message names the axis and the rank
    #[test]
    fn an_axis_outside_the_rank_is_refused() {
        let layer = Concatenate::new(2);
        let message = layer
            .compute_output_shape_many(&[Shape::known(&[2, 3])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("axis 2"), "{message}");
        assert!(message.contains("rank 2"), "{message}");

        // A negative axis counts back from the end, and -3 reaches past the front
        let layer = Concatenate::new(-3);
        assert!(
            layer
                .compute_output_shape_many(&[Shape::known(&[2, 3])])
                .is_err()
        );

        // A shape of rank 0 holds no axis at all
        let layer = Concatenate::new(0);
        assert!(
            layer
                .compute_output_shape_many(&[Shape::new(Vec::new())])
                .is_err()
        );
    }

    /// A negative axis counts back from the end of the full rank
    #[test]
    fn a_negative_axis_counts_from_the_end() {
        let layer = Concatenate::new(-2);
        assert_eq!(
            layer
                .compute_output_shape_many(&[Shape::known(&[2, 3, 4]), Shape::known(&[2, 5, 4])])
                .unwrap(),
            Shape::known(&[2, 8, 4])
        );

        // The axis -3 of a rank-3 shape is the batch axis
        let layer = Concatenate::new(-3);
        assert_eq!(
            layer
                .compute_output_shape_many(&[Shape::known(&[2, 3, 4]), Shape::known(&[5, 3, 4])])
                .unwrap(),
            Shape::known(&[7, 3, 4])
        );
    }

    /// A free extent on the joined axis frees that axis of the output
    #[test]
    fn a_free_extent_on_the_joined_axis_frees_the_output() {
        let layer = Concatenate::new(-1);
        let free = Shape::new(vec![Some(2), None]);
        assert_eq!(
            layer
                .compute_output_shape_many(&[free, Shape::known(&[2, 3])])
                .unwrap(),
            Shape::new(vec![Some(2), None])
        );
    }

    /// A free axis beside the joined axis takes the fixed extent of the other side
    #[test]
    fn a_free_axis_agrees_with_any_extent() {
        let layer = Concatenate::new(0);
        assert_eq!(
            layer
                .compute_output_shape_many(&[
                    Shape::with_free_batch(&[2, 3]),
                    Shape::known(&[4, 3]),
                ])
                .unwrap(),
            Shape::new(vec![None, Some(3)]),
            "1 free batch extent leaves the joined axis free"
        );

        let layer = Concatenate::new(-1);
        assert_eq!(
            layer
                .compute_output_shape_many(&[
                    Shape::with_free_batch(&[2, 3]),
                    Shape::known(&[4, 5]),
                ])
                .unwrap(),
            Shape::known(&[4, 8]),
            "the free batch axis takes the fixed extent of the other side"
        );
    }

    /// The layer takes 1 input at least, and every entry point says so
    #[test]
    fn the_layer_takes_one_input_at_least() {
        let mut layer = Concatenate::new(-1);
        assert_eq!(layer.arity(), Arity::AtLeast(1));

        let message = layer.build_many(&[]).unwrap_err().to_string();
        assert!(message.contains("Concatenate"), "{message}");
        assert!(message.contains("1 or more"), "{message}");

        assert!(layer.compute_output_shape_many(&[]).is_err());

        let mut ctx = Ctx::training();
        assert!(layer.forward_many(&[], &mut ctx).is_err());
        assert_eq!(ctx.pending_caches(), 0, "a refused pass parks no cache");
    }

    /// A second build for other shapes is refused, and another batch extent is accepted
    #[test]
    fn a_second_build_for_other_shapes_is_refused() {
        let mut layer = Concatenate::new(-1);
        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 4])])
            .unwrap();

        // 1 layer serves every batch size, so the batch extent is not part of the build
        layer
            .build_many(&[Shape::known(&[8, 3]), Shape::known(&[8, 4])])
            .unwrap();

        let message = layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 5])])
            .unwrap_err()
            .to_string();
        assert!(message.contains("already built"), "{message}");
    }

    /// A gradient of another shape than the output is refused, and the message names both
    #[test]
    fn a_gradient_of_another_shape_is_refused() {
        let left = tensor(&[2, 2], &[1.0; 4]);
        let right = tensor(&[2, 1], &[1.0; 2]);

        let mut layer = Concatenate::new(-1);
        let mut ctx = Ctx::training();
        layer.forward_many_mut(&[&left, &right], &mut ctx).unwrap();

        let message = layer
            .backward_many(&tensor(&[2, 4], &[1.0; 8]), &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(message.contains("[2, 3]"), "{message}");
        assert!(message.contains("[2, 4]"), "{message}");
    }

    /// The layer holds no array, and it reports 1 build shape per input
    #[test]
    fn the_layer_holds_no_array_and_reports_every_build_shape() {
        let mut layer = Concatenate::new(-1);
        assert_eq!(layer.param_count(), ParamCounts::none());
        assert!(layer.weights().is_empty());
        assert!(layer.weights_mut().is_empty());
        assert!(!layer.is_built());
        assert!(layer.build_config().is_none());
        assert_eq!(layer.output_shape(), "Unknown");

        layer
            .build_many(&[Shape::known(&[2, 3]), Shape::known(&[2, 4])])
            .unwrap();

        assert!(layer.is_built());
        assert_eq!(
            layer.known_input_shapes(),
            Some(vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 4]),
            ]),
            "every reported shape carries a free batch axis"
        );
        assert_eq!(layer.output_shape(), "(None, 7)");
        assert_eq!(
            layer.build_config().unwrap().input_shapes,
            vec![
                Shape::with_free_batch(&[2, 3]),
                Shape::with_free_batch(&[2, 4]),
            ]
        );
    }
}
