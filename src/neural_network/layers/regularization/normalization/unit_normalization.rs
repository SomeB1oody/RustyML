//! Unit normalization layer and its axis configuration, with support for single-axis and
//! multi-axis normalization

use super::folds::{rows_per_block, segment_dot};
use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::no_trainable_parameters_layer_functions;
use crate::neural_network::traits::Layer;
use crate::parallel_gates::cheap_map_parallel_threshold;
use ndarray::{Axis, IxDyn, Zip};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

/// Largest scale the layer applies. It equals `1 / epsilon` for a fixed `epsilon` of `1e-12`
///
/// The bound keeps a group of near-zero elements from being scaled up without limit. Written as
/// a literal because `1e12` and `1.0 / 1e-12` round to the same `f32`, which is
/// `999999995904.0`
const MAX_SCALE: f32 = 1e12;

/// Which axes an L2 norm reduces over
///
/// Defaults to [`UnitNormalizationAxis::Default`], the last axis. Under the channels-last
/// layout, that is the channel axis
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum UnitNormalizationAxis {
    /// Normalize along the last axis (the feature or channel axis)
    #[default]
    Default,
    /// Normalize along a single named axis, counting from 0 and including the batch axis
    Custom(usize),
    /// Normalize jointly over several axes. 1 norm covers the combined elements of those axes,
    /// so the axes need not be next to each other
    Multiple(Vec<usize>),
}

/// Scales each group of elements to an L2 norm of 1
///
/// The output has the shape of the input. A group is 1 set of elements the norm reduces over,
/// picked by the axis configuration. With the default axis and a `[batch_size, features]`
/// input, each sample becomes a unit vector. The layer holds no parameter, and it behaves the
/// same in training and in inference
///
/// The scale is `1 / sqrt(sum of squares)`, capped at `1e12`. The cap is `1 / epsilon` for a
/// fixed `epsilon` of `1e-12`, and it takes over for every group whose norm falls below
/// `1e-12`. An all-zero group is the limiting case. It is scaled by `1e12` and stays all zero,
/// rather than dividing by zero
///
/// Unlike [`LayerNormalization`](crate::neural_network::layers::LayerNormalization), this layer
/// subtracts no mean and learns no `gamma` or `beta`. It changes the length of each group and
/// never its direction. This makes it the layer for cosine similarity, for a metric or
/// retrieval head. Use it anywhere a downstream dot product must read as an angle
///
/// # Notes
///
/// A group whose scale hit the cap returns the derivative of the function this layer computed,
/// which is the cap itself. The value is finite, not `NaN`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array2;
///
/// // A rank-2 input: 2 samples of 3 features each
/// let x = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 0.0, 5.0, 12.0])
///     .unwrap()
///     .into_dyn();
///
/// let mut model = Sequential::new();
/// model
///     .add(UnitNormalization::new(UnitNormalizationAxis::Default).unwrap())
///     .compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // View model structure
/// model.summary();
///
/// let unit = model.predict(&x).unwrap();
///
/// // The 3-4-5 triangle becomes 0.6, 0.8, and the 5-12-13 triangle becomes 5/13, 12/13
/// assert!((unit[[0, 0]] - 0.6).abs() < 1e-6);
/// assert!((unit[[0, 1]] - 0.8).abs() < 1e-6);
/// assert!((unit[[1, 2]] - 12.0 / 13.0).abs() < 1e-6);
///
/// // Every row now has a length of 1
/// for row in 0..2 {
///     let length: f32 = (0..3).map(|c| unit[[row, c]] * unit[[row, c]]).sum();
///     assert!((length - 1.0).abs() < 1e-6);
/// }
/// ```
///
/// # Performance
///
/// The layer takes a fused row path when the normalized axes are the trailing block of the
/// shape, which the default axis always is. Each group is then 1 contiguous run. The pass costs
/// 2 linear walks of the input, 1 for the norm and 1 for the scale. Any other axis choice leaves
/// the groups as strided lanes. That path reduces the groups in place instead of paying for a
/// transpose in and a transpose back out. It holds 1 more tensor of the input size while it
/// folds the squares
///
/// `forward` also keeps a copy of its output, which the backward pass reads. `predict` keeps
/// nothing
#[derive(Debug)]
pub struct UnitNormalization {
    /// Axes the L2 norm reduces over
    axis: UnitNormalizationAxis,
    /// Shape of the most recent forward input. The backward pass needs it to check the gradient
    input_shape: Option<Vec<usize>>,
    /// Scale the most recent forward pass applied, 1 element per group, in the input shape with
    /// a 1 at every normalized axis
    scale: Option<Tensor>,
    /// 1.0 for a group whose scale came from its norm, and 0.0 for a group the cap took over.
    /// Same shape as `scale`
    from_norm: Option<Tensor>,
    /// Output of the most recent forward pass, which the backward pass reads
    output: Option<Tensor>,
}

impl UnitNormalization {
    /// Creates a new UnitNormalization layer
    ///
    /// # Parameters
    ///
    /// - `axis` - Axes the L2 norm reduces over. Use [`UnitNormalizationAxis::Default`] for the
    ///   last axis
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `UnitNormalization` layer instance
    ///
    /// # Notes
    ///
    /// An axis is only checked against the input rank at the forward pass, because the
    /// constructor sees no tensor. Only the shape of the axis list itself is checked here
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `axis` is a [`UnitNormalizationAxis::Multiple`] that is
    ///   empty or that names an axis more than once
    pub fn new(axis: UnitNormalizationAxis) -> Result<Self, Error> {
        if let UnitNormalizationAxis::Multiple(axes) = &axis {
            if axes.is_empty() {
                return Err(Error::invalid_parameter(
                    "axis",
                    "is an empty list, and a norm must reduce over at least 1 axis",
                ));
            }
            for (position, &a) in axes.iter().enumerate() {
                if axes[..position].contains(&a) {
                    return Err(Error::invalid_parameter(
                        "axis",
                        format!("names axis {a} more than once"),
                    ));
                }
            }
        }

        Ok(UnitNormalization {
            axis,
            input_shape: None,
            scale: None,
            from_norm: None,
            output: None,
        })
    }

    /// Resolves the axis configuration against a concrete input rank
    ///
    /// # Returns
    ///
    /// - `Result<Vec<usize>, Error>` - The normalized axes, in ascending order
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the rank is below 2, or if an axis is out of bounds for it
    fn resolve_axes(&self, ndim: usize) -> Result<Vec<usize>, Error> {
        if ndim < 2 {
            return Err(Error::invalid_input(format!(
                "UnitNormalization layer expects an input with a batch axis and at least 1 more axis, got a {ndim}D tensor"
            )));
        }

        let mut axes = match &self.axis {
            UnitNormalizationAxis::Default => vec![ndim - 1],
            UnitNormalizationAxis::Custom(a) => vec![*a],
            UnitNormalizationAxis::Multiple(list) => list.clone(),
        };
        if let Some(&a) = axes.iter().find(|&&a| a >= ndim) {
            return Err(Error::invalid_input(format!(
                "UnitNormalization layer normalizes axis {a}, which is out of bounds for a {ndim}D input"
            )));
        }
        axes.sort_unstable();
        Ok(axes)
    }

    /// Checks a tensor entering the layer and resolves its normalized axes
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the rank is below 2, or if an axis is out of bounds for it
    /// - `Error::EmptyInput` - If any axis has an extent of 0
    fn validate(&self, input: &Tensor) -> Result<Vec<usize>, Error> {
        let axes = self.resolve_axes(input.ndim())?;
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }
        Ok(axes)
    }

    /// Runs the forward arithmetic without touching any cache
    ///
    /// Returns `(output, scale, from_norm)`, where the last 2 carry 1 element per group in the
    /// input shape with a 1 at every normalized axis
    fn normalize(input: &Tensor, axes: &[usize]) -> (Tensor, Tensor, Tensor) {
        let shape = input.shape();
        let group_shape = group_shape(shape, axes);

        let Some(group_len) = trailing_group_len(shape, axes) else {
            return strided_forward(input, axes, &group_shape);
        };

        let input_standard = input.as_standard_layout();
        let x = input_standard.as_slice().unwrap();
        let groups = x.len() / group_len;
        let mut output = Tensor::zeros(IxDyn(shape));
        let mut scale = vec![0.0f32; groups];
        let mut from_norm = vec![0.0f32; groups];
        row_forward(
            x,
            group_len,
            x.len() >= cheap_map_parallel_threshold(),
            output.as_slice_mut().unwrap(),
            &mut scale,
            &mut from_norm,
        );

        (
            output,
            into_group_tensor(scale, &group_shape),
            into_group_tensor(from_norm, &group_shape),
        )
    }
}

/// The input shape with a 1 at every normalized axis, which is the shape of 1 value per group
fn group_shape(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let mut group_shape = shape.to_vec();
    for &a in axes {
        group_shape[a] = 1;
    }
    group_shape
}

/// Elements per group when the normalized axes are the trailing block of the shape, and `None`
/// otherwise
///
/// Under that layout a group is 1 contiguous run of a C-order buffer, so the whole pass reduces
/// to a walk over equal-length rows. `axes` is in ascending order and holds no duplicate. The
/// axes form the trailing block exactly when the first of them sits `axes.len()` positions from
/// the end
fn trailing_group_len(shape: &[usize], axes: &[usize]) -> Option<usize> {
    (axes[0] == shape.len() - axes.len()).then(|| shape[axes[0]..].iter().product())
}

/// Wraps 1 value per group in the group shape, so it broadcasts back over the input
fn into_group_tensor(values: Vec<f32>, group_shape: &[usize]) -> Tensor {
    Tensor::from_shape_vec(IxDyn(group_shape), values).unwrap()
}

/// The scale for a group, and whether its norm or the cap decided it
///
/// The comparison is written so that a `square_sum` of `NaN` takes the norm branch and carries
/// the `NaN` into the output. It also sends an infinite reciprocal, which an all-zero group
/// gives, to the cap
fn group_scale(square_sum: f32) -> (f32, f32) {
    let from_norm = 1.0 / square_sum.sqrt();
    if from_norm > MAX_SCALE {
        (MAX_SCALE, 0.0)
    } else {
        (from_norm, 1.0)
    }
}

/// Scales each row of a standard-layout `[groups, group_len]` slice to an L2 norm of 1
///
/// Rows are independent, and each row runs entirely inside 1 task, so neither the `parallel`
/// flag nor the rows-per-task chunking changes the result bits
fn row_forward(
    x: &[f32],
    group_len: usize,
    parallel: bool,
    output: &mut [f32],
    scale: &mut [f32],
    from_norm: &mut [f32],
) {
    let rows = rows_per_block(group_len);
    let chunk = rows * group_len;
    type ForwardChunks<'a> = ((&'a mut [f32], &'a [f32]), (&'a mut [f32], &'a mut [f32]));
    let task = |((out_c, x_c), (scale_c, norm_c)): ForwardChunks| {
        let row_iter = x_c
            .chunks_exact(group_len)
            .zip(out_c.chunks_exact_mut(group_len))
            .zip(scale_c.iter_mut().zip(norm_c.iter_mut()));
        for ((x_row, out_row), (s, n)) in row_iter {
            let (row_scale, row_from_norm) = group_scale(segment_dot(x_row, x_row, 1.0));
            for (o, &v) in out_row.iter_mut().zip(x_row) {
                *o = v * row_scale;
            }
            *s = row_scale;
            *n = row_from_norm;
        }
    };
    if parallel {
        output
            .par_chunks_mut(chunk)
            .zip(x.par_chunks(chunk))
            .zip(
                scale
                    .par_chunks_mut(rows)
                    .zip(from_norm.par_chunks_mut(rows)),
            )
            .for_each(task);
    } else {
        output
            .chunks_mut(chunk)
            .zip(x.chunks(chunk))
            .zip(scale.chunks_mut(rows).zip(from_norm.chunks_mut(rows)))
            .for_each(task);
    }
}

/// Composes the input gradient of a standard-layout `[groups, group_len]` slice
///
/// `dx = scale * (grad_output - output * dot(grad_output, output))`, where the dot product runs
/// over the group. A group the cap took over was scaled by a constant, so its second term drops
fn row_backward(
    grad_output: &[f32],
    output: &[f32],
    group_len: usize,
    scale: &[f32],
    from_norm: &[f32],
    parallel: bool,
    grad_input: &mut [f32],
) {
    let rows = rows_per_block(group_len);
    let chunk = rows * group_len;
    type BackwardChunks<'a> = (
        (&'a mut [f32], (&'a [f32], &'a [f32])),
        (&'a [f32], &'a [f32]),
    );
    let task = |((dx_c, (g_c, out_c)), (scale_c, norm_c)): BackwardChunks| {
        let row_iter = g_c
            .chunks_exact(group_len)
            .zip(out_c.chunks_exact(group_len))
            .zip(dx_c.chunks_exact_mut(group_len))
            .zip(scale_c.iter().zip(norm_c.iter()));
        for (((g_row, out_row), dx_row), (&s, &n)) in row_iter {
            let dot = segment_dot(g_row, out_row, 1.0) * n;
            for ((d, &gv), &yv) in dx_row.iter_mut().zip(g_row).zip(out_row) {
                *d = s * (gv - yv * dot);
            }
        }
    };
    if parallel {
        grad_input
            .par_chunks_mut(chunk)
            .zip(grad_output.par_chunks(chunk).zip(output.par_chunks(chunk)))
            .zip(scale.par_chunks(rows).zip(from_norm.par_chunks(rows)))
            .for_each(task);
    } else {
        grad_input
            .chunks_mut(chunk)
            .zip(grad_output.chunks(chunk).zip(output.chunks(chunk)))
            .zip(scale.chunks(rows).zip(from_norm.chunks(rows)))
            .for_each(task);
    }
}

/// Reduces `full` over `axes` and returns the result in the group shape
///
/// The reduction drops the named axes from the back, which leaves the index of every axis still
/// to come untouched. Reinserting them from the front then rebuilds the group shape
fn reduce_to_groups(full: Tensor, axes: &[usize], group_shape: &[usize]) -> Tensor {
    let mut reduced = full;
    for &a in axes.iter().rev() {
        reduced = reduced.sum_axis(Axis(a));
    }
    for &a in axes {
        reduced = reduced.insert_axis(Axis(a));
    }
    debug_assert_eq!(reduced.shape(), group_shape);
    reduced
}

/// Runs the forward pass for axes that are not the trailing block of the shape
///
/// Such groups are strided lanes rather than contiguous rows. The pass reduces them in place
/// through ndarray, because a transpose to the row layout would cost 2 more copies of the whole
/// input
fn strided_forward(
    input: &Tensor,
    axes: &[usize],
    group_shape: &[usize],
) -> (Tensor, Tensor, Tensor) {
    let mut squares = Tensor::zeros(input.raw_dim());
    Zip::from(&mut squares)
        .and(input)
        .for_each(|o, &v| *o = v * v);

    let mut scale = reduce_to_groups(squares, axes, group_shape);
    let mut from_norm = Tensor::zeros(scale.raw_dim());
    Zip::from(&mut scale)
        .and(&mut from_norm)
        .for_each(|s, n| (*s, *n) = group_scale(*s));

    let mut output = Tensor::zeros(input.raw_dim());
    let broadcast_scale = scale.broadcast(input.raw_dim()).unwrap();
    Zip::from(&mut output)
        .and(input)
        .and(&broadcast_scale)
        .for_each(|o, &v, &s| *o = v * s);

    (output, scale, from_norm)
}

/// Runs the backward pass for axes that are not the trailing block of the shape
///
/// Computes the same `scale * (grad_output - output * dot(grad_output, output))` the row path
/// does, with the per-group dot product reduced in place
fn strided_backward(
    grad_output: &Tensor,
    output: &Tensor,
    scale: &Tensor,
    from_norm: &Tensor,
    axes: &[usize],
) -> Tensor {
    let mut products = Tensor::zeros(grad_output.raw_dim());
    Zip::from(&mut products)
        .and(grad_output)
        .and(output)
        .for_each(|o, &g, &y| *o = g * y);

    let mut dot = reduce_to_groups(products, axes, scale.shape());
    // A group the cap took over was scaled by a constant, so its second term drops
    Zip::from(&mut dot).and(from_norm).for_each(|d, &n| *d *= n);

    let mut grad_input = Tensor::zeros(grad_output.raw_dim());
    let broadcast_scale = scale.broadcast(grad_output.raw_dim()).unwrap();
    let broadcast_dot = dot.broadcast(grad_output.raw_dim()).unwrap();
    Zip::from(&mut grad_input)
        .and(grad_output)
        .and(output)
        .and(&broadcast_scale)
        .and(&broadcast_dot)
        .for_each(|dx, &g, &y, &s, &d| *dx = s * (g - y * d));

    grad_input
}

impl Layer for UnitNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let axes = self.validate(input)?;
        let (output, scale, from_norm) = Self::normalize(input, &axes);

        self.input_shape = Some(input.shape().to_vec());
        self.scale = Some(scale);
        self.from_norm = Some(from_norm);
        self.output = Some(output.clone());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        let axes = self.validate(input)?;
        Ok(Self::normalize(input, &axes).0)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let (Some(input_shape), Some(scale), Some(from_norm), Some(output)) = (
            &self.input_shape,
            &self.scale,
            &self.from_norm,
            &self.output,
        ) else {
            return Err(Error::forward_pass_not_run("UnitNormalization"));
        };

        if grad_output.shape() != input_shape.as_slice() {
            return Err(Error::shape_mismatch(
                input_shape.clone(),
                grad_output.shape(),
            ));
        }

        let axes = self.resolve_axes(input_shape.len())?;
        let Some(group_len) = trailing_group_len(input_shape, &axes) else {
            return Ok(strided_backward(
                grad_output,
                output,
                scale,
                from_norm,
                &axes,
            ));
        };

        let grad_standard = grad_output.as_standard_layout();
        let g = grad_standard.as_slice().unwrap();
        let mut grad_input = Tensor::zeros(IxDyn(input_shape));
        row_backward(
            g,
            output.as_slice().unwrap(),
            group_len,
            scale.as_slice().unwrap(),
            from_norm.as_slice().unwrap(),
            g.len() >= cheap_map_parallel_threshold(),
            grad_input.as_slice_mut().unwrap(),
        );
        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "UnitNormalization"
    }

    fn output_shape(&self) -> String {
        match &self.input_shape {
            // Element 0 is the batch axis, which `summary()` prints as "None"
            Some(shape) => {
                let axes: Vec<String> = shape[1..].iter().map(|e| e.to_string()).collect();
                format!("(None, {})", axes.join(", "))
            }
            None => "Unknown".to_string(),
        }
    }

    no_trainable_parameters_layer_functions!();
}
