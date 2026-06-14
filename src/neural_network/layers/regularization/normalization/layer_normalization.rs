//! Layer Normalization layer and its axis configuration, including support for
//! single-axis and multi-axis (merged) normalization

use super::folds::{
    par_col_dot, par_col_sum, rows_per_block, segment_dot, segment_dot3, segment_sq_dev,
    segment_sum,
};
use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::{LayerNormalizationLayerWeight, LayerWeight};
use crate::neural_network::layers::regularization::mode_dependent_layer_set_training;
use crate::neural_network::layers::regularization::mode_dependent_layer_trait;
use crate::neural_network::layers::regularization::normalization::normalization_layer_output_shape;
use crate::neural_network::layers::regularization::validation::{
    validate_epsilon, validate_input_shape,
};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use ndarray::{Array1, Axis, IxDyn};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::borrow::Cow;

/// Total-element count above which the trailing-axis row passes (per-row statistics +
/// normalize, and the backward composition) run on rayon.
///
/// Each normalization group is one contiguous row computed entirely inside one task with
/// fixed-order kernels, so the gate is a pure performance knob: the bits are identical at any
/// thread count and on either side of the gate.
///
/// Measured on the fused row pass itself (AMD Ryzen 9 9950X, 16C/32T, 32 rayon threads,
/// 2026-06-12; see benches/RESULTS.md "LayerNorm fused row pass"): crossover bracket 64K-256K
/// elements (0.73x at 64K, 1.48x at 256K), 2.5-4.1x at 1M, fading toward memory bandwidth
/// (1.2x) at 12.6M
const LN_ROW_PARALLEL_MIN_ELEMS: usize = 262_144;

/// Element count above which the gamma/beta gradient column folds run on rayon.
///
/// The same row-block fold kernel class as BatchNorm's measured
/// `BN_COL_STATS_PARALLEL_MIN_ELEMS` (AMD Ryzen 9 9950X, 16C/32T, 32 rayon threads,
/// 2026-06-12; see benches/RESULTS.md "BatchNorm column stats, row-block fold": crossover
/// bracket 64K-256K elements), mapped from that measurement rather than calibrated separately
const LN_COL_STATS_PARALLEL_MIN_ELEMS: usize = 262_144;

/// Axis selection for layer normalization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerNormalizationAxis {
    /// Normalize along the last dimension (feature dimension)
    Default,
    /// Normalize along a single custom specified axis
    Custom(usize),
    /// Normalize jointly over several axes (Keras-style axis list); statistics are computed over
    /// the combined elements of those axes, and `gamma`/`beta` are 1-D with length equal to the
    /// product of those axes' sizes
    Multiple(Vec<usize>),
}

/// Validates a `Multiple` axis list against the input rank and returns the merge permutation:
/// the non-normalized axes (original order) followed by the normalized axes (given order).
/// An identity permutation means the normalized axes already form the trailing contiguous
/// block, so merging is pure reshape semantics and needs no transpose copy
fn multiple_axes_perm(ndim: usize, axes: &[usize]) -> Result<Vec<usize>, Error> {
    if axes.is_empty() {
        return Err(Error::invalid_parameter(
            "normalized_axis",
            "LayerNormalization Multiple axis list must be non-empty",
        ));
    }
    for (i, &a) in axes.iter().enumerate() {
        if a >= ndim {
            return Err(Error::invalid_parameter(
                "normalized_axis",
                format!("Normalization axis {a} is out of bounds for input with {ndim} dimensions"),
            ));
        }
        if axes[..i].contains(&a) {
            return Err(Error::invalid_parameter(
                "normalized_axis",
                format!("Duplicate normalization axis {a}"),
            ));
        }
    }
    let mut perm: Vec<usize> = (0..ndim).filter(|ax| !axes.contains(ax)).collect();
    perm.extend_from_slice(axes);
    Ok(perm)
}

/// How the configured axis maps onto the input layout
enum LayoutPlan {
    /// The normalized elements form the contiguous trailing block of each group: the input is
    /// logically `[R, N]` and takes the fused row path with no layout transform
    Rows { n: usize },
    /// `Multiple` axes that need a genuine permutation before they form a trailing block: one
    /// transpose copy in, the row path on the merged layout, one transpose copy back out
    MergedRows { axes: Vec<usize>, n: usize },
    /// A non-trailing `Custom` axis: stays on the broadcast ndarray path — its groups are
    /// strided mid-axis lanes, and ndarray reduces them in place, so a transpose would add the
    /// two copies the row path exists to avoid
    Strided { axis: usize },
}

/// Fused per-row normalization over a standard-layout `[R, N]` slice: for each row, the mean
/// and variance fold with the fixed-order segment kernels, then one streaming sweep writes the
/// centered, normalized, and scaled-shifted values (plus the row's mean/std into `[R]`
/// buffers). Rows are independent and each is computed entirely inside one task, so the
/// `parallel` flag — and the rows-per-task chunking — never changes the result bits
#[allow(clippy::too_many_arguments)]
fn row_forward(
    x: &[f32],
    n: usize,
    epsilon: f32,
    gamma: &[f32],
    beta: &[f32],
    parallel: bool,
    xc: &mut [f32],
    xn: &mut [f32],
    out: &mut [f32],
    mean: &mut [f32],
    std_dev: &mut [f32],
) {
    let rows = rows_per_block(n);
    let chunk = rows * n;
    type ForwardChunks<'a> = (
        (((&'a mut [f32], &'a mut [f32]), &'a mut [f32]), &'a [f32]),
        (&'a mut [f32], &'a mut [f32]),
    );
    let task = |((((xc_c, xn_c), out_c), x_c), (mean_c, std_c)): ForwardChunks| {
        let row_iter = x_c
            .chunks_exact(n)
            .zip(xc_c.chunks_exact_mut(n))
            .zip(xn_c.chunks_exact_mut(n))
            .zip(out_c.chunks_exact_mut(n))
            .zip(mean_c.iter_mut().zip(std_c.iter_mut()));
        for ((((x_row, xc_row), xn_row), out_row), (m, s)) in row_iter {
            let mean_val = segment_sum(x_row, 1.0) / n as f32;
            for (o, &v) in xc_row.iter_mut().zip(x_row) {
                *o = v - mean_val;
            }
            let var = segment_dot(xc_row, xc_row, 1.0) / n as f32;
            let std_val = (var + epsilon).sqrt();
            for (((xn_v, out_v), &xc_v), (&gamma_v, &beta_v)) in xn_row
                .iter_mut()
                .zip(out_row.iter_mut())
                .zip(xc_row.iter())
                .zip(gamma.iter().zip(beta))
            {
                *xn_v = xc_v / std_val;
                *out_v = *xn_v * gamma_v + beta_v;
            }
            *m = mean_val;
            *s = std_val;
        }
    };
    if parallel {
        xc.par_chunks_mut(chunk)
            .zip(xn.par_chunks_mut(chunk))
            .zip(out.par_chunks_mut(chunk))
            .zip(x.par_chunks(chunk))
            .zip(mean.par_chunks_mut(rows).zip(std_dev.par_chunks_mut(rows)))
            .for_each(task);
    } else {
        xc.chunks_mut(chunk)
            .zip(xn.chunks_mut(chunk))
            .zip(out.chunks_mut(chunk))
            .zip(x.chunks(chunk))
            .zip(mean.chunks_mut(rows).zip(std_dev.chunks_mut(rows)))
            .for_each(task);
    }
}

/// The inference twin of [`row_forward`]: identical per-row arithmetic bit for bit, but writes
/// only the output — the variance folds over the deviations in registers
/// ([`segment_sq_dev`]) instead of through a centered buffer, and nothing else is allocated
fn row_predict(
    x: &[f32],
    n: usize,
    epsilon: f32,
    gamma: &[f32],
    beta: &[f32],
    parallel: bool,
    out: &mut [f32],
) {
    let chunk = rows_per_block(n) * n;
    let task = |(out_c, x_c): (&mut [f32], &[f32])| {
        for (x_row, out_row) in x_c.chunks_exact(n).zip(out_c.chunks_exact_mut(n)) {
            let mean_val = segment_sum(x_row, 1.0) / n as f32;
            let var = segment_sq_dev(x_row, mean_val) / n as f32;
            let std_val = (var + epsilon).sqrt();
            for ((out_v, &v), (&gamma_v, &beta_v)) in
                out_row.iter_mut().zip(x_row).zip(gamma.iter().zip(beta))
            {
                *out_v = ((v - mean_val) / std_val) * gamma_v + beta_v;
            }
        }
    };
    if parallel {
        out.par_chunks_mut(chunk)
            .zip(x.par_chunks(chunk))
            .for_each(task);
    } else {
        out.chunks_mut(chunk).zip(x.chunks(chunk)).for_each(task);
    }
}

/// Fused per-row backward over standard-layout `[R, N]` slices: the three per-row reductions
/// (variance-gradient, mean-gradient, and centered sums) fold with the fixed-order segment
/// kernels — `grad_x_normalized` is fused as `g * gamma` per term instead of materialized —
/// and one streaming sweep composes the input gradient. Same flag semantics as [`row_forward`]
fn row_backward(
    g: &[f32],
    xc: &[f32],
    std_dev: &[f32],
    gamma: &[f32],
    n: usize,
    parallel: bool,
    gi: &mut [f32],
) {
    let rows = rows_per_block(n);
    let chunk = rows * n;
    let nf = n as f32;
    type BackwardChunks<'a> = (((&'a mut [f32], &'a [f32]), &'a [f32]), &'a [f32]);
    let task = |(((gi_c, g_c), xc_c), std_c): BackwardChunks| {
        let row_iter = g_c
            .chunks_exact(n)
            .zip(gi_c.chunks_exact_mut(n))
            .zip(xc_c.chunks_exact(n))
            .zip(std_c.iter());
        for (((g_row, gi_row), xc_row), &std_val) in row_iter {
            let inv_std = 1.0 / std_val;
            let grad_var = segment_dot3(g_row, gamma, xc_row, -0.5) * inv_std * inv_std * inv_std;
            let grad_mean_1 = segment_dot(g_row, gamma, -1.0) * inv_std;
            let x_sum = segment_sum(xc_row, 1.0);
            let grad_mean = grad_mean_1 + grad_var * ((x_sum * -2.0) / nf);
            let grad_mean_term = grad_mean / nf;
            for (((gi_v, &g_v), &xc_v), &gamma_v) in
                gi_row.iter_mut().zip(g_row).zip(xc_row).zip(gamma)
            {
                *gi_v = (g_v * gamma_v) * inv_std + grad_var * ((xc_v * 2.0) / nf) + grad_mean_term;
            }
        }
    };
    if parallel {
        gi.par_chunks_mut(chunk)
            .zip(g.par_chunks(chunk))
            .zip(xc.par_chunks(chunk))
            .zip(std_dev.par_chunks(rows))
            .for_each(task);
    } else {
        gi.chunks_mut(chunk)
            .zip(g.chunks(chunk))
            .zip(xc.chunks(chunk))
            .zip(std_dev.chunks(rows))
            .for_each(task);
    }
}

/// Permutes the `axes` to the trailing positions and merges them into a single axis, returning the
/// transformed contiguous tensor plus the permutation and pre-merge shape needed to invert it
///
/// Multi-axis layer normalization reduces to single-axis normalization of this merged axis, so the
/// public methods bracket the row core with this transform and its inverse,
/// [`unmerge_normalized_axes`] — but only when the permutation is **not** the identity; trailing
/// in-order axes skip the copies entirely (see [`LayoutPlan`])
fn merge_normalized_axes(
    input: &Tensor,
    axes: &[usize],
) -> Result<(Tensor, Vec<usize>, Vec<usize>), Error> {
    let perm = multiple_axes_perm(input.ndim(), axes)?;
    let permuted = input
        .view()
        .permuted_axes(perm.clone())
        .as_standard_layout()
        .to_owned();
    let permuted_shape = permuted.shape().to_vec();

    let outer = input.ndim() - axes.len();
    let inner: usize = axes.iter().map(|&a| input.shape()[a]).product();
    let mut merged_shape: Vec<usize> = permuted_shape[..outer].to_vec();
    merged_shape.push(inner);
    let merged = permuted
        .into_shape_with_order(merged_shape)
        .context("layer-norm merge reshape")?;
    Ok((merged, perm, permuted_shape))
}

/// Inverts [`merge_normalized_axes`]: un-merges back to `permuted_shape`, then applies the inverse
/// of `perm`, returning a contiguous tensor in the original layout
fn unmerge_normalized_axes(
    output_merged: Tensor,
    perm: &[usize],
    permuted_shape: &[usize],
) -> Tensor {
    let unmerged = output_merged
        .into_shape_with_order(permuted_shape.to_vec())
        .expect("unmerge reshape must succeed (element count is unchanged)");
    let ndim = perm.len();
    let mut inv = vec![0usize; ndim];
    for (new_pos, &old_ax) in perm.iter().enumerate() {
        inv[old_ax] = new_pos;
    }
    unmerged
        .view()
        .permuted_axes(inv)
        .as_standard_layout()
        .to_owned()
}

/// Layer Normalization layer for neural networks
///
/// Normalizes across feature dimensions for each sample, which is effective when batch sizes are
/// small or variable
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::traits::Layer;
/// use ndarray::Array2;
///
/// // Create a LayerNormalization layer
/// let mut ln = LayerNormalization::new(vec![32, 128], LayerNormalizationAxis::Default, 1e-5).unwrap();
///
/// // Create input tensor
/// let input = Array2::ones((32, 128)).into_dyn();
///
/// // During training, normalizes the input
/// let output = ln.forward(&input).unwrap();
/// ```
#[derive(Debug)]
pub struct LayerNormalization {
    /// Small constant for numerical stability in normalization
    epsilon: f32,
    /// Axis along which to normalize
    normalized_axis: LayerNormalizationAxis,
    /// Shape of the input tensor
    input_shape: Vec<usize>,
    /// Scale parameter (trainable)
    gamma: Tensor,
    /// Shift parameter (trainable)
    beta: Tensor,
    /// Whether the layer is in training mode or inference mode
    training: bool,
    /// Normalized input (cached for backward pass)
    x_normalized: Option<Tensor>,
    /// Centered input (cached for backward pass)
    x_centered: Option<Tensor>,
    /// Mean computed during forward pass (cached for backward pass)
    mean: Option<Tensor>,
    /// Standard deviation computed during forward pass (cached for backward pass)
    std_dev: Option<Tensor>,
    /// Gradient for the gamma parameter
    grad_gamma: Option<Tensor>,
    /// Gradient for the beta parameter
    grad_beta: Option<Tensor>,
}

impl LayerNormalization {
    /// Creates a new LayerNormalization layer
    ///
    /// # Parameters
    ///
    /// - `input_shape` - Shape of the input tensor
    /// - `normalized_axis` - Axis along which to normalize
    /// - `epsilon` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New LayerNormalization layer instance or a validation error
    ///
    /// # Errors
    ///
    /// - `Error::invalid_parameter` - If `epsilon` is not positive or not finite, or if a
    ///   `Multiple` axis list is empty, contains a duplicate, or has an out-of-bounds axis
    pub fn new(
        input_shape: Vec<usize>,
        normalized_axis: LayerNormalizationAxis,
        epsilon: f32,
    ) -> Result<Self, Error> {
        validate_epsilon(epsilon)?;

        // gamma/beta are 1-D over the normalized dimension(s): the axis size for a single axis, or
        // the product of the listed axes' sizes for Multiple (merged into one axis at runtime)
        let param_shape = match &normalized_axis {
            LayerNormalizationAxis::Default => {
                if input_shape.is_empty() {
                    vec![1]
                } else {
                    vec![input_shape[input_shape.len() - 1]]
                }
            }
            LayerNormalizationAxis::Custom(axis) => {
                if input_shape.len() > *axis {
                    vec![input_shape[*axis]]
                } else {
                    vec![1]
                }
            }
            LayerNormalizationAxis::Multiple(axes) => {
                if axes.is_empty() {
                    return Err(Error::invalid_parameter(
                        "normalized_axis",
                        "LayerNormalization Multiple axis list must be non-empty",
                    ));
                }
                for (i, &a) in axes.iter().enumerate() {
                    if a >= input_shape.len() {
                        return Err(Error::invalid_parameter(
                            "normalized_axis",
                            format!(
                                "Normalization axis {a} is out of bounds for input shape with {} dimensions",
                                input_shape.len()
                            ),
                        ));
                    }
                    if axes[..i].contains(&a) {
                        return Err(Error::invalid_parameter(
                            "normalized_axis",
                            format!("Duplicate normalization axis {a}"),
                        ));
                    }
                }
                vec![axes.iter().map(|&a| input_shape[a]).product()]
            }
        };

        let param_shape_ndarray = param_shape.as_slice();

        Ok(LayerNormalization {
            epsilon,
            normalized_axis,
            input_shape,
            gamma: Tensor::ones(param_shape_ndarray),
            beta: Tensor::zeros(param_shape_ndarray),
            training: true,
            x_normalized: None,
            x_centered: None,
            mean: None,
            std_dev: None,
            grad_gamma: None,
            grad_beta: None,
        })
    }

    mode_dependent_layer_set_training!();

    /// Sets the weights for the LayerNormalization layer
    ///
    /// # Parameters
    ///
    /// - `gamma` - Scale parameter (trainable)
    /// - `beta` - Shift parameter (trainable)
    ///
    /// # Errors
    ///
    /// - Returns an error if `gamma` or `beta` does not match the expected parameter shape
    pub fn set_weights(&mut self, gamma: Tensor, beta: Tensor) -> Result<(), Error> {
        validate_weight_shape("gamma", self.gamma.shape(), gamma.shape())?;
        validate_weight_shape("beta", self.beta.shape(), beta.shape())?;
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }

    /// Maps the configured axis onto the given tensor's layout (see [`LayoutPlan`]). The plan
    /// depends only on the configuration and the shape, so forward and backward always resolve
    /// to the same path and the caches they share stay layout-consistent
    fn resolve_plan(&self, t: &Tensor) -> Result<LayoutPlan, Error> {
        match &self.normalized_axis {
            LayerNormalizationAxis::Default => {
                if t.ndim() == 0 {
                    return Err(Error::invalid_input("Cannot normalize a scalar tensor"));
                }
                Ok(LayoutPlan::Rows {
                    n: t.shape()[t.ndim() - 1],
                })
            }
            LayerNormalizationAxis::Custom(axis) => {
                if *axis >= t.ndim() {
                    return Err(Error::invalid_parameter(
                        "normalized_axis",
                        format!(
                            "Normalization axis {} is out of bounds for input with {} dimensions",
                            axis,
                            t.ndim()
                        ),
                    ));
                }
                if *axis + 1 == t.ndim() {
                    Ok(LayoutPlan::Rows {
                        n: t.shape()[*axis],
                    })
                } else {
                    Ok(LayoutPlan::Strided { axis: *axis })
                }
            }
            LayerNormalizationAxis::Multiple(axes) => {
                let perm = multiple_axes_perm(t.ndim(), axes)?;
                let n: usize = axes.iter().map(|&a| t.shape()[a]).product();
                if perm.iter().enumerate().all(|(i, &a)| i == a) {
                    // The normalized axes already form the trailing block: merged layout ==
                    // original layout, so the row path runs with no transpose copies at all
                    Ok(LayoutPlan::Rows { n })
                } else {
                    Ok(LayoutPlan::MergedRows {
                        axes: axes.clone(),
                        n,
                    })
                }
            }
        }
    }

    /// Training forward on the fused row path over the logically `[R, N]` input
    fn forward_rows(&mut self, input: &Tensor, n: usize) -> Result<Tensor, Error> {
        let shape = input.shape().to_vec();
        let total = input.len();
        if total == 0 {
            // Degenerate empty input: nothing to normalize (and no caches to write)
            return Ok(Tensor::zeros(IxDyn(&shape)));
        }
        let r = total / n;

        // The row passes need contiguous data; standardize a non-contiguous input once
        let std_input;
        let x: &[f32] = match input.as_slice() {
            Some(s) => s,
            None => {
                std_input = input.as_standard_layout().into_owned();
                std_input.as_slice().unwrap()
            }
        };

        let parallel = total >= LN_ROW_PARALLEL_MIN_ELEMS;
        let mut x_centered = Tensor::zeros(IxDyn(&shape));
        let mut x_normalized = Tensor::zeros(IxDyn(&shape));
        let mut output = Tensor::zeros(IxDyn(&shape));
        let mut mean = Array1::<f32>::zeros(r);
        let mut std_dev = Array1::<f32>::zeros(r);
        row_forward(
            x,
            n,
            self.epsilon,
            self.gamma.as_slice().unwrap(),
            self.beta.as_slice().unwrap(),
            parallel,
            x_centered.as_slice_mut().unwrap(),
            x_normalized.as_slice_mut().unwrap(),
            output.as_slice_mut().unwrap(),
            mean.as_slice_mut().unwrap(),
            std_dev.as_slice_mut().unwrap(),
        );

        // Cache values for backward pass (mean/std as [R], one scalar per row)
        self.x_normalized = Some(x_normalized);
        self.x_centered = Some(x_centered);
        self.mean = Some(mean.into_dyn());
        self.std_dev = Some(std_dev.into_dyn());

        Ok(output)
    }

    /// Inference forward on the row path (writes no caches, allocates only the output)
    fn predict_rows(&self, input: &Tensor, n: usize) -> Result<Tensor, Error> {
        let shape = input.shape().to_vec();
        if input.is_empty() {
            return Ok(Tensor::zeros(IxDyn(&shape)));
        }
        let std_input;
        let x: &[f32] = match input.as_slice() {
            Some(s) => s,
            None => {
                std_input = input.as_standard_layout().into_owned();
                std_input.as_slice().unwrap()
            }
        };
        let parallel = input.len() >= LN_ROW_PARALLEL_MIN_ELEMS;
        let mut output = Tensor::zeros(IxDyn(&shape));
        row_predict(
            x,
            n,
            self.epsilon,
            self.gamma.as_slice().unwrap(),
            self.beta.as_slice().unwrap(),
            parallel,
            output.as_slice_mut().unwrap(),
        );
        Ok(output)
    }

    /// Backward on the row path, mirroring [`Self::forward_rows`]: the gamma/beta gradients
    /// are column folds over `[R, N]` and the input gradient composes per row
    fn backward_rows(&mut self, grad_output: &Tensor, n: usize) -> Result<Tensor, Error> {
        let shape = grad_output.shape().to_vec();
        let total = grad_output.len();
        if total == 0 {
            return Ok(Tensor::zeros(IxDyn(&shape)));
        }

        let std_grad;
        let g: &[f32] = match grad_output.as_slice() {
            Some(s) => s,
            None => {
                std_grad = grad_output.as_standard_layout().into_owned();
                std_grad.as_slice().unwrap()
            }
        };

        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;
        let x_centered = self
            .x_centered
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;
        let std_dev = self
            .std_dev
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;
        // The caches were built by forward_rows, so they are standard layout ([R] for std_dev)
        let xn_s = x_normalized.as_slice().unwrap();
        let xc_s = x_centered.as_slice().unwrap();
        let std_s = std_dev.as_slice().unwrap();

        // Gradients for gamma and beta: fused column folds over [R, N] (no product temporary)
        let col_parallel = total >= LN_COL_STATS_PARALLEL_MIN_ELEMS;
        self.grad_gamma = Some(par_col_dot(g, xn_s, n, col_parallel, 1.0));
        self.grad_beta = Some(par_col_sum(g, n, col_parallel, 1.0));

        let row_parallel = total >= LN_ROW_PARALLEL_MIN_ELEMS;
        let mut grad_input = Tensor::zeros(IxDyn(&shape));
        row_backward(
            g,
            xc_s,
            std_s,
            self.gamma.as_slice().unwrap(),
            n,
            row_parallel,
            grad_input.as_slice_mut().unwrap(),
        );
        Ok(grad_input)
    }
}

impl Layer for LayerNormalization {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        match self.resolve_plan(input)? {
            LayoutPlan::Rows { n } => self.forward_rows(input, n),
            LayoutPlan::MergedRows { axes, n } => {
                let (merged, perm, permuted_shape) = merge_normalized_axes(input, &axes)?;
                let output = self.forward_rows(&merged, n)?;
                Ok(unmerge_normalized_axes(output, &perm, &permuted_shape))
            }
            LayoutPlan::Strided { axis } => self.forward_strided(input, axis),
        }
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_shape(input.shape(), &self.input_shape)?;

        match self.resolve_plan(input)? {
            LayoutPlan::Rows { n } => self.predict_rows(input, n),
            LayoutPlan::MergedRows { axes, n } => {
                let (merged, perm, permuted_shape) = merge_normalized_axes(input, &axes)?;
                let output = self.predict_rows(&merged, n)?;
                Ok(unmerge_normalized_axes(output, &perm, &permuted_shape))
            }
            LayoutPlan::Strided { axis } => self.predict_strided(input, axis),
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        if !self.training {
            // During inference, pass gradient through unchanged
            return Ok(grad_output.clone());
        }

        // The plan depends only on configuration and shape, so it matches the forward pass
        // that produced the caches
        match self.resolve_plan(grad_output)? {
            LayoutPlan::Rows { n } => self.backward_rows(grad_output, n),
            LayoutPlan::MergedRows { axes, n } => {
                let (merged, perm, permuted_shape) = merge_normalized_axes(grad_output, &axes)?;
                let grad_input = self.backward_rows(&merged, n)?;
                Ok(unmerge_normalized_axes(grad_input, &perm, &permuted_shape))
            }
            LayoutPlan::Strided { axis } => self.backward_strided(grad_output, axis),
        }
    }

    fn layer_type(&self) -> &str {
        "LayerNormalization"
    }

    fn output_shape(&self) -> String {
        normalization_layer_output_shape!(self)
    }

    fn param_count(&self) -> TrainingParameters {
        TrainingParameters::Trainable(self.gamma.len() + self.beta.len())
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            gamma,
            beta,
            grad_gamma,
            grad_beta,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(grad_a), Some(grad_b)) = (grad_gamma.as_ref(), grad_beta.as_ref()) {
            params.push(ParamGrad {
                value: gamma.as_slice_mut().expect("gamma must be contiguous"),
                grad: grad_a.as_slice().expect("grad_gamma must be contiguous"),
            });
            params.push(ParamGrad {
                value: beta.as_slice_mut().expect("beta must be contiguous"),
                grad: grad_b.as_slice().expect("grad_beta must be contiguous"),
            });
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::LayerNormalization(LayerNormalizationLayerWeight {
            gamma: Cow::Borrowed(&self.gamma),
            beta: Cow::Borrowed(&self.beta),
        })
    }

    mode_dependent_layer_trait!();
}

impl LayerNormalization {
    /// Training forward for a non-trailing `Custom` axis: the broadcast ndarray path. The
    /// groups are strided mid-axis lanes that ndarray reduces in place, so this path stays
    /// transpose-free by construction; it remains serial because its access pattern, not
    /// compute, is the cost
    fn forward_strided(&mut self, input: &Tensor, axis_idx: usize) -> Result<Tensor, Error> {
        // Mean along the axis, then insert the axis back so broadcasting works
        let mean = input.mean_axis(Axis(axis_idx)).unwrap();
        let mean = mean.insert_axis(Axis(axis_idx));

        // Center the data and compute variance over the normalized axis (broadcasting `mean`)
        let x_centered = input - &mean;
        let var = (&x_centered * &x_centered)
            .mean_axis(Axis(axis_idx))
            .unwrap()
            .insert_axis(Axis(axis_idx));

        // Normalize (broadcasting the per-row std over the normalized axis)
        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = &x_centered / &std_dev;

        // Scale and shift: reshape gamma/beta to broadcast over the input shape
        let mut gamma_shape = vec![1; input.ndim()];
        let mut beta_shape = vec![1; input.ndim()];

        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
            beta_shape[axis_idx + i] = dim;
        }

        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();
        let beta_broadcast = self
            .beta
            .clone()
            .into_shape_with_order(beta_shape.as_slice())
            .unwrap();

        let output = &x_normalized * &gamma_broadcast + &beta_broadcast;

        // Cache values for backward pass (broadcast-ready shapes with the axis kept)
        self.x_normalized = Some(x_normalized);
        self.x_centered = Some(x_centered);
        self.mean = Some(mean);
        self.std_dev = Some(std_dev);

        Ok(output)
    }

    /// Inference forward for a non-trailing `Custom` axis (no caches); see
    /// [`Self::forward_strided`]
    fn predict_strided(&self, input: &Tensor, axis_idx: usize) -> Result<Tensor, Error> {
        let mean = input.mean_axis(Axis(axis_idx)).unwrap();
        let mean = mean.insert_axis(Axis(axis_idx));

        let x_centered = input - &mean;
        let var = (&x_centered * &x_centered)
            .mean_axis(Axis(axis_idx))
            .unwrap()
            .insert_axis(Axis(axis_idx));

        let std_dev = (&var + self.epsilon).mapv(|x| x.sqrt());
        let x_normalized = &x_centered / &std_dev;

        let mut gamma_shape = vec![1; input.ndim()];
        let mut beta_shape = vec![1; input.ndim()];

        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
            beta_shape[axis_idx + i] = dim;
        }

        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();
        let beta_broadcast = self
            .beta
            .clone()
            .into_shape_with_order(beta_shape.as_slice())
            .unwrap();

        Ok(&x_normalized * &gamma_broadcast + &beta_broadcast)
    }

    /// Backward for a non-trailing `Custom` axis, mirroring [`Self::forward_strided`]
    fn backward_strided(&mut self, grad_output: &Tensor, axis_idx: usize) -> Result<Tensor, Error> {
        let x_normalized = self
            .x_normalized
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;

        let x_centered = self
            .x_centered
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;

        let std_dev = self
            .std_dev
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("LayerNormalization"))?;

        // gamma/beta are 1-D over the normalized axis, so reduce every other axis to leave a
        // gradient of shape `[axis_size]`
        let mut grad_gamma = grad_output * x_normalized;
        let mut grad_beta = grad_output.clone();

        let ndim = grad_gamma.ndim();
        // Sum the axes after the normalized axis (from the end so lower indices are unaffected),
        // then the axes before it
        for i in (axis_idx + 1..ndim).rev() {
            grad_gamma = grad_gamma.sum_axis(Axis(i));
            grad_beta = grad_beta.sum_axis(Axis(i));
        }
        for i in (0..axis_idx).rev() {
            grad_gamma = grad_gamma.sum_axis(Axis(i));
            grad_beta = grad_beta.sum_axis(Axis(i));
        }

        self.grad_gamma = Some(grad_gamma);
        self.grad_beta = Some(grad_beta);

        // Gradient with respect to normalized input: reshape gamma for broadcasting
        let mut gamma_shape = vec![1; grad_output.ndim()];
        for (i, &dim) in self.gamma.shape().iter().enumerate() {
            gamma_shape[axis_idx + i] = dim;
        }
        let gamma_broadcast = self
            .gamma
            .clone()
            .into_shape_with_order(gamma_shape.as_slice())
            .unwrap();

        let grad_x_normalized = grad_output * &gamma_broadcast;

        // Inverse standard deviation and size of the normalization dimension
        let inv_std = std_dev.mapv(|x| 1.0 / x);
        let norm_size = grad_output.shape()[axis_idx] as f32;

        // Gradient with respect to variance
        let grad_var = (&grad_x_normalized * x_centered * -0.5).sum_axis(Axis(axis_idx));
        let grad_var = grad_var.insert_axis(Axis(axis_idx));
        let grad_var = &grad_var * &inv_std * &inv_std * &inv_std;

        // Gradient with respect to mean
        let grad_mean_1 = (&grad_x_normalized * -1.0).sum_axis(Axis(axis_idx));
        let grad_mean_1 = grad_mean_1.insert_axis(Axis(axis_idx));
        let grad_mean_1 = &grad_mean_1 * &inv_std;

        let x_sum = x_centered
            .sum_axis(Axis(axis_idx))
            .insert_axis(Axis(axis_idx));
        let grad_mean_2 = &grad_var * (&x_sum * -2.0 / norm_size);
        let grad_mean = grad_mean_1 + grad_mean_2;

        // Gradient with respect to input
        let grad_input = &grad_x_normalized * &inv_std
            + &grad_var * (x_centered * 2.0 / norm_size)
            + &grad_mean / norm_size;

        Ok(grad_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    // Helper: build a Tensor from a flat Vec and shape
    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        ArrayD::from_shape_vec(shape, data).expect("shape/data mismatch in test helper")
    }

    // merge_normalized_axes: output shape

    /// merge_normalized_axes on [3,4,5] with axes=[1,2] yields merged shape [3,20]
    #[test]
    fn test_merge_normalized_axes_shape() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (merged, _perm, _permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        assert_eq!(merged.shape(), &[3, 20]);
    }

    /// merge_normalized_axes returns perm [0,1,2] for [3,4,5] with axes=[1,2]
    #[test]
    fn test_merge_normalized_axes_perm() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (_merged, perm, _permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        assert_eq!(perm, vec![0usize, 1, 2]);
    }

    /// merge_normalized_axes returns permuted_shape [3,4,5] for [3,4,5] with axes=[1,2]
    #[test]
    fn test_merge_normalized_axes_permuted_shape() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (_merged, _perm, permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        assert_eq!(permuted_shape, vec![3usize, 4, 5]);
    }

    // merge + unmerge round-trip

    /// merge then unmerge recovers the original tensor elementwise for [3,4,5] with axes=[1,2]
    #[test]
    fn test_merge_unmerge_round_trip() {
        let n = 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 1.1).collect();
        let x = make_tensor(data, &[3, 4, 5]);

        let (merged, perm, permuted_shape) =
            merge_normalized_axes(&x, &[1, 2]).expect("merge_normalized_axes failed");

        let recovered = unmerge_normalized_axes(merged, &perm, &permuted_shape);

        assert_eq!(recovered.shape(), x.shape());
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    /// merge then unmerge round-trip holds when axes require a non-trivial permutation ([2,3,4], axes=[0,2])
    #[test]
    fn test_merge_unmerge_round_trip_nontrivial_perm() {
        let n = 2 * 3 * 4;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.7 + 0.3).collect();
        let x = make_tensor(data, &[2, 3, 4]);

        let (merged, perm, permuted_shape) =
            merge_normalized_axes(&x, &[0, 2]).expect("merge_normalized_axes failed");

        // Verify intermediate merged shape: outer=1 (axis 1), inner=2*4=8 -> [3,8]
        assert_eq!(merged.shape(), &[3, 8]);

        let recovered = unmerge_normalized_axes(merged, &perm, &permuted_shape);

        assert_eq!(recovered.shape(), x.shape());
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "non-trivial round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    // merge_normalized_axes: error cases

    /// merge_normalized_axes returns Err for an empty axes slice
    #[test]
    fn test_merge_normalized_axes_empty_axes_error() {
        let x = make_tensor(vec![1.0f32, 2.0, 3.0], &[3]);
        let result = merge_normalized_axes(&x, &[]);
        assert!(result.is_err(), "expected Err for empty axes");
    }

    /// merge_normalized_axes returns Err when an axis is out of bounds
    #[test]
    fn test_merge_normalized_axes_out_of_bounds_error() {
        let x = make_tensor(vec![1.0f32; 6], &[2, 3]);
        let result = merge_normalized_axes(&x, &[2]); // ndim=2, axis 2 is OOB
        assert!(result.is_err(), "expected Err for out-of-bounds axis");
    }

    /// merge_normalized_axes returns Err for duplicate axes
    #[test]
    fn test_merge_normalized_axes_duplicate_axes_error() {
        let x = make_tensor(vec![1.0f32; 12], &[3, 4]);
        let result = merge_normalized_axes(&x, &[0, 0]);
        assert!(result.is_err(), "expected Err for duplicate axis");
    }

    fn test_rows(r: usize, n: usize, salt: f32) -> Vec<f32> {
        (0..r * n).map(|i| (i as f32 * salt).sin()).collect()
    }

    /// The row passes compute each row entirely inside one task with fixed-order kernels, so
    /// the parallel flag must never change the bits — including rows that straddle task-chunk
    /// boundaries, sub-8 rows, and rows larger than one chunk
    #[test]
    fn row_passes_parallel_flag_invariant() {
        for &(r, n) in &[
            (7usize, 5usize),
            (33, 130),
            (3, 16_384),
            (2, 20_000),
            (1, 1_000),
            (4_097, 8),
        ] {
            let x = test_rows(r, n, 0.731);
            let g = test_rows(r, n, 0.433);
            let gamma: Vec<f32> = (0..n).map(|j| 1.5 - 0.001 * j as f32).collect();
            let beta: Vec<f32> = (0..n).map(|j| -0.25 + 0.002 * j as f32).collect();
            let eps = 1e-5f32;

            type PassOutputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
            let mut results: Vec<PassOutputs> = Vec::new();
            for parallel in [false, true] {
                let mut xc = vec![0.0f32; r * n];
                let mut xn = vec![0.0f32; r * n];
                let mut out = vec![0.0f32; r * n];
                let mut mean = vec![0.0f32; r];
                let mut std_dev = vec![0.0f32; r];
                row_forward(
                    &x,
                    n,
                    eps,
                    &gamma,
                    &beta,
                    parallel,
                    &mut xc,
                    &mut xn,
                    &mut out,
                    &mut mean,
                    &mut std_dev,
                );
                let mut predict_out = vec![0.0f32; r * n];
                row_predict(&x, n, eps, &gamma, &beta, parallel, &mut predict_out);
                let mut gi = vec![0.0f32; r * n];
                row_backward(&g, &xc, &std_dev, &gamma, n, parallel, &mut gi);

                assert_eq!(
                    out, predict_out,
                    "predict must equal the training output bit for bit at [{r}x{n}] \
                     (parallel={parallel})"
                );
                results.push((xc, xn, out, mean, std_dev, gi));
            }
            assert_eq!(
                results[0], results[1],
                "the parallel flag changed row-pass bits at [{r}x{n}]"
            );
        }
    }

    /// On integer-valued data with a power-of-two row length the per-row statistics are exact,
    /// so the row-path forward must reproduce the broadcast ndarray reference bit for bit
    #[test]
    fn row_path_exact_on_integer_data_matches_broadcast_reference() {
        use ndarray::Array2;
        let (r, n) = (6usize, 64usize);
        let x2 = Array2::from_shape_fn((r, n), |(i, j)| ((i * 7 + j * 5) % 4) as f32);
        let x = x2.clone().into_dyn();
        let gamma = Array1::from_shape_fn(n, |j| 1.5 - 0.01 * j as f32);
        let beta = Array1::from_shape_fn(n, |j| -0.75 + 0.02 * j as f32);

        let mut ln =
            LayerNormalization::new(vec![r, n], LayerNormalizationAxis::Default, 1e-5).unwrap();
        ln.set_weights(gamma.clone().into_dyn(), beta.clone().into_dyn())
            .unwrap();
        let out = ln.forward(&x).unwrap();

        // Reference with the old broadcast forms; exact statistics make the grouping moot
        let mean = x2.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let x_centered = &x2 - &mean;
        let var = (&x_centered * &x_centered)
            .mean_axis(Axis(1))
            .unwrap()
            .insert_axis(Axis(1));
        let std_dev = (&var + 1e-5f32).mapv(|v| v.sqrt());
        let x_normalized = &x_centered / &std_dev;
        let expected = (&x_normalized * &gamma + &beta).into_dyn();

        assert_eq!(
            out, expected,
            "row-path forward must match the broadcast reference exactly on exact statistics"
        );
    }

    /// A trailing-and-in-order Multiple axis list resolves to the zero-copy row path, which
    /// must agree bit for bit with an explicit Default-axis layer on the reshaped input
    #[test]
    fn multiple_identity_perm_matches_default_on_reshaped_input() {
        let (b, h, w) = (3usize, 4usize, 5usize);
        let n = h * w;
        let data: Vec<f32> = (0..b * n).map(|i| (i as f32 * 0.617).sin()).collect();
        let x3 = make_tensor(data.clone(), &[b, h, w]);
        let x2 = make_tensor(data, &[b, n]);

        let mut ln_multi = LayerNormalization::new(
            vec![b, h, w],
            LayerNormalizationAxis::Multiple(vec![1, 2]),
            1e-5,
        )
        .unwrap();
        let mut ln_default =
            LayerNormalization::new(vec![b, n], LayerNormalizationAxis::Default, 1e-5).unwrap();

        let out_multi = ln_multi.forward(&x3).unwrap();
        let out_default = ln_default.forward(&x2).unwrap();
        assert_eq!(out_multi.shape(), &[b, h, w]);
        assert_eq!(
            out_multi.as_slice().unwrap(),
            out_default.as_slice().unwrap(),
            "identity-perm Multiple must equal Default on the reshaped input bit for bit"
        );

        let grad: Vec<f32> = (0..b * n).map(|i| (i as f32 * 0.433).sin()).collect();
        let gi_multi = ln_multi
            .backward(&make_tensor(grad.clone(), &[b, h, w]))
            .unwrap();
        let gi_default = ln_default.backward(&make_tensor(grad, &[b, n])).unwrap();
        assert_eq!(
            gi_multi.as_slice().unwrap(),
            gi_default.as_slice().unwrap(),
            "identity-perm Multiple backward must equal Default on the reshaped input"
        );
        for (pg_m, pg_d) in ln_multi
            .parameters()
            .iter()
            .zip(ln_default.parameters().iter())
        {
            assert_eq!(
                pg_m.grad, pg_d.grad,
                "parameter grads must match bit for bit"
            );
        }
    }

    /// A Multiple axis list that genuinely needs a permutation must agree bit for bit with
    /// merging by hand and running a Default-axis layer on the merged tensor
    #[test]
    fn multiple_nontrivial_perm_matches_default_on_merged_input() {
        let (d0, d1, d2) = (2usize, 3usize, 4usize);
        let axes = vec![0usize, 2usize];
        let n = d0 * d2;
        let data: Vec<f32> = (0..d0 * d1 * d2)
            .map(|i| (i as f32 * 0.519).sin())
            .collect();
        let x = make_tensor(data, &[d0, d1, d2]);

        let mut ln_multi = LayerNormalization::new(
            vec![d0, d1, d2],
            LayerNormalizationAxis::Multiple(axes.clone()),
            1e-5,
        )
        .unwrap();
        let out_multi = ln_multi.forward(&x).unwrap();

        let (merged, perm, permuted_shape) = merge_normalized_axes(&x, &axes).unwrap();
        let mut ln_default =
            LayerNormalization::new(vec![d1, n], LayerNormalizationAxis::Default, 1e-5).unwrap();
        let expected =
            unmerge_normalized_axes(ln_default.forward(&merged).unwrap(), &perm, &permuted_shape);

        assert_eq!(
            out_multi, expected,
            "non-trivial-perm Multiple must equal merge -> Default -> unmerge bit for bit"
        );
    }
}
