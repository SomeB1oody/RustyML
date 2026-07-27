//! Normalization layers (batch, group, instance, layer) and the shared group-normalization core
//!
//! Group and instance normalization are the same operation at different group counts, so both
//! delegate to [`group_norm_forward_core`] / [`group_norm_backward_core`] here. Everything works
//! directly on the channels-last buffer: a channel group is a contiguous sub-run of each position's
//! channel vector, never a plane that has to be gathered out

use self::folds::{par_col_dot, par_col_sum, rows_per_block};
use crate::neural_network::Tensor;
use ndarray::{Array1, IxDyn};
use rayon::prelude::*;

tunable_gate! {
    /// Total-element count above which the group-normalization per-instance row passes run on rayon
    ///
    /// Each instance (one sample's channel group) is one contiguous row computed entirely inside
    /// one task with fixed-order kernels, so the gate is a performance knob: the result is the
    /// same on either side of the gate. Shares the fused row-pass
    /// kernel class of LayerNorm's `LN_ROW_PARALLEL_MIN_ELEMS` (crossover bracket 64K-256K
    /// elements), mapped from that measurement
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) GN_ROW_PARALLEL_MIN_ELEMS => gn_row_parallel_min_elems / set_gn_row_parallel_min_elems = 262_144
}

tunable_gate! {
    /// Total-element count above which group normalization's per-channel parameter-gradient
    /// folds run on rayon
    ///
    /// `grad_gamma`/`grad_beta` are column folds over the `[M, C]` view of the gradient, the same
    /// deterministic row-block kernel BatchNorm uses, so the flag is a pure performance knob
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) GN_PARAM_GRAD_PARALLEL_MIN_ELEMS => gn_param_grad_parallel_min_elems / set_gn_param_grad_parallel_min_elems = 262_144
}

/// Positions per statistics block, sized from the channel count so one block is about
/// [`DET_REDUCE_BLOCK`](crate::math::reduction::DET_REDUCE_BLOCK) elements
///
/// A function of the input shape only, so the block grouping - and with it the float summation
/// order - never depends on how many threads happened to run
fn positions_per_block(channels: usize) -> usize {
    rows_per_block(channels)
}

/// Per-group `(shift, sum, sum_of_squares)` accumulators for one batch item
struct GroupStats {
    /// The value each group's deviations are measured from
    shift: Vec<f32>,
    /// `sum(x - shift)` per group
    s1: Vec<f32>,
    /// `sum((x - shift)^2)` per group
    s2: Vec<f32>,
}

/// One-pass per-group mean and variance for batch item `b` of a `[batch, positions, channels]`
/// buffer, by the shifted-sum method
///
/// The textbook two-pass form reads the instance once for the mean and again for the squared
/// deviations. Under this layout an instance is `positions` strided runs of `channels_per_group`,
/// so that second read is a second walk of the whole sample - which is exactly the traffic worth
/// removing. Accumulating `sum(x - K)` and `sum((x - K)^2)` together gets both statistics from one
/// walk, and `var = s2/n - (s1/n)^2` recovers the variance.
///
/// `K` is the instance's first element rather than zero, and that choice is what makes the
/// subtraction safe: with `K` on the order of the data, `s1/n` is the small residual `mean - K`
/// instead of the mean itself, so the two terms differ by roughly the variance rather than by the
/// square of the mean. A plain `E[x^2] - E[x]^2` would cancel catastrophically in f32 on exactly
/// the inputs a normalization layer sees, where the mean often dwarfs the spread.
///
/// Blocks are merged in block order, so the result depends on the shape and not on the schedule
fn group_stats(x: &[f32], b: usize, layout: &GroupLayout, parallel: bool) -> GroupStats {
    let (positions, channels, num_groups) = (layout.positions, layout.channels, layout.num_groups);
    let cpg = layout.channels_per_group;
    let base = b * positions * channels;

    let shift: Vec<f32> = (0..num_groups).map(|gr| x[base + gr * cpg]).collect();
    let block = positions_per_block(channels);

    // The shift broadcast to one value per channel, so the block fold below is a flat pass
    let shift_c: Vec<f32> = shift
        .iter()
        .flat_map(|&k| std::iter::repeat_n(k, cpg))
        .collect();

    // A block accumulates per *channel* and reduces to per group once at the end, rather than
    // looping groups-outer with a `channels_per_group`-long inner loop. The nested form degenerates
    // completely for instance normalization, where `num_groups == channels` makes that inner loop a
    // single element and leaves a scalar walk with a dependent load-store per group; accumulating
    // per channel keeps the inner loop one flat pass over contiguous rows at every group count
    let fold = |p0: usize| -> (Vec<f32>, Vec<f32>) {
        let len = block.min(positions - p0);
        let mut c1 = vec![0.0f32; channels];
        let mut c2 = vec![0.0f32; channels];
        for p in p0..p0 + len {
            let row = &x[base + p * channels..base + (p + 1) * channels];
            for (((a1, a2), &v), &k) in c1
                .iter_mut()
                .zip(c2.iter_mut())
                .zip(row)
                .zip(shift_c.iter())
            {
                let d = v - k;
                *a1 += d;
                *a2 += d * d;
            }
        }
        // Reduce the channels of each group, in channel order
        let mut b1 = vec![0.0f32; num_groups];
        let mut b2 = vec![0.0f32; num_groups];
        for gr in 0..num_groups {
            for j in gr * cpg..(gr + 1) * cpg {
                b1[gr] += c1[j];
                b2[gr] += c2[j];
            }
        }
        (b1, b2)
    };

    let starts: Vec<usize> = (0..positions).step_by(block.max(1)).collect();
    let parts: Vec<(Vec<f32>, Vec<f32>)> = if parallel {
        starts.par_iter().map(|&p0| fold(p0)).collect()
    } else {
        starts.iter().map(|&p0| fold(p0)).collect()
    };

    let mut s1 = vec![0.0f32; num_groups];
    let mut s2 = vec![0.0f32; num_groups];
    for (b1, b2) in parts {
        for gr in 0..num_groups {
            s1[gr] += b1[gr];
            s2[gr] += b2[gr];
        }
    }
    GroupStats { shift, s1, s2 }
}

/// Per-channel mean and inverse standard deviation, broadcast out of the per-group statistics
///
/// Expanding the `num_groups`-long statistics to `channels`-long vectors up front turns the affine
/// sweep below into a flat elementwise pass over contiguous rows: no per-element group lookup, no
/// modulo, and both operands line up with `gamma`/`beta`
fn per_channel_stats(
    stats: &GroupStats,
    layout: &GroupLayout,
    epsilon: f32,
) -> (Vec<f32>, Vec<f32>) {
    let cpg = layout.channels_per_group;
    let n = (layout.positions * cpg) as f32;
    let mut mean_c = vec![0.0f32; layout.channels];
    let mut inv_std_c = vec![0.0f32; layout.channels];
    for gr in 0..layout.num_groups {
        let m = stats.s1[gr] / n;
        // Rounding can leave the difference a hair below zero on a near-constant instance; the
        // clamp keeps the square root real without changing any well-conditioned result
        let var = (stats.s2[gr] / n - m * m).max(0.0);
        let inv = 1.0 / (var + epsilon).sqrt();
        for j in 0..cpg {
            mean_c[gr * cpg + j] = stats.shift[gr] + m;
            inv_std_c[gr * cpg + j] = inv;
        }
    }
    (mean_c, inv_std_c)
}

/// One batch item's slabs in the forward sweep: `(x_normalized, output)` to fill from `input`
type ForwardSlabs<'a> = (usize, ((&'a mut [f32], &'a mut [f32]), &'a [f32]));

/// One batch item's slabs in the backward sweep: `grad_input` to fill from `(grad_output, x_norm)`
type BackwardSlabs<'a> = (usize, (&'a mut [f32], (&'a [f32], &'a [f32])));

/// Shape facts a group-normalization pass needs, derived once from a channels-last input
pub(super) struct GroupLayout {
    /// Batch items
    pub batch: usize,
    /// Spatial positions per batch item (the product of the middle axes)
    pub positions: usize,
    /// Channels, the trailing axis
    pub channels: usize,
    /// Groups the channel axis is split into
    pub num_groups: usize,
    /// Channels in each group
    pub channels_per_group: usize,
}

impl GroupLayout {
    /// Derives the layout of a `[batch, spatial..., channels]` input split into `num_groups`
    pub(super) fn new(shape: &[usize], num_groups: usize) -> Self {
        let channels = shape[shape.len() - 1];
        let positions: usize = shape[1..shape.len() - 1].iter().product();
        Self {
            batch: shape[0],
            positions,
            channels,
            num_groups,
            channels_per_group: channels / num_groups.max(1),
        }
    }
}

/// Group-normalization forward core
///
/// Two linear passes over each batch item: [`group_stats`] for the per-group statistics, then one
/// sweep writing `x_normalized` and the affine output. Both walk contiguous `channels`-long rows,
/// so nothing is gathered, permuted, or copied out to make a group contiguous
///
/// Returns `(output, x_normalized, inv_std)` with `inv_std` shaped `[batch * num_groups]`
pub(super) fn group_norm_forward_core(
    input: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
    beta: &Tensor,
    epsilon: f32,
) -> (Tensor, Tensor, Tensor) {
    let shape = input.shape().to_vec();
    let layout = GroupLayout::new(&shape, num_groups);
    let total = input.len();
    if total == 0 {
        return (
            Tensor::zeros(IxDyn(&shape)),
            Tensor::zeros(IxDyn(&shape)),
            Array1::<f32>::zeros(layout.batch * num_groups).into_dyn(),
        );
    }

    let input_std = input.as_standard_layout();
    let x = input_std.as_slice().unwrap();
    let parallel = total >= gn_row_parallel_min_elems();
    let (gamma_s, beta_s) = (gamma.as_slice().unwrap(), beta.as_slice().unwrap());

    let mut x_normalized = Tensor::zeros(IxDyn(&shape));
    let mut output = Tensor::zeros(IxDyn(&shape));
    let mut inv_std = Array1::<f32>::zeros(layout.batch * num_groups);
    let item = layout.positions * layout.channels;

    // Batch items are independent, so spread them across rayon once the batch alone fills the pool
    // and let each item's statistics fold run serially inside its task. Below that the batch is too
    // short to fill the machine, so the split moves inside: one item at a time, with its fold
    // spread over position blocks. Same choice the convolution engine makes for its per-item GEMMs
    let batch_parallel = parallel && layout.batch >= rayon::current_num_threads();
    let stats_parallel = parallel && !batch_parallel;

    // Per-item statistics, broadcast to per-channel so the sweep below is flat elementwise
    let per_item = |b: usize| {
        per_channel_stats(
            &group_stats(x, b, &layout, stats_parallel),
            &layout,
            epsilon,
        )
    };
    let stats: Vec<(Vec<f32>, Vec<f32>)> = if batch_parallel {
        (0..layout.batch).into_par_iter().map(per_item).collect()
    } else {
        (0..layout.batch).map(per_item).collect()
    };
    for (b, (_, inv_std_c)) in stats.iter().enumerate() {
        for gr in 0..num_groups {
            inv_std[b * num_groups + gr] = inv_std_c[gr * layout.channels_per_group];
        }
    }

    let sweep = |(b, ((xn, out), src)): ForwardSlabs| {
        let (mean_c, inv_std_c) = &stats[b];
        for ((x_row, xn_row), out_row) in src
            .chunks_exact(layout.channels)
            .zip(xn.chunks_exact_mut(layout.channels))
            .zip(out.chunks_exact_mut(layout.channels))
        {
            for c in 0..layout.channels {
                let v = (x_row[c] - mean_c[c]) * inv_std_c[c];
                xn_row[c] = v;
                out_row[c] = v * gamma_s[c] + beta_s[c];
            }
        }
    };
    {
        let xn = x_normalized.as_slice_mut().unwrap();
        let out = output.as_slice_mut().unwrap();
        if parallel {
            xn.par_chunks_mut(item)
                .zip(out.par_chunks_mut(item))
                .zip(x.par_chunks(item))
                .enumerate()
                .for_each(sweep);
        } else {
            xn.chunks_mut(item)
                .zip(out.chunks_mut(item))
                .zip(x.chunks(item))
                .enumerate()
                .for_each(sweep);
        }
    }

    (output, x_normalized, inv_std.into_dyn())
}

/// Group-normalization backward core
///
/// Inverse of [`group_norm_forward_core`]. The parameter gradients are the same column folds the
/// rest of the module uses - under this layout `[batch, spatial..., channels]` is already the
/// `[M, C]` matrix they read - and the input gradient is two passes per batch item: the per-group
/// reductions, then the fused composition
/// `dx = inv_std * (g * gamma - (sum_g + x_norm * sum_g_xnorm) / group_size)`
///
/// Returns `(grad_input, grad_gamma, grad_beta)` with the parameter gradients shaped `[channels]`
pub(super) fn group_norm_backward_core(
    grad_output: &Tensor,
    x_normalized: &Tensor,
    inv_std: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let shape = grad_output.shape().to_vec();
    let layout = GroupLayout::new(&shape, num_groups);
    let total = grad_output.len();
    if total == 0 {
        return (
            Tensor::zeros(IxDyn(&shape)),
            Array1::<f32>::zeros(layout.channels).into_dyn(),
            Array1::<f32>::zeros(layout.channels).into_dyn(),
        );
    }

    let grad_std = grad_output.as_standard_layout();
    let g = grad_std.as_slice().unwrap();
    let xn_std = x_normalized.as_standard_layout();
    let xn = xn_std.as_slice().unwrap();
    let inv_std_s = inv_std.as_slice().unwrap();
    let gamma_s = gamma.as_slice().unwrap();

    let col_parallel = total >= gn_param_grad_parallel_min_elems();
    let grad_beta = par_col_sum(g, layout.channels, col_parallel, 1.0);
    let grad_gamma = par_col_dot(g, xn, layout.channels, col_parallel, 1.0);

    let (channels, cpg) = (layout.channels, layout.channels_per_group);
    let n = (layout.positions * cpg) as f32;
    let item = layout.positions * channels;
    let mut grad_input = Tensor::zeros(IxDyn(&shape));

    // One task per batch item: their slabs are disjoint, so no merge is needed
    let per_item = |(b, (dx, (g_item, xn_item))): BackwardSlabs| {
        // Per-group reductions of `g * gamma` and `g * gamma * x_norm`, merged in row order
        let mut sum_g = vec![0.0f32; num_groups];
        let mut sum_g_xn = vec![0.0f32; num_groups];
        for (g_row, xn_row) in g_item
            .chunks_exact(channels)
            .zip(xn_item.chunks_exact(channels))
        {
            for gr in 0..num_groups {
                let (mut a1, mut a2) = (0.0f32, 0.0f32);
                for c in gr * cpg..(gr + 1) * cpg {
                    let t = g_row[c] * gamma_s[c];
                    a1 += t;
                    a2 += t * xn_row[c];
                }
                sum_g[gr] += a1;
                sum_g_xn[gr] += a2;
            }
        }

        // Broadcast the per-group scalars to per-channel so the sweep below is flat elementwise
        let mut is_c = vec![0.0f32; channels];
        let mut sg_c = vec![0.0f32; channels];
        let mut sgx_c = vec![0.0f32; channels];
        for gr in 0..num_groups {
            let inv = inv_std_s[b * num_groups + gr];
            for j in 0..cpg {
                is_c[gr * cpg + j] = inv;
                sg_c[gr * cpg + j] = sum_g[gr] / n;
                sgx_c[gr * cpg + j] = sum_g_xn[gr] / n;
            }
        }

        for ((dx_row, g_row), xn_row) in dx
            .chunks_exact_mut(channels)
            .zip(g_item.chunks_exact(channels))
            .zip(xn_item.chunks_exact(channels))
        {
            for c in 0..channels {
                dx_row[c] = is_c[c] * (g_row[c] * gamma_s[c] - (sg_c[c] + xn_row[c] * sgx_c[c]));
            }
        }
    };
    {
        let dx_all = grad_input.as_slice_mut().unwrap();
        if total >= gn_row_parallel_min_elems() {
            dx_all
                .par_chunks_mut(item)
                .zip(g.par_chunks(item).zip(xn.par_chunks(item)))
                .enumerate()
                .for_each(per_item);
        } else {
            dx_all
                .chunks_mut(item)
                .zip(g.chunks(item).zip(xn.chunks(item)))
                .enumerate()
                .for_each(per_item);
        }
    }

    (grad_input, grad_gamma, grad_beta)
}

/// Deterministic fold kernels shared by the normalization layers
mod folds;

/// Batch Normalization layer for neural networks
pub mod batch_normalization;
/// Group Normalization layer for neural networks
pub mod group_normalization;
/// Instance Normalization layer for neural networks
pub mod instance_normalization;
/// Layer Normalization layer for neural networks
pub mod layer_normalization;

pub use batch_normalization::BatchNormalization;
pub use group_normalization::GroupNormalization;
pub use instance_normalization::InstanceNormalization;
pub use layer_normalization::{LayerNormalization, LayerNormalizationAxis};

// Macros are defined after the `mod` declarations and path-exported via a `pub(in ...) use`
// re-export, so callers import them explicitly rather than relying on textual macro ordering
/// Common implementation for `output_shape` method in normalization layers
macro_rules! normalization_layer_output_shape {
    ($self:expr) => {
        if !$self.input_shape.is_empty() {
            format!(
                "({})",
                $self
                    .input_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::from("Unknown")
        }
    };
}

pub(in crate::neural_network::layers::regularization::normalization) use normalization_layer_output_shape;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_shape_vec(IxDyn(shape), data).unwrap()
    }

    /// Hand-derived group statistics on a channels-last input
    ///
    /// `[1, 2, 4]` is one sample with 2 positions and 4 channels split into 2 groups. Group 0 sees
    /// {1, 2, 3, 4} (mean 2.5, variance 1.25) and group 1 sees {10, 20, 30, 40} (mean 25, variance
    /// 125). The two groups have wildly different scales, so a group boundary read anywhere but
    /// across the trailing channel axis would blend them and neither number would survive
    #[test]
    fn group_norm_forward_hand_derived() {
        // Position 0 = [1, 2, 10, 20], position 1 = [3, 4, 30, 40]
        let x = tensor(vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0], &[1, 2, 4]);
        let gamma = tensor(vec![1.0; 4], &[4]);
        let beta = tensor(vec![0.0; 4], &[4]);

        let (out, _xn, inv_std) = group_norm_forward_core(&x, 2, &gamma, &beta, 1e-5);

        let inv0 = 1.0 / (1.25f32 + 1e-5).sqrt();
        let inv1 = 1.0 / (125.0f32 + 1e-5).sqrt();
        assert_abs_diff_eq!(inv_std.as_slice().unwrap()[0], inv0, epsilon = 1e-6);
        assert_abs_diff_eq!(inv_std.as_slice().unwrap()[1], inv1, epsilon = 1e-6);

        let got: Vec<f32> = out.iter().copied().collect();
        let want = [
            -1.5 * inv0,
            -0.5 * inv0,
            -15.0 * inv1,
            -5.0 * inv1,
            0.5 * inv0,
            1.5 * inv0,
            5.0 * inv1,
            15.0 * inv1,
        ];
        for (g, w) in got.iter().zip(want) {
            assert_abs_diff_eq!(*g, w, epsilon = 1e-5);
        }
    }

    /// The variance survives a mean that dwarfs the spread
    ///
    /// This is the case the shifted sum exists for. The data sits around `1e6` with a spread of a
    /// few units, so `E[x^2]` is about `1e12` - past f32's ~7 significant digits - and the textbook
    /// `E[x^2] - E[x]^2` would subtract two nearly equal huge numbers and return garbage or a
    /// negative variance. Measuring deviations from an element of the data keeps both accumulators
    /// on the order of the spread, so the normalized output is still the plain z-score
    #[test]
    fn group_norm_variance_survives_a_large_mean() {
        const BASE: f32 = 1.0e6;
        let x = tensor(
            vec![BASE + 1.0, BASE + 2.0, BASE + 3.0, BASE + 4.0],
            &[1, 4, 1],
        );
        let gamma = tensor(vec![1.0], &[1]);
        let beta = tensor(vec![0.0], &[1]);

        let (out, _xn, _inv) = group_norm_forward_core(&x, 1, &gamma, &beta, 1e-5);

        // mean 2.5 above BASE, variance 1.25
        let inv = 1.0 / (1.25f32).sqrt();
        let want = [-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv];
        for (g, w) in out.iter().zip(want) {
            assert_abs_diff_eq!(*g, w, epsilon = 2e-3);
        }
    }

    /// A single group over every channel reduces to per-sample normalization, and one group per
    /// channel to per-channel normalization - the two extremes the group count spans
    #[test]
    fn group_norm_group_count_extremes() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
        let gamma = tensor(vec![1.0, 1.0], &[2]);
        let beta = tensor(vec![0.0, 0.0], &[2]);

        // 1 group: statistics over all 6 elements, mean 3.5
        let (one, _, _) = group_norm_forward_core(&x, 1, &gamma, &beta, 0.0);
        let mean_all = 3.5f32;
        let var_all = (0..6)
            .map(|i| (i as f32 + 1.0 - mean_all).powi(2))
            .sum::<f32>()
            / 6.0;
        assert_abs_diff_eq!(
            one.iter().copied().next().unwrap(),
            (1.0 - mean_all) / var_all.sqrt(),
            epsilon = 1e-5
        );

        // 2 groups (one channel each): channel 0 is {1, 3, 5} and channel 1 is {2, 4, 6}
        let (two, _, _) = group_norm_forward_core(&x, 2, &gamma, &beta, 0.0);
        let var_ch = ((1.0f32 - 3.0).powi(2) + 0.0 + (5.0f32 - 3.0).powi(2)) / 3.0;
        assert_abs_diff_eq!(
            two.iter().copied().next().unwrap(),
            (1.0 - 3.0) / var_ch.sqrt(),
            epsilon = 1e-5
        );
    }

    /// The statistics fold groups its blocks by shape, so the parallel flag cannot move a bit
    #[test]
    fn group_stats_parallel_matches_serial_bitwise() {
        let (b, p, c) = (2usize, 300usize, 8usize);
        let data: Vec<f32> = (0..b * p * c)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.375)
            .collect();
        let x = tensor(data, &[b, p, c]);
        let xs = x.as_slice().unwrap();
        let layout = GroupLayout::new(x.shape(), 4);

        for item in 0..b {
            let serial = group_stats(xs, item, &layout, false);
            let par = group_stats(xs, item, &layout, true);
            assert_eq!(serial.s1, par.s1, "sums differ across the gate");
            assert_eq!(serial.s2, par.s2, "squared sums differ across the gate");
        }
    }

    /// Backward agrees with a central finite difference of the forward pass
    ///
    /// The forward is pinned against hand-derived values above, so differencing it is an
    /// independent check of the gradient rather than a self-consistency loop
    #[test]
    fn group_norm_backward_matches_finite_difference() {
        let shape = [1usize, 3, 4];
        let base: Vec<f32> = (0..12)
            .map(|i| (i as f32 * 0.7).sin() * 2.0 + 0.3)
            .collect();
        let gamma = tensor(vec![1.3, 0.7, -0.4, 1.1], &[4]);
        let beta = tensor(vec![0.2, -0.1, 0.5, 0.0], &[4]);
        let groups = 2;
        let eps = 1e-5;

        // Upstream gradient: a fixed asymmetric pattern so nothing cancels by symmetry
        let g_vec: Vec<f32> = (0..12).map(|i| ((i * 5 % 7) as f32 - 3.0) * 0.25).collect();
        let g = tensor(g_vec.clone(), &shape);

        let x = tensor(base.clone(), &shape);
        let (_, xn, inv_std) = group_norm_forward_core(&x, groups, &gamma, &beta, eps);
        let (grad_in, _, _) = group_norm_backward_core(&g, &xn, &inv_std, groups, &gamma);

        let h = 1e-2f32;
        for i in 0..12 {
            let mut up = base.clone();
            let mut dn = base.clone();
            up[i] += h;
            dn[i] -= h;
            let (out_up, _, _) =
                group_norm_forward_core(&tensor(up, &shape), groups, &gamma, &beta, eps);
            let (out_dn, _, _) =
                group_norm_forward_core(&tensor(dn, &shape), groups, &gamma, &beta, eps);
            // d/dx of sum(g * out)
            let num: f32 = out_up
                .iter()
                .zip(out_dn.iter())
                .zip(&g_vec)
                .map(|((u, d), gv)| gv * (u - d))
                .sum::<f32>()
                / (2.0 * h);
            assert_abs_diff_eq!(grad_in.as_slice().unwrap()[i], num, epsilon = 5e-3);
        }
    }
}
