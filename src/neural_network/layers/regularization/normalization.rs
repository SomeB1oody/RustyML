//! Normalization layers (batch, group, instance, layer) and shared layout/gradient helpers
//!
//! Provides channels-first permutation utilities and macros reused across the
//! normalization layer implementations

use self::folds::{
    par_plane_dot, par_plane_sum, rows_per_block, segment_dot, segment_sq_dev, segment_sum,
};
use crate::neural_network::Tensor;
use ndarray::{Array1, IxDyn};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::borrow::Cow;

/// Total-element count above which the group-normalization per-instance row passes run on
/// rayon.
///
/// Each instance (one sample's channel group) is one contiguous row computed entirely inside
/// one task with fixed-order kernels, so the gate is a pure performance knob: the bits are
/// identical at any thread count and on either side of the gate. The same fused row-pass
/// kernel class as LayerNorm's measured `LN_ROW_PARALLEL_MIN_ELEMS` (AMD Ryzen 9 9950X,
/// 16C/32T, 32 rayon threads, 2026-06-12; see benches/RESULTS.md "LayerNorm fused row pass":
/// crossover bracket 64K-256K elements), mapped from that measurement rather than calibrated
/// separately
const GN_ROW_PARALLEL_MIN_ELEMS: usize = 262_144;

/// Element count above which the per-channel gamma/beta gradient plane folds run on rayon.
///
/// The same plane-fold kernel as BatchNorm's measured `BN_PLANE_STATS_PARALLEL_MIN_ELEMS`
/// (same machine and date; see benches/RESULTS.md "BatchNorm plane stats, native-layout
/// fold": crossover bracket 64K-256K elements), mapped from that measurement
const GN_PLANE_STATS_PARALLEL_MIN_ELEMS: usize = 262_144;

/// Returns the axis permutation that moves `channel_axis` to position 1 (channels-first), keeping
/// the batch axis at 0 and the spatial axes in their original relative order
///
/// Instance/Group normalization is layout-equivariant: normalizing a tensor with the channel at
/// `channel_axis` is identical to permuting it channels-first, normalizing, and permuting back. The
/// channels-first numeric cores assume contiguous `[batch, channel, spatial...]` layout, so the
/// public methods bracket them with [`to_channels_first`] / [`from_channels_first`]
fn channels_first_perm(ndim: usize, channel_axis: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(ndim);
    perm.push(0);
    perm.push(channel_axis);
    perm.extend((1..ndim).filter(|&ax| ax != channel_axis));
    perm
}

/// Permutes `input` so the channel axis sits at position 1, returning a contiguous owned array
///
/// For `channel_axis == 1` the input is borrowed unchanged (no copy - the common fast path)
pub(super) fn to_channels_first(input: &Tensor, channel_axis: usize) -> Cow<'_, Tensor> {
    if channel_axis == 1 {
        Cow::Borrowed(input)
    } else {
        let perm = channels_first_perm(input.ndim(), channel_axis);
        Cow::Owned(
            input
                .view()
                .permuted_axes(perm)
                .as_standard_layout()
                .to_owned(),
        )
    }
}

/// Inverse of [`to_channels_first`]: moves the channel axis from position 1 back to `channel_axis`,
/// returning a contiguous owned array
///
/// A no-op for `channel_axis == 1`
pub(super) fn from_channels_first(output_cf: Tensor, channel_axis: usize) -> Tensor {
    if channel_axis == 1 {
        return output_cf;
    }
    let ndim = output_cf.ndim();
    let fwd = channels_first_perm(ndim, channel_axis);
    // Invert the forward permutation: channels-first position `new_pos` came from original axis
    // `old_ax`, so original axis `old_ax` reads from `new_pos`
    let mut inv = vec![0usize; ndim];
    for (new_pos, &old_ax) in fwd.iter().enumerate() {
        inv[old_ax] = new_pos;
    }
    output_cf
        .view()
        .permuted_axes(inv)
        .as_standard_layout()
        .to_owned()
}

/// Fused per-instance group-normalization forward over a contiguous channels-first slice
/// viewed as `[num_instances, group_size]` rows (one row = one sample's channel group, a
/// contiguous `channels_per_group * spatial` block): per row, the mean and variance fold with
/// the fixed-order segment kernels (the deviations square in registers, no centered
/// temporary), then one streaming sweep writes `x_normalized` and the per-channel affine
/// output. Rows are independent and each is computed entirely inside one task, so the
/// `parallel` flag — and the rows-per-task chunking — never changes the result bits
#[allow(clippy::too_many_arguments)]
fn gn_row_forward(
    x: &[f32],
    channels_per_group: usize,
    spatial: usize,
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
    parallel: bool,
    xn: &mut [f32],
    out: &mut [f32],
    inv_std: &mut [f32],
) {
    let n = channels_per_group * spatial;
    let rows = rows_per_block(n);
    let chunk = rows * n;
    type ForwardChunks<'a> = (
        usize,
        (((&'a mut [f32], &'a mut [f32]), &'a [f32]), &'a mut [f32]),
    );
    let task = |(ci, (((xn_c, out_c), x_c), is_c)): ForwardChunks| {
        let row0 = ci * rows;
        let row_iter = x_c
            .chunks_exact(n)
            .zip(xn_c.chunks_exact_mut(n))
            .zip(out_c.chunks_exact_mut(n))
            .zip(is_c.iter_mut())
            .enumerate();
        for (j, (((x_row, xn_row), out_row), is_out)) in row_iter {
            let mean = segment_sum(x_row, 1.0) / n as f32;
            let var = segment_sq_dev(x_row, mean) / n as f32;
            let inv_std_val = 1.0 / (var + epsilon).sqrt();
            // The row's channels are contiguous `spatial`-length segments; hoist each
            // channel's affine scalars over its segment
            let ch_base = ((row0 + j) % num_groups) * channels_per_group;
            let seg_iter = x_row
                .chunks_exact(spatial)
                .zip(xn_row.chunks_exact_mut(spatial))
                .zip(out_row.chunks_exact_mut(spatial))
                .enumerate();
            for (k, ((x_seg, xn_seg), out_seg)) in seg_iter {
                let (gamma_v, beta_v) = (gamma[ch_base + k], beta[ch_base + k]);
                for ((xn_v, out_v), &v) in xn_seg.iter_mut().zip(out_seg.iter_mut()).zip(x_seg) {
                    *xn_v = (v - mean) * inv_std_val;
                    *out_v = *xn_v * gamma_v + beta_v;
                }
            }
            *is_out = inv_std_val;
        }
    };
    if parallel {
        xn.par_chunks_mut(chunk)
            .zip(out.par_chunks_mut(chunk))
            .zip(x.par_chunks(chunk))
            .zip(inv_std.par_chunks_mut(rows))
            .enumerate()
            .for_each(task);
    } else {
        xn.chunks_mut(chunk)
            .zip(out.chunks_mut(chunk))
            .zip(x.chunks(chunk))
            .zip(inv_std.chunks_mut(rows))
            .enumerate()
            .for_each(task);
    }
}

/// Fused per-instance group-normalization backward over channels-first slices viewed as
/// `[num_instances, group_size]` rows: the two per-row reductions fold `g * gamma` per term
/// (no `grad_x_normalized` temporary) and one streaming sweep composes the input gradient
/// with the standard identity
/// `dx = inv_std * (g * gamma - (sum_g + x_norm * sum_g_xnorm) / group_size)`.
/// Same flag semantics as [`gn_row_forward`]
#[allow(clippy::too_many_arguments)]
fn gn_row_backward(
    g: &[f32],
    xn: &[f32],
    inv_std: &[f32],
    channels_per_group: usize,
    spatial: usize,
    num_groups: usize,
    gamma: &[f32],
    parallel: bool,
    gi: &mut [f32],
) {
    let n = channels_per_group * spatial;
    let gs = n as f32;
    let rows = rows_per_block(n);
    let chunk = rows * n;
    type BackwardChunks<'a> = (usize, (((&'a mut [f32], &'a [f32]), &'a [f32]), &'a [f32]));
    let task = |(ci, (((gi_c, g_c), xn_c), is_c)): BackwardChunks| {
        let row0 = ci * rows;
        let row_iter = g_c
            .chunks_exact(n)
            .zip(gi_c.chunks_exact_mut(n))
            .zip(xn_c.chunks_exact(n))
            .zip(is_c.iter())
            .enumerate();
        for (j, (((g_row, gi_row), xn_row), &inv_std_val)) in row_iter {
            let ch_base = ((row0 + j) % num_groups) * channels_per_group;
            let mut sum_g = 0.0f32;
            let mut sum_gx = 0.0f32;
            for (k, (g_seg, xn_seg)) in g_row
                .chunks_exact(spatial)
                .zip(xn_row.chunks_exact(spatial))
                .enumerate()
            {
                let gamma_v = gamma[ch_base + k];
                sum_g += segment_sum(g_seg, gamma_v);
                sum_gx += segment_dot(g_seg, xn_seg, gamma_v);
            }
            let seg_iter = gi_row
                .chunks_exact_mut(spatial)
                .zip(g_row.chunks_exact(spatial))
                .zip(xn_row.chunks_exact(spatial))
                .enumerate();
            for (k, ((gi_seg, g_seg), xn_seg)) in seg_iter {
                let gamma_v = gamma[ch_base + k];
                for ((gi_v, &g_v), &xn_v) in gi_seg.iter_mut().zip(g_seg).zip(xn_seg) {
                    *gi_v = (g_v * gamma_v - (sum_g + xn_v * sum_gx) / gs) * inv_std_val;
                }
            }
        }
    };
    if parallel {
        gi.par_chunks_mut(chunk)
            .zip(g.par_chunks(chunk))
            .zip(xn.par_chunks(chunk))
            .zip(inv_std.par_chunks(rows))
            .enumerate()
            .for_each(task);
    } else {
        gi.chunks_mut(chunk)
            .zip(g.chunks(chunk))
            .zip(xn.chunks(chunk))
            .zip(inv_std.chunks(rows))
            .enumerate()
            .for_each(task);
    }
}

/// Channels-first group-normalization forward core
///
/// `input` is channels-first `[batch, channels, spatial...]`. Each sample's channels are split
/// into `num_groups` contiguous groups and normalized within the group. Because a group is a
/// contiguous `channels_per_group * spatial` block in this layout, the whole op runs as the
/// fused per-instance row pass [`gn_row_forward`] — no reshape copies or broadcast
/// temporaries
///
/// Returns `(output, x_normalized, inv_std)`:
/// - `output` already includes the per-channel affine (`gamma * x_norm + beta`)
/// - `x_normalized` is channels-first `[batch, channels, spatial...]`, cached for backward
/// - `inv_std` is `1 / sqrt(var + epsilon)`, one value per instance `[batch * num_groups]`
pub(super) fn group_norm_forward_core(
    input: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
    beta: &Tensor,
    epsilon: f32,
) -> (Tensor, Tensor, Tensor) {
    let shape = input.shape().to_vec();
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let channels_per_group = channels / num_groups;
    let num_instances = shape[0] * num_groups;
    let total = input.len();
    if total == 0 {
        // Degenerate empty input: nothing to normalize
        return (
            Tensor::zeros(IxDyn(&shape)),
            Tensor::zeros(IxDyn(&shape)),
            Array1::<f32>::zeros(num_instances).into_dyn(),
        );
    }

    // The row pass needs contiguous data; a borrowed channel_axis == 1 input may be any view
    let input_std = input.as_standard_layout();
    let x = input_std.as_slice().unwrap();
    let parallel = total >= GN_ROW_PARALLEL_MIN_ELEMS;

    let mut x_normalized = Tensor::zeros(IxDyn(&shape));
    let mut output = Tensor::zeros(IxDyn(&shape));
    let mut inv_std = Array1::<f32>::zeros(num_instances);
    gn_row_forward(
        x,
        channels_per_group,
        spatial,
        num_groups,
        gamma.as_slice().unwrap(),
        beta.as_slice().unwrap(),
        epsilon,
        parallel,
        x_normalized.as_slice_mut().unwrap(),
        output.as_slice_mut().unwrap(),
        inv_std.as_slice_mut().unwrap(),
    );

    (output, x_normalized, inv_std.into_dyn())
}

/// Channels-first group-normalization backward core
///
/// Inverse of [`group_norm_forward_core`]; all arguments are channels-first. The per-channel
/// parameter gradients are deterministic plane folds over the native `[batch, channels,
/// spatial]` layout and the input gradient is the fused per-instance row pass
/// [`gn_row_backward`] — no reshape copies or broadcast temporaries
///
/// Returns `(grad_input, grad_gamma, grad_beta)` with `grad_input` channels-first and the
/// parameter gradients shaped `[channels]`
pub(super) fn group_norm_backward_core(
    grad_output: &Tensor,
    x_normalized: &Tensor,
    inv_std: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let shape = grad_output.shape().to_vec();
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let channels_per_group = channels / num_groups;
    let total = grad_output.len();
    if total == 0 {
        return (
            Tensor::zeros(IxDyn(&shape)),
            Array1::<f32>::zeros(channels).into_dyn(),
            Array1::<f32>::zeros(channels).into_dyn(),
        );
    }

    let grad_std = grad_output.as_standard_layout();
    let g = grad_std.as_slice().unwrap();
    let xn_std = x_normalized.as_standard_layout();
    let xn = xn_std.as_slice().unwrap();
    let inv_std_s = inv_std.as_slice().unwrap();
    let gamma_s = gamma.as_slice().unwrap();

    // Parameter gradients: per-channel plane folds over the native [B, C, P] layout
    let plane_parallel = total >= GN_PLANE_STATS_PARALLEL_MIN_ELEMS;
    let grad_beta = par_plane_sum(g, channels, spatial, plane_parallel, 1.0);
    let grad_gamma = par_plane_dot(g, xn, channels, spatial, plane_parallel, 1.0);

    // Input gradient: fused per-instance row pass
    let row_parallel = total >= GN_ROW_PARALLEL_MIN_ELEMS;
    let mut grad_input = Tensor::zeros(IxDyn(&shape));
    gn_row_backward(
        g,
        xn,
        inv_std_s,
        channels_per_group,
        spatial,
        num_groups,
        gamma_s,
        row_parallel,
        grad_input.as_slice_mut().unwrap(),
    );

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
    use ndarray::ArrayD;

    // Helper: build a Tensor from a flat Vec and shape
    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        ArrayD::from_shape_vec(shape, data).expect("shape/data mismatch in test helper")
    }

    // Helper: assert a tensor's elements (logical order) match `expected` within `tol`
    fn assert_close(actual: &Tensor, expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (got, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= tol,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    // channels_first_perm

    /// channels_first_perm(4, 3) puts channel axis 3 at position 1, yielding [0, 3, 1, 2]
    #[test]
    fn test_channels_first_perm_4_3() {
        let perm = channels_first_perm(4, 3);
        assert_eq!(perm, vec![0usize, 3, 1, 2]);
    }

    /// channels_first_perm(4, 1) is the identity [0, 1, 2, 3] since the channel is already at position 1
    #[test]
    fn test_channels_first_perm_4_1_identity() {
        let perm = channels_first_perm(4, 1);
        assert_eq!(perm, vec![0usize, 1, 2, 3]);
    }

    /// channels_first_perm(3, 2) puts channel axis 2 at position 1, yielding [0, 2, 1]
    #[test]
    fn test_channels_first_perm_3_2() {
        let perm = channels_first_perm(3, 2);
        assert_eq!(perm, vec![0usize, 2, 1]);
    }

    // to_channels_first + from_channels_first round-trip

    /// from_channels_first(to_channels_first(x), ca) recovers x for shape [2,4,3,3] with channel_axis=3
    #[test]
    fn test_to_from_channels_first_roundtrip() {
        // Non-constant tensor so any element swap is caught
        let n: usize = 2 * 4 * 3 * 3;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = make_tensor(data, &[2, 4, 3, 3]);

        let channel_axis = 3usize;
        let cf = to_channels_first(&x, channel_axis);
        let recovered = from_channels_first(cf.into_owned(), channel_axis);

        // Elementwise equality: every element must return to its original slot
        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(
                orig, got,
                "round-trip mismatch at flat index {i}: orig={orig}, got={got}"
            );
        }
    }

    /// When channel_axis==1, to_channels_first borrows without copying and the round-trip still holds
    #[test]
    fn test_to_from_channels_first_channel_axis_1_noop() {
        let data: Vec<f32> = (0..24usize).map(|i| i as f32 * 0.5).collect();
        let x = make_tensor(data, &[2, 4, 3]);
        let channel_axis = 1usize;

        let cf = to_channels_first(&x, channel_axis);
        // Must be a Borrow (no copy)
        assert!(matches!(cf, std::borrow::Cow::Borrowed(_)));
        let recovered = from_channels_first(cf.into_owned(), channel_axis);

        let x_flat: &[f32] = x.as_slice().unwrap();
        let r_flat: &[f32] = recovered.as_slice().unwrap();
        for (i, (&orig, &got)) in x_flat.iter().zip(r_flat.iter()).enumerate() {
            assert_eq!(orig, got, "noop round-trip mismatch at flat index {i}");
        }
    }

    // group_norm_forward_core

    /// num_groups=1 normalizes all C*spatial values of a sample together, then applies
    /// the per-channel affine gamma*x_norm + beta. Input [1,2,3,4]: mean=2.5, var=1.25
    #[test]
    fn test_group_norm_forward_core_single_group() {
        // [batch=1, channels=2, spatial=2], channels-first flat [1,2,3,4]
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![2.0, 3.0], &[2]);
        let beta = make_tensor(vec![10.0, 20.0], &[2]);

        let (output, x_norm, inv_std) = group_norm_forward_core(&input, 1, &gamma, &beta, 0.0);

        // x_norm = (x - 2.5) * inv_std, inv_std = 1/sqrt(1.25)
        let inv = 1.0_f32 / 1.25_f32.sqrt();
        assert_close(
            &x_norm,
            &[-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv],
            1e-5,
        );
        assert_close(&inv_std, &[inv], 1e-6);
        // channel 0 (elems 0,1) affine 2x+10; channel 1 (elems 2,3) affine 3x+20
        assert_close(
            &output,
            &[
                2.0 * (-1.5 * inv) + 10.0,
                2.0 * (-0.5 * inv) + 10.0,
                3.0 * (0.5 * inv) + 20.0,
                3.0 * (1.5 * inv) + 20.0,
            ],
            1e-5,
        );
    }

    /// num_groups=2: each group is one channel of 2 spatial values, so every group
    /// [a,b] normalizes to [-1, 1] (var=0.25, inv_std=2) independently
    #[test]
    fn test_group_norm_forward_core_two_groups() {
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![1.0, 1.0], &[2]);
        let beta = make_tensor(vec![0.0, 0.0], &[2]);

        let (output, x_norm, inv_std) = group_norm_forward_core(&input, 2, &gamma, &beta, 0.0);

        assert_close(&x_norm, &[-1.0, 1.0, -1.0, 1.0], 1e-5);
        assert_close(&inv_std, &[2.0, 2.0], 1e-6);
        assert_close(&output, &[-1.0, 1.0, -1.0, 1.0], 1e-5);
    }

    // group_norm_backward_core

    /// Input gradient for one group, checked against the hand-evaluated identity
    /// dx = inv_std * (g - (sum(g) + x_norm*sum(g*x_norm))/group_size) with
    /// grad_output = [1,0,0,0] and gamma = 1; combo works out to [0.7, 0.4, 0.1, -0.2]
    #[test]
    fn test_group_norm_backward_core_single_group() {
        let inv = 1.0_f32 / 1.25_f32.sqrt();
        let x_norm = make_tensor(
            vec![-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv],
            &[1, 2, 2],
        );
        let inv_std = make_tensor(vec![inv], &[1]);
        let grad_output = make_tensor(vec![1.0, 0.0, 0.0, 0.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![1.0, 1.0], &[2]);

        let (grad_input, grad_gamma, grad_beta) =
            group_norm_backward_core(&grad_output, &x_norm, &inv_std, 1, &gamma);

        // grad = (g - combo) * inv_std
        assert_close(
            &grad_input,
            &[0.3 * inv, -0.4 * inv, -0.1 * inv, 0.2 * inv],
            1e-5,
        );
        // grad_gamma[c] = sum(grad_output * x_norm) over the channel
        assert_close(&grad_gamma, &[-1.5 * inv, 0.0], 1e-5);
        // grad_beta[c] = sum(grad_output) over the channel
        assert_close(&grad_beta, &[1.0, 0.0], 1e-6);
    }

    /// A group of two elements is fully determined (+/-1) after normalization, so its
    /// input gradient is exactly zero for any grad_output; parameter grads are the sums
    #[test]
    fn test_group_norm_backward_core_two_groups_zero_input_grad() {
        let x_norm = make_tensor(vec![-1.0, 1.0, -1.0, 1.0], &[1, 2, 2]);
        let inv_std = make_tensor(vec![2.0, 2.0], &[2]);
        let grad_output = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gamma = make_tensor(vec![1.0, 1.0], &[2]);

        let (grad_input, grad_gamma, grad_beta) =
            group_norm_backward_core(&grad_output, &x_norm, &inv_std, 2, &gamma);

        assert_close(&grad_input, &[0.0, 0.0, 0.0, 0.0], 1e-6);
        // grad_gamma[c] = sum(grad_output * x_norm): ch0 = 1*-1+2*1 = 1; ch1 = 3*-1+4*1 = 1
        assert_close(&grad_gamma, &[1.0, 1.0], 1e-6);
        // grad_beta[c] = sum(grad_output): ch0 = 1+2 = 3; ch1 = 3+4 = 7
        assert_close(&grad_beta, &[3.0, 7.0], 1e-6);
    }

    fn group_data(b: usize, c: usize, p: usize, salt: f32) -> Tensor {
        ArrayD::from_shape_vec(
            vec![b, c, p],
            (0..b * c * p).map(|i| (i as f32 * salt).sin()).collect(),
        )
        .unwrap()
    }

    /// The group-norm row passes compute each instance entirely inside one task with
    /// fixed-order kernels, so the parallel flag must never change the bits - including
    /// sub-8 spatial sizes, chunk-boundary-exact group sizes, and instance-norm-like
    /// single-channel groups
    #[test]
    fn gn_row_passes_parallel_flag_invariant() {
        for &(b, groups, cpg, spatial) in &[
            (2usize, 3usize, 5usize, 7usize),
            (1, 2, 4, 4_096),
            (3, 4, 1, 5_000),
            (2, 1, 8, 3),
            (5, 2, 3, 8_191),
        ] {
            let c = groups * cpg;
            let total = b * c * spatial;
            let x: Vec<f32> = (0..total).map(|i| (i as f32 * 0.731).sin()).collect();
            let g: Vec<f32> = (0..total).map(|i| (i as f32 * 0.433).sin()).collect();
            let gamma: Vec<f32> = (0..c).map(|j| 1.5 - 0.05 * j as f32).collect();
            let beta: Vec<f32> = (0..c).map(|j| -0.25 + 0.1 * j as f32).collect();
            let n_inst = b * groups;
            let eps = 1e-5f32;

            type PassOutputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
            let mut results: Vec<PassOutputs> = Vec::new();
            for parallel in [false, true] {
                let mut xn = vec![0.0f32; total];
                let mut out = vec![0.0f32; total];
                let mut inv_std = vec![0.0f32; n_inst];
                gn_row_forward(
                    &x,
                    cpg,
                    spatial,
                    groups,
                    &gamma,
                    &beta,
                    eps,
                    parallel,
                    &mut xn,
                    &mut out,
                    &mut inv_std,
                );
                let mut gi = vec![0.0f32; total];
                gn_row_backward(
                    &g, &xn, &inv_std, cpg, spatial, groups, &gamma, parallel, &mut gi,
                );
                results.push((xn, out, inv_std, gi));
            }
            assert_eq!(
                results[0], results[1],
                "the parallel flag changed group-norm row-pass bits at \
                 [b={b} groups={groups} cpg={cpg} spatial={spatial}]"
            );
        }
    }

    /// On integer-valued data with a power-of-two group size the per-instance statistics are
    /// exact, so the forward core must reproduce the reshape + broadcast reference bit for bit
    #[test]
    fn gn_forward_core_exact_on_integer_data_matches_broadcast_reference() {
        use ndarray::{Array2, Axis};
        let (b, groups, cpg, spatial) = (2usize, 2usize, 4usize, 16usize);
        let c = groups * cpg;
        let group_size = cpg * spatial;
        let n_inst = b * groups;
        let eps = 1e-5f32;
        let x = ArrayD::from_shape_vec(
            vec![b, c, spatial],
            (0..b * c * spatial).map(|i| ((i * 7) % 4) as f32).collect(),
        )
        .unwrap();
        let gamma =
            ArrayD::from_shape_vec(vec![c], (0..c).map(|j| 1.5 - 0.25 * j as f32).collect())
                .unwrap();
        let beta =
            ArrayD::from_shape_vec(vec![c], (0..c).map(|j| -0.75 + 0.5 * j as f32).collect())
                .unwrap();

        let (out, xn, inv_std) = group_norm_forward_core(&x, groups, &gamma, &beta, eps);

        // Reference with the old reshape + broadcast forms; exact statistics make the
        // kernel grouping moot
        let flat: Array2<f32> = x.to_shape((n_inst, group_size)).unwrap().to_owned();
        let mean = flat.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let centered = &flat - &mean;
        let var = centered.mapv(|v| v * v).mean_axis(Axis(1)).unwrap();
        let inv_std_ref = var.mapv(|v| 1.0 / (v + eps).sqrt());
        let normalized = &centered * &inv_std_ref.clone().insert_axis(Axis(1));
        let xn_ref = normalized.to_shape(x.shape()).unwrap().to_owned();
        let gamma_b = gamma.to_shape([1, c, 1]).unwrap().to_owned();
        let beta_b = beta.to_shape([1, c, 1]).unwrap().to_owned();
        let out_ref = &xn_ref * &gamma_b + &beta_b;

        assert_eq!(xn, xn_ref.into_dyn(), "x_normalized must match exactly");
        assert_eq!(out, out_ref.into_dyn(), "output must match exactly");
        assert_eq!(
            inv_std.as_slice().unwrap(),
            inv_std_ref.as_slice().unwrap(),
            "inv_std must match exactly"
        );
    }

    /// The fused cores agree with the reshape + broadcast reference formulas to rounding on
    /// arbitrary float data, forward and backward (including the plane-fold parameter
    /// gradients)
    #[test]
    fn gn_cores_match_broadcast_reference_closely() {
        use ndarray::{Array2, Axis};
        let (b, groups, cpg, spatial) = (3usize, 4usize, 2usize, 10usize);
        let c = groups * cpg;
        let group_size = cpg * spatial;
        let n_inst = b * groups;
        let eps = 1e-5f32;
        let x = group_data(b, c, spatial, 0.731);
        let grad = group_data(b, c, spatial, 0.433);
        let gamma = ArrayD::from_shape_vec(vec![c], (0..c).map(|j| 1.2 - 0.1 * j as f32).collect())
            .unwrap();
        let beta = ArrayD::from_shape_vec(vec![c], (0..c).map(|j| 0.3 * j as f32 - 0.5).collect())
            .unwrap();

        let (out, xn, inv_std) = group_norm_forward_core(&x, groups, &gamma, &beta, eps);
        let (gi, gg, gb) = group_norm_backward_core(&grad, &xn, &inv_std, groups, &gamma);

        // Forward reference
        let flat: Array2<f32> = x.to_shape((n_inst, group_size)).unwrap().to_owned();
        let mean = flat.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let centered = &flat - &mean;
        let var = centered.mapv(|v| v * v).mean_axis(Axis(1)).unwrap();
        let inv_std_ref = var.mapv(|v| 1.0 / (v + eps).sqrt());
        let xn_ref: Array2<f32> = &centered * &inv_std_ref.clone().insert_axis(Axis(1));

        // Backward reference (old identity forms)
        let g_flat: Array2<f32> = grad.to_shape((n_inst, group_size)).unwrap().to_owned();
        let mut gxn = g_flat.clone();
        for r in 0..n_inst {
            let gidx = r % groups;
            for j in 0..group_size {
                let ch = gidx * cpg + j / spatial;
                gxn[(r, j)] *= gamma[ch];
            }
        }
        let sum_g = gxn.sum_axis(Axis(1)).insert_axis(Axis(1));
        let sum_gx = (&gxn * &xn_ref).sum_axis(Axis(1)).insert_axis(Axis(1));
        let combo = (&sum_g + &(&xn_ref * &sum_gx)) / group_size as f32;
        let gi_ref = (&gxn - &combo) * &inv_std_ref.clone().insert_axis(Axis(1));

        let close = |a: f32, e: f32| (a - e).abs() <= 1e-5 + 1e-4 * e.abs();
        for (i, (&a, &e)) in out
            .as_slice()
            .unwrap()
            .iter()
            .zip(
                (&xn_ref.to_shape(x.shape()).unwrap().to_owned()
                    * &gamma.to_shape([1, c, 1]).unwrap().to_owned()
                    + &beta.to_shape([1, c, 1]).unwrap().to_owned())
                    .as_slice()
                    .unwrap(),
            )
            .enumerate()
        {
            assert!(close(a, e), "forward mismatch at {i}: {a} vs {e}");
        }
        for (i, (&a, &e)) in gi
            .as_slice()
            .unwrap()
            .iter()
            .zip(gi_ref.as_slice().unwrap())
            .enumerate()
        {
            assert!(close(a, e), "grad_input mismatch at {i}: {a} vs {e}");
        }
        // Parameter gradients vs axis-sum reference on the [B, C, P] view
        let go3 = grad.to_shape((b, c, spatial)).unwrap().to_owned();
        let xn3 = xn.to_shape((b, c, spatial)).unwrap().to_owned();
        let gb_ref = go3.sum_axis(Axis(2)).sum_axis(Axis(0));
        let gg_ref = (&go3 * &xn3).sum_axis(Axis(2)).sum_axis(Axis(0));
        for (i, (&a, &e)) in gb
            .as_slice()
            .unwrap()
            .iter()
            .zip(gb_ref.as_slice().unwrap())
            .enumerate()
        {
            assert!(close(a, e), "grad_beta mismatch at {i}: {a} vs {e}");
        }
        for (i, (&a, &e)) in gg
            .as_slice()
            .unwrap()
            .iter()
            .zip(gg_ref.as_slice().unwrap())
            .enumerate()
        {
            assert!(close(a, e), "grad_gamma mismatch at {i}: {a} vs {e}");
        }
    }
}
