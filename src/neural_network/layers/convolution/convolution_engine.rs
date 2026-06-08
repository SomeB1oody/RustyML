//! Dimension-generic convolution engine shared by `Conv1D`, `Conv2D`, and `Conv3D`.
//!
//! A plain convolution is the same operation at every rank — only the number of spatial axes
//! changes — so one implementation, generic over the spatial rank `R = ndim - 2`, serves all
//! three layers. The layer wrappers keep their public API, weight storage, activation, and caches;
//! they delegate just the numeric forward/backward to [`conv_forward`] and [`conv_backward`].
//!
//! # Conventions (match the previous per-rank code exactly)
//!
//! - `Valid` output: `(in - k) / stride + 1`; `Same` output: `ceil(in / stride)`.
//! - `Same` padding splits the total evenly with the extra cell on the trailing edge
//!   (`pad_before = pad_total / 2`).
//! - Cross-correlation (no kernel flip); the bias is added last so results are bit-compatible with
//!   the previous implementations.
//!
//! # Layout
//!
//! Tensors are `[batch, channels, spatial...]`; weights are flat row-major `[F, Cin, k...]`. Work is
//! parallelized over `batch * filters` (forward), `filters` (weight/bias gradients), and
//! `batch * channels` (input gradient) — each task writes a disjoint output region.

use super::PaddingType;
use crate::neural_network::Tensor;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

/// Workload (output-element count) at or above which an engine pass runs in parallel.
const CONV_PARALLEL_THRESHOLD: usize = 10_000;

/// Analytic gradients returned by [`conv_backward`].
pub(super) struct ConvGradients {
    /// Weight gradient, flat row-major `[F, Cin, k...]` (reshape to the layer's weight array).
    pub weight_grad: Vec<f32>,
    /// Bias gradient, one value per filter `[F]`.
    pub bias_grad: Vec<f32>,
    /// Input gradient, shape `[batch, Cin, spatial...]`.
    pub input_grad: Tensor,
}

/// Row-major (C-order) strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Advances a multi-index `idx` (row-major, last axis fastest) within `dims`; `false` when it wraps.
#[inline]
fn increment_index(idx: &mut [usize], dims: &[usize]) -> bool {
    for k in (0..idx.len()).rev() {
        idx[k] += 1;
        if idx[k] < dims[k] {
            return true;
        }
        idx[k] = 0;
    }
    false
}

/// Runs `f` over `0..n`, in parallel when `parallel`, preserving index order.
fn map_indexed<R, F>(n: usize, parallel: bool, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if parallel {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Per-spatial-axis output size, leading padding, and padded size for the given padding mode.
fn conv_geometry(
    sp: &[usize],
    k_dims: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let r = sp.len();
    match padding {
        PaddingType::Valid => {
            let out_sp: Vec<usize> = (0..r).map(|d| (sp[d] - k_dims[d]) / strides[d] + 1).collect();
            (out_sp, vec![0; r], sp.to_vec())
        }
        PaddingType::Same => {
            let out_sp: Vec<usize> = (0..r).map(|d| sp[d].div_ceil(strides[d])).collect();
            let pad_before: Vec<usize> = (0..r)
                .map(|d| (((out_sp[d] - 1) * strides[d] + k_dims[d]).saturating_sub(sp[d])) / 2)
                .collect();
            let padded_sp: Vec<usize> = (0..r)
                .map(|d| ((out_sp[d] - 1) * strides[d] + k_dims[d]).max(sp[d]))
                .collect();
            (out_sp, pad_before, padded_sp)
        }
    }
}

/// Builds a zero-padded copy of the flat `[bc, sp...]` channel-plane data.
fn build_padded(
    in_flat: &[f32],
    bc: usize,
    sp: &[usize],
    padded_sp: &[usize],
    pad_before: &[usize],
) -> Vec<f32> {
    let r = sp.len();
    let in_plane: usize = sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(padded_sp);
    let mut out = vec![0.0f32; bc * padded_plane];

    for chan in 0..bc {
        let in_base = chan * in_plane;
        let pad_base = chan * padded_plane;
        let mut si = vec![0usize; r];
        let mut si_flat = 0usize;
        loop {
            let mut pidx = 0usize;
            for d in 0..r {
                pidx += (si[d] + pad_before[d]) * padded_strides[d];
            }
            out[pad_base + pidx] = in_flat[in_base + si_flat];
            si_flat += 1;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
    }
    out
}

/// Forward convolution. `weight_shape` is `[F, Cin, k...]`; `bias` is `[F]`; `strides` has length `R`.
pub(super) fn conv_forward(
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    strides: &[usize],
    padding: PaddingType,
) -> Tensor {
    let in_shape = input.shape();
    let (batch, cin) = (in_shape[0], in_shape[1]);
    let sp = &in_shape[2..];
    let r = sp.len();
    let filters = weight_shape[0];
    let k_dims = &weight_shape[2..];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding);
    let out_plane: usize = out_sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(&padded_sp);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let padded_storage = if padded_sp.as_slice() != sp {
        Some(build_padded(in_flat, batch * cin, sp, &padded_sp, &pad_before))
    } else {
        None
    };
    let padded: &[f32] = padded_storage.as_deref().unwrap_or(in_flat);

    // One [out_plane] tile per (batch, filter) pair.
    let process_bf = |bf: usize| -> Vec<f32> {
        let b = bf / filters;
        let f = bf % filters;
        let mut tile = vec![0.0f32; out_plane];
        let mut o = vec![0usize; r];
        let mut kk = vec![0usize; r];
        let mut o_flat = 0usize;
        loop {
            let mut sum = 0.0f32;
            for c in 0..cin {
                let w_base = (f * cin + c) * k_plane;
                let p_base = (b * cin + c) * padded_plane;
                kk.iter_mut().for_each(|x| *x = 0);
                let mut kk_flat = 0usize;
                loop {
                    let mut pidx = 0usize;
                    for d in 0..r {
                        pidx += (o[d] * strides[d] + kk[d]) * padded_strides[d];
                    }
                    sum += padded[p_base + pidx] * weights[w_base + kk_flat];
                    kk_flat += 1;
                    if !increment_index(&mut kk, k_dims) {
                        break;
                    }
                }
            }
            // Bias added last to match the previous per-rank accumulation order.
            tile[o_flat] = sum + bias[f];
            o_flat += 1;
            if !increment_index(&mut o, &out_sp) {
                break;
            }
        }
        tile
    };

    let parallel = batch * filters * out_plane >= CONV_PARALLEL_THRESHOLD;
    let tiles = map_indexed(batch * filters, parallel, process_bf);

    let mut out_flat = Vec::with_capacity(batch * filters * out_plane);
    for tile in tiles {
        out_flat.extend(tile);
    }
    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.push(filters);
    out_shape.extend_from_slice(&out_sp);
    ArrayD::from_shape_vec(IxDyn(&out_shape), out_flat).expect("conv output length matches shape")
}

/// Backward convolution. `input` is the original (unpadded) forward input; `grad_output` is the
/// gradient w.r.t. the convolution output (i.e. after the activation backward).
pub(super) fn conv_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weights: &[f32],
    weight_shape: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> ConvGradients {
    let in_shape = input.shape();
    let (batch, cin) = (in_shape[0], in_shape[1]);
    let sp = &in_shape[2..];
    let r = sp.len();
    let filters = weight_shape[0];
    let k_dims = &weight_shape[2..];
    let k_plane: usize = k_dims.iter().product();

    let (out_sp, pad_before, padded_sp) = conv_geometry(sp, k_dims, strides, padding);
    let out_plane: usize = out_sp.iter().product();
    let in_plane: usize = sp.iter().product();
    let padded_plane: usize = padded_sp.iter().product();
    let padded_strides = row_major_strides(&padded_sp);

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");
    let padded_storage = if padded_sp.as_slice() != sp {
        Some(build_padded(in_flat, batch * cin, sp, &padded_sp, &pad_before))
    } else {
        None
    };
    let padded: &[f32] = padded_storage.as_deref().unwrap_or(in_flat);

    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // --- weight and bias gradients: one filter per task (reduces over batch and output). ---
    let per_f = |f: usize| -> (Vec<f32>, f32) {
        let mut wg = vec![0.0f32; cin * k_plane];
        let mut bias_sum = 0.0f32;
        let mut o = vec![0usize; r];
        let mut kk = vec![0usize; r];
        for b in 0..batch {
            let g_base = (b * filters + f) * out_plane;
            o.iter_mut().for_each(|x| *x = 0);
            let mut o_flat = 0usize;
            loop {
                let g = grad_flat[g_base + o_flat];
                bias_sum += g;
                for c in 0..cin {
                    let p_base = (b * cin + c) * padded_plane;
                    let w_base = c * k_plane;
                    kk.iter_mut().for_each(|x| *x = 0);
                    let mut kk_flat = 0usize;
                    loop {
                        let mut pidx = 0usize;
                        for d in 0..r {
                            pidx += (o[d] * strides[d] + kk[d]) * padded_strides[d];
                        }
                        wg[w_base + kk_flat] += g * padded[p_base + pidx];
                        kk_flat += 1;
                        if !increment_index(&mut kk, k_dims) {
                            break;
                        }
                    }
                }
                o_flat += 1;
                if !increment_index(&mut o, &out_sp) {
                    break;
                }
            }
        }
        (wg, bias_sum)
    };

    let parallel_w = batch * out_plane * cin * k_plane >= CONV_PARALLEL_THRESHOLD;
    let f_results = map_indexed(filters, parallel_w, per_f);
    let mut weight_grad = Vec::with_capacity(filters * cin * k_plane);
    let mut bias_grad = Vec::with_capacity(filters);
    for (wg, b) in f_results {
        weight_grad.extend(wg);
        bias_grad.push(b);
    }

    // --- input gradient: one channel-plane per (batch, channel), gathered per original position. ---
    let process_bc = |bc: usize| -> Vec<f32> {
        let b = bc / cin;
        let c = bc % cin;
        let mut plane = vec![0.0f32; in_plane];
        let mut si = vec![0usize; r];
        let mut kk = vec![0usize; r];
        let mut si_flat = 0usize;
        loop {
            let mut sum = 0.0f32;
            for f in 0..filters {
                let g_base = (b * filters + f) * out_plane;
                let w_base = (f * cin + c) * k_plane;
                kk.iter_mut().for_each(|x| *x = 0);
                let mut kk_flat = 0usize;
                loop {
                    // Map padded position (si + pad_before) back to the output position it came from.
                    let mut valid = true;
                    let mut o_flat = 0usize;
                    for d in 0..r {
                        let pd = si[d] + pad_before[d];
                        if pd < kk[d] {
                            valid = false;
                            break;
                        }
                        let diff = pd - kk[d];
                        if !diff.is_multiple_of(strides[d]) {
                            valid = false;
                            break;
                        }
                        let od = diff / strides[d];
                        if od >= out_sp[d] {
                            valid = false;
                            break;
                        }
                        o_flat = o_flat * out_sp[d] + od;
                    }
                    if valid {
                        sum += weights[w_base + kk_flat] * grad_flat[g_base + o_flat];
                    }
                    kk_flat += 1;
                    if !increment_index(&mut kk, k_dims) {
                        break;
                    }
                }
            }
            plane[si_flat] = sum;
            si_flat += 1;
            if !increment_index(&mut si, sp) {
                break;
            }
        }
        plane
    };

    let parallel_ig = batch * cin * in_plane >= CONV_PARALLEL_THRESHOLD;
    let ig_planes = map_indexed(batch * cin, parallel_ig, process_bc);
    let mut in_grad_flat = Vec::with_capacity(batch * cin * in_plane);
    for plane in ig_planes {
        in_grad_flat.extend(plane);
    }
    let mut ig_shape = Vec::with_capacity(2 + r);
    ig_shape.push(batch);
    ig_shape.push(cin);
    ig_shape.extend_from_slice(sp);
    let input_grad =
        ArrayD::from_shape_vec(IxDyn(&ig_shape), in_grad_flat).expect("input grad matches shape");

    ConvGradients {
        weight_grad,
        bias_grad,
        input_grad,
    }
}
