//! Dimension-generic pooling engine shared by every pooling layer.
//!
//! All pooling layers (max/average, windowed/global, 1D/2D/3D) reduce to four functions here. The
//! spatial rank is derived at run time from `input.ndim() - 2`, so a single implementation serves
//! every dimensionality; the only per-layer difference is the [`PoolKind`] and how the public
//! `pool_size`/`strides` tuples are flattened into slices.
//!
//! # Layout
//!
//! Inputs are `[batch, channels, spatial...]`. Work is parallelized over the `batch * channels`
//! "channel planes"; each plane is processed as a contiguous `&[f32]` (row-major over the spatial
//! dimensions) and the results are concatenated in `bc` order into the output buffer. This keeps
//! the hot path allocation-light and lets the output be built with a single `from_shape_vec`
//! instead of scalar-indexed writes.

use crate::neural_network::Tensor;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

/// Threshold (in `batch * channels` units) at or above which a pooling pass runs in parallel.
pub(super) const POOL_PARALLEL_THRESHOLD: usize = 32;

/// The reduction performed over each pooling window.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PoolKind {
    /// Take the maximum element (records the arg-max for backprop).
    Max,
    /// Take the mean of the elements in the window.
    Average,
}

/// Row-major (C-order) strides for `shape`: the number of flat elements per unit step on each axis.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Advances a multi-index `idx` (row-major, last axis fastest) within bounds `dims`.
///
/// Returns `true` while there are more indices, `false` once it wraps back to all-zero (done).
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

/// Runs `f` for every `bc` in `0..bc_total`, in parallel above the threshold, preserving order.
fn map_planes<R, F>(bc_total: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if bc_total >= POOL_PARALLEL_THRESHOLD {
        (0..bc_total).into_par_iter().map(f).collect()
    } else {
        (0..bc_total).map(f).collect()
    }
}

/// Forward pass for windowed pooling (`MaxPooling{1,2,3}D` / `AveragePooling{1,2,3}D`).
///
/// `pool` and `strides` are the per-spatial-axis window sizes and steps (length = spatial rank).
/// Returns the pooled tensor and, for [`PoolKind::Max`], the flat per-output arg-max indices into
/// each input plane (used by [`windowed_pool_backward`]); `None` for averaging.
pub(super) fn windowed_pool_forward(
    input: &Tensor,
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
) -> (Tensor, Option<Vec<usize>>) {
    let shape = input.shape();
    let (batch, channels) = (shape[0], shape[1]);
    let sp = &shape[2..];
    let r = sp.len();
    let out_sp: Vec<usize> = (0..r).map(|k| (sp[k] - pool[k]) / strides[k] + 1).collect();
    let plane_in: usize = sp.iter().product();
    let plane_out: usize = out_sp.iter().product();
    let in_strides = row_major_strides(sp);
    let bc_total = batch * channels;
    let track = kind == PoolKind::Max;

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    let process_bc = |bc: usize| -> (Vec<f32>, Vec<usize>) {
        let plane = &in_flat[bc * plane_in..(bc + 1) * plane_in];
        let mut out_plane = vec![0.0f32; plane_out];
        let mut arg_plane = if track { vec![0usize; plane_out] } else { Vec::new() };

        let mut o = vec![0usize; r];
        let mut w = vec![0usize; r];
        let mut o_flat = 0usize;
        loop {
            // Reduce the window anchored at output position `o`.
            w.iter_mut().for_each(|x| *x = 0);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;
            loop {
                let mut in_idx = 0usize;
                let mut in_bounds = true;
                for k in 0..r {
                    let p = o[k] * strides[k] + w[k];
                    if p >= sp[k] {
                        in_bounds = false;
                        break;
                    }
                    in_idx += p * in_strides[k];
                }
                if in_bounds {
                    let v = plane[in_idx];
                    match kind {
                        PoolKind::Max => {
                            if v > max_val {
                                max_val = v;
                                max_idx = in_idx;
                            }
                        }
                        PoolKind::Average => {
                            sum += v;
                            count += 1;
                        }
                    }
                }
                if !increment_index(&mut w, pool) {
                    break;
                }
            }
            match kind {
                PoolKind::Max => {
                    out_plane[o_flat] = max_val;
                    arg_plane[o_flat] = max_idx;
                }
                PoolKind::Average => {
                    out_plane[o_flat] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
            o_flat += 1;
            if !increment_index(&mut o, &out_sp) {
                break;
            }
        }
        (out_plane, arg_plane)
    };

    let planes = map_planes(bc_total, process_bc);

    let mut out_flat = Vec::with_capacity(bc_total * plane_out);
    let mut argmax = if track {
        Vec::with_capacity(bc_total * plane_out)
    } else {
        Vec::new()
    };
    for (out_plane, arg_plane) in planes {
        out_flat.extend(out_plane);
        if track {
            argmax.extend(arg_plane);
        }
    }

    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.push(channels);
    out_shape.extend_from_slice(&out_sp);
    let output = ArrayD::from_shape_vec(IxDyn(&out_shape), out_flat)
        .expect("pool output length matches its shape");

    (output, if track { Some(argmax) } else { None })
}

/// Backward pass for windowed pooling.
///
/// `input_shape` is the full forward input shape `[batch, channels, spatial...]`. For
/// [`PoolKind::Max`], `argmax` must be the indices returned by [`windowed_pool_forward`];
/// averaging redistributes each output gradient evenly over its window and ignores `argmax`.
pub(super) fn windowed_pool_backward(
    grad_output: &Tensor,
    input_shape: &[usize],
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
    argmax: Option<&[usize]>,
) -> Tensor {
    let (batch, channels) = (input_shape[0], input_shape[1]);
    let sp = &input_shape[2..];
    let r = sp.len();
    let out_sp = &grad_output.shape()[2..];
    let plane_in: usize = sp.iter().product();
    let plane_out: usize = out_sp.iter().product();
    let in_strides = row_major_strides(sp);
    let bc_total = batch * channels;

    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    let process_bc = |bc: usize| -> Vec<f32> {
        let grad_out_plane = &grad_flat[bc * plane_out..(bc + 1) * plane_out];
        let mut grad_in_plane = vec![0.0f32; plane_in];

        match kind {
            PoolKind::Max => {
                let arg = argmax.expect("max pooling backward requires arg-max positions");
                let arg_plane = &arg[bc * plane_out..(bc + 1) * plane_out];
                for (o_flat, &g) in grad_out_plane.iter().enumerate() {
                    grad_in_plane[arg_plane[o_flat]] += g;
                }
            }
            PoolKind::Average => {
                let mut o = vec![0usize; r];
                let mut w = vec![0usize; r];
                let mut o_flat = 0usize;
                loop {
                    // Count the in-bounds elements of this window, then spread the gradient evenly.
                    w.iter_mut().for_each(|x| *x = 0);
                    let mut count = 0usize;
                    loop {
                        if (0..r).all(|k| o[k] * strides[k] + w[k] < sp[k]) {
                            count += 1;
                        }
                        if !increment_index(&mut w, pool) {
                            break;
                        }
                    }
                    if count > 0 {
                        let grad_per_element = grad_out_plane[o_flat] / count as f32;
                        w.iter_mut().for_each(|x| *x = 0);
                        loop {
                            let mut in_idx = 0usize;
                            let mut in_bounds = true;
                            for k in 0..r {
                                let p = o[k] * strides[k] + w[k];
                                if p >= sp[k] {
                                    in_bounds = false;
                                    break;
                                }
                                in_idx += p * in_strides[k];
                            }
                            if in_bounds {
                                grad_in_plane[in_idx] += grad_per_element;
                            }
                            if !increment_index(&mut w, pool) {
                                break;
                            }
                        }
                    }
                    o_flat += 1;
                    if !increment_index(&mut o, out_sp) {
                        break;
                    }
                }
            }
        }
        grad_in_plane
    };

    let planes = map_planes(bc_total, process_bc);

    let mut grad_in = Vec::with_capacity(bc_total * plane_in);
    for plane in planes {
        grad_in.extend(plane);
    }
    ArrayD::from_shape_vec(IxDyn(input_shape), grad_in)
        .expect("grad-input length matches the input shape")
}

/// Forward pass for global pooling (`GlobalMaxPooling{1,2,3}D` / `GlobalAveragePooling{1,2,3}D`).
///
/// Reduces every spatial dimension to one value per channel, producing a `[batch, channels]`
/// tensor. For [`PoolKind::Max`], returns the flat per-channel arg-max index into each input plane.
pub(super) fn global_pool_forward(input: &Tensor, kind: PoolKind) -> (Tensor, Option<Vec<usize>>) {
    let shape = input.shape();
    let (batch, channels) = (shape[0], shape[1]);
    let plane_in: usize = shape[2..].iter().product();
    let bc_total = batch * channels;
    let track = kind == PoolKind::Max;

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    let process_bc = |bc: usize| -> (f32, usize) {
        let plane = &in_flat[bc * plane_in..(bc + 1) * plane_in];
        match kind {
            PoolKind::Max => {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0usize;
                for (i, &v) in plane.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }
                (max_val, max_idx)
            }
            PoolKind::Average => {
                let sum: f32 = plane.iter().sum();
                (sum / plane_in as f32, 0)
            }
        }
    };

    let results = map_planes(bc_total, process_bc);

    let mut out_flat = Vec::with_capacity(bc_total);
    let mut argmax = if track {
        Vec::with_capacity(bc_total)
    } else {
        Vec::new()
    };
    for (val, idx) in results {
        out_flat.push(val);
        if track {
            argmax.push(idx);
        }
    }

    let output = ArrayD::from_shape_vec(IxDyn(&[batch, channels]), out_flat)
        .expect("global-pool output length matches [batch, channels]");
    (output, if track { Some(argmax) } else { None })
}

/// Backward pass for global pooling.
///
/// `grad_output` has shape `[batch, channels]`. Averaging spreads each gradient evenly over its
/// channel plane; [`PoolKind::Max`] routes it to the stored arg-max element.
pub(super) fn global_pool_backward(
    grad_output: &Tensor,
    input_shape: &[usize],
    kind: PoolKind,
    argmax: Option<&[usize]>,
) -> Tensor {
    let (batch, channels) = (input_shape[0], input_shape[1]);
    let plane_in: usize = input_shape[2..].iter().product();
    let bc_total = batch * channels;

    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    let process_bc = |bc: usize| -> Vec<f32> {
        let g = grad_flat[bc];
        let mut plane = vec![0.0f32; plane_in];
        match kind {
            PoolKind::Max => {
                let arg = argmax.expect("global max pooling backward requires arg-max positions");
                plane[arg[bc]] += g;
            }
            PoolKind::Average => {
                let grad_per_element = g / plane_in as f32;
                plane.iter_mut().for_each(|x| *x = grad_per_element);
            }
        }
        plane
    };

    let planes = map_planes(bc_total, process_bc);

    let mut grad_in = Vec::with_capacity(bc_total * plane_in);
    for plane in planes {
        grad_in.extend(plane);
    }
    ArrayD::from_shape_vec(IxDyn(input_shape), grad_in)
        .expect("grad-input length matches the input shape")
}
