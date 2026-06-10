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
        let mut arg_plane = if track {
            vec![0usize; plane_out]
        } else {
            Vec::new()
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── row_major_strides ──────────────────────────────────────────────────────

    /// For shape [3, 4, 5] the row-major strides are [4*5, 5, 1] = [20, 5, 1].
    #[test]
    fn test_row_major_strides_3d() {
        let s = row_major_strides(&[3, 4, 5]);
        assert_eq!(s, vec![20, 5, 1]);
    }

    /// For shape [2, 3] the strides are [3, 1].
    #[test]
    fn test_row_major_strides_2d() {
        let s = row_major_strides(&[2, 3]);
        assert_eq!(s, vec![3, 1]);
    }

    /// A 1-D shape has a single stride of 1.
    #[test]
    fn test_row_major_strides_1d() {
        let s = row_major_strides(&[7]);
        assert_eq!(s, vec![1]);
    }

    /// An empty shape returns an empty stride vector.
    #[test]
    fn test_row_major_strides_empty() {
        let s = row_major_strides(&[]);
        assert_eq!(s, Vec::<usize>::new());
    }

    // ── increment_index ────────────────────────────────────────────────────────

    /// Incrementing [0, 0] in dims [2, 3] gives [0, 1] and returns true.
    #[test]
    fn test_increment_index_normal() {
        let mut idx = vec![0usize, 0];
        let more = increment_index(&mut idx, &[2, 3]);
        assert!(more);
        assert_eq!(idx, vec![0, 1]);
    }

    /// When the last index reaches its bound it wraps and carries: [0, 2] → [1, 0].
    #[test]
    fn test_increment_index_carry() {
        let mut idx = vec![0usize, 2];
        let more = increment_index(&mut idx, &[2, 3]);
        assert!(more);
        assert_eq!(idx, vec![1, 0]);
    }

    /// At the very last position [1, 2] everything wraps back to [0, 0] and false is returned.
    #[test]
    fn test_increment_index_exhausted() {
        let mut idx = vec![1usize, 2];
        let more = increment_index(&mut idx, &[2, 3]);
        assert!(!more);
        assert_eq!(idx, vec![0, 0]);
    }

    /// Single-dimension: [3] in dims [4] advances to [4] but that wraps to [0] → false is only
    /// returned when the dimension is fully exhausted. Here [3] → increments to 4 which equals
    /// dim 4, so it wraps to [0] and returns false.
    #[test]
    fn test_increment_index_1d_last_step() {
        let mut idx = vec![3usize];
        let more = increment_index(&mut idx, &[4]);
        assert!(!more);
        assert_eq!(idx, vec![0]);
    }

    /// Mid-range 1D step: [2] in [4] → [3], returns true.
    #[test]
    fn test_increment_index_1d_mid() {
        let mut idx = vec![2usize];
        let more = increment_index(&mut idx, &[4]);
        assert!(more);
        assert_eq!(idx, vec![3]);
    }

    // ── windowed_pool_forward (Max, 1-D) ──────────────────────────────────────

    /// 1-D Max-pool: input [1,1,4] = [3,1,4,1], pool=[2], stride=[2].
    /// Window 0: positions 0,1 → values 3,1 → max=3 at flat-idx 0.
    /// Window 1: positions 2,3 → values 4,1 → max=4 at flat-idx 2.
    /// Expected output values: [3, 4]; argmax: Some([0, 2]).
    #[test]
    fn test_windowed_pool_forward_1d_max() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![3.0f32, 1.0, 4.0, 1.0]).unwrap();
        let (out, argmax) = windowed_pool_forward(&data, &[2], &[2], PoolKind::Max);
        // Output shape must be [1,1,2]
        assert_eq!(out.shape(), &[1, 1, 2]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 4.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(am, vec![0, 2]);
    }

    // ── windowed_pool_forward (Max, 2-D) ──────────────────────────────────────

    /// 2-D Max-pool: input [1,1,2,2] = [[1,2],[3,4]], pool=[2,2], stride=[2,2].
    /// Single window covers all 4 positions (row-major indices 0..3).
    /// max=4 at flat-idx 3 (row 1, col 1: 1*2+1=3).
    /// Expected output: [[4]], argmax: Some([3]).
    #[test]
    fn test_windowed_pool_forward_2d_max() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (out, argmax) = windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 4.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(am, vec![3]);
    }

    // ── windowed_pool_forward (Average, 1-D) ──────────────────────────────────

    /// 1-D Avg-pool: input [1,1,4] = [3,1,4,1], pool=[2], stride=[2].
    /// Window 0: mean(3,1) = 2.0.
    /// Window 1: mean(4,1) = 2.5.
    /// argmax must be None.
    #[test]
    fn test_windowed_pool_forward_1d_avg() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![3.0f32, 1.0, 4.0, 1.0]).unwrap();
        let (out, argmax) = windowed_pool_forward(&data, &[2], &[2], PoolKind::Average);
        assert_eq!(out.shape(), &[1, 1, 2]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 2.5, epsilon = 1e-6);
        assert!(argmax.is_none(), "Average pool must not return argmax");
    }

    /// 2-D Avg-pool: input [1,1,2,2] = [[1,2],[3,4]], pool=[2,2], stride=[2,2].
    /// Single window: mean(1+2+3+4) = 10/4 = 2.5. argmax must be None.
    #[test]
    fn test_windowed_pool_forward_2d_avg() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (out, argmax) = windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Average);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 2.5, epsilon = 1e-6);
        assert!(argmax.is_none(), "Average pool must not return argmax");
    }

    // ── windowed_pool_backward (Max, non-overlapping) ──────────────────────────

    /// Max-pool backward, non-overlapping (stride == pool).
    /// Input shape [1,1,4], argmax=[0,2], grad_output=[1.0, 1.0].
    /// Each upstream grad routes to exactly its argmax; all others stay 0.
    /// Expected grad_input: [1.0, 0.0, 1.0, 0.0].
    #[test]
    fn test_windowed_pool_backward_1d_max_nonoverlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, 1.0]).unwrap();
        let argmax = vec![0usize, 2];
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 1, 4],
            &[2],
            &[2],
            PoolKind::Max,
            Some(&argmax),
        );
        assert_eq!(grad_in.shape(), &[1, 1, 4]);
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 1.0, epsilon = 1e-6); // argmax of window 0
        assert_abs_diff_eq!(flat[1], 0.0, epsilon = 1e-6); // not selected
        assert_abs_diff_eq!(flat[2], 1.0, epsilon = 1e-6); // argmax of window 1
        assert_abs_diff_eq!(flat[3], 0.0, epsilon = 1e-6); // not selected
    }

    /// Larger upstream grad values are correctly routed to argmax positions.
    /// Input shape [1,1,4], argmax=[1,3], grad_output=[2.0, 5.0].
    /// Expected grad_input: [0.0, 2.0, 0.0, 5.0].
    #[test]
    fn test_windowed_pool_backward_1d_max_varied_grads() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![2.0f32, 5.0]).unwrap();
        let argmax = vec![1usize, 3];
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 1, 4],
            &[2],
            &[2],
            PoolKind::Max,
            Some(&argmax),
        );
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[3], 5.0, epsilon = 1e-6);
    }

    // ── windowed_pool_backward (Average, overlapping) ─────────────────────────

    /// Avg-pool backward, overlapping stride (pool=[2], stride=[1], length=4).
    ///
    /// out_sp = (4-2)/1+1 = 3.  grad_output = [1.0, 1.0, 1.0] (all ones).
    /// Each window spreads its gradient (1.0) evenly over 2 positions (1.0/2 = 0.5 each).
    ///   window o=0 covers input positions 0,1  → each receives +0.5
    ///   window o=1 covers input positions 1,2  → each receives +0.5
    ///   window o=2 covers input positions 2,3  → each receives +0.5
    /// Accumulated: pos0=0.5, pos1=1.0, pos2=1.0, pos3=0.5.
    #[test]
    fn test_windowed_pool_backward_1d_avg_overlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0f32, 1.0, 1.0]).unwrap();
        let grad_in =
            windowed_pool_backward(&grad_out, &[1, 1, 4], &[2], &[1], PoolKind::Average, None);
        assert_eq!(grad_in.shape(), &[1, 1, 4]);
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 0.5, epsilon = 1e-6); // edge
        assert_abs_diff_eq!(flat[1], 1.0, epsilon = 1e-6); // interior
        assert_abs_diff_eq!(flat[2], 1.0, epsilon = 1e-6); // interior
        assert_abs_diff_eq!(flat[3], 0.5, epsilon = 1e-6); // edge
    }

    /// Avg-pool backward, non-overlapping (pool=[2], stride=[2], length=4).
    /// grad_output = [1.0, 1.0].  Each output spreads 0.5 to both positions in its window.
    ///   window 0 → positions 0,1 get 0.5 each
    ///   window 1 → positions 2,3 get 0.5 each
    /// No overlap, so final: [0.5, 0.5, 0.5, 0.5].
    #[test]
    fn test_windowed_pool_backward_1d_avg_nonoverlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, 1.0]).unwrap();
        let grad_in =
            windowed_pool_backward(&grad_out, &[1, 1, 4], &[2], &[2], PoolKind::Average, None);
        assert_eq!(grad_in.shape(), &[1, 1, 4]);
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[2], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[3], 0.5, epsilon = 1e-6);
    }

    // ── output shape checks ───────────────────────────────────────────────────

    /// out_sp formula: (sp - pool) / stride + 1.
    /// Input [1,1,6], pool=[3], stride=[2] → out = (6-3)/2+1 = 2. Shape must be [1,1,2].
    #[test]
    fn test_windowed_pool_output_shape_1d() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 6]), vec![1.0f32; 6]).unwrap();
        let (out, _) = windowed_pool_forward(&data, &[3], &[2], PoolKind::Average);
        assert_eq!(out.shape(), &[1, 1, 2]);
    }

    /// 2-D shape: input [2,3,4,4], pool=[2,2], stride=[2,2] → out = (4-2)/2+1=2 per axis.
    /// Shape must be [2,3,2,2].
    #[test]
    fn test_windowed_pool_output_shape_2d_batched() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 4, 4]), vec![1.0f32; 2 * 3 * 4 * 4]).unwrap();
        let (out, _) = windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max);
        assert_eq!(out.shape(), &[2, 3, 2, 2]);
    }
    // ── Max forward tie-breaking (first occurrence wins) ──────────────────────

    /// Windowed Max-pool tie-break: a window containing two equal maxima must record the
    /// FIRST occurrence as the arg-max. The reduction uses the strict `if v > max_val`, so an
    /// equal value never overwrites the running maximum.
    ///
    /// Input [1,1,2,2] row-major = [5, 5, 1, 2]:
    ///   (0,0)=5 → flat 0, (0,1)=5 → flat 1, (1,0)=1 → flat 2, (1,1)=2 → flat 3.
    /// One window (pool=[2,2], stride=[2,2]) scans the offsets w in order
    ///   (0,0),(0,1),(1,0),(1,1) → in_idx = w0*2 + w1*1 = 0,1,2,3 (last axis fastest).
    /// flat 0 sets max=5 @ argmax 0; flat 1 (=5) is NOT `> 5`, so argmax stays 0.
    /// Expected: max value 5.0, argmax = Some([0]) (the FIRST of the two maxima).
    /// A change to `>=`/last-occurrence would record argmax 1 and fail.
    #[test]
    fn test_windowed_pool_forward_2d_max_tie_breaks_to_first() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![5.0f32, 5.0, 1.0, 2.0]).unwrap();
        let (out, argmax) = windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 5.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(
            am,
            vec![0],
            "tie must resolve to the FIRST maximum (flat index 0)"
        );
    }

    /// Global Max-pool tie-break: scanning the whole plane with the strict `if v > max_val`,
    /// duplicate maxima resolve to the FIRST index.
    ///
    /// Input [1,1,4] = [5, 5, 1, 2]; the plane is scanned by `enumerate()` (i = 0,1,2,3).
    /// i=0 sets max=5 @ argmax 0; i=1 (=5) is NOT `> 5`, so argmax stays 0.
    /// Expected: max value 5.0, argmax = Some([0]). A `>=`/last rule would give argmax 1 and fail.
    #[test]
    fn test_global_pool_forward_max_tie_breaks_to_first() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![5.0f32, 5.0, 1.0, 2.0]).unwrap();
        let (out, argmax) = global_pool_forward(&data, PoolKind::Max);
        assert_eq!(out.shape(), &[1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 5.0, epsilon = 1e-6);
        let am = argmax.expect("Global max pool must return argmax");
        assert_eq!(
            am,
            vec![0],
            "tie must resolve to the FIRST maximum (index 0)"
        );
    }
}
