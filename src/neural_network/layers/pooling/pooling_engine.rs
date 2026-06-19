//! Dimension-generic pooling engine shared by every pooling layer
//!
//! All pooling layers (max/average, windowed/global, 1D/2D/3D) reduce to 4 functions here. The
//! spatial rank is derived at run time from `input.ndim() - 2`, so a single implementation serves
//! every dimensionality. The only per-layer difference is the [`PoolKind`] and how the public
//! `pool_size`/`strides` tuples are flattened into slices
//!
//! # Layout
//!
//! Inputs are `[batch, channels, spatial...]`. Work is parallelized over the `batch * channels`
//! "channel planes". Each plane is processed as a contiguous `&[f32]` (row-major over the spatial
//! dimensions) and the results are concatenated in `bc` order into the output buffer. This keeps
//! the hot path allocation-light and builds the output with a single `from_shape_vec` instead of
//! scalar-indexed writes

use crate::neural_network::Tensor;
use crate::neural_network::layers::convolution::PaddingType;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

tunable_gate! {
    /// Estimated total element ops (`bc_total * work_per_plane`) at or above which a pooling pass
    /// runs in parallel
    ///
    /// Counting per-plane work rather than plane count keeps the gate meaningful when a few planes
    /// carry large spatial dims (e.g. batch == 1 on a big image) and when many planes are tiny. The
    /// measured crossover bracket is 4.1K-12.3K window taps on both the few-large-planes and
    /// many-tiny-planes ladders
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) POOL_PARALLEL_MIN_OPS => pool_parallel_min_ops / set_pool_parallel_min_ops = 12_000
}

/// Per-spatial-axis pooling output sizes and leading padding for a given padding mode
///
/// `Valid` drops the trailing remainder and pads nothing. `Same` rounds the output up to
/// `ceil(in / stride)` and splits the padding evenly with the extra cell on the trailing edge
/// (`pad_before = pad_total / 2`), matching the convolution engine. Padding cells are virtual:
/// the forward/backward passes skip out-of-bounds positions, so average pooling divides by the
/// count of real (in-bounds) elements, matching Keras `count_include_pad=False` behavior
fn pool_geometry(
    sp: &[usize],
    pool: &[usize],
    strides: &[usize],
    padding: PaddingType,
) -> (Vec<usize>, Vec<usize>) {
    let r = sp.len();
    match padding {
        PaddingType::Valid => {
            let out_sp = (0..r).map(|k| (sp[k] - pool[k]) / strides[k] + 1).collect();
            (out_sp, vec![0; r])
        }
        PaddingType::Same => {
            let out_sp: Vec<usize> = (0..r).map(|k| sp[k].div_ceil(strides[k])).collect();
            let pad_before = (0..r)
                .map(|k| (((out_sp[k] - 1) * strides[k] + pool[k]).saturating_sub(sp[k])) / 2)
                .collect();
            (out_sp, pad_before)
        }
    }
}

/// The reduction performed over each pooling window
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    /// Take the maximum element (records the arg-max for backprop)
    Max,
    /// Take the mean of the elements in the window
    Average,
}

/// Row-major (C-order) strides for `shape`: the number of flat elements per unit step on each axis
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Advances a multi-index `idx` (row-major, last axis fastest) within bounds `dims`
///
/// # Returns
///
/// - `bool` - `true` while there are more indices, `false` once it wraps back to all-zero
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

/// Decomposes a flat row-major index into a multi-index for `dims` (inverse of the flat
/// counter that `increment_index` advances), so a chunked task can start mid-plane
fn decode_index(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; dims.len()];
    for k in (0..dims.len()).rev() {
        if dims[k] > 0 {
            idx[k] = flat % dims[k];
            flat /= dims[k];
        }
    }
    idx
}

/// Minimum output positions per forward task: small enough that a `batch * channels == 1` plane
/// still splits across threads, large enough (~12 us of window reduction) to amortize the rayon
/// task overhead
const POOL_MIN_CHUNK_OUT: usize = 1024;

/// Runs `f` for every `bc` in `0..bc_total`, preserving order, in parallel once the estimated
/// total work (`bc_total * work_per_plane` element ops) clears the gate
///
/// `force_parallel` overrides the gate decision; production passes `None`
fn map_planes<R, F>(
    bc_total: usize,
    work_per_plane: usize,
    force_parallel: Option<bool>,
    f: F,
) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    let parallel = force_parallel
        .unwrap_or(bc_total.saturating_mul(work_per_plane) >= pool_parallel_min_ops());
    if parallel {
        (0..bc_total).into_par_iter().map(f).collect()
    } else {
        (0..bc_total).map(f).collect()
    }
}

/// Forward pass for windowed pooling (`MaxPooling{1,2,3}D` / `AveragePooling{1,2,3}D`)
///
/// `pool` and `strides` are the per-spatial-axis window sizes and steps (length = spatial rank)
///
/// # Returns
///
/// The pooled tensor and, for [`PoolKind::Max`], the flat per-output arg-max indices into each
/// input plane (used by [`windowed_pool_backward`]); `None` for averaging
pub(super) fn windowed_pool_forward(
    input: &Tensor,
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
    padding: PaddingType,
) -> (Tensor, Option<Vec<usize>>) {
    windowed_pool_forward_impl(input, pool, strides, kind, padding, None)
}

/// `windowed_pool_forward` with an optional override of the parallel/serial gate decision
///
/// `force_parallel` selects the parallel or serial path regardless of the work estimate;
/// production passes `None`. Reachable outside the crate only through `bench_internals`
pub fn windowed_pool_forward_impl(
    input: &Tensor,
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
    padding: PaddingType,
    force_parallel: Option<bool>,
) -> (Tensor, Option<Vec<usize>>) {
    let shape = input.shape();
    let (batch, channels) = (shape[0], shape[1]);
    let sp = &shape[2..];
    let r = sp.len();
    let (out_sp, pad_before) = pool_geometry(sp, pool, strides, padding);
    let plane_in: usize = sp.iter().product();
    let plane_out: usize = out_sp.iter().product();
    let in_strides = row_major_strides(sp);
    let bc_total = batch * channels;
    let track = kind == PoolKind::Max;

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // One task per (plane, output-position chunk): splitting the output positions lets a single
    // large plane (e.g. batch == 1 with few channels) use every thread, while many planes get one
    // chunk apiece. Every output element is reduced by the same serial loop regardless of the
    // chunk boundaries, so the result is bitwise-independent of the thread count
    let process_range = |bc: usize, c0: usize, len: usize| -> (Vec<f32>, Vec<usize>) {
        let plane = &in_flat[bc * plane_in..(bc + 1) * plane_in];
        let mut out_chunk = vec![0.0f32; len];
        let mut arg_chunk = if track { vec![0usize; len] } else { Vec::new() };

        let mut o = decode_index(c0, &out_sp);
        let mut w = vec![0usize; r];
        for i in 0..len {
            // Reduce the window anchored at output position `o`
            w.iter_mut().for_each(|x| *x = 0);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;
            loop {
                let mut in_idx = 0usize;
                let mut in_bounds = true;
                for k in 0..r {
                    // Window position in padded coordinates, shifted back into real input space
                    let p = (o[k] * strides[k] + w[k]) as isize - pad_before[k] as isize;
                    if p < 0 || p as usize >= sp[k] {
                        in_bounds = false;
                        break;
                    }
                    in_idx += p as usize * in_strides[k];
                }
                if in_bounds {
                    let v = plane[in_idx];
                    match kind {
                        PoolKind::Max => {
                            // Propagate NaN: once seen it wins and sticks (matches PyTorch/TF and
                            // the sibling activations); a bare `v > max_val` would silently drop it
                            // since NaN is never `>` anything
                            if v.is_nan() {
                                if !max_val.is_nan() {
                                    max_val = v;
                                    max_idx = in_idx;
                                }
                            } else if v > max_val {
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
                    out_chunk[i] = max_val;
                    arg_chunk[i] = max_idx;
                }
                PoolKind::Average => {
                    out_chunk[i] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
            increment_index(&mut o, &out_sp);
        }
        (out_chunk, arg_chunk)
    };

    let total_ops = bc_total
        .saturating_mul(plane_out)
        .saturating_mul(pool.iter().product::<usize>());
    let parallel = force_parallel.unwrap_or(total_ops >= pool_parallel_min_ops());

    // Chunk size: enough chunks to feed every thread once the planes alone cannot, but never so
    // small the task overhead dominates
    let chunk_len = if parallel && bc_total > 0 && plane_out > 0 {
        let chunks_per_plane = rayon::current_num_threads().div_ceil(bc_total);
        plane_out.div_ceil(chunks_per_plane).max(POOL_MIN_CHUNK_OUT)
    } else {
        plane_out.max(1)
    };
    let tasks: Vec<(usize, usize, usize)> = (0..bc_total)
        .flat_map(|bc| {
            (0..plane_out)
                .step_by(chunk_len.max(1))
                .map(move |c0| (bc, c0, chunk_len.min(plane_out - c0)))
        })
        .collect();

    let results: Vec<(Vec<f32>, Vec<usize>)> = if parallel {
        tasks
            .par_iter()
            .map(|&(bc, c0, len)| process_range(bc, c0, len))
            .collect()
    } else {
        tasks
            .iter()
            .map(|&(bc, c0, len)| process_range(bc, c0, len))
            .collect()
    };

    let mut out_flat = vec![0.0f32; bc_total * plane_out];
    let mut argmax = if track {
        vec![0usize; bc_total * plane_out]
    } else {
        Vec::new()
    };
    for (&(bc, c0, len), (out_chunk, arg_chunk)) in tasks.iter().zip(results) {
        let base = bc * plane_out + c0;
        out_flat[base..base + len].copy_from_slice(&out_chunk);
        if track {
            argmax[base..base + len].copy_from_slice(&arg_chunk);
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

/// Backward pass for windowed pooling
///
/// `input_shape` is the full forward input shape `[batch, channels, spatial...]`. For
/// [`PoolKind::Max`], `argmax` must be the indices returned by [`windowed_pool_forward`];
/// averaging redistributes each output gradient evenly over its window and ignores `argmax`
pub(super) fn windowed_pool_backward(
    grad_output: &Tensor,
    input_shape: &[usize],
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
    argmax: Option<&[usize]>,
    padding: PaddingType,
) -> Tensor {
    let (batch, channels) = (input_shape[0], input_shape[1]);
    let sp = &input_shape[2..];
    let r = sp.len();
    let out_sp = &grad_output.shape()[2..];
    // Leading padding per axis (Max backward uses the recorded arg-max, so only Average needs it)
    let (_, pad_before) = pool_geometry(sp, pool, strides, padding);
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
                    // Count the in-bounds elements of this window, then spread the gradient evenly
                    w.iter_mut().for_each(|x| *x = 0);
                    let mut count = 0usize;
                    loop {
                        if (0..r).all(|k| {
                            let p = (o[k] * strides[k] + w[k]) as isize - pad_before[k] as isize;
                            p >= 0 && (p as usize) < sp[k]
                        }) {
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
                                let p =
                                    (o[k] * strides[k] + w[k]) as isize - pad_before[k] as isize;
                                if p < 0 || p as usize >= sp[k] {
                                    in_bounds = false;
                                    break;
                                }
                                in_idx += p as usize * in_strides[k];
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

    let planes = map_planes(
        bc_total,
        plane_out * pool.iter().product::<usize>(),
        None,
        process_bc,
    );

    let mut grad_in = Vec::with_capacity(bc_total * plane_in);
    for plane in planes {
        grad_in.extend(plane);
    }
    ArrayD::from_shape_vec(IxDyn(input_shape), grad_in)
        .expect("grad-input length matches the input shape")
}

/// Forward pass for global pooling (`GlobalMaxPooling{1,2,3}D` / `GlobalAveragePooling{1,2,3}D`)
///
/// Reduces every spatial dimension to one value per channel, producing a `[batch, channels]`
/// tensor
///
/// # Returns
///
/// The pooled tensor and, for [`PoolKind::Max`], the flat per-channel arg-max index into each
/// input plane; `None` for averaging
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
                    // Propagate NaN (see `windowed_pool_forward`): NaN wins and sticks
                    if v.is_nan() {
                        if !max_val.is_nan() {
                            max_val = v;
                            max_idx = i;
                        }
                    } else if v > max_val {
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

    let results = map_planes(bc_total, plane_in, None, process_bc);

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

/// Backward pass for global pooling
///
/// `grad_output` has shape `[batch, channels]`. Averaging spreads each gradient evenly over its
/// channel plane. [`PoolKind::Max`] routes it to the stored arg-max element
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

    let planes = map_planes(bc_total, plane_in, None, process_bc);

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

    // row_major_strides

    /// Row-major strides of a 3-D shape
    #[test]
    fn test_row_major_strides_3d() {
        let s = row_major_strides(&[3, 4, 5]);
        assert_eq!(s, vec![20, 5, 1]);
    }

    /// Row-major strides of a 2-D shape
    #[test]
    fn test_row_major_strides_2d() {
        let s = row_major_strides(&[2, 3]);
        assert_eq!(s, vec![3, 1]);
    }

    /// A 1-D shape has a single stride of 1
    #[test]
    fn test_row_major_strides_1d() {
        let s = row_major_strides(&[7]);
        assert_eq!(s, vec![1]);
    }

    /// An empty shape returns an empty stride vector
    #[test]
    fn test_row_major_strides_empty() {
        let s = row_major_strides(&[]);
        assert_eq!(s, Vec::<usize>::new());
    }

    // increment_index

    /// Incrementing a non-final index advances the last axis and returns true
    #[test]
    fn test_increment_index_normal() {
        let mut idx = vec![0usize, 0];
        let more = increment_index(&mut idx, &[2, 3]);
        assert!(more);
        assert_eq!(idx, vec![0, 1]);
    }

    /// When the last index reaches its bound it wraps to 0 and carries into the next axis
    #[test]
    fn test_increment_index_carry() {
        let mut idx = vec![0usize, 2];
        let more = increment_index(&mut idx, &[2, 3]);
        assert!(more);
        assert_eq!(idx, vec![1, 0]);
    }

    /// At the final position every axis wraps back to zero and false is returned
    #[test]
    fn test_increment_index_exhausted() {
        let mut idx = vec![1usize, 2];
        let more = increment_index(&mut idx, &[2, 3]);
        assert!(!more);
        assert_eq!(idx, vec![0, 0]);
    }

    /// A 1-D index at its last value wraps to zero and returns false
    #[test]
    fn test_increment_index_1d_last_step() {
        let mut idx = vec![3usize];
        let more = increment_index(&mut idx, &[4]);
        assert!(!more);
        assert_eq!(idx, vec![0]);
    }

    /// A mid-range 1-D index advances and returns true
    #[test]
    fn test_increment_index_1d_mid() {
        let mut idx = vec![2usize];
        let more = increment_index(&mut idx, &[4]);
        assert!(more);
        assert_eq!(idx, vec![3]);
    }

    // windowed_pool_forward (Max, 1-D)

    /// 1-D max-pool returns per-window maxima and their flat arg-max indices
    #[test]
    fn test_windowed_pool_forward_1d_max() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![3.0f32, 1.0, 4.0, 1.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2], &[2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 1, 2]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 4.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(am, vec![0, 2]);
    }

    // windowed_pool_forward (Max, 2-D)

    /// 2-D max-pool over a single window returns the max and its flat arg-max index
    #[test]
    fn test_windowed_pool_forward_2d_max() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 4.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(am, vec![3]);
    }

    // windowed_pool_forward (Average, 1-D)

    /// 1-D average-pool returns per-window means and no arg-max
    #[test]
    fn test_windowed_pool_forward_1d_avg() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![3.0f32, 1.0, 4.0, 1.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2], &[2], PoolKind::Average, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 1, 2]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 2.5, epsilon = 1e-6);
        assert!(argmax.is_none(), "Average pool must not return argmax");
    }

    /// 2-D average-pool over a single window returns the mean and no arg-max
    #[test]
    fn test_windowed_pool_forward_2d_avg() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (out, argmax) = windowed_pool_forward(
            &data,
            &[2, 2],
            &[2, 2],
            PoolKind::Average,
            PaddingType::Valid,
        );
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 2.5, epsilon = 1e-6);
        assert!(argmax.is_none(), "Average pool must not return argmax");
    }

    // windowed_pool_backward (Max, non-overlapping)

    /// Max-pool backward routes each upstream gradient to its arg-max position
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
            PaddingType::Valid,
        );
        assert_eq!(grad_in.shape(), &[1, 1, 4]);
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 1.0, epsilon = 1e-6); // argmax of window 0
        assert_abs_diff_eq!(flat[1], 0.0, epsilon = 1e-6); // not selected
        assert_abs_diff_eq!(flat[2], 1.0, epsilon = 1e-6); // argmax of window 1
        assert_abs_diff_eq!(flat[3], 0.0, epsilon = 1e-6); // not selected
    }

    /// Distinct upstream gradient values reach their respective arg-max positions
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
            PaddingType::Valid,
        );
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[3], 5.0, epsilon = 1e-6);
    }

    // windowed_pool_backward (Average, overlapping)

    /// Average-pool backward spreads each gradient over its window and accumulates overlaps
    #[test]
    fn test_windowed_pool_backward_1d_avg_overlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0f32, 1.0, 1.0]).unwrap();
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 1, 4],
            &[2],
            &[1],
            PoolKind::Average,
            None,
            PaddingType::Valid,
        );
        assert_eq!(grad_in.shape(), &[1, 1, 4]);
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 0.5, epsilon = 1e-6); // edge
        assert_abs_diff_eq!(flat[1], 1.0, epsilon = 1e-6); // interior
        assert_abs_diff_eq!(flat[2], 1.0, epsilon = 1e-6); // interior
        assert_abs_diff_eq!(flat[3], 0.5, epsilon = 1e-6); // edge
    }

    /// Non-overlapping average-pool backward spreads each gradient evenly with no overlap
    #[test]
    fn test_windowed_pool_backward_1d_avg_nonoverlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, 1.0]).unwrap();
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 1, 4],
            &[2],
            &[2],
            PoolKind::Average,
            None,
            PaddingType::Valid,
        );
        assert_eq!(grad_in.shape(), &[1, 1, 4]);
        let flat: Vec<f32> = grad_in.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[2], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[3], 0.5, epsilon = 1e-6);
    }

    // output shape checks

    /// 1-D output shape follows (sp - pool) / stride + 1
    #[test]
    fn test_windowed_pool_output_shape_1d() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 6]), vec![1.0f32; 6]).unwrap();
        let (out, _) =
            windowed_pool_forward(&data, &[3], &[2], PoolKind::Average, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 1, 2]);
    }

    /// 2-D batched output shape follows (sp - pool) / stride + 1 per axis
    #[test]
    fn test_windowed_pool_output_shape_2d_batched() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 4, 4]), vec![1.0f32; 2 * 3 * 4 * 4]).unwrap();
        let (out, _) =
            windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(out.shape(), &[2, 3, 2, 2]);
    }

    // Max forward tie-breaking (first occurrence wins)

    /// Windowed max-pool tie-break records the first of equal maxima (strict `>`)
    #[test]
    fn test_windowed_pool_forward_2d_max_tie_breaks_to_first() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![5.0f32, 5.0, 1.0, 2.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max, PaddingType::Valid);
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

    /// Global max-pool tie-break records the first of equal maxima (strict `>`)
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
