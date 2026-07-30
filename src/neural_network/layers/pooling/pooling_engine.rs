//! Dimension-generic pooling engine shared by every pooling layer
//!
//! All pooling layers (max or average, windowed or global, at rank 1, 2, or 3) reduce to 4
//! functions here. The spatial rank comes from `input.ndim() - 2` at run time, so a single
//! implementation serves every dimensionality. The only per-layer difference is the [`PoolKind`]
//! and how the public `pool_size` and `strides` tuples flatten into slices
//!
//! # Layout
//!
//! Inputs are channels-last: `[batch, spatial..., channels]`. The unit of work is a single output
//! position. A window reduction reads `channels` contiguous floats per tap and folds them into a
//! `channels`-wide accumulator. This reduces every channel of a position together
//!
//! This layout pays off because of the window geometry: the bounds checks, the index arithmetic,
//! and the in-bounds element count. The engine computes this geometry once per output position
//! and reuses it for every channel, instead of once per `(channel, position)` pair
//!
//! Forward work splits over `(batch item, output-position block)`, each with a disjoint output
//! slab
//!
//! Backward work splits over `(batch item, channel slab)` instead. A slab that owns channels
//! `[j0, j1)` writes only input addresses congruent to that range modulo `channels`. The scatter
//! is therefore conflict-free without a halo, a merge, or atomics. This split still fills the
//! machine when `batch == 1`, unlike a split on batch alone

use crate::neural_network::Tensor;
use crate::neural_network::layers::convolution::PaddingType;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

tunable_gate! {
    /// Total element ops (`batch * out_positions * channels * window` taps) at or above which a
    /// pooling pass runs in parallel
    ///
    /// The gate counts element ops rather than tasks, so it stays meaningful when a few positions
    /// carry many channels or when many positions carry few
    ///
    /// Overridable through [`crate::tuning`]
    pub(crate) POOL_PARALLEL_MIN_OPS => pool_parallel_min_ops / set_pool_parallel_min_ops = 12_000
}

/// Per-spatial-axis pooling output sizes and leading padding for a given padding mode
///
/// `Valid` drops the trailing remainder and pads nothing. `Same` rounds the output up to
/// `ceil(in / stride)`. It splits the padding evenly, with the extra cell on the trailing edge
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
/// counter that `increment_index` advances), so a chunked task can start mid-item
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

/// Minimum output positions per forward task: small enough that a single-image batch still splits
/// across threads, large enough to amortize the rayon task overhead
const POOL_MIN_CHUNK_OUT: usize = 256;

/// Minimum channels per backward task, so a slab is at least a few vector registers wide
const POOL_MIN_CHUNK_CHANNELS: usize = 16;

/// Positions whose channels the accumulator visits per fold block in [`global_pool_forward`]
///
/// The block size is a function of the channel count only. The fold grouping, and with it the
/// summation order, depends on the input shape and never on the thread count
fn rows_per_block(channels: usize) -> usize {
    (16_384 / channels.max(1)).max(1)
}

/// Folds one window's element into a running max, matching the serial scan's NaN rule
///
/// Once a NaN appears, it wins and stays. A later NaN does not replace it, so the index of the
/// first NaN is kept. No finite value can displace it either, because `v > NaN` is false. A bare
/// `v > max_val` would silently drop NaNs instead of propagating them
#[inline]
fn fold_max(max_val: &mut f32, max_idx: &mut usize, v: f32, idx: usize) {
    if v.is_nan() {
        if !max_val.is_nan() {
            *max_val = v;
            *max_idx = idx;
        }
    } else if v > *max_val {
        *max_val = v;
        *max_idx = idx;
    }
}

/// Forward pass for windowed pooling (`MaxPooling{1,2,3}D` / `AveragePooling{1,2,3}D`)
///
/// `pool` and `strides` are the per-spatial-axis window sizes and steps (length = spatial rank)
///
/// # Returns
///
/// - `Tensor` - the pooled output
/// - `Option<Vec<usize>>` - for [`PoolKind::Max`], one arg-max per output element as a flat
///   element offset into the batch item. It already carries the channel, so
///   [`windowed_pool_backward`] needs no further arithmetic. `None` for averaging
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
/// `force_parallel` selects the parallel or serial path regardless of the work estimate.
/// Production code passes `None`. Reachable outside the crate only through `bench_internals`
pub fn windowed_pool_forward_impl(
    input: &Tensor,
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
    padding: PaddingType,
    force_parallel: Option<bool>,
) -> (Tensor, Option<Vec<usize>>) {
    let shape = input.shape();
    let r = shape.len() - 2;
    let batch = shape[0];
    let sp = &shape[1..1 + r];
    let channels = shape[1 + r];
    let (out_sp, pad_before) = pool_geometry(sp, pool, strides, padding);
    let plane_out: usize = out_sp.iter().product();
    let in_strides = row_major_strides(sp);
    let item_in: usize = sp.iter().product::<usize>() * channels;
    let track = kind == PoolKind::Max;

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // Reduces output positions `[c0, c0 + len)` of batch item `b`. The same serial window loop
    // folds every position, whatever the block boundaries are. The result therefore does not
    // depend on how the work was split
    let process_range = |b: usize, c0: usize, len: usize| -> (Vec<f32>, Vec<usize>) {
        let item_base = b * item_in;
        let mut out_chunk = vec![0.0f32; len * channels];
        let mut arg_chunk = if track {
            vec![0usize; len * channels]
        } else {
            Vec::new()
        };

        let mut o = decode_index(c0, &out_sp);
        let mut w = vec![0usize; r];
        for i in 0..len {
            let acc = &mut out_chunk[i * channels..(i + 1) * channels];
            let arg = if track {
                &mut arg_chunk[i * channels..(i + 1) * channels]
            } else {
                &mut [][..]
            };
            match kind {
                PoolKind::Max => acc.fill(f32::NEG_INFINITY),
                PoolKind::Average => acc.fill(0.0),
            }

            // The loop evaluates the window geometry below once for the whole position, and
            // every channel reuses it. That reuse is what pays for this layout
            w.iter_mut().for_each(|x| *x = 0);
            let mut count = 0usize;
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
                    let off = item_base + in_idx * channels;
                    let x = &in_flat[off..off + channels];
                    match kind {
                        PoolKind::Max => {
                            for c in 0..channels {
                                fold_max(&mut acc[c], &mut arg[c], x[c], off - item_base + c);
                            }
                        }
                        PoolKind::Average => {
                            for (a, &v) in acc.iter_mut().zip(x) {
                                *a += v;
                            }
                            count += 1;
                        }
                    }
                }
                if !increment_index(&mut w, pool) {
                    break;
                }
            }
            if kind == PoolKind::Average {
                // The in-bounds count is a property of the window, not of any channel
                let scale = if count > 0 { 1.0 / count as f32 } else { 0.0 };
                acc.iter_mut().for_each(|a| *a *= scale);
            }
            increment_index(&mut o, &out_sp);
        }
        (out_chunk, arg_chunk)
    };

    let total_ops = batch
        .saturating_mul(plane_out)
        .saturating_mul(channels)
        .saturating_mul(pool.iter().product::<usize>());
    let parallel = force_parallel.unwrap_or(total_ops >= pool_parallel_min_ops());

    // Enough blocks to feed every thread once the batch alone cannot, but never so small that the
    // task overhead dominates
    let chunk_len = if parallel && batch > 0 && plane_out > 0 {
        let chunks_per_item = rayon::current_num_threads().div_ceil(batch);
        plane_out.div_ceil(chunks_per_item).max(POOL_MIN_CHUNK_OUT)
    } else {
        plane_out.max(1)
    };
    let tasks: Vec<(usize, usize, usize)> = (0..batch)
        .flat_map(|b| {
            (0..plane_out)
                .step_by(chunk_len.max(1))
                .map(move |c0| (b, c0, chunk_len.min(plane_out - c0)))
        })
        .collect();

    let results: Vec<(Vec<f32>, Vec<usize>)> = if parallel {
        tasks
            .par_iter()
            .map(|&(b, c0, len)| process_range(b, c0, len))
            .collect()
    } else {
        tasks
            .iter()
            .map(|&(b, c0, len)| process_range(b, c0, len))
            .collect()
    };

    let mut out_flat = vec![0.0f32; batch * plane_out * channels];
    let mut argmax = if track {
        vec![0usize; batch * plane_out * channels]
    } else {
        Vec::new()
    };
    for (&(b, c0, len), (out_chunk, arg_chunk)) in tasks.iter().zip(results) {
        let base = (b * plane_out + c0) * channels;
        out_flat[base..base + len * channels].copy_from_slice(&out_chunk);
        if track {
            argmax[base..base + len * channels].copy_from_slice(&arg_chunk);
        }
    }

    let mut out_shape = Vec::with_capacity(2 + r);
    out_shape.push(batch);
    out_shape.extend_from_slice(&out_sp);
    out_shape.push(channels);
    let output = ArrayD::from_shape_vec(IxDyn(&out_shape), out_flat)
        .expect("pool output length matches its shape");

    (output, if track { Some(argmax) } else { None })
}

/// Backward pass for windowed pooling
///
/// `input_shape` is the full forward input shape `[batch, spatial..., channels]`. For
/// [`PoolKind::Max`], `argmax` must be the offsets that [`windowed_pool_forward`] returns.
/// Averaging redistributes each output gradient evenly over its window and ignores `argmax`
pub(super) fn windowed_pool_backward(
    grad_output: &Tensor,
    input_shape: &[usize],
    pool: &[usize],
    strides: &[usize],
    kind: PoolKind,
    argmax: Option<&[usize]>,
    padding: PaddingType,
) -> Tensor {
    let r = input_shape.len() - 2;
    let batch = input_shape[0];
    let sp = &input_shape[1..1 + r];
    let channels = input_shape[1 + r];
    let out_sp = &grad_output.shape()[1..1 + r];
    // Leading padding per axis (Max backward uses the recorded arg-max, so only Average needs it)
    let (_, pad_before) = pool_geometry(sp, pool, strides, padding);
    let plane_in: usize = sp.iter().product();
    let plane_out: usize = out_sp.iter().product();
    let in_strides = row_major_strides(sp);
    let item_in = plane_in * channels;

    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // Accumulates the gradient for channels `[j0, j1)` of batch item `b` into `out`, a slab of the
    // item laid out `[plane_in, j1 - j0]`. 2 slabs never touch the same address, so this is safe
    // to run concurrently. Within a slab, the accumulation runs in output-position order
    let process_slab = |b: usize, j0: usize, width: usize, out: &mut [f32]| {
        let g_item = b * plane_out * channels;
        match kind {
            PoolKind::Max => {
                let arg = argmax.expect("max pooling backward requires arg-max positions");
                let arg_item = &arg[g_item..g_item + plane_out * channels];
                for o_flat in 0..plane_out {
                    for j in j0..j0 + width {
                        let g = grad_flat[g_item + o_flat * channels + j];
                        // The stored offset already carries the channel, so dividing it back out
                        // gives the position and the remainder is `j` by construction
                        let target = arg_item[o_flat * channels + j];
                        out[(target / channels) * width + (j - j0)] += g;
                    }
                }
            }
            PoolKind::Average => {
                let mut o = vec![0usize; r];
                let mut w = vec![0usize; r];
                let mut o_flat = 0usize;
                loop {
                    // Count the in-bounds elements of this window, then spread the gradient evenly.
                    // Every channel of the position shares the count
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
                        let scale = 1.0 / count as f32;
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
                                let dst = &mut out[in_idx * width..(in_idx + 1) * width];
                                let src = &grad_flat[g_item + o_flat * channels + j0..][..width];
                                for (d, &g) in dst.iter_mut().zip(src) {
                                    *d += g * scale;
                                }
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
    };

    // One task per (batch item, channel slab)
    let total_ops = batch
        .saturating_mul(plane_out)
        .saturating_mul(channels)
        .saturating_mul(pool.iter().product::<usize>());
    let parallel = total_ops >= pool_parallel_min_ops();
    let slab = if parallel && batch > 0 && channels > 0 {
        let slabs_per_item = rayon::current_num_threads().div_ceil(batch);
        channels
            .div_ceil(slabs_per_item)
            .max(POOL_MIN_CHUNK_CHANNELS)
            .min(channels)
    } else {
        channels.max(1)
    };
    let tasks: Vec<(usize, usize, usize)> = (0..batch)
        .flat_map(|b| {
            (0..channels)
                .step_by(slab.max(1))
                .map(move |j0| (b, j0, slab.min(channels - j0)))
        })
        .collect();

    let run = |&(b, j0, width): &(usize, usize, usize)| {
        let mut out = vec![0.0f32; plane_in * width];
        process_slab(b, j0, width, &mut out);
        out
    };
    let slabs: Vec<Vec<f32>> = if parallel {
        tasks.par_iter().map(run).collect()
    } else {
        tasks.iter().map(run).collect()
    };

    let mut grad_in = vec![0.0f32; batch * item_in];
    for (&(b, j0, width), slab_data) in tasks.iter().zip(slabs) {
        let item_base = b * item_in;
        for p in 0..plane_in {
            let dst = item_base + p * channels + j0;
            grad_in[dst..dst + width].copy_from_slice(&slab_data[p * width..(p + 1) * width]);
        }
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
/// - `Tensor` - the pooled output, with shape `[batch, channels]`
/// - `Option<Vec<usize>>` - for [`PoolKind::Max`], one arg-max per output element as a flat
///   element offset into the batch item. `None` for averaging
pub(super) fn global_pool_forward(input: &Tensor, kind: PoolKind) -> (Tensor, Option<Vec<usize>>) {
    let shape = input.shape();
    let r = shape.len() - 2;
    let batch = shape[0];
    let channels = shape[1 + r];
    let positions: usize = shape[1..1 + r].iter().product();
    let item_in = positions * channels;
    let track = kind == PoolKind::Max;

    let input_std = input.as_standard_layout();
    let in_flat = input_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    // One task per (batch item, block of positions). The block size depends only on the channel
    // count. The fold grouping, and with it the float summation order, therefore depends on the
    // input shape and never on the schedule
    let block = rows_per_block(channels);
    let tasks: Vec<(usize, usize, usize)> = (0..batch)
        .flat_map(|b| {
            (0..positions)
                .step_by(block.max(1))
                .map(move |p0| (b, p0, block.min(positions - p0)))
        })
        .collect();

    let run = |&(b, p0, len): &(usize, usize, usize)| -> (Vec<f32>, Vec<usize>) {
        let base = b * item_in + p0 * channels;
        let mut acc = match kind {
            PoolKind::Max => vec![f32::NEG_INFINITY; channels],
            PoolKind::Average => vec![0.0f32; channels],
        };
        let mut arg = if track {
            vec![0usize; channels]
        } else {
            Vec::new()
        };
        for p in 0..len {
            let off = base + p * channels;
            let x = &in_flat[off..off + channels];
            match kind {
                PoolKind::Max => {
                    for c in 0..channels {
                        fold_max(&mut acc[c], &mut arg[c], x[c], (p0 + p) * channels + c);
                    }
                }
                PoolKind::Average => {
                    for (a, &v) in acc.iter_mut().zip(x) {
                        *a += v;
                    }
                }
            }
        }
        (acc, arg)
    };

    let parallel = batch.saturating_mul(item_in) >= pool_parallel_min_ops();
    let partials: Vec<(Vec<f32>, Vec<usize>)> = if parallel {
        tasks.par_iter().map(run).collect()
    } else {
        tasks.iter().map(run).collect()
    };

    // Merge the block partials in block order. Folding them with the same rule the serial scan
    // uses reproduces a single pass exactly. A `Max` block that saw a NaN keeps it. Ties still
    // resolve to the earliest position
    let mut out_flat = match kind {
        PoolKind::Max => vec![f32::NEG_INFINITY; batch * channels],
        PoolKind::Average => vec![0.0f32; batch * channels],
    };
    let mut argmax = if track {
        vec![0usize; batch * channels]
    } else {
        Vec::new()
    };
    for (&(b, _, _), (acc, arg)) in tasks.iter().zip(partials) {
        let out = &mut out_flat[b * channels..(b + 1) * channels];
        match kind {
            PoolKind::Max => {
                let dst_arg = &mut argmax[b * channels..(b + 1) * channels];
                for c in 0..channels {
                    fold_max(&mut out[c], &mut dst_arg[c], acc[c], arg[c]);
                }
            }
            PoolKind::Average => {
                for (o, v) in out.iter_mut().zip(acc) {
                    *o += v;
                }
            }
        }
    }
    if kind == PoolKind::Average {
        let scale = 1.0 / positions as f32;
        out_flat.iter_mut().for_each(|v| *v *= scale);
    }

    let output = ArrayD::from_shape_vec(IxDyn(&[batch, channels]), out_flat)
        .expect("global-pool output length matches [batch, channels]");
    (output, if track { Some(argmax) } else { None })
}

/// Backward pass for global pooling
///
/// `grad_output` has shape `[batch, channels]`. Averaging spreads each gradient evenly over every
/// position of its channel. [`PoolKind::Max`] routes it to the stored arg-max element
pub(super) fn global_pool_backward(
    grad_output: &Tensor,
    input_shape: &[usize],
    kind: PoolKind,
    argmax: Option<&[usize]>,
) -> Tensor {
    let r = input_shape.len() - 2;
    let batch = input_shape[0];
    let channels = input_shape[1 + r];
    let positions: usize = input_shape[1..1 + r].iter().product();
    let item_in = positions * channels;

    let grad_std = grad_output.as_standard_layout();
    let grad_flat = grad_std
        .as_slice()
        .expect("standard-layout array is contiguous");

    let mut grad_in = vec![0.0f32; batch * item_in];
    for b in 0..batch {
        let item = &mut grad_in[b * item_in..(b + 1) * item_in];
        let g = &grad_flat[b * channels..(b + 1) * channels];
        match kind {
            PoolKind::Max => {
                let arg = argmax.expect("global max pooling backward requires arg-max positions");
                let arg_item = &arg[b * channels..(b + 1) * channels];
                for c in 0..channels {
                    item[arg_item[c]] += g[c];
                }
            }
            PoolKind::Average => {
                // Every position of the item gets the same `[channels]` vector. This pass builds
                // a tile that repeats the vector, then copies it into the item in tile-sized
                // strides. Writing the item row by row would call `memcpy` once per position.
                // That access pattern is the wrong shape when `channels` is small
                let scale = 1.0 / positions as f32;
                let reps = (1024 / channels.max(1)).clamp(1, positions.max(1));
                let mut tile = Vec::with_capacity(reps * channels);
                for _ in 0..reps {
                    tile.extend(g.iter().map(|&v| v * scale));
                }
                for chunk in item.chunks_mut(tile.len()) {
                    chunk.copy_from_slice(&tile[..chunk.len()]);
                }
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(input_shape), grad_in)
        .expect("grad-input length matches the input shape")
}

/// Unit tests for the pooling engine
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

    /// An empty shape yields no strides
    #[test]
    fn test_row_major_strides_empty() {
        let s = row_major_strides(&[]);
        assert_eq!(s, Vec::<usize>::new());
    }

    // increment_index

    /// A multi-index advances its last axis first
    #[test]
    fn test_increment_index_normal() {
        let dims = [2usize, 3];
        let mut idx = vec![0usize, 0];
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 1]);
    }

    /// Overflowing the last axis carries into the previous one
    #[test]
    fn test_increment_index_carry() {
        let dims = [2usize, 3];
        let mut idx = vec![0usize, 2];
        assert!(increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![1, 0]);
    }

    /// The final index wraps to the origin and reports exhaustion
    #[test]
    fn test_increment_index_exhausted() {
        let dims = [2usize, 3];
        let mut idx = vec![1usize, 2];
        assert!(!increment_index(&mut idx, &dims));
        assert_eq!(idx, vec![0, 0]);
    }

    // decode_index

    /// A flat index decomposes into the multi-index `increment_index` would have reached
    #[test]
    fn test_decode_index_matches_increment() {
        let dims = [2usize, 3, 4];
        let mut idx = vec![0usize; 3];
        for flat in 0..24 {
            assert_eq!(decode_index(flat, &dims), idx, "flat {flat}");
            increment_index(&mut idx, &dims);
        }
    }

    // windowed_pool_forward (Max)

    /// 1-D max-pool returns per-window maxima and their flat arg-max offsets
    #[test]
    fn test_windowed_pool_forward_1d_max() {
        // [batch, length, channels] with 1 channel
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![3.0f32, 1.0, 4.0, 1.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2], &[2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 2, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 4.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(am, vec![0, 2]);
    }

    /// 2-D max-pool over a single window returns the max and its flat arg-max offset
    #[test]
    fn test_windowed_pool_forward_2d_max() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 4.0, epsilon = 1e-6);
        let am = argmax.expect("Max pool must return argmax");
        assert_eq!(am, vec![3]);
    }

    /// Each channel is pooled on its own, and the arg-max offset carries the channel
    ///
    /// Channel 0 peaks in the first window position and channel 1 in the last. A single shared
    /// arg-max, or channels bleeding into each other, would both appear here
    #[test]
    fn test_windowed_pool_forward_1d_max_two_channels() {
        // [1, 4, 2]: channel 0 is 4, 3, 2, 1 and channel 1 is 1, 2, 3, 4
        let data = ArrayD::from_shape_vec(
            IxDyn(&[1, 4, 2]),
            vec![4.0f32, 1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0],
        )
        .unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2], &[2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 2, 2]);
        // Window 0 covers positions 0-1: max is 4 on channel 0 and 2 on channel 1
        // Window 1 covers positions 2-3: max is 2 on channel 0 and 4 on channel 1
        assert_eq!(
            out.iter().copied().collect::<Vec<f32>>(),
            vec![4.0, 2.0, 2.0, 4.0]
        );
        // Offsets are `position * channels + channel`
        assert_eq!(argmax.unwrap(), vec![0, 3, 4, 7]);
    }

    // windowed_pool_forward (Average)

    /// 1-D average-pool returns per-window means and no arg-max
    #[test]
    fn test_windowed_pool_forward_1d_avg() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![3.0f32, 1.0, 4.0, 1.0]).unwrap();
        let (out, argmax) =
            windowed_pool_forward(&data, &[2], &[2], PoolKind::Average, PaddingType::Valid);
        assert_eq!(out.shape(), &[1, 2, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 2.5, epsilon = 1e-6);
        assert!(argmax.is_none(), "Average pool must not return argmax");
    }

    /// 2-D average-pool over a single window returns the mean and no arg-max
    #[test]
    fn test_windowed_pool_forward_2d_avg() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
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

    /// `Same` padding divides by the count of real elements, not the window size
    ///
    /// The trailing window of a length-3 input under a size-2 stride-2 window holds 1 real
    /// element and 1 padding cell. Its mean is therefore that element itself
    #[test]
    fn test_windowed_pool_forward_1d_avg_same_padding_excludes_pad() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), vec![1.0f32, 2.0, 6.0]).unwrap();
        let (out, _) =
            windowed_pool_forward(&data, &[2], &[2], PoolKind::Average, PaddingType::Same);
        assert_eq!(out.shape(), &[1, 2, 1]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 6.0, epsilon = 1e-6);
    }

    /// Ties resolve to the first position scanned, on every channel
    #[test]
    fn test_windowed_pool_forward_2d_max_tie_breaks_to_first() {
        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1]), vec![5.0f32, 5.0, 5.0, 5.0]).unwrap();
        let (_, argmax) =
            windowed_pool_forward(&data, &[2, 2], &[2, 2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(argmax.unwrap(), vec![0]);
    }

    // windowed_pool_backward

    /// Max-pool backward routes each upstream gradient to its arg-max position
    #[test]
    fn test_windowed_pool_backward_1d_max_nonoverlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![1.0f32, 1.0]).unwrap();
        let argmax = vec![0usize, 2];
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 4, 1],
            &[2],
            &[2],
            PoolKind::Max,
            Some(&argmax),
            PaddingType::Valid,
        );
        assert_eq!(grad_in.shape(), &[1, 4, 1]);
        assert_eq!(
            grad_in.iter().copied().collect::<Vec<f32>>(),
            vec![1.0, 0.0, 1.0, 0.0]
        );
    }

    /// Max-pool backward keeps each channel's gradient on its own channel
    #[test]
    fn test_windowed_pool_backward_1d_max_two_channels() {
        // Mirrors `test_windowed_pool_forward_1d_max_two_channels`
        let grad_out =
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let argmax = vec![0usize, 3, 4, 7];
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 4, 2],
            &[2],
            &[2],
            PoolKind::Max,
            Some(&argmax),
            PaddingType::Valid,
        );
        assert_eq!(grad_in.shape(), &[1, 4, 2]);
        assert_eq!(
            grad_in.iter().copied().collect::<Vec<f32>>(),
            vec![10.0, 0.0, 0.0, 20.0, 30.0, 0.0, 0.0, 40.0]
        );
    }

    /// Average-pool backward spreads each gradient evenly across its window
    #[test]
    fn test_windowed_pool_backward_1d_avg_nonoverlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![2.0f32, 4.0]).unwrap();
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 4, 1],
            &[2],
            &[2],
            PoolKind::Average,
            None,
            PaddingType::Valid,
        );
        assert_eq!(
            grad_in.iter().copied().collect::<Vec<f32>>(),
            vec![1.0, 1.0, 2.0, 2.0]
        );
    }

    /// Overlapping windows accumulate where they overlap
    #[test]
    fn test_windowed_pool_backward_1d_avg_overlapping() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), vec![2.0f32, 2.0, 2.0]).unwrap();
        let grad_in = windowed_pool_backward(
            &grad_out,
            &[1, 4, 1],
            &[2],
            &[1],
            PoolKind::Average,
            None,
            PaddingType::Valid,
        );
        // Each window contributes 1.0 to both of its positions. The interior is covered twice
        assert_eq!(
            grad_in.iter().copied().collect::<Vec<f32>>(),
            vec![1.0, 2.0, 2.0, 1.0]
        );
    }

    // global_pool_forward / global_pool_backward

    /// Global average pooling reduces every position and keeps the channels apart
    #[test]
    fn test_global_pool_forward_avg_two_channels() {
        // [1, 4, 2]: channel 0 is 1, 2, 3, 4 and channel 1 is 10, 20, 30, 40
        let data = ArrayD::from_shape_vec(
            IxDyn(&[1, 4, 2]),
            vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        )
        .unwrap();
        let (out, argmax) = global_pool_forward(&data, PoolKind::Average);
        assert_eq!(out.shape(), &[1, 2]);
        let flat: Vec<f32> = out.iter().copied().collect();
        assert_abs_diff_eq!(flat[0], 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(flat[1], 25.0, epsilon = 1e-6);
        assert!(argmax.is_none());
    }

    /// Global max pooling records a per-channel arg-max offset
    #[test]
    fn test_global_pool_forward_max_two_channels() {
        let data = ArrayD::from_shape_vec(
            IxDyn(&[1, 4, 2]),
            vec![4.0f32, 1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0],
        )
        .unwrap();
        let (out, argmax) = global_pool_forward(&data, PoolKind::Max);
        assert_eq!(out.iter().copied().collect::<Vec<f32>>(), vec![4.0, 4.0]);
        // Channel 0 peaks at position 0, channel 1 at position 3
        assert_eq!(argmax.unwrap(), vec![0, 7]);
    }

    /// Global max ties resolve to the first position scanned
    #[test]
    fn test_global_pool_forward_max_tie_breaks_to_first() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), vec![5.0f32, 5.0, 5.0]).unwrap();
        let (_, argmax) = global_pool_forward(&data, PoolKind::Max);
        assert_eq!(argmax.unwrap(), vec![0]);
    }

    /// Global average backward spreads each channel's gradient over every position
    #[test]
    fn test_global_pool_backward_avg() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![8.0f32, 40.0]).unwrap();
        let grad_in = global_pool_backward(&grad_out, &[1, 4, 2], PoolKind::Average, None);
        assert_eq!(grad_in.shape(), &[1, 4, 2]);
        // 8 / 4 = 2 on channel 0, 40 / 4 = 10 on channel 1, at every position
        assert_eq!(
            grad_in.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 10.0, 2.0, 10.0, 2.0, 10.0, 2.0, 10.0]
        );
    }

    /// Global max backward routes each channel's gradient to its recorded winner
    #[test]
    fn test_global_pool_backward_max() {
        let grad_out = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![7.0f32, 9.0]).unwrap();
        let argmax = vec![0usize, 7];
        let grad_in = global_pool_backward(&grad_out, &[1, 4, 2], PoolKind::Max, Some(&argmax));
        assert_eq!(
            grad_in.iter().copied().collect::<Vec<f32>>(),
            vec![7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]
        );
    }

    /// Output shapes put the channel axis last at every rank
    #[test]
    fn test_windowed_pool_output_shapes() {
        let d1 = ArrayD::<f32>::zeros(IxDyn(&[2, 8, 3]));
        let (o1, _) = windowed_pool_forward(&d1, &[2], &[2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(o1.shape(), &[2, 4, 3]);

        let d2 = ArrayD::<f32>::zeros(IxDyn(&[2, 8, 6, 3]));
        let (o2, _) =
            windowed_pool_forward(&d2, &[2, 2], &[2, 2], PoolKind::Max, PaddingType::Valid);
        assert_eq!(o2.shape(), &[2, 4, 3, 3]);

        let d3 = ArrayD::<f32>::zeros(IxDyn(&[2, 8, 6, 4, 3]));
        let (o3, _) = windowed_pool_forward(
            &d3,
            &[2, 2, 2],
            &[2, 2, 2],
            PoolKind::Max,
            PaddingType::Valid,
        );
        assert_eq!(o3.shape(), &[2, 4, 3, 2, 3]);
    }

    /// The parallel and serial forward paths agree bit for bit
    ///
    /// The same serial window loop folds every output position, whatever the block boundaries
    /// are, so the gate is a pure performance knob
    #[test]
    fn test_windowed_pool_forward_parallel_matches_serial() {
        let data: Vec<f32> = (0..2 * 20 * 20 * 5)
            .map(|i| (i % 37) as f32 * 0.5)
            .collect();
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 20, 20, 5]), data).unwrap();
        for kind in [PoolKind::Max, PoolKind::Average] {
            let (serial, s_arg) = windowed_pool_forward_impl(
                &input,
                &[3, 3],
                &[2, 2],
                kind,
                PaddingType::Same,
                Some(false),
            );
            let (par, p_arg) = windowed_pool_forward_impl(
                &input,
                &[3, 3],
                &[2, 2],
                kind,
                PaddingType::Same,
                Some(true),
            );
            assert_eq!(serial, par, "{kind:?} values differ across the gate");
            assert_eq!(s_arg, p_arg, "{kind:?} arg-max differs across the gate");
        }
    }
}
