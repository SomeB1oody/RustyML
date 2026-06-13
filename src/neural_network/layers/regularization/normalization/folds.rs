//! Deterministic fold kernels shared by the normalization layers
//!
//! Everything here computes with a fixed, shape-derived grouping and accumulation order, so a
//! caller's `parallel` flag (or task chunking) only decides whether work runs on rayon — never
//! the result bits.

use crate::math::reduction::DET_REDUCE_BLOCK;
use crate::neural_network::Tensor;
use ndarray::Array1;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::ops::Range;

/// Sums one contiguous segment of scaled terms in eight independent lanes combined in a fixed
/// order (the serial kernel the deterministic folds share). The result depends only on the
/// segment boundaries, which derive from the input shape, never on scheduling; the lanes keep
/// the dependency chain short enough to vectorize
pub(super) fn segment_sum(seg: &[f32], scale: f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks = seg.chunks_exact(8);
    for ch in chunks.by_ref() {
        for (l, &v) in lanes.iter_mut().zip(ch) {
            *l += v * scale;
        }
    }
    let mut tail = 0.0f32;
    for &v in chunks.remainder() {
        tail += v * scale;
    }
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        + tail
}

/// The product twin of [`segment_sum`]: `sum_i a[i] * b[i] * scale` over two equal-length
/// contiguous segments
pub(super) fn segment_dot(a: &[f32], b: &[f32], scale: f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);
    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        for ((l, &va), &vb) in lanes.iter_mut().zip(ca).zip(cb) {
            *l += va * vb * scale;
        }
    }
    let mut tail = 0.0f32;
    for (&va, &vb) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        tail += va * vb * scale;
    }
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        + tail
}

/// The squared-deviation twin of [`segment_dot`] with the deviation fused in:
/// `sum_i (seg[i] - mean)^2`. Bitwise identical to centering the segment into a buffer and
/// taking `segment_dot(buf, buf, 1.0)` — the subtraction result is the same f32 whether stored
/// or kept in a register — so callers can skip the centered temporary
pub(super) fn segment_sq_dev(seg: &[f32], mean: f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks = seg.chunks_exact(8);
    for ch in chunks.by_ref() {
        for (l, &v) in lanes.iter_mut().zip(ch) {
            let d = v - mean;
            *l += d * d;
        }
    }
    let mut tail = 0.0f32;
    for &v in chunks.remainder() {
        let d = v - mean;
        tail += d * d;
    }
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        + tail
}

/// The triple-product sibling of [`segment_dot`]: `sum_i a[i] * b[i] * c[i] * scale` over
/// three equal-length contiguous segments, left-associated per term so fusing matches the
/// two-step `(a * b)` -> `segment_dot` composition bit for bit
pub(super) fn segment_dot3(a: &[f32], b: &[f32], c: &[f32], scale: f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);
    let mut chunks_c = c.chunks_exact(8);
    for ((ca, cb), cc) in chunks_a
        .by_ref()
        .zip(chunks_b.by_ref())
        .zip(chunks_c.by_ref())
    {
        for (((l, &va), &vb), &vc) in lanes.iter_mut().zip(ca).zip(cb).zip(cc) {
            *l += va * vb * vc * scale;
        }
    }
    let mut tail = 0.0f32;
    for ((&va, &vb), &vc) in chunks_a
        .remainder()
        .iter()
        .zip(chunks_b.remainder())
        .zip(chunks_c.remainder())
    {
        tail += va * vb * vc * scale;
    }
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        + tail
}

/// Rows per block for the column folds: whole rows, sized so one block holds about
/// [`DET_REDUCE_BLOCK`] elements. A function of the input shape only, so the (deterministic)
/// fold grouping never depends on scheduling
pub(super) fn rows_per_block(c: usize) -> usize {
    (DET_REDUCE_BLOCK / c).max(1)
}

/// Folds one chunk of whole rows into a local per-column accumulator (the serial kernel both
/// paths of the column folds share)
fn col_sum_chunk(chunk: &[f32], c: usize, scale: f32) -> Vec<f32> {
    let mut acc = vec![0.0f32; c];
    for row in chunk.chunks_exact(c) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v * scale;
        }
    }
    acc
}

/// The product-fold twin of [`col_sum_chunk`]
fn col_dot_chunk(chunk_a: &[f32], chunk_b: &[f32], c: usize, scale: f32) -> Vec<f32> {
    let mut acc = vec![0.0f32; c];
    for (row_a, row_b) in chunk_a.chunks_exact(c).zip(chunk_b.chunks_exact(c)) {
        for ((s, &va), &vb) in acc.iter_mut().zip(row_a).zip(row_b) {
            *s += va * vb * scale;
        }
    }
    acc
}

/// Merges per-block partial column sums in block order
fn merge_col_parts(parts: Vec<Vec<f32>>, c: usize) -> Tensor {
    let mut out = vec![0.0f32; c];
    for part in parts {
        for (o, p) in out.iter_mut().zip(part) {
            *o += p;
        }
    }
    Array1::from_vec(out).into_dyn()
}

/// Per-column sums of scaled terms over a standard-layout `[M, C]` slice:
/// `out[j] = sum_r x[r, j] * scale`, computed as a row-block deterministic fold. Each block
/// streams whole rows into a local `[C]` accumulator and block partials merge in block order;
/// the `parallel` flag only decides whether the blocks run on rayon, never the result bits.
/// `scale` is applied per term, matching the serial `(x * scale).sum_axis(Axis(0))` form
pub(super) fn par_col_sum(x: &[f32], c: usize, parallel: bool, scale: f32) -> Tensor {
    let block = rows_per_block(c) * c;
    let parts: Vec<Vec<f32>> = if parallel {
        x.par_chunks(block)
            .map(|chunk| col_sum_chunk(chunk, c, scale))
            .collect()
    } else {
        x.chunks(block)
            .map(|chunk| col_sum_chunk(chunk, c, scale))
            .collect()
    };
    merge_col_parts(parts, c)
}

/// Per-column sums of scaled products over two standard-layout `[M, C]` slices:
/// `out[j] = sum_r a[r, j] * b[r, j] * scale`, as the same row-block deterministic fold as
/// [`par_col_sum`] (same flag semantics). Fusing the product into the fold avoids
/// materializing the `[M, C]` temp the serial `(a * b * scale).sum_axis(Axis(0))` form
/// requires
pub(super) fn par_col_dot(a: &[f32], b: &[f32], c: usize, parallel: bool, scale: f32) -> Tensor {
    let block = rows_per_block(c) * c;
    let parts: Vec<Vec<f32>> = if parallel {
        a.par_chunks(block)
            .zip(b.par_chunks(block))
            .map(|(ca, cb)| col_dot_chunk(ca, cb, c, scale))
            .collect()
    } else {
        a.chunks(block)
            .zip(b.chunks(block))
            .map(|(ca, cb)| col_dot_chunk(ca, cb, c, scale))
            .collect()
    };
    merge_col_parts(parts, c)
}

/// Folds the logical element range `[start, end)` of channel `ch` into a scalar partial, where
/// a channel's logical sequence is its `[P]` planes of the `[B, C, P]` slice in batch order.
/// Plane-crossing ranges chain their per-plane [`segment_sum`] partials in order
pub(super) fn plane_range_sum(
    x: &[f32],
    ch: usize,
    c: usize,
    p: usize,
    range: Range<usize>,
    scale: f32,
) -> f32 {
    let mut acc = 0.0f32;
    let mut pos = range.start;
    while pos < range.end {
        let (bi, off) = (pos / p, pos % p);
        let take = (p - off).min(range.end - pos);
        let base = (bi * c + ch) * p + off;
        acc += segment_sum(&x[base..base + take], scale);
        pos += take;
    }
    acc
}

/// The product twin of [`plane_range_sum`] over two `[B, C, P]` slices of the same shape
pub(super) fn plane_range_dot(
    a: &[f32],
    b: &[f32],
    ch: usize,
    c: usize,
    p: usize,
    range: Range<usize>,
    scale: f32,
) -> f32 {
    let mut acc = 0.0f32;
    let mut pos = range.start;
    while pos < range.end {
        let (bi, off) = (pos / p, pos % p);
        let take = (p - off).min(range.end - pos);
        let base = (bi * c + ch) * p + off;
        acc += segment_dot(&a[base..base + take], &b[base..base + take], scale);
        pos += take;
    }
    acc
}

/// Per-channel sums of scaled terms over a standard-layout `[B, C, P]` slice (channel axis 1,
/// planes contiguous): `out[ch] = sum_{b,i} x[b, ch, i] * scale`, computed without transposing
/// to channel-last. Each channel's logical sequence (its planes in batch order) folds in
/// [`DET_REDUCE_BLOCK`]-element blocks whose partials merge in block order, so the grouping
/// depends only on the shape; the `parallel` flag only decides whether the (channel, block)
/// tasks run on rayon, never the result bits
pub(super) fn par_plane_sum(x: &[f32], c: usize, p: usize, parallel: bool, scale: f32) -> Tensor {
    if x.is_empty() {
        return Array1::zeros(c).into_dyn();
    }
    let len_per_chan = x.len() / c;
    let n_blocks = len_per_chan.div_ceil(DET_REDUCE_BLOCK);
    let fold = |t: usize| {
        let (ch, blk) = (t / n_blocks, t % n_blocks);
        let start = blk * DET_REDUCE_BLOCK;
        let end = (start + DET_REDUCE_BLOCK).min(len_per_chan);
        plane_range_sum(x, ch, c, p, start..end, scale)
    };
    let partials: Vec<f32> = if parallel {
        (0..c * n_blocks).into_par_iter().map(fold).collect()
    } else {
        (0..c * n_blocks).map(fold).collect()
    };
    let out: Vec<f32> = partials
        .chunks(n_blocks)
        .map(|parts| parts.iter().fold(0.0f32, |acc, &v| acc + v))
        .collect();
    Array1::from_vec(out).into_dyn()
}

/// Per-channel sums of scaled products over two standard-layout `[B, C, P]` slices of the same
/// shape: `out[ch] = sum_{b,i} a[b, ch, i] * b[b, ch, i] * scale`, with the same block
/// grouping and flag semantics as [`par_plane_sum`]. Fusing the product into the fold avoids
/// materializing the elementwise-product temporary
pub(super) fn par_plane_dot(
    a: &[f32],
    b: &[f32],
    c: usize,
    p: usize,
    parallel: bool,
    scale: f32,
) -> Tensor {
    if a.is_empty() {
        return Array1::zeros(c).into_dyn();
    }
    let len_per_chan = a.len() / c;
    let n_blocks = len_per_chan.div_ceil(DET_REDUCE_BLOCK);
    let fold = |t: usize| {
        let (ch, blk) = (t / n_blocks, t % n_blocks);
        let start = blk * DET_REDUCE_BLOCK;
        let end = (start + DET_REDUCE_BLOCK).min(len_per_chan);
        plane_range_dot(a, b, ch, c, p, start..end, scale)
    };
    let partials: Vec<f32> = if parallel {
        (0..c * n_blocks).into_par_iter().map(fold).collect()
    } else {
        (0..c * n_blocks).map(fold).collect()
    };
    let out: Vec<f32> = partials
        .chunks(n_blocks)
        .map(|parts| parts.iter().fold(0.0f32, |acc, &v| acc + v))
        .collect();
    Array1::from_vec(out).into_dyn()
}
