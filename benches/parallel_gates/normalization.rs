//! Normalization-layer statistics gates: the BatchNorm column-stats folds (channel-chunked
//! negative result and the viable row-block fold), the rank>=3 native-layout plane fold, and the
//! trailing-axis LayerNorm fused row pass.

use crate::harness::{Row, Section, random_matrix, time_per_call_ns};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use std::hint::black_box;

// ---- BatchNorm column statistics: channel-chunked parallel vs serial mean_axis ----

/// One cache line of `f32` per channel chunk, mirroring the production helper
const BENCH_CHANNEL_CHUNK: usize = 16;

/// Channel-chunked parallel column sums of a standard-layout [M, C] f32 matrix. Each channel
/// accumulates in row order (bitwise identical to ndarray's serial `sum_axis(Axis(0))`);
/// parallelism only splits the channel axis, so the task count is C / 16
fn bench_par_col_sum(x: &Array2<f32>) -> Array1<f32> {
    let c = x.ncols();
    let slice = x.as_slice().unwrap();
    let mut out = Array1::<f32>::zeros(c);
    out.as_slice_mut()
        .unwrap()
        .par_chunks_mut(BENCH_CHANNEL_CHUNK)
        .enumerate()
        .for_each(|(g, acc)| {
            let j0 = g * BENCH_CHANNEL_CHUNK;
            let width = acc.len();
            for row in slice.chunks_exact(c) {
                for (a, &v) in acc.iter_mut().zip(&row[j0..j0 + width]) {
                    *a += v;
                }
            }
        });
    out
}

/// The BatchNorm statistics reduction: per-channel sums over batch x spatial rows. The win is
/// capped by the channel-chunk task count (C / 16), so narrow-C rungs document where the
/// parallel path merely ties.
///
/// **Negative result, kept as the record of why production uses row blocks instead:** the
/// channel split preserves the serial per-channel accumulation order (bitwise identical to
/// `mean_axis`) but loses 2-3x - the serial row-streaming fold is already bandwidth-efficient
/// and SIMD-wide, and column-chunk tasks break exactly that
pub fn calibrate_bn_col_stats() -> Section {
    let mut rows = Vec::new();
    for &(m, c) in &[
        (4_096usize, 64usize),
        (16_384, 64),
        (65_536, 64),
        (524_288, 64),
        (2_048, 512),
        (16_384, 512),
        (262_144, 8),
    ] {
        let x = random_matrix(m, c, 103);
        let s = time_per_call_ns(|| {
            black_box(x.mean_axis(Axis(0)).unwrap());
        });
        let p = time_per_call_ns(|| {
            let sums = bench_par_col_sum(&x);
            black_box(sums / m as f32);
        });
        rows.push(Row {
            label: format!("M={m} C={c}"),
            work: m * c,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "BatchNorm column stats, channel-chunked (negative result; see row-block section)",
        work_unit: "elements (M x C)",
        pick_fastest: false,
        rows,
    }
}

/// Row-block deterministic fold for the same per-channel sums: each block streams whole rows
/// (the same bandwidth-friendly, SIMD-across-channels pattern as the serial fold) into a local
/// [C] accumulator; block partials merge in block order. Rows per block scale as
/// DET_REDUCE_BLOCK / C, so the grouping depends only on the input shape, never on scheduling
fn bench_par_col_sum_rowblock(x: &Array2<f32>) -> Array1<f32> {
    let (m, c) = x.dim();
    let slice = x.as_slice().unwrap();
    let rows_per_block = (16_384usize / c).max(1);
    let parts: Vec<Array1<f32>> = slice
        .par_chunks(rows_per_block * c)
        .map(|chunk| {
            let mut acc = Array1::<f32>::zeros(c);
            let acc_slice = acc.as_slice_mut().unwrap();
            for row in chunk.chunks_exact(c) {
                for (a, &v) in acc_slice.iter_mut().zip(row) {
                    *a += v;
                }
            }
            acc
        })
        .collect();
    let mut out = Array1::<f32>::zeros(c);
    for p in parts {
        out += &p;
    }
    debug_assert_eq!(m * c, slice.len());
    out
}

/// The viable BatchNorm stats parallelization (BN_COL_STATS_PARALLEL_MIN_ELEMS): row-block
/// deterministic fold vs serial mean_axis. Changes the per-channel accumulation grouping
/// (a versioned behavior change), but is bitwise identical at any thread count
pub fn calibrate_bn_col_stats_rowblock() -> Section {
    let mut rows = Vec::new();
    for &(m, c) in &[
        (1_024usize, 64usize),
        (4_096, 64),
        (16_384, 64),
        (65_536, 64),
        (524_288, 64),
        (2_048, 512),
        (16_384, 512),
        (262_144, 8),
    ] {
        let x = random_matrix(m, c, 103);
        let s = time_per_call_ns(|| {
            black_box(x.mean_axis(Axis(0)).unwrap());
        });
        let p = time_per_call_ns(|| {
            let sums = bench_par_col_sum_rowblock(&x);
            black_box(sums / m as f32);
        });
        rows.push(Row {
            label: format!("M={m} C={c}"),
            work: m * c,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "BatchNorm column stats, row-block fold (BN_COL_STATS_PARALLEL_MIN_ELEMS)",
        work_unit: "elements (M x C)",
        pick_fastest: false,
        rows,
    }
}

// ---- BatchNorm plane statistics (rank >= 3 native layout): BN_PLANE_STATS_PARALLEL_MIN_ELEMS ----

/// Mirrors the production plane fold: per-channel sums over the native `[B, C, P]` layout,
/// each channel's logical sequence (its planes in batch order) folded in 16K-element blocks
/// whose contiguous segments accumulate in eight SIMD-friendly lanes; block partials merge in
/// block order. The `parallel` flag only moves the (channel, block) tasks onto rayon
fn bench_plane_sum(x: &[f32], c: usize, p: usize, parallel: bool) -> Array1<f32> {
    const BLOCK: usize = 16_384;
    let len_per_chan = x.len() / c;
    let n_blocks = len_per_chan.div_ceil(BLOCK);
    let segment_sum = |seg: &[f32]| -> f32 {
        let mut lanes = [0.0f32; 8];
        let mut chunks = seg.chunks_exact(8);
        for ch in chunks.by_ref() {
            for (l, &v) in lanes.iter_mut().zip(ch) {
                *l += v;
            }
        }
        let mut tail = 0.0f32;
        for &v in chunks.remainder() {
            tail += v;
        }
        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
            + tail
    };
    let fold = |t: usize| {
        let (ch, blk) = (t / n_blocks, t % n_blocks);
        let (start, end) = (blk * BLOCK, ((blk + 1) * BLOCK).min(len_per_chan));
        let mut acc = 0.0f32;
        let mut pos = start;
        while pos < end {
            let (bi, off) = (pos / p, pos % p);
            let take = (p - off).min(end - pos);
            let base = (bi * c + ch) * p + off;
            acc += segment_sum(&x[base..base + take]);
            pos += take;
        }
        acc
    };
    let partials: Vec<f32> = if parallel {
        (0..c * n_blocks).into_par_iter().map(fold).collect()
    } else {
        (0..c * n_blocks).map(fold).collect()
    };
    Array1::from_iter(
        partials
            .chunks(n_blocks)
            .map(|parts| parts.iter().fold(0.0f32, |acc, &v| acc + v)),
    )
}

/// The rank >= 3 BatchNorm statistics reduction on the native layout: forced serial vs forced
/// parallel of the same plane fold (the flag never changes the bits, so the gate is a pure
/// performance knob). Spans conv-scale shapes plus narrow-channel and wide-channel extremes
pub fn calibrate_bn_plane_stats() -> Section {
    let mut rows = Vec::new();
    for &(b, c, p) in &[
        (4usize, 16usize, 256usize),
        (8, 16, 512),
        (8, 32, 1_024),
        (16, 32, 2_048),
        (32, 8, 4_096),
        (8, 512, 256),
        (16, 64, 4_096),
        (32, 64, 4_096),
    ] {
        // random_matrix(b * c, p) flattens to the same standard-layout [B, C, P] slice
        let x = random_matrix(b * c, p, 107);
        let xs = x.as_slice().unwrap();
        let s = time_per_call_ns(|| {
            black_box(bench_plane_sum(xs, c, p, false));
        });
        let par = time_per_call_ns(|| {
            black_box(bench_plane_sum(xs, c, p, true));
        });
        rows.push(Row {
            label: format!("B={b} C={c} P={p}"),
            work: b * c * p,
            serial_ns: s,
            parallel_ns: par,
        });
    }
    Section {
        title: "BatchNorm plane stats, native-layout fold (BN_PLANE_STATS_PARALLEL_MIN_ELEMS)",
        work_unit: "elements (B x C x P)",
        pick_fastest: false,
        rows,
    }
}

// ---- LayerNorm fused row pass (trailing axis): LN_ROW_PARALLEL_MIN_ELEMS ----

/// Mirrors the production LayerNorm row pass: per row of a `[R, N]` slice, eight-lane mean
/// and variance folds plus the fused center/normalize/scale-shift sweep writing three
/// buffers. Rows are independent, so the flag is scheduling-only
fn bench_ln_row_pass(
    x: &[f32],
    n: usize,
    gamma: &[f32],
    beta: &[f32],
    parallel: bool,
    bufs: &mut (Vec<f32>, Vec<f32>, Vec<f32>),
) {
    let segment_sum = |seg: &[f32]| -> f32 {
        let mut lanes = [0.0f32; 8];
        let mut chunks = seg.chunks_exact(8);
        for ch in chunks.by_ref() {
            for (l, &v) in lanes.iter_mut().zip(ch) {
                *l += v;
            }
        }
        let mut tail = 0.0f32;
        for &v in chunks.remainder() {
            tail += v;
        }
        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
            + tail
    };
    let (xc, xn, out) = (&mut bufs.0, &mut bufs.1, &mut bufs.2);
    let chunk = (16_384usize / n).max(1) * n;
    type RowChunks<'a> = (((&'a mut [f32], &'a mut [f32]), &'a mut [f32]), &'a [f32]);
    let task = |(((xc_c, xn_c), out_c), x_c): RowChunks| {
        let rows = x_c
            .chunks_exact(n)
            .zip(xc_c.chunks_exact_mut(n))
            .zip(xn_c.chunks_exact_mut(n))
            .zip(out_c.chunks_exact_mut(n));
        for (((x_row, xc_row), xn_row), out_row) in rows {
            let mean = segment_sum(x_row) / n as f32;
            for (o, &v) in xc_row.iter_mut().zip(x_row) {
                *o = v - mean;
            }
            let var = segment_sum(xc_row) / n as f32; // stand-in for the dot fold, same traffic
            let std_val = (var.abs() + 1e-5).sqrt();
            for (((xn_v, out_v), &xc_v), (&g, &b)) in xn_row
                .iter_mut()
                .zip(out_row.iter_mut())
                .zip(xc_row.iter())
                .zip(gamma.iter().zip(beta))
            {
                *xn_v = xc_v / std_val;
                *out_v = *xn_v * g + b;
            }
        }
    };
    if parallel {
        xc.par_chunks_mut(chunk)
            .zip(xn.par_chunks_mut(chunk))
            .zip(out.par_chunks_mut(chunk))
            .zip(x.par_chunks(chunk))
            .for_each(task);
    } else {
        xc.chunks_mut(chunk)
            .zip(xn.chunks_mut(chunk))
            .zip(out.chunks_mut(chunk))
            .zip(x.chunks(chunk))
            .for_each(task);
    }
}

/// The trailing-axis LayerNorm forward pass: forced serial vs forced parallel of the same
/// fused row sweep (per-row bits are scheduling-invariant, so the gate is a pure performance
/// knob). Spans transformer-scale shapes plus wide-row and narrow-row extremes
pub fn calibrate_ln_row_pass() -> Section {
    let mut rows = Vec::new();
    for &(r, n) in &[
        (64usize, 256usize),
        (128, 512),
        (512, 512),
        (2_048, 512),
        (64, 16_384),
        (32_768, 32),
        (16_384, 768),
    ] {
        let x = random_matrix(r, n, 109);
        let xs = x.as_slice().unwrap();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut bufs = (
            vec![0.0f32; r * n],
            vec![0.0f32; r * n],
            vec![0.0f32; r * n],
        );
        let s = time_per_call_ns(|| {
            bench_ln_row_pass(xs, n, &gamma, &beta, false, &mut bufs);
            black_box(&bufs.2);
        });
        let p = time_per_call_ns(|| {
            bench_ln_row_pass(xs, n, &gamma, &beta, true, &mut bufs);
            black_box(&bufs.2);
        });
        rows.push(Row {
            label: format!("R={r} N={n}"),
            work: r * n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "LayerNorm fused row pass, trailing axis (LN_ROW_PARALLEL_MIN_ELEMS)",
        work_unit: "elements (R x N)",
        pick_fastest: false,
        rows,
    }
}
