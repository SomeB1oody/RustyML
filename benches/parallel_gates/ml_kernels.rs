//! Classical-ML / utils gates: the f64 elementwise classes (centering, kernel transforms,
//! arg-min row scans, reference parallel sums) and the deterministic blocked reductions
//! (block size, exp-heavy log-loss, k-means assign-accumulate, f32->f64 grad-norm square-sum)

use crate::harness::{
    Row, Section, random_matrix_f64, random_vector_f32, random_vector_f64, time_per_call_ns,
};
use ndarray::{Array1, Array2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustyml::math::reduction::{det_reduce, det_reduce_range};
use std::hint::black_box;
use std::ops::AddAssign;

// f64 elementwise classes (the ML/utils gates)

pub fn calibrate_elementwise_f64() -> Vec<Section> {
    let sizes = [
        512usize, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576, 4194304,
    ];
    let mut sections = Vec::new();

    // Cheap map (centering / scaling): converges to 0.5, so in-place reapplication is stable
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut buf = Array1::from_elem(len, 0.25f64);
        let s = time_per_call_ns(|| {
            buf.mapv_inplace(|x| x * 0.9999 + 0.00005);
            black_box(&buf);
        });
        let mut buf = Array1::from_elem(len, 0.25f64);
        let p = time_per_call_ns(|| {
            buf.par_mapv_inplace(|x| x * 0.9999 + 0.00005);
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("f64 center/scale {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "f64 cheap map (centering/normalize/standardize class)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Exp map (the logistic sigmoid, RBF/tanh kernel transforms)
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut buf = Array1::from_elem(len, 0.5f64);
        let s = time_per_call_ns(|| {
            buf.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            black_box(&buf);
        });
        let mut buf = Array1::from_elem(len, 0.5f64);
        let p = time_per_call_ns(|| {
            buf.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("f64 sigmoid {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "f64 exp map (sigmoid / kernel-transform class)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Arg-min row scan (the KMeans/LDA label pick): n rows of k = 16 candidates
    let mut rows = Vec::new();
    let k = 16usize;
    for &n in &[256usize, 1024, 4096, 16384, 65536, 262144] {
        let proj = random_matrix_f64(n, k, 31);
        let scan_row = |i: usize| -> usize {
            let row = proj.row(i);
            let mut best = 0;
            let mut best_val = f64::MAX;
            for (j, &v) in row.iter().enumerate() {
                if v < best_val {
                    best_val = v;
                    best = j;
                }
            }
            best
        };
        let s = time_per_call_ns(|| {
            let labels: Vec<usize> = (0..n).map(scan_row).collect();
            black_box(labels);
        });
        let p = time_per_call_ns(|| {
            let labels: Vec<usize> = (0..n).into_par_iter().map(scan_row).collect();
            black_box(labels);
        });
        rows.push(Row {
            label: format!("f64 argmin {n}x{k}"),
            work: n * k,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "f64 arg-min row scan (KMeans assignment / LDA label-pick class)",
        work_unit: "elements scanned",
        pick_fastest: false,
        rows,
    });

    // Parallel f64 sum (reference only: rayon's reduction grouping is scheduling-dependent, so
    // a parallel float sum is not bitwise reproducible - these sites get serialized unless the
    // win is overwhelming and a deterministic blocked reduction is used instead)
    let mut rows = Vec::new();
    for &len in &sizes {
        let buf = random_vector_f64(len, 33);
        let s = time_per_call_ns(|| {
            black_box(buf.iter().map(|v| v * v).sum::<f64>());
        });
        let slice = buf.as_slice().unwrap();
        let p = time_per_call_ns(|| {
            black_box(slice.par_iter().map(|v| v * v).sum::<f64>());
        });
        rows.push(Row {
            label: format!("f64 sum of squares {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "f64 parallel sum (reference only: non-deterministic reduction order)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    sections
}

// deterministic blocked reduction: DET_REDUCE_BLOCK

pub fn calibrate_det_reduce_block() -> Section {
    let data = random_vector_f64(4_194_304, 69);
    let slice = data.as_slice().unwrap();
    let serial = time_per_call_ns(|| {
        black_box(slice.iter().map(|v| v * v).sum::<f64>());
    });
    let mut rows = Vec::new();
    for &block in &[1024usize, 2048, 4096, 8192, 16384, 32768, 65536, 262144] {
        let p = time_per_call_ns(|| {
            let parts: Vec<f64> = slice
                .par_chunks(block)
                .map(|c| c.iter().map(|v| v * v).sum::<f64>())
                .collect();
            black_box(parts.into_iter().fold(0.0, |a, b| a + b));
        });
        rows.push(Row {
            label: format!("4.2M sum-of-squares, block {block}"),
            work: block,
            serial_ns: serial,
            parallel_ns: p,
        });
    }
    Section {
        title: "deterministic blocked reduction block size (DET_REDUCE_BLOCK)",
        work_unit: "elements per block",
        pick_fastest: true,
        rows,
    }
}

// exp-heavy reduction: EXP_REDUCE_MIN_ELEMS (math::logistic_loss)

/// Per-element work is the numerically stable log-loss term (`exp` + `ln` + a few flops), an
/// order of magnitude heavier than the cheap sum-of-squares class, so its crossover sits well
/// below SUM_F64_PARALLEL_MIN_ELEMS. The parallel side is the deterministic blocked range fold
/// the production path uses
pub fn calibrate_exp_reduction() -> Section {
    let max_n = 1_048_576usize;
    let logits = random_vector_f64(max_n, 81).mapv(|v| v * 4.0);
    let labels = random_vector_f64(max_n, 82).mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    let logit_slice = logits.as_slice().unwrap();
    let label_slice = labels.as_slice().unwrap();

    let loss_term = |x: f64, y: f64| x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln();

    let mut rows = Vec::new();
    for &n in &[
        8_192usize, 16_384, 32_768, 65_536, 131_072, 262_144, 1_048_576,
    ] {
        let s = time_per_call_ns(|| {
            black_box(
                logit_slice[..n]
                    .iter()
                    .zip(label_slice[..n].iter())
                    .map(|(&x, &y)| loss_term(x, y))
                    .sum::<f64>(),
            );
        });
        let p = time_per_call_ns(|| {
            black_box(det_reduce_range(
                n,
                true,
                |range| {
                    range
                        .map(|i| loss_term(logit_slice[i], label_slice[i]))
                        .sum::<f64>()
                },
                |a, b| a + b,
                0.0,
            ));
        });
        rows.push(Row {
            label: format!("logistic loss, n={n}"),
            work: n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "exp-heavy f64 reduction (EXP_REDUCE_MIN_ELEMS: logistic loss)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    }
}

// k-means assign-accumulate: SUM gate on samples x features

/// The per-iteration k-means pass that folds every sample's row into its cluster's centroid sum
/// (plus counts and inertia). Serial is the production scatter loop; parallel is the
/// deterministic blocked range fold with an (Array2 sums, counts, inertia) accumulator per
/// block. Work metric is samples x features - the d=8 rungs land in the same bracket as the
/// d=32 rungs, which is what justifies gating on the product
pub fn calibrate_kmeans_accumulate() -> Section {
    let mut rows = Vec::new();
    for &(n, d, k) in &[
        (2_048usize, 32usize, 16usize),
        (4_096, 32, 16),
        (8_192, 32, 16),
        (16_384, 32, 16),
        (65_536, 32, 16),
        (8_192, 8, 8),
        (32_768, 8, 8),
        (131_072, 8, 8),
    ] {
        let data = random_matrix_f64(n, d, 91);
        let mut rng = StdRng::seed_from_u64(92);
        let results: Vec<(usize, f64)> = (0..n)
            .map(|_| (rng.random_range(0..k), rng.random_range(0.0..2.0)))
            .collect();

        let s = time_per_call_ns(|| {
            let mut sums = Array2::<f64>::zeros((k, d));
            let mut counts = vec![0usize; k];
            let mut inertia = 0.0;
            for (i, &(cluster, dist)) in results.iter().enumerate() {
                inertia += dist;
                sums.row_mut(cluster).add_assign(&data.row(i));
                counts[cluster] += 1;
            }
            black_box((sums, counts, inertia));
        });
        let p = time_per_call_ns(|| {
            let folded = det_reduce_range(
                n,
                true,
                |range| {
                    let mut sums = Array2::<f64>::zeros((k, d));
                    let mut counts = vec![0usize; k];
                    let mut inertia = 0.0;
                    for i in range {
                        let (cluster, dist) = results[i];
                        inertia += dist;
                        sums.row_mut(cluster).add_assign(&data.row(i));
                        counts[cluster] += 1;
                    }
                    (sums, counts, inertia)
                },
                |(mut sa, mut ca, ia), (sb, cb, ib)| {
                    sa += &sb;
                    for (a, b) in ca.iter_mut().zip(cb) {
                        *a += b;
                    }
                    (sa, ca, ia + ib)
                },
                (Array2::<f64>::zeros((k, d)), vec![0usize; k], 0.0),
            );
            black_box(folded);
        });
        rows.push(Row {
            label: format!("n={n} d={d} k={k}"),
            work: n * d,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "k-means assign-accumulate (SUM gate on samples x features)",
        work_unit: "samples x features",
        pick_fastest: false,
        rows,
    }
}

// f32 -> f64 widening square-sum: SQ_SUM_F32_PARALLEL_MIN_ELEMS (global_grad_norm)

/// The clip-by-global-norm reduction: f32 gradients squared and accumulated in f64. Serial
/// baseline is the production flat chain; the parallel side is the generic deterministic
/// blocked fold over f32 blocks with f64 partials
pub fn calibrate_f32_sq_sum() -> Section {
    let data = random_vector_f32(4_194_304, 101);
    let slice = data.as_slice().unwrap();

    let mut rows = Vec::new();
    for &n in &[
        32_768usize,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
        4_194_304,
    ] {
        let s = time_per_call_ns(|| {
            black_box(
                slice[..n]
                    .iter()
                    .map(|&g| (g as f64) * (g as f64))
                    .sum::<f64>(),
            );
        });
        let p = time_per_call_ns(|| {
            black_box(det_reduce(
                &slice[..n],
                true,
                |block| block.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>(),
                |a, b| a + b,
                0.0,
            ));
        });
        rows.push(Row {
            label: format!("f32 sq-sum (f64 acc), n={n}"),
            work: n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "f32 -> f64 square-sum (SQ_SUM_F32_PARALLEL_MIN_ELEMS: grad norm)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    }
}

/// DET_REDUCE_BLOCK validation on f32 elements (the constant counts elements, so an f32 block
/// is half the bytes of the calibrated f64 one - this confirms 16K still sits on the plateau)
pub fn calibrate_det_reduce_block_f32() -> Section {
    let data = random_vector_f32(4_194_304, 102);
    let slice = data.as_slice().unwrap();
    let serial = time_per_call_ns(|| {
        black_box(slice.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>());
    });
    let mut rows = Vec::new();
    for &block in &[4096usize, 8192, 16384, 32768, 65536] {
        let p = time_per_call_ns(|| {
            let parts: Vec<f64> = slice
                .par_chunks(block)
                .map(|c| c.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>())
                .collect();
            black_box(parts.into_iter().fold(0.0, |a, b| a + b));
        });
        rows.push(Row {
            label: format!("4.2M f32 sq-sum, block {block}"),
            work: block,
            serial_ns: serial,
            parallel_ns: p,
        });
    }
    Section {
        title: "deterministic blocked reduction block size, f32 elements (DET_REDUCE_BLOCK)",
        work_unit: "elements per block",
        pick_fastest: true,
        rows,
    }
}
