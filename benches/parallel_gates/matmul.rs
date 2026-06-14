//! Dense linear-algebra gates: GEMM and GEMV FLOP/block thresholds (f32 and f64), plus the two
//! GEMM-strategy shootouts (tiled-GEMM chunk budget and the pairwise-distance strategy) that pick
//! between `gemm`, a GEMV swarm, and per-pair scalar loops

use crate::harness::{
    Row, Section, random_matrix, random_matrix_f64, random_vector_f32, random_vector_f64,
    time_per_call_ns,
};
use rayon::prelude::*;
use rustyml::math::matmul::{gemm, gemm_par, gemv_par};
use std::hint::black_box;

/// Mirrors the crate-internal `MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS` so the strategy
/// shootouts emulate the production gating (the trait itself is not public)
const GEMM_F64_GATE: usize = 2_000_000;

// ---- gemm: PAR_GEMM_MIN_FLOPS ----

pub fn calibrate_par_matmul_flops() -> Section {
    let mut rows = Vec::new();
    // Square-ish ladder
    for &n in &[32usize, 48, 64, 96, 128, 192, 256, 384, 512] {
        let a = random_matrix(n, n, 1);
        let b = random_matrix(n, n, 2);
        let s = time_per_call_ns(|| {
            black_box(a.dot(&b));
        });
        let p = time_per_call_ns(|| {
            black_box(gemm_par(&a, &b, 1));
        });
        rows.push(Row {
            label: format!("square {n}x{n}x{n}"),
            work: 2 * n * n * n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    // Skinny ladders: tall-A (projection), wide-B (input grad), huge-k (weight reduction)
    for &(m, k, n) in &[
        (256usize, 64usize, 64usize),
        (1024, 64, 64),
        (4096, 64, 64),
        (16384, 64, 64),
        (64, 64, 1024),
        (64, 64, 16384),
        (64, 4096, 64),
        (64, 65536, 64),
    ] {
        let a = random_matrix(m, k, 3);
        let b = random_matrix(k, n, 4);
        let s = time_per_call_ns(|| {
            black_box(a.dot(&b));
        });
        let p = time_per_call_ns(|| {
            black_box(gemm_par(&a, &b, 1));
        });
        rows.push(Row {
            label: format!("skinny {m}x{k}x{n}"),
            work: 2 * m * k * n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "gemm FLOPs gate (PAR_GEMM_MIN_FLOPS)",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

// ---- gemm: PAR_GEMM_MIN_BLOCK ----

pub fn calibrate_par_matmul_min_block() -> Section {
    let a = random_matrix(2048, 512, 5);
    let b = random_matrix(512, 512, 6);
    let serial = time_per_call_ns(|| {
        black_box(a.dot(&b));
    });
    let mut rows = Vec::new();
    for &blk in &[8usize, 16, 32, 64, 128, 256, 512] {
        let p = time_per_call_ns(|| {
            black_box(gemm_par(&a, &b, blk));
        });
        rows.push(Row {
            label: format!("2048x512x512 min_block {blk}"),
            work: blk,
            serial_ns: serial,
            parallel_ns: p,
        });
    }
    Section {
        title: "gemm block size (PAR_GEMM_MIN_BLOCK)",
        work_unit: "rows per block",
        pick_fastest: true,
        rows,
    }
}

// ---- f64 GEMM: MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS ----

pub fn calibrate_par_matmul_flops_f64() -> Section {
    let mut rows = Vec::new();
    for &n in &[32usize, 48, 64, 96, 128, 192, 256, 384, 512] {
        let a = random_matrix_f64(n, n, 11);
        let b = random_matrix_f64(n, n, 12);
        let s = time_per_call_ns(|| {
            black_box(a.dot(&b));
        });
        let p = time_per_call_ns(|| {
            black_box(gemm_par(&a, &b, 1));
        });
        rows.push(Row {
            label: format!("f64 square {n}x{n}x{n}"),
            work: 2 * n * n * n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    for &(m, k, n) in &[
        (1024usize, 64usize, 64usize),
        (4096, 64, 64),
        (16384, 64, 64),
        (64, 64, 4096),
        (64, 16384, 64),
    ] {
        let a = random_matrix_f64(m, k, 13);
        let b = random_matrix_f64(k, n, 14);
        let s = time_per_call_ns(|| {
            black_box(a.dot(&b));
        });
        let p = time_per_call_ns(|| {
            black_box(gemm_par(&a, &b, 1));
        });
        rows.push(Row {
            label: format!("f64 skinny {m}x{k}x{n}"),
            work: 2 * m * k * n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "f64 gemm FLOPs gate (MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS)",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

pub fn calibrate_par_matmul_min_block_f64() -> Section {
    let a = random_matrix_f64(2048, 512, 15);
    let b = random_matrix_f64(512, 512, 16);
    let serial = time_per_call_ns(|| {
        black_box(a.dot(&b));
    });
    let mut rows = Vec::new();
    for &blk in &[8usize, 16, 32, 64, 128, 256, 512] {
        let p = time_per_call_ns(|| {
            black_box(gemm_par(&a, &b, blk));
        });
        rows.push(Row {
            label: format!("f64 2048x512x512 min_block {blk}"),
            work: blk,
            serial_ns: serial,
            parallel_ns: p,
        });
    }
    Section {
        title: "f64 gemm block size (PAR_GEMM_MIN_BLOCK check)",
        work_unit: "rows per block",
        pick_fastest: true,
        rows,
    }
}

// ---- matvec: MatmulElem::PAR_GEMV_MIN_FLOPS (f32 + f64) ----

/// `(m, k)` ladder shared by both element types: square-ish plus the tall (`X . w`) and
/// short-wide (`X^T . e`) shapes the linear models produce
const MATVEC_SHAPES: &[(usize, usize)] = &[
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (16384, 64),
    (65536, 64),
    (262144, 64),
    (256, 16384),
    (128, 65536),
];

pub fn calibrate_par_matvec_flops_f64() -> Section {
    let mut rows = Vec::new();
    for &(m, k) in MATVEC_SHAPES {
        let a = random_matrix_f64(m, k, 21);
        let x = random_vector_f64(k, 22);
        let s = time_per_call_ns(|| {
            black_box(a.dot(&x));
        });
        let p = time_per_call_ns(|| {
            black_box(gemv_par(&a, &x, 1));
        });
        rows.push(Row {
            label: format!("f64 matvec {m}x{k}"),
            work: 2 * m * k,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "f64 matvec FLOPs gate (MatmulElem::<f64>::PAR_GEMV_MIN_FLOPS)",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

pub fn calibrate_par_matvec_flops_f32() -> Section {
    let mut rows = Vec::new();
    for &(m, k) in MATVEC_SHAPES {
        let a = random_matrix(m, k, 23);
        let x = random_vector_f32(k, 24);
        let s = time_per_call_ns(|| {
            black_box(a.dot(&x));
        });
        let p = time_per_call_ns(|| {
            black_box(gemv_par(&a, &x, 1));
        });
        rows.push(Row {
            label: format!("f32 matvec {m}x{k}"),
            work: 2 * m * k,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "f32 matvec FLOPs gate (MatmulElem::<f32>::PAR_GEMV_MIN_FLOPS)",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

pub fn calibrate_par_matvec_min_block() -> Vec<Section> {
    let mut sections = Vec::new();
    // Tall (the X . w projection) and short-wide (the X^T . e reduction over few features) -
    // the wide case is where a high row floor forfeits parallelism entirely
    for &(m, k, tag) in &[(262144usize, 64usize, "tall"), (128, 65536, "short-wide")] {
        let a = random_matrix_f64(m, k, 25);
        let x = random_vector_f64(k, 26);
        let serial = time_per_call_ns(|| {
            black_box(a.dot(&x));
        });
        let mut rows = Vec::new();
        for &blk in &[1usize, 2, 4, 8, 16, 32, 64, 128] {
            let p = time_per_call_ns(|| {
                black_box(gemv_par(&a, &x, blk));
            });
            rows.push(Row {
                label: format!("f64 {tag} {m}x{k} min_block {blk}"),
                work: blk,
                serial_ns: serial,
                parallel_ns: p,
            });
        }
        sections.push(Section {
            title: match tag {
                "tall" => "f64 matvec block floor, tall shape (PAR_GEMV_MIN_BLOCK)",
                _ => "f64 matvec block floor, short-wide shape (PAR_GEMV_MIN_BLOCK)",
            },
            work_unit: "rows per block",
            pick_fastest: true,
            rows,
        });
    }
    sections
}

// ---- tiled-GEMM chunk budget: matmul::GEMM_CHUNK_ELEMS ----

pub fn calibrate_gemm_chunk_budget() -> Vec<Section> {
    let mut sections = Vec::new();
    // KNN-predict-shaped workloads: queries against a training set, distance scan per row
    // Baseline ("serial" column) is the pre-rewrite path: one GEMV per query, parallel over
    // queries. The sweep finds the chunk budget for the tiled-GEMM replacement
    // The 50k/200k training sets (~25 MB) fit in the 9950X's 64 MB L3, where the GEMV swarm
    // re-reads X from cache for free; the 500k set (256 MB) overflows it - the case tiling
    // exists for
    for &(n_train, d, n_query, tag) in &[
        (50_000usize, 64usize, 2048usize, "50k train, d=64"),
        (200_000, 16, 1024, "200k train, d=16"),
        (500_000, 64, 128, "500k train, d=64 (L3 overflow)"),
    ] {
        let x_train = random_matrix_f64(n_train, d, 41);
        let queries = random_matrix_f64(n_query, d, 42);
        let train_sq: Vec<f64> = x_train.rows().into_iter().map(|r| r.dot(&r)).collect();

        // Pre-rewrite baseline: per-query GEMV swarm
        let baseline = time_per_call_ns(|| {
            let nearest: Vec<usize> = (0..n_query)
                .into_par_iter()
                .map(|qi| {
                    let q = queries.row(qi);
                    let proj = x_train.dot(&q);
                    let mut best = 0;
                    let mut best_val = f64::MAX;
                    for (j, &p) in proj.iter().enumerate() {
                        let dist = train_sq[j] - 2.0 * p;
                        if dist < best_val {
                            best_val = dist;
                            best = j;
                        }
                    }
                    best
                })
                .collect();
            black_box(nearest);
        });

        let mut rows = Vec::new();
        for &chunk_rows in &[16usize, 64, 256, 1024, 4096] {
            let chunk_rows = chunk_rows.min(n_query);
            let tiled = time_per_call_ns(|| {
                let mut nearest: Vec<usize> = Vec::with_capacity(n_query);
                for start in (0..n_query).step_by(chunk_rows) {
                    let end = (start + chunk_rows).min(n_query);
                    let proj = gemm(
                        &queries.slice(ndarray::s![start..end, ..]),
                        &x_train.t(),
                        GEMM_F64_GATE,
                    );
                    let chunk: Vec<usize> = (start..end)
                        .into_par_iter()
                        .map(|qi| {
                            let row = proj.row(qi - start);
                            let mut best = 0;
                            let mut best_val = f64::MAX;
                            for (j, &p) in row.iter().enumerate() {
                                let dist = train_sq[j] - 2.0 * p;
                                if dist < best_val {
                                    best_val = dist;
                                    best = j;
                                }
                            }
                            best
                        })
                        .collect();
                    nearest.extend(chunk);
                }
                black_box(nearest);
            });
            rows.push(Row {
                label: format!("chunk {chunk_rows} rows ({tag})"),
                work: chunk_rows * n_train,
                serial_ns: baseline,
                parallel_ns: tiled,
            });
        }
        sections.push(Section {
            title: match tag {
                "50k train, d=64" => {
                    "tiled-GEMM chunk budget, 50k train / d=64 / 2048 queries (GEMM_CHUNK_ELEMS)"
                }
                "200k train, d=16" => {
                    "tiled-GEMM chunk budget, 200k train / d=16 / 1024 queries (GEMM_CHUNK_ELEMS)"
                }
                _ => "tiled-GEMM chunk budget, 500k train / d=64 / 128 queries, L3 overflow (GEMM_CHUNK_ELEMS)",
            },
            work_unit: "buffer elements (rows x n_train)",
            pick_fastest: true,
            rows,
        });
    }
    sections
}

// ---- pairwise-distance strategy (the t-SNE / mean-shift all-pairs shapes) ----

/// Compares three ways of producing the full per-row distance rows that t-SNE's neighbor
/// search and mean-shift's bandwidth estimation consume: the per-pair scalar loops (the
/// pre-rewrite code), a per-row GEMV swarm with the norms identity, and the one-shot
/// block-parallel GEMM with the same identity. The "serial" column is the per-pair scalar
/// baseline for every row
pub fn calibrate_pairwise_strategy() -> Section {
    let n = 5000usize;
    let d = 50usize;
    let x = random_matrix_f64(n, d, 51);
    let x_sq: Vec<f64> = x.rows().into_iter().map(|r| r.dot(&r)).collect();

    // Strategy 0 (baseline): per-pair scalar distance loops, parallel over rows
    let scalar = time_per_call_ns(|| {
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row_i = x.row(i);
                (0..n)
                    .map(|j| {
                        let row_j = x.row(j);
                        row_i
                            .iter()
                            .zip(row_j.iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect();
        black_box(rows);
    });

    // Strategy 1: per-row GEMV swarm + norms identity, parallel over rows
    let gemv_swarm = time_per_call_ns(|| {
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let proj = x.dot(&x.row(i));
                proj.iter()
                    .zip(x_sq.iter())
                    .map(|(&p, &j_sq)| (x_sq[i] + j_sq - 2.0 * p).max(0.0))
                    .collect()
            })
            .collect();
        black_box(rows);
    });

    // Strategy 2: one-shot block-parallel GEMM + norms identity (the t-SNE exact path)
    let full_gemm = time_per_call_ns(|| {
        let gram = gemm(&x, &x.t(), GEMM_F64_GATE);
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                gram.row(i)
                    .iter()
                    .zip(x_sq.iter())
                    .map(|(&g, &j_sq)| (x_sq[i] + j_sq - 2.0 * g).max(0.0))
                    .collect()
            })
            .collect();
        black_box(rows);
    });

    let rows = vec![
        Row {
            label: format!("per-pair scalar {n}x{n} d={d}"),
            work: 1,
            serial_ns: scalar,
            parallel_ns: scalar,
        },
        Row {
            label: format!("per-row GEMV swarm {n}x{n} d={d}"),
            work: 2,
            serial_ns: scalar,
            parallel_ns: gemv_swarm,
        },
        Row {
            label: format!("one-shot GEMM {n}x{n} d={d}"),
            work: 3,
            serial_ns: scalar,
            parallel_ns: full_gemm,
        },
    ];
    Section {
        title: "pairwise-distance strategy, 5000 points / d=50 (t-SNE / mean-shift shapes)",
        work_unit: "strategy id",
        pick_fastest: true,
        rows,
    }
}
