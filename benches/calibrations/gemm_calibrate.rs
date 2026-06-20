//! Calibrates the serial-`matrixmultiply`-vs-`gemm`-crate crossover that
//! `MatmulElem::GEMM_CRATE_MIN_FLOPS` should sit at.
//!
//! For each shape it times two backends and prints `serial/gemm` (>1 means the `gemm` crate is
//! faster, so the product belongs above the threshold):
//!   - serial : `a.dot(&b)` (ndarray -> single-threaded `matrixmultiply`)
//!   - gemm   : `gemm::gemm(..., Parallelism::Rayon)`, the exact call `gemm_internal` makes
//!
//! ```bash
//! cargo bench --bench gemm_calibrate --features math
//! ```
//! harness = false: prints a table to stdout.

use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::{Array2, Axis};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::hint::black_box;
use std::time::Instant;

/// Minimum rows per block for the row-split path (mirrors `matmul::PAR_ROWSPLIT_MIN_BLOCK`).
const PAR_ROWSPLIT_MIN_BLOCK: usize = 8;

fn time_ns<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..8 {
        f();
    }
    let mut inner = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..inner {
            f();
        }
        let el = t.elapsed().as_secs_f64();
        if el >= 0.05 || inner >= 1 << 30 {
            break;
        }
        inner = (inner * 2).max(((0.05 / el.max(1e-9)) as usize).max(inner + 1));
    }
    let mut total = 0.0;
    let n = 12;
    for _ in 0..n {
        let t = Instant::now();
        for _ in 0..inner {
            f();
        }
        total += t.elapsed().as_secs_f64() / inner as f64;
    }
    (total / n as f64) * 1e9
}

/// One `gemm` crate call producing a fresh standard-layout `[m, n]` array (mirrors `gemm_internal`).
macro_rules! gemm_crate {
    ($name:ident, $t:ty) => {
        fn $name(a: &Array2<$t>, b: &Array2<$t>, par: gemm::Parallelism) -> Array2<$t> {
            let (m, k) = a.dim();
            let n = b.ncols();
            let mut out = Array2::<$t>::zeros((m, n));
            let (as_, bs) = (a.strides(), b.strides());
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    out.as_mut_ptr(),
                    1,
                    n as isize,
                    false,
                    a.as_ptr(),
                    as_[1],
                    as_[0],
                    b.as_ptr(),
                    bs[1],
                    bs[0],
                    0.0,
                    1.0,
                    false,
                    false,
                    false,
                    par,
                );
            }
            out
        }
    };
}
gemm_crate!(gemm_crate_f32, f32);
gemm_crate!(gemm_crate_f64, f64);

/// Row-split over rayon, each block a serial `gemm` call (mirrors `matmul::gemm_rowsplit`).
/// The block kernel runs directly on the row-block view (no copy), exactly like the real code.
macro_rules! gemm_rowsplit {
    ($name:ident, $t:ty) => {
        fn $name(a: &Array2<$t>, b: &Array2<$t>) -> Array2<$t> {
            let (m, n) = (a.nrows(), b.ncols());
            let k = a.ncols();
            let threads = rayon::current_num_threads();
            let chunk = m.div_ceil(threads.max(1)).max(PAR_ROWSPLIT_MIN_BLOCK);
            let mut out = Array2::<$t>::zeros((m, n));
            if chunk >= m {
                let (as_, bs) = (a.strides(), b.strides());
                unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        out.as_mut_ptr(),
                        1,
                        n as isize,
                        false,
                        a.as_ptr(),
                        as_[1],
                        as_[0],
                        b.as_ptr(),
                        bs[1],
                        bs[0],
                        0.0,
                        1.0,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                return out;
            }
            let bv = b.view();
            out.axis_chunks_iter_mut(Axis(0), chunk)
                .into_par_iter()
                .zip(a.axis_chunks_iter(Axis(0), chunk).into_par_iter())
                .for_each(|(mut c_blk, a_blk)| {
                    let mb = a_blk.nrows();
                    let (as_, bs) = (a_blk.strides(), bv.strides());
                    let (cs0, cs1) = (c_blk.strides()[0], c_blk.strides()[1]);
                    unsafe {
                        gemm::gemm(
                            mb,
                            n,
                            k,
                            c_blk.as_mut_ptr(),
                            cs1,
                            cs0,
                            false,
                            a_blk.as_ptr(),
                            as_[1],
                            as_[0],
                            bv.as_ptr(),
                            bs[1],
                            bs[0],
                            0.0,
                            1.0,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                });
            out
        }
    };
}
gemm_rowsplit!(gemm_rowsplit_f32, f32);
gemm_rowsplit!(gemm_rowsplit_f64, f64);

/// Row-split capped at `max_blocks` rayon tasks (instead of `current_num_threads()`). GEMV is
/// bandwidth-bound, so its speedup plateaus once enough cores saturate memory bandwidth; this lets
/// the calibration sweep the block count to find that knee (C2: cap the GEMV split).
macro_rules! gemm_rowsplit_cap {
    ($name:ident, $t:ty) => {
        fn $name(a: &Array2<$t>, b: &Array2<$t>, max_blocks: usize) -> Array2<$t> {
            let (m, n) = (a.nrows(), b.ncols());
            let k = a.ncols();
            let blocks = rayon::current_num_threads().min(max_blocks).max(1);
            let chunk = m.div_ceil(blocks).max(PAR_ROWSPLIT_MIN_BLOCK);
            let mut out = Array2::<$t>::zeros((m, n));
            if chunk >= m {
                let (as_, bs) = (a.strides(), b.strides());
                unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        out.as_mut_ptr(),
                        1,
                        n as isize,
                        false,
                        a.as_ptr(),
                        as_[1],
                        as_[0],
                        b.as_ptr(),
                        bs[1],
                        bs[0],
                        0.0,
                        1.0,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                return out;
            }
            let bv = b.view();
            out.axis_chunks_iter_mut(Axis(0), chunk)
                .into_par_iter()
                .zip(a.axis_chunks_iter(Axis(0), chunk).into_par_iter())
                .for_each(|(mut c_blk, a_blk)| {
                    let mb = a_blk.nrows();
                    let (as_, bs) = (a_blk.strides(), bv.strides());
                    let (cs0, cs1) = (c_blk.strides()[0], c_blk.strides()[1]);
                    unsafe {
                        gemm::gemm(
                            mb,
                            n,
                            k,
                            c_blk.as_mut_ptr(),
                            cs1,
                            cs0,
                            false,
                            a_blk.as_ptr(),
                            as_[1],
                            as_[0],
                            bv.as_ptr(),
                            bs[1],
                            bs[0],
                            0.0,
                            1.0,
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                });
            out
        }
    };
}
gemm_rowsplit_cap!(gemm_rowsplit_cap_f64, f64);

fn rand_f32(r: usize, c: usize, s: u64) -> Array2<f32> {
    Array2::from_shape_fn((r, c), |(i, j)| {
        (((s as f64) * 0.731 + (i * c + j) as f64 * 0.618).sin() * 43758.5).fract() as f32 - 0.5
    })
}
fn rand_f64(r: usize, c: usize, s: u64) -> Array2<f64> {
    Array2::from_shape_fn((r, c), |(i, j)| {
        ((s as f64) * 0.731 + (i * c + j) as f64 * 0.618).sin() * 43758.5 % 1.0 - 0.5
    })
}

fn main() {
    println!(
        "# gemm-crate vs serial matrixmultiply crossover (rayon threads: {})",
        rayon::current_num_threads()
    );
    // (label, m, k, n): a square ladder (clean crossover) plus the real skinny call patterns
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("sq_32", 32, 32, 32),
        ("sq_48", 48, 48, 48),
        ("sq_64", 64, 64, 64),
        ("sq_96", 96, 96, 96),
        ("sq_128", 128, 128, 128),
        ("sq_160", 160, 160, 160),
        ("sq_192", 192, 192, 192),
        ("sq_224", 224, 224, 224),
        ("sq_256", 256, 256, 256),
        ("sq_320", 320, 320, 320),
        ("sq_384", 384, 384, 384),
        ("sq_512", 512, 512, 512),
        ("lstm_32x128x512", 32, 128, 512),
        ("lstm_32x64x512", 32, 64, 512),
        // MLP training-loop GEMMs (the 0.86x regression): the three ~33.5M-flop products per step.
        ("mlp_fwd_512x256x128", 512, 256, 128), // L1 forward          n=128
        ("mlp_dx_512x128x256", 512, 128, 256),  // L1 grad-input dX    n=256
        ("mlp_dw_256x512x128", 256, 512, 128),  // L1 grad-weight dW   n=128
        ("mlp_512x128x10", 512, 128, 10),       // L2 (small/thin)
        // medium-band ladder around the 33.5M point, to find the row-split <-> column-par crossover
        ("med_512x512x128", 512, 512, 128), // 67M    n=128
        ("med_512x512x256", 512, 512, 256), // 134M   n=256
        ("dense_256x784x512", 256, 784, 512),
    ];

    let none = gemm::Parallelism::None;
    let threads = rayon::current_num_threads();
    let rayon_par = gemm::Parallelism::Rayon(threads);

    for dtype in ["f32", "f64"] {
        // serial_mm: ndarray dot; g_none: gemm serial; g_par: gemm on rayon. The threshold that
        // matters is the g_none -> g_par crossover (same kernel, only rayon dispatch differs).
        // serial_mm: ndarray dot; gnone: gemm serial; grow: rayon row-split of serial gemm blocks
        // (HEAD-style); gpar: gemm-crate column parallel. Best of {gnone,grow,gpar} per row marked.
        println!(
            "\n{:<22} {:>12} {:>10} {:>10} {:>10} {:>10}  best",
            format!("shape ({dtype})"),
            "flops",
            "serial_us",
            "gnone_us",
            "grow_us",
            "gpar_us",
        );
        for &(label, m, k, n) in shapes {
            let flops = 2 * m * k * n;
            let (serial, gn, gr, gp) = if dtype == "f32" {
                let a = rand_f32(m, k, 1);
                let b = rand_f32(k, n, 2);
                (
                    time_ns(|| {
                        black_box(a.dot(&b));
                    }),
                    time_ns(|| {
                        black_box(gemm_crate_f32(&a, &b, none));
                    }),
                    time_ns(|| {
                        black_box(gemm_rowsplit_f32(&a, &b));
                    }),
                    time_ns(|| {
                        black_box(gemm_crate_f32(&a, &b, rayon_par));
                    }),
                )
            } else {
                let a = rand_f64(m, k, 1);
                let b = rand_f64(k, n, 2);
                (
                    time_ns(|| {
                        black_box(a.dot(&b));
                    }),
                    time_ns(|| {
                        black_box(gemm_crate_f64(&a, &b, none));
                    }),
                    time_ns(|| {
                        black_box(gemm_rowsplit_f64(&a, &b));
                    }),
                    time_ns(|| {
                        black_box(gemm_crate_f64(&a, &b, rayon_par));
                    }),
                )
            };
            let best = if gn <= gr && gn <= gp {
                "gnone"
            } else if gr <= gp {
                "grow"
            } else {
                "gpar"
            };
            println!(
                "{label:<22} {flops:>12} {:>10.3} {:>10.3} {:>10.3} {:>10.3}  {best}",
                serial / 1e3,
                gn / 1e3,
                gr / 1e3,
                gp / 1e3,
            );
        }
    }

    // GEMV (n = 1): the `gemm` crate never parallelizes a matvec, so the only parallel option is the
    // row split (`gemv_par_strategy`). The decision that matters is gnone (serial, one gemm call =
    // `gemv_par_switch(false)`) vs grow (row-split = `gemv_par_switch(true)`). The current gate keys
    // on raw flops `2*m*k`, but the splittable axis is `m`: short-output shapes (small `m`, large
    // `k`) clear the flop gate yet split into tiny-work blocks. This sweep finds the real `m`-driven
    // crossover. `none/row > 1` means the row split wins (belongs on the parallel path).
    let _ = rayon_par; // GEMV has no useful column-parallel variant; keep the GEMM section's binding
    let gemv_shapes: &[(&str, usize, usize)] = &[
        // tall regime (k = 64): sweep m to find where the row split starts to pay
        ("tall_k64_m1k", 1_000, 64),
        ("tall_k64_m2k", 2_000, 64),
        ("tall_k64_m5k", 5_000, 64),
        ("tall_k64_m10k", 10_000, 64),
        ("tall_k64_m50k", 50_000, 64), // logistic predict
        ("tall_k64_m200k", 200_000, 64),
        ("tall_k64_m500k", 500_000, 64),
        // short-output regime (k = 50000): sweep m - does the split EVER beat serial here?
        ("short_k50k_m64", 64, 50_000), // logistic / linear_reg gradient (X^T . e)
        ("short_k50k_m128", 128, 50_000),
        ("short_k50k_m256", 256, 50_000),
        ("short_k50k_m512", 512, 50_000),
        ("short_k50k_m1k", 1_024, 50_000),
        ("short_k50k_m2k", 2_048, 50_000),
        // mid / square real call shapes
        ("svc_2000x750", 2_000, 750), // svc decision_values_batch
        ("sq_512", 512, 512),
        ("sq_1024", 1_024, 1_024),
        ("sq_1500", 1_500, 1_500), // kernel_pca / svc eigensolver matvec
        ("sq_2048", 2_048, 2_048),
    ];
    for dtype in ["f32", "f64"] {
        println!(
            "\n{:<20} {:>12} {:>8} {:>10} {:>10} {:>10}  best",
            format!("GEMV ({dtype})"),
            "flops",
            "m",
            "gnone_us",
            "grow_us",
            "none/row",
        );
        for &(label, m, k) in gemv_shapes {
            let flops = 2 * m * k;
            let (gn, gr) = if dtype == "f32" {
                let a = rand_f32(m, k, 1);
                let b = rand_f32(k, 1, 2);
                (
                    time_ns(|| {
                        black_box(gemm_crate_f32(&a, &b, none));
                    }),
                    time_ns(|| {
                        black_box(gemm_rowsplit_f32(&a, &b));
                    }),
                )
            } else {
                let a = rand_f64(m, k, 1);
                let b = rand_f64(k, 1, 2);
                (
                    time_ns(|| {
                        black_box(gemm_crate_f64(&a, &b, none));
                    }),
                    time_ns(|| {
                        black_box(gemm_rowsplit_f64(&a, &b));
                    }),
                )
            };
            let best = if gn <= gr { "gnone" } else { "grow" };
            println!(
                "{label:<20} {flops:>12} {m:>8} {:>10.3} {:>10.3} {:>10.2}  {best}",
                gn / 1e3,
                gr / 1e3,
                gn / gr
            );
        }
    }

    // C2 - GEMV row-split block-count sweep (f64): how many rayon blocks before bandwidth
    // saturates? Each column is the row-split capped at that many blocks; `1` is serial. The knee
    // (where adding blocks stops helping) is the bandwidth-saturation core count - capping the real
    // gemv_par_strategy there would shed fork overhead the extra blocks only add.
    let caps: &[usize] = &[1, 2, 4, 8, 16, 32];
    let cap_shapes: &[(&str, usize, usize)] = &[
        ("tall_k64_m50k", 50_000, 64),
        ("tall_k64_m500k", 500_000, 64),
        ("short_k50k_m256", 256, 50_000),
        ("svc_2000x750", 2_000, 750),
        ("sq_1500", 1_500, 1_500),
    ];
    print!("\n{:<18} {:>10}", "GEMV cap (f64)", "");
    for c in caps {
        print!(" {:>9}", format!("b{c}_us"));
    }
    println!("   best_b");
    for &(label, m, k) in cap_shapes {
        let a = rand_f64(m, k, 1);
        let b = rand_f64(k, 1, 2);
        let mut times = Vec::with_capacity(caps.len());
        for &cap in caps {
            times.push(time_ns(|| {
                black_box(gemm_rowsplit_cap_f64(&a, &b, cap));
            }));
        }
        let best_i = (0..times.len())
            .min_by(|&x, &y| times[x].partial_cmp(&times[y]).unwrap())
            .unwrap();
        print!("{label:<18} {:>10}", "");
        for t in &times {
            print!(" {:>9.3}", t / 1e3);
        }
        println!("   b{}", caps[best_i]);
    }
}
