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
                        m, n, k, out.as_mut_ptr(), 1, n as isize, false,
                        a.as_ptr(), as_[1], as_[0], b.as_ptr(), bs[1], bs[0],
                        0.0, 1.0, false, false, false, gemm::Parallelism::None,
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
                            mb, n, k, c_blk.as_mut_ptr(), cs1, cs0, false,
                            a_blk.as_ptr(), as_[1], as_[0], bv.as_ptr(), bs[1], bs[0],
                            0.0, 1.0, false, false, false, gemm::Parallelism::None,
                        );
                    }
                });
            out
        }
    };
}
gemm_rowsplit!(gemm_rowsplit_f32, f32);
gemm_rowsplit!(gemm_rowsplit_f64, f64);

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
        ("mlp_512x128x10", 512, 128, 10),        // L2 (small/thin)
        // medium-band ladder around the 33.5M point, to find the row-split <-> column-par crossover
        ("med_512x512x128", 512, 512, 128),     // 67M    n=128
        ("med_512x512x256", 512, 512, 256),     // 134M   n=256
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

    // GEMV (n = 1): memory-bound, so the None -> Rayon crossover sits lower than for GEMM.
    let gemv_shapes: &[(&str, usize, usize)] = &[
        ("gv_256x256", 256, 256),
        ("gv_512x512", 512, 512),
        ("gv_1024x1024", 1024, 1024),
        ("gv_1500x1500", 1500, 1500), // kernel_pca / svc eigensolver matvec
        ("gv_2048x2048", 2048, 2048),
        ("gv_4096x4096", 4096, 4096),
        ("gv_50000x64", 50_000, 64), // logistic predict
        ("gv_64x50000", 64, 50_000), // logistic gradient (X^T . e)
        ("gv_500000x64", 500_000, 64),
    ];
    for dtype in ["f32", "f64"] {
        println!(
            "\n{:<20} {:>12} {:>10} {:>10} {:>10}",
            format!("GEMV ({dtype})"),
            "flops",
            "gnone_us",
            "gpar_us",
            "none/par"
        );
        for &(label, m, k) in gemv_shapes {
            let flops = 2 * m * k;
            let (gn, gp) = if dtype == "f32" {
                let a = rand_f32(m, k, 1);
                let b = rand_f32(k, 1, 2);
                (
                    time_ns(|| {
                        black_box(gemm_crate_f32(&a, &b, none));
                    }),
                    time_ns(|| {
                        black_box(gemm_crate_f32(&a, &b, rayon_par));
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
                        black_box(gemm_crate_f64(&a, &b, rayon_par));
                    }),
                )
            };
            println!(
                "{label:<20} {flops:>12} {:>10.3} {:>10.3} {:>10.2}",
                gn / 1e3,
                gp / 1e3,
                gn / gp
            );
        }
    }
}
