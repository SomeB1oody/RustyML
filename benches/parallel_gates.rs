//! Parallel/serial gate calibration for the neural-network kernels
//!
//! Every `*_MIN_FLOPS` / `*_MIN_OPS` / `*_PARALLEL_THRESHOLD` constant in the crate decides when
//! a pass is worth spreading across rayon. This bench times the **forced-serial** and
//! **forced-parallel** implementations of each kernel class across a size ladder, prints the
//! tables, and rewrites `benches/RESULTS.md` with the measurements and the observed crossovers,
//! so the constants can be set from data instead of estimates.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench parallel_gates
//! ```
//!
//! Calibration is machine-specific; the generated report records the CPU and thread count.

use ndarray::{Array1, Array2, Axis, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;
use rustyml::bench_internals::{KdTree, PoolKind, conv_forward_forced, windowed_pool_forward_impl};
use rustyml::math::matmul::{gemm, gemm_par, gemv_par};

/// Mirrors the crate-internal `MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS` so the strategy
/// shootouts emulate the production gating (the trait itself is not public)
const GEMM_F64_GATE: usize = 2_000_000;
use rustyml::math::reduction::{det_reduce, det_reduce_range};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::PaddingType;
use rustyml::types::DistanceCalculationMetric;
use std::fmt::Write as _;
use std::hint::black_box;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

/// Nanoseconds per call of `f`: the batch size grows until one batch takes >= 5 ms, then the
/// minimum over three batches filters scheduler noise
fn time_per_call_ns<F: FnMut()>(mut f: F) -> f64 {
    f(); // warmup
    let mut k = 1usize;
    let batch_ns = loop {
        let t = Instant::now();
        for _ in 0..k {
            f();
        }
        let elapsed = t.elapsed();
        if elapsed >= Duration::from_millis(5) {
            break elapsed.as_nanos() as f64 / k as f64;
        }
        k *= 4;
    };
    let mut best = batch_ns;
    for _ in 0..2 {
        let t = Instant::now();
        for _ in 0..k {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / k as f64);
    }
    best
}

/// One measured ladder rung
struct Row {
    label: String,
    work: usize,
    serial_ns: f64,
    parallel_ns: f64,
}

impl Row {
    fn speedup(&self) -> f64 {
        self.serial_ns / self.parallel_ns
    }
}

/// One calibration table plus the work units its `work` column counts
struct Section {
    title: &'static str,
    work_unit: &'static str,
    /// When true the table is a parameter sweep: report the fastest rung instead of a crossover
    pick_fastest: bool,
    rows: Vec<Row>,
}

impl Section {
    fn print(&self) {
        println!("\n== {} ==", self.title);
        for r in &self.rows {
            println!(
                "{:>28}  work {:>12}  serial {:>10.1} us  parallel {:>10.1} us  speedup {:>5.2}x",
                r.label,
                r.work,
                r.serial_ns / 1e3,
                r.parallel_ns / 1e3,
                r.speedup()
            );
        }
        println!("   -> {}", self.conclusion());
    }

    /// Human-readable takeaway line for the section
    fn conclusion(&self) -> String {
        if self.pick_fastest {
            let best = self
                .rows
                .iter()
                .max_by(|a, b| a.speedup().total_cmp(&b.speedup()))
                .expect("sweep has rows");
            return format!(
                "fastest: {} {} ({:.2}x)",
                best.work,
                self.work_unit,
                best.speedup()
            );
        }
        match self.crossover() {
            Some((0, hi)) => format!(
                "crossover below {hi} {} (parallel wins everywhere)",
                self.work_unit
            ),
            Some((lo, hi)) => format!("crossover between {lo} and {hi} {}", self.work_unit),
            None => "no crossover observed in this ladder".to_string(),
        }
    }

    /// The work bracket where the parallel path starts winning for good: the rung after the
    /// *last* rung (in work order) whose speedup stays within the noise margin of losing.
    /// Requiring 1.05x filters ~1.00x ties that would otherwise read as early crossovers
    fn crossover(&self) -> Option<(usize, usize)> {
        let mut sorted: Vec<&Row> = self.rows.iter().collect();
        sorted.sort_by_key(|r| r.work);
        let last_loss = sorted.iter().rposition(|r| r.speedup() <= 1.05);
        match last_loss {
            None => Some((0, sorted.first()?.work)),
            Some(i) if i + 1 < sorted.len() => Some((sorted[i].work, sorted[i + 1].work)),
            Some(_) => None,
        }
    }

    fn to_markdown(&self, out: &mut String) {
        let _ = writeln!(out, "## {}\n", self.title);
        let _ = writeln!(
            out,
            "| shape | work ({}) | serial (us) | parallel (us) | speedup |",
            self.work_unit
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|");
        for r in &self.rows {
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} | {:.1} | {:.2}x |",
                r.label,
                r.work,
                r.serial_ns / 1e3,
                r.parallel_ns / 1e3,
                r.speedup()
            );
        }
        let _ = writeln!(out, "\n**Takeaway:** {}.\n", self.conclusion());
    }
}

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

fn random_vector_f64(len: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::random_using(len, Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

fn random_vector_f32(len: usize, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::random_using(len, Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

// ---- gemm: PAR_GEMM_MIN_FLOPS ----

fn calibrate_par_matmul_flops() -> Section {
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

fn calibrate_par_matmul_min_block() -> Section {
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

// ---- conv engine: CONV_PARALLEL_MIN_FLOPS ----

fn calibrate_conv_forward() -> Section {
    let mut rows = Vec::new();
    // (cin, filters, img, k) at batch == 1, the case the gate must serve hardest
    for &(cin, f, img, k) in &[
        (3usize, 8usize, 16usize, 3usize),
        (3, 16, 32, 3),
        (8, 16, 32, 3),
        (16, 32, 32, 3),
        (16, 32, 64, 3),
        (32, 64, 64, 3),
        (64, 64, 128, 3),
    ] {
        let input = Tensor::from_elem(IxDyn(&[1, cin, img, img]), 1.0f32);
        let weights = vec![0.5f32; f * cin * k * k];
        let bias = vec![0.0f32; f];
        let out = img - k + 1;
        let flops = 2 * f * out * out * cin * k * k;
        let run = |force: bool| {
            black_box(
                conv_forward_forced(
                    &input,
                    &weights,
                    &[f, cin, k, k],
                    &bias,
                    &[1, 1],
                    PaddingType::Valid,
                    Some(force),
                )
                .unwrap(),
            );
        };
        let s = time_per_call_ns(|| run(false));
        let p = time_per_call_ns(|| run(true));
        rows.push(Row {
            label: format!("conv {cin}c->{f}f {img}px k{k}"),
            work: flops,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "conv engine FLOPs gate (CONV_PARALLEL_MIN_FLOPS), batch == 1",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

// ---- pooling engine: POOL_PARALLEL_MIN_OPS ----

fn calibrate_pooling() -> Section {
    let mut rows = Vec::new();
    let run = |input: &Tensor, force: bool| {
        black_box(windowed_pool_forward_impl(
            input,
            &[2, 2],
            &[2, 2],
            PoolKind::Max,
            PaddingType::Valid,
            Some(force),
        ));
    };
    // Few large planes (batch 1, 3 channels - the case a plane-count gate starves)
    for &img in &[32usize, 64, 128, 256, 512, 1024] {
        let input = Tensor::from_elem(IxDyn(&[1, 3, img, img]), 1.0f32);
        let work = 3 * (img / 2) * (img / 2) * 4; // bc * plane_out * window taps
        let s = time_per_call_ns(|| run(&input, false));
        let p = time_per_call_ns(|| run(&input, true));
        rows.push(Row {
            label: format!("maxpool 1x3x{img}x{img}"),
            work,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    // Many tiny planes
    for &bc in &[16usize, 64, 256, 1024] {
        let input = Tensor::from_elem(IxDyn(&[bc, 1, 16, 16]), 1.0f32);
        let work = bc * 8 * 8 * 4;
        let s = time_per_call_ns(|| run(&input, false));
        let p = time_per_call_ns(|| run(&input, true));
        rows.push(Row {
            label: format!("maxpool {bc}x1x16x16"),
            work,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "pooling ops gate (POOL_PARALLEL_MIN_OPS)",
        work_unit: "window taps",
        pick_fastest: false,
        rows,
    }
}

// ---- elementwise kernel classes: activation / dropout / optimizer thresholds ----

fn calibrate_elementwise() -> Vec<Section> {
    let sizes = [
        512usize, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576,
    ];
    let mut sections = Vec::new();

    // Cheap op (ReLU-like): x.max(0) is idempotent, so in-place reapplication is stable
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut buf = Array1::from_elem(len, 0.5f32);
        let s = time_per_call_ns(|| {
            buf.mapv_inplace(|x| x.max(0.0));
            black_box(&buf);
        });
        let mut buf = Array1::from_elem(len, 0.5f32);
        let p = time_per_call_ns(|| {
            buf.par_mapv_inplace(|x| x.max(0.0));
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("relu-like max(0) {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "relu-like cheap map (CHEAP_MAP_PARALLEL_THRESHOLD class: ReLU, dropout masks)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Exp op (sigmoid-like): repeated sigmoid converges to a benign fixpoint in (0, 1)
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut buf = Array1::from_elem(len, 0.5f32);
        let s = time_per_call_ns(|| {
            buf.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            black_box(&buf);
        });
        let mut buf = Array1::from_elem(len, 0.5f32);
        let p = time_per_call_ns(|| {
            buf.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("sigmoid-like exp {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "sigmoid/tanh-like exp map (EXP_MAP_PARALLEL_THRESHOLD class: sigmoid, tanh, softmax)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Per-element RNG (dropout-like): Bernoulli draw per element; the parallel path mirrors the
    // per-chunk-rng pattern a parallel dropout needs
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut rng = StdRng::seed_from_u64(7);
        let mut buf = vec![0.0f32; len];
        let s = time_per_call_ns(|| {
            for x in buf.iter_mut() {
                *x = if rng.random::<f32>() < 0.5 { 0.0 } else { 2.0 };
            }
            black_box(&buf);
        });
        let mut buf = vec![0.0f32; len];
        let p = time_per_call_ns(|| {
            buf.par_chunks_mut(4096).enumerate().for_each(|(i, chunk)| {
                let mut rng = StdRng::seed_from_u64(7 + i as u64);
                for x in chunk.iter_mut() {
                    *x = if rng.random::<f32>() < 0.5 { 0.0 } else { 2.0 };
                }
            });
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("dropout-like rng {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "dropout-like per-element rng (reference only: the dropout layers keep one serial rng stream for seed reproducibility)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Adam-style fused multi-slice update (optimizer kernels' PARALLEL_THRESHOLD)
    let mut rows = Vec::new();
    for &len in &sizes {
        let grad = vec![0.01f32; len];
        let step = |p: &mut f32, g: f32, m: &mut f32, v: &mut f32| {
            *m = 0.9 * *m + 0.1 * g;
            *v = 0.999 * *v + 0.001 * g * g;
            *p -= 0.001 * *m / (v.sqrt() + 1e-8);
        };
        let mut param = vec![1.0f32; len];
        let mut m = vec![0.0f32; len];
        let mut v = vec![0.0f32; len];
        let s = time_per_call_ns(|| {
            for ((p, &g), (m, v)) in param
                .iter_mut()
                .zip(grad.iter())
                .zip(m.iter_mut().zip(v.iter_mut()))
            {
                step(p, g, m, v);
            }
            black_box(&param);
        });
        let mut param = vec![1.0f32; len];
        let mut m = vec![0.0f32; len];
        let mut v = vec![0.0f32; len];
        let p = time_per_call_ns(|| {
            param
                .par_iter_mut()
                .zip(grad.par_iter())
                .zip(m.par_iter_mut().zip(v.par_iter_mut()))
                .for_each(|((p, &g), (m, v))| step(p, g, m, v));
            black_box(&param);
        });
        rows.push(Row {
            label: format!("adam-like fused {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "adam-like fused slice update (FUSED_SLICE_PARALLEL_THRESHOLD class: optimizer kernels)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    sections
}

// ---- f64 GEMM: MatmulElem::<f64>::PAR_GEMM_MIN_FLOPS ----

fn calibrate_par_matmul_flops_f64() -> Section {
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

fn calibrate_par_matmul_min_block_f64() -> Section {
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

fn calibrate_par_matvec_flops_f64() -> Section {
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

fn calibrate_par_matvec_flops_f32() -> Section {
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

fn calibrate_par_matvec_min_block() -> Vec<Section> {
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

// ---- f64 elementwise classes (the ML/utils gates) ----

fn calibrate_elementwise_f64() -> Vec<Section> {
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
    // win is overwhelming AND a deterministic blocked reduction is used instead)
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

// ---- tiled-GEMM chunk budget: matmul::GEMM_CHUNK_ELEMS ----

fn calibrate_gemm_chunk_budget() -> Vec<Section> {
    let mut sections = Vec::new();
    // KNN-predict-shaped workloads: queries against a training set, distance scan per row.
    // Baseline ("serial" column) is the pre-rewrite path: one GEMV per query, parallel over
    // queries. The sweep finds the chunk budget for the tiled-GEMM replacement
    // The 50k/200k training sets (~25 MB) fit in the 9950X's 64 MB L3, where the GEMV swarm
    // re-reads X from cache for free; the 500k set (256 MB) overflows it, the case tiling
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
fn calibrate_pairwise_strategy() -> Section {
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

// ---- coarse-task classes: tree traversal / sort-scan / tree build ----

/// Synthetic decision-tree predict kernel: per-sample root-to-leaf walk over a heap-layout
/// binary tree (depth 16, ~65K nodes), the same pointer-chasing shape as DecisionTree and
/// IsolationForest prediction
fn calibrate_tree_traversal() -> Section {
    let depth = 16usize;
    let n_nodes = (1usize << depth) - 1;
    let d = 8usize;
    let nodes: Vec<(usize, f64)> = (0..n_nodes)
        .map(|i| {
            let t = i as f64 * 0.618;
            (i % d, (t.sin() * 43758.5453).fract() * 0.4 - 0.2)
        })
        .collect();

    let mut rows = Vec::new();
    for &n in &[64usize, 256, 1024, 4096, 16384, 65536] {
        let x = random_matrix_f64(n, d, 63);
        let walk = |i: usize| -> usize {
            let row = x.row(i);
            let mut node = 0usize;
            while node < n_nodes {
                let (f, thr) = nodes[node];
                node = if row[f] < thr {
                    2 * node + 1
                } else {
                    2 * node + 2
                };
            }
            node
        };
        let s = time_per_call_ns(|| {
            let leaves: Vec<usize> = (0..n).map(walk).collect();
            black_box(leaves);
        });
        let p = time_per_call_ns(|| {
            let leaves: Vec<usize> = (0..n).into_par_iter().map(walk).collect();
            black_box(leaves);
        });
        rows.push(Row {
            label: format!("tree-walk {n} samples, depth {depth}"),
            work: n * depth,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "tree-traversal class (DecisionTree/IsolationForest predict)",
        work_unit: "node visits (samples x depth)",
        pick_fastest: false,
        rows,
    }
}

/// Synthetic split-search kernel: per-feature copy + sort + scan over the node's samples, the
/// shape of DecisionTree's `find_best_split` (one task per feature)
fn calibrate_sort_scan() -> Section {
    let features = 8usize;
    let mut rows = Vec::new();
    for &n in &[64usize, 256, 1024, 4096, 16384] {
        let x = random_matrix_f64(n, features, 65);
        let task = |f: usize| -> f64 {
            let mut col: Vec<f64> = x.column(f).to_vec();
            col.sort_unstable_by(|a, b| a.total_cmp(b));
            // prefix scan standing in for the running-impurity sweep
            let mut acc = 0.0;
            let mut best = f64::MAX;
            for (i, &v) in col.iter().enumerate() {
                acc += v;
                let split_score = (acc / (i + 1) as f64).abs();
                if split_score < best {
                    best = split_score;
                }
            }
            best
        };
        let s = time_per_call_ns(|| {
            let scores: Vec<f64> = (0..features).map(task).collect();
            black_box(scores);
        });
        let p = time_per_call_ns(|| {
            let scores: Vec<f64> = (0..features).into_par_iter().map(task).collect();
            black_box(scores);
        });
        rows.push(Row {
            label: format!("sort-scan {n} samples x {features} features"),
            work: n * features,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "sort-scan class (DecisionTree find_best_split, one task per feature)",
        work_unit: "sorted elements (samples x features)",
        pick_fastest: false,
        rows,
    }
}

/// Synthetic isolation-tree build kernel: each task recursively random-splits a 256-sample
/// subsample (the IsolationForest default), the shape of parallel forest construction
fn calibrate_tree_build() -> Section {
    let psi = 256usize;
    let d = 8usize;
    let x = random_matrix_f64(psi, d, 67);

    fn build_rec(x: &Array2<f64>, idx: &mut [usize], depth: usize, rng: &mut u64) -> usize {
        if idx.len() <= 1 || depth == 0 {
            return idx.len();
        }
        // LCG for the random feature/threshold pick
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = (*rng >> 33) as usize % x.ncols();
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let t = ((*rng >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        let mid = itertools_partition(idx, |&i| x[[i, f]] < t);
        if mid == 0 || mid == idx.len() {
            return idx.len();
        }
        let (l, r) = idx.split_at_mut(mid);
        build_rec(x, l, depth - 1, rng) + build_rec(x, r, depth - 1, rng)
    }

    /// Stable partition returning the split point (no std `partition_point` on unsorted data)
    fn itertools_partition(idx: &mut [usize], pred: impl Fn(&usize) -> bool) -> usize {
        let mut split = 0;
        for i in 0..idx.len() {
            if pred(&idx[i]) {
                idx.swap(split, i);
                split += 1;
            }
        }
        split
    }

    let mut rows = Vec::new();
    for &trees in &[2usize, 4, 8, 16, 32, 64] {
        let build_one = |t: usize| -> usize {
            let mut idx: Vec<usize> = (0..psi).collect();
            let mut rng = 0x9E3779B97F4A7C15u64 ^ (t as u64);
            build_rec(&x, &mut idx, 8, &mut rng)
        };
        let s = time_per_call_ns(|| {
            let sizes: Vec<usize> = (0..trees).map(build_one).collect();
            black_box(sizes);
        });
        let p = time_per_call_ns(|| {
            let sizes: Vec<usize> = (0..trees).into_par_iter().map(build_one).collect();
            black_box(sizes);
        });
        rows.push(Row {
            label: format!("build {trees} trees (psi=256)"),
            work: trees,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "tree-build class (IsolationForest fit, one task per tree)",
        work_unit: "trees",
        pick_fastest: false,
        rows,
    }
}

// ---- deterministic blocked reduction: DET_REDUCE_BLOCK ----

fn calibrate_det_reduce_block() -> Section {
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

// ---- exp-heavy reduction: EXP_REDUCE_MIN_ELEMS (math::logistic_loss) ----

/// Per-element work is the numerically stable log-loss term (`exp` + `ln` + a few flops), an
/// order of magnitude heavier than the cheap sum-of-squares class, so its crossover sits well
/// below SUM_F64_PARALLEL_MIN_ELEMS. The parallel side is the deterministic blocked range fold
/// the production path uses
fn calibrate_exp_reduction() -> Section {
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

// ---- k-means assign-accumulate: SUM gate on samples x features ----

/// The per-iteration k-means pass that folds every sample's row into its cluster's centroid sum
/// (plus counts and inertia). Serial is the production scatter loop; parallel is the
/// deterministic blocked range fold with an (Array2 sums, counts, inertia) accumulator per
/// block. Work metric is samples x features - the d=8 rungs land in the same bracket as the
/// d=32 rungs, which is what justifies gating on the product
fn calibrate_kmeans_accumulate() -> Section {
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

// ---- f32 -> f64 widening square-sum: SQ_SUM_F32_PARALLEL_MIN_ELEMS (global_grad_norm) ----

/// The clip-by-global-norm reduction: f32 gradients squared and accumulated in f64. Serial
/// baseline is the production flat chain; the parallel side is the generic deterministic
/// blocked fold over f32 blocks with f64 partials
fn calibrate_f32_sq_sum() -> Section {
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
fn calibrate_det_reduce_block_f32() -> Section {
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
fn calibrate_bn_col_stats() -> Section {
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
fn calibrate_bn_col_stats_rowblock() -> Section {
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
fn calibrate_bn_plane_stats() -> Section {
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
fn calibrate_ln_row_pass() -> Section {
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

// ---- kd-tree vs brute force by dimension: KNN/DBSCAN_KD_TREE_MAX_DIMS ----

/// The "serial" column is the kd-tree path and the "parallel" column the brute-force scan, so
/// the crossover reads "the dimension bracket where brute force starts winning for good".
/// Uniform data; clustered data shifts the boundary, so this is a same-distribution comparison,
/// not a universal constant
fn calibrate_kd_tree_dims() -> Section {
    let n_train = 20_000usize;
    let n_query = 512usize;
    let k = 8usize;
    let mut rows = Vec::new();
    for &d in &[2usize, 4, 8, 12, 16, 20, 24, 32] {
        let x_train = random_matrix_f64(n_train, d, 71);
        let queries = random_matrix_f64(n_query, d, 72);
        let train_sq: Vec<f64> = x_train.rows().into_iter().map(|r| r.dot(&r)).collect();
        let tree = KdTree::build(x_train.view(), DistanceCalculationMetric::Euclidean);

        let t_tree = time_per_call_ns(|| {
            let res: Vec<usize> = (0..n_query)
                .into_par_iter()
                .map(|qi| tree.k_nearest(queries.row(qi), k)[0].0)
                .collect();
            black_box(res);
        });
        let t_brute = time_per_call_ns(|| {
            let res: Vec<usize> = (0..n_query)
                .into_par_iter()
                .map(|qi| {
                    let q = queries.row(qi);
                    let proj = x_train.dot(&q);
                    let mut dists: Vec<(f64, usize)> = proj
                        .iter()
                        .zip(train_sq.iter())
                        .enumerate()
                        .map(|(j, (&p, &sq))| (sq - 2.0 * p, j))
                        .collect();
                    dists.select_nth_unstable_by(k - 1, |a, b| {
                        a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
                    });
                    dists[0].1
                })
                .collect();
            black_box(res);
        });
        rows.push(Row {
            label: format!("d={d} (20k train, 512 queries, k=8)"),
            work: d,
            serial_ns: t_tree,
            parallel_ns: t_brute,
        });
    }
    Section {
        title: "kd-tree (serial col) vs brute force (parallel col) by dimension (KD_TREE_MAX_DIMS); uniform data",
        work_unit: "dimensions",
        pick_fastest: false,
        rows,
    }
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn today() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    // `cargo bench -- <filter>` passes harness args; this custom harness ignores them
    let threads = rayon::current_num_threads();
    println!("calibrating parallel gates (rayon threads: {threads}) ...");

    let mut sections = vec![
        calibrate_par_matmul_flops(),
        calibrate_par_matmul_min_block(),
        calibrate_conv_forward(),
        calibrate_pooling(),
    ];
    sections.extend(calibrate_elementwise());
    sections.push(calibrate_par_matmul_flops_f64());
    sections.push(calibrate_par_matmul_min_block_f64());
    sections.push(calibrate_par_matvec_flops_f32());
    sections.push(calibrate_par_matvec_flops_f64());
    sections.extend(calibrate_par_matvec_min_block());
    sections.extend(calibrate_elementwise_f64());
    sections.extend(calibrate_gemm_chunk_budget());
    sections.push(calibrate_pairwise_strategy());
    sections.push(calibrate_tree_traversal());
    sections.push(calibrate_sort_scan());
    sections.push(calibrate_tree_build());
    sections.push(calibrate_det_reduce_block());
    sections.push(calibrate_exp_reduction());
    sections.push(calibrate_kmeans_accumulate());
    sections.push(calibrate_f32_sq_sum());
    sections.push(calibrate_det_reduce_block_f32());
    sections.push(calibrate_bn_col_stats());
    sections.push(calibrate_bn_col_stats_rowblock());
    sections.push(calibrate_bn_plane_stats());
    sections.push(calibrate_ln_row_pass());
    sections.push(calibrate_kd_tree_dims());

    for s in &sections {
        s.print();
    }

    // Rewrite the report
    let mut md = String::new();
    let _ = writeln!(md, "# Parallel-gate calibration results\n");
    let _ = writeln!(
        md,
        "Generated by `cargo bench --bench parallel_gates`. Do not edit by hand - rerun instead.\n"
    );
    let _ = writeln!(md, "- CPU: {}", cpu_model());
    let _ = writeln!(md, "- rayon threads: {threads}");
    let _ = writeln!(md, "- date: {}\n", today());
    let _ = writeln!(
        md,
        "Each table forces the **serial** and **parallel** implementation of one kernel class on \
         either side of its gate and reports the speedup per ladder rung; the gate constant \
         should sit at the crossover (with a small safety margin toward serial).\n"
    );
    for s in &sections {
        s.to_markdown(&mut md);
    }
    let _ = writeln!(
        md,
        "## End-to-end benchmarks\n\nThe end-to-end numbers are tracked separately by criterion: \
         `cargo bench --bench nn_end_to_end` (neural-network layer forwards, a training epoch) and \
         `cargo bench --bench ml_end_to_end --features full` (classical-ML/utils fits, predicts, \
         and transforms); detailed reports and saved baselines live under `target/criterion/` \
         (use `-- --save-baseline <name>` / `-- --baseline <name>` to compare across changes)."
    );

    let path = std::path::Path::new("benches/RESULTS.md");
    std::fs::write(path, md).expect("write benches/RESULTS.md");
    println!("\nwrote {}", path.display());
}
