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

use ndarray::{Array1, Array2, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;
use rustyml::bench_internals::{
    PoolKind, conv_forward_forced, split_matmul, windowed_pool_forward_impl,
};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::PaddingType;
use std::fmt::Write as _;
use std::hint::black_box;
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
            Some((0, hi)) => format!("crossover below {hi} {} (parallel wins everywhere)", self.work_unit),
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

// ---- par_matmul: PAR_GEMM_MIN_FLOPS ----

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
            black_box(split_matmul(a.view(), b.view(), 1));
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
            black_box(split_matmul(a.view(), b.view(), 1));
        });
        rows.push(Row {
            label: format!("skinny {m}x{k}x{n}"),
            work: 2 * m * k * n,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "par_matmul FLOPs gate (PAR_GEMM_MIN_FLOPS)",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

// ---- par_matmul: PAR_GEMM_MIN_BLOCK ----

fn calibrate_par_matmul_min_block() -> Section {
    let a = random_matrix(2048, 512, 5);
    let b = random_matrix(512, 512, 6);
    let serial = time_per_call_ns(|| {
        black_box(a.dot(&b));
    });
    let mut rows = Vec::new();
    for &blk in &[8usize, 16, 32, 64, 128, 256, 512] {
        let p = time_per_call_ns(|| {
            black_box(split_matmul(a.view(), b.view(), blk));
        });
        rows.push(Row {
            label: format!("2048x512x512 min_block {blk}"),
            work: blk,
            serial_ns: serial,
            parallel_ns: p,
        });
    }
    Section {
        title: "par_matmul block size (PAR_GEMM_MIN_BLOCK)",
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
        "## End-to-end benchmarks\n\nThe end-to-end numbers (public API: layer forwards, a \
         training epoch) are tracked separately by criterion via `cargo bench --bench \
         nn_end_to_end`; detailed reports and saved baselines live under `target/criterion/` \
         (use `-- --save-baseline <name>` / `-- --baseline <name>` to compare across changes)."
    );

    let path = std::path::Path::new("benches/RESULTS.md");
    std::fs::write(path, md).expect("write benches/RESULTS.md");
    println!("\nwrote {}", path.display());
}
