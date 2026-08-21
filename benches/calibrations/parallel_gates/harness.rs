//! Shared calibration harness: the timing loop, the `Row`/`Section` table model that every
//! calibration produces, and the seeded random-data generators the ladders feed on

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Nanoseconds per call of `f`: the batch size grows until 1 batch takes >= 5 ms, then the
/// minimum over 3 batches filters scheduler noise
pub fn time_per_call_ns<F: FnMut()>(mut f: F) -> f64 {
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

/// Speedup at or below which a rung counts as a loss. The margin filters the ~1.00x ties that
/// would otherwise read as a crossover
const WIN_MARGIN: f64 = 1.05;

/// Losing rungs above the crossover that do not invalidate it. A ladder usually has at most 1:
/// either a noisy measurement, or a largest shape whose working set overflows the cache, where a
/// wide machine returns single-thread throughput
const MAX_INTERIOR_DIPS: usize = 1;

/// 1 measured ladder rung
pub struct Row {
    /// Label for this rung, used in the printed table and the markdown report
    pub label: String,
    /// Work units this rung represents, counted in the section's `work_unit`
    pub work: usize,
    /// Nanoseconds per call, forced serial
    pub serial_ns: f64,
    /// Nanoseconds per call, forced parallel
    pub parallel_ns: f64,
}

impl Row {
    fn speedup(&self) -> f64 {
        self.serial_ns / self.parallel_ns
    }
}

/// 1 calibration table plus the work units its `work` column counts
pub struct Section {
    /// Heading for this table, printed above its rows
    pub title: &'static str,
    /// Name of the unit the `work` column counts
    pub work_unit: &'static str,
    /// When true the table is a parameter sweep: report the fastest rung instead of a crossover
    pub pick_fastest: bool,
    /// The measured ladder rungs, in run order
    pub rows: Vec<Row>,
}

impl Section {
    pub fn print(&self) {
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
        let ladder = self.ladder();
        match self.crossover(&ladder) {
            Some((None, hi, dips)) => format!(
                "crossover at or below {hi} {} (parallel wins from the first rung){}",
                self.work_unit,
                Self::dip_note(&dips, self.work_unit)
            ),
            Some((Some(lo), hi, dips)) => format!(
                "crossover between {lo} and {hi} {}{}",
                self.work_unit,
                Self::dip_note(&dips, self.work_unit)
            ),
            None => {
                let best = ladder
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("ladder has rows");
                format!(
                    "no crossover in this ladder (best {:.2}x at {} {})",
                    best.1, best.0, self.work_unit
                )
            }
        }
    }

    /// The rungs in work order, with tied work values collapsed to their slowest rung
    ///
    /// Several ladders repeat a work value at different shapes, because a work estimate is 1
    /// number and a shape is not. A tie collapses to its slowest rung: a crossover claim must
    /// hold for every shape that reaches that work, and not only for the best of them
    fn ladder(&self) -> Vec<(usize, f64)> {
        let mut rungs: Vec<(usize, f64)> =
            self.rows.iter().map(|r| (r.work, r.speedup())).collect();
        rungs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        rungs.dedup_by(|later, first| first.0 == later.0);
        rungs
    }

    /// The work bracket from which the parallel path wins, plus every rung above it that falls
    /// back to a loss
    ///
    /// The rule finds the first rung that wins and has no more than [`MAX_INTERIOR_DIPS`] losing
    /// rungs above it. An earlier rule keyed on the LAST loss instead, which let 1 slow top rung
    /// erase a whole ladder. That is exactly what a largest shape produces when its working set
    /// stops fitting in cache, so the rule discarded the ladders that needed it most. The same
    /// rule also pushed a bracket past a real win whenever 1 interior rung dipped.
    fn crossover(&self, ladder: &[(usize, f64)]) -> Option<(Option<usize>, usize, Vec<usize>)> {
        for i in 0..ladder.len() {
            if ladder[i].1 <= WIN_MARGIN {
                continue;
            }
            let dips: Vec<usize> = ladder[i..]
                .iter()
                .filter(|(_, sp)| *sp <= WIN_MARGIN)
                .map(|(w, _)| *w)
                .collect();
            if dips.len() <= MAX_INTERIOR_DIPS {
                let lo = if i == 0 { None } else { Some(ladder[i - 1].0) };
                return Some((lo, ladder[i].0, dips));
            }
        }
        None
    }

    /// Names the tolerated losing rungs, so a bracket never hides one
    fn dip_note(dips: &[usize], unit: &str) -> String {
        match dips {
            [] => String::new(),
            [w] => format!("; the {w} {unit} rung above it falls back to a loss"),
            _ => format!(
                "; these rungs above it fall back to a loss: {}",
                dips.iter()
                    .map(|w| w.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    pub fn to_markdown(&self, out: &mut String) {
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

pub fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

pub fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

pub fn random_vector_f64(len: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::random_using(len, Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

pub fn random_vector_f32(len: usize, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::random_using(len, Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}
