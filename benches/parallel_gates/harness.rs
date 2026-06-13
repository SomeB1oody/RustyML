//! Shared calibration harness: the timing loop, the `Row`/`Section` table model that every
//! calibration produces, and the seeded random-data generators the ladders feed on.

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Nanoseconds per call of `f`: the batch size grows until one batch takes >= 5 ms, then the
/// minimum over three batches filters scheduler noise
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

/// One measured ladder rung
pub struct Row {
    pub label: String,
    pub work: usize,
    pub serial_ns: f64,
    pub parallel_ns: f64,
}

impl Row {
    fn speedup(&self) -> f64 {
        self.serial_ns / self.parallel_ns
    }
}

/// One calibration table plus the work units its `work` column counts
pub struct Section {
    pub title: &'static str,
    pub work_unit: &'static str,
    /// When true the table is a parameter sweep: report the fastest rung instead of a crossover
    pub pick_fastest: bool,
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
