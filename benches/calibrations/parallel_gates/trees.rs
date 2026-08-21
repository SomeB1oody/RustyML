//! Coarse-task tree gates: per-sample tree traversal (DecisionTree/IsolationForest predict),
//! the per-feature sort-scan of `find_best_split`, and parallel isolation-tree construction

use crate::harness::{Row, Section, random_matrix_f64, time_per_call_ns};
use ndarray::Array2;
use rayon::prelude::*;
use std::hint::black_box;

// coarse-task classes: tree traversal / sort-scan / tree build

/// Synthetic decision-tree predict kernel: per-sample root-to-leaf walk over a heap-layout
/// binary tree (depth 16, ~65K nodes), the same pointer-chasing shape as DecisionTree and
/// IsolationForest prediction
pub fn calibrate_tree_traversal() -> Section {
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
/// shape of DecisionTree's `find_best_split` (1 task per feature)
pub fn calibrate_sort_scan() -> Section {
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
pub fn calibrate_tree_build() -> Section {
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
