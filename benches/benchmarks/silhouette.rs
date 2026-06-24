//! `silhouette_score` pairwise-distance benchmarks for the `metrics` clustering module.
//!
//! Isolates the dominant cost of [`rustyml::metrics::silhouette_score`]: the `O(n^2 * d)`
//! pairwise-distance fill of `dist_to_cluster`. Configs are sized so `scan_work = n * n * d`
//! clears the `silhouette_parallel_min_elems` gate (262_144), so the parallel fill path is the
//! one being measured. Each metric is benchmarked separately because the per-pair cost differs
//! (`Manhattan` < `Euclidean` (a `sqrt`) < `Minkowski(p)` (a `powf`)), and the symmetric
//! optimization removes half of these calls - so the more expensive the metric, the larger the
//! expected win.
//!
//! ```bash
//! cargo bench --bench silhouette -- --save-baseline before   # before the change
//! cargo bench --bench silhouette -- --baseline before        # after the change
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use rustyml::math::DistanceCalculationMetric;
use rustyml::metrics::silhouette_score;
use std::hint::black_box;

/// Deterministic pseudo-random feature matrix (hash-based, no rng dependency) so the bench builds
/// with just the `metrics` feature. Mirrors the matmul-test generator.
fn pseudo_random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let t = (seed as f64) * 0.731 + (i * cols + j) as f64 * 0.618_033_988_7;
        (t.sin() * 43758.5453).fract() - 0.5
    })
}

/// Round-robin cluster labels: exactly `k` clusters, each of size ~n/k (no singletons), with
/// `2 <= k < n` so `validate_clustering_inputs` accepts them.
fn round_robin_labels(n: usize, k: usize) -> Array1<usize> {
    Array1::from_shape_fn(n, |i| i % k)
}

fn bench_silhouette(c: &mut Criterion) {
    let mut group = c.benchmark_group("silhouette_score");
    group.sample_size(10);

    // (label, n_samples, n_features, n_clusters). All clear the n*n*d >= 262_144 parallel gate.
    let configs: &[(&str, usize, usize, usize)] = &[
        ("n2000_d50_k8", 2000, 50, 8),
        ("n3000_d50_k10", 3000, 50, 10),
        ("n2000_d100_k8", 2000, 100, 8),
    ];

    let metrics: &[(&str, DistanceCalculationMetric)] = &[
        ("euclidean", DistanceCalculationMetric::Euclidean),
        ("manhattan", DistanceCalculationMetric::Manhattan),
        ("minkowski3", DistanceCalculationMetric::Minkowski(3.0)),
    ];

    for &(shape_label, n, d, k) in configs {
        let x = pseudo_random_matrix(n, d, 7);
        let labels = round_robin_labels(n, k);
        for &(metric_label, metric) in metrics {
            let id = format!("{shape_label}_{metric_label}");
            group.bench_function(id, |b| {
                b.iter(|| black_box(silhouette_score(black_box(&x), black_box(&labels), metric)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_silhouette);
criterion_main!(benches);
