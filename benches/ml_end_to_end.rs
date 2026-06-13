//! End-to-end criterion benchmarks over the classical-ML and utils public API
//!
//! Tracks the wall-clock effect of the GEMM/parallelism work (and guards against regressions)
//! at the level a user sees: whole `fit`/`predict`/`transform` calls. Detailed reports and
//! saved baselines live under `target/criterion/`; compare across changes with
//! `cargo bench --bench ml_end_to_end -- --save-baseline <name>` and `-- --baseline <name>`.
//!
//! The micro-level serial/parallel crossovers behind the gate constants are calibrated
//! separately by `cargo bench --bench parallel_gates` (see benches/RESULTS.md).

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use rustyml::machine_learning::{
    KMeans, KNN, KernelType, LogisticRegression, SVC, WeightingStrategy,
};
use rustyml::types::DistanceCalculationMetric;
use rustyml::utils::kernel_pca::{EigenSolver, KernelPCA};
use rustyml::utils::pca::{PCA, SVDSolver};
use rustyml::utils::t_sne::{Init, TSNE, TSNEMethod};
use std::hint::black_box;

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

/// KMeans fit: per-iteration assignment GEMM + arg-min scan over 20k x 32, k = 16
fn bench_kmeans_fit(c: &mut Criterion) {
    let x = random_matrix(20_000, 32, 1);
    c.bench_function("kmeans_fit_20000x32_k16", |b| {
        b.iter(|| {
            let mut model = KMeans::new(16, 50, 1e-4, Some(42)).unwrap();
            model.fit(black_box(&x)).unwrap();
            black_box(model.get_inertia());
        })
    });
}

/// KNN predict on the brute-force Euclidean path (64 features > the kd-tree ceiling):
/// cache-resident training set, so the per-query GEMV swarm serves the projections
fn bench_knn_predict(c: &mut Criterion) {
    let x_train = random_matrix(20_000, 64, 2);
    let y_train: Array1<i64> = Array1::from_iter((0..20_000).map(|i| (i % 4) as i64));
    let queries = random_matrix(512, 64, 3);

    let mut model = KNN::new(
        8,
        WeightingStrategy::Uniform,
        DistanceCalculationMetric::Euclidean,
    )
    .unwrap();
    model.fit(&x_train, &y_train).unwrap();

    c.bench_function("knn_predict_512q_20000x64", |b| {
        b.iter(|| {
            black_box(model.predict_parallel(black_box(&queries)).unwrap());
        })
    });
}

/// SVC fit with the RBF kernel: one batched kernel-matrix GEMM + SMO on 1500 x 16
fn bench_svc_fit(c: &mut Criterion) {
    let x = random_matrix(1500, 16, 4);
    let y: Array1<f64> = Array1::from_iter(
        x.rows()
            .into_iter()
            .map(|r| if r.sum() > 0.0 { 1.0 } else { -1.0 }),
    );

    let mut group = c.benchmark_group("svc");
    group.sample_size(10);
    group.bench_function("svc_fit_rbf_1500x16", |b| {
        b.iter(|| {
            let mut model =
                SVC::new(KernelType::RBF { gamma: 0.5 }, 1.0, 1e-3, 100, Some(42)).unwrap();
            model.fit(black_box(&x), black_box(&y)).unwrap();
            black_box(model.get_bias());
        })
    });
    group.finish();
}

/// Logistic regression fit: per-iteration GEMV + sigmoid map over 50k x 64
fn bench_logistic_fit(c: &mut Criterion) {
    let x = random_matrix(50_000, 64, 5);
    let y: Array1<f64> = Array1::from_iter(
        x.rows()
            .into_iter()
            .map(|r| if r.sum() > 0.0 { 1.0 } else { 0.0 }),
    );

    c.bench_function("logistic_fit_50000x64_100it", |b| {
        b.iter(|| {
            let mut model = LogisticRegression::new(true, 0.1, 100, 1e-9, None).unwrap();
            model.fit(black_box(&x), black_box(&y)).unwrap();
            black_box(model.get_actual_iterations());
        })
    });
}

/// PCA fit + transform: covariance GEMM (power-iteration solver) and projection GEMM
fn bench_pca_fit_transform(c: &mut Criterion) {
    let x = random_matrix(10_000, 128, 6);
    c.bench_function("pca_fit_transform_10000x128_16c", |b| {
        b.iter(|| {
            let mut model = PCA::new(16, SVDSolver::PowerIteration).unwrap();
            black_box(model.fit_transform(black_box(&x)).unwrap());
        })
    });
}

/// Kernel PCA fit + transform: batched RBF kernel matrix + Lanczos eigensolver matvecs
fn bench_kernel_pca(c: &mut Criterion) {
    let x = random_matrix(1500, 32, 7);
    let mut group = c.benchmark_group("kernel_pca");
    group.sample_size(10);
    group.bench_function("kernel_pca_fit_transform_rbf_1500x32_8c", |b| {
        b.iter(|| {
            let mut model =
                KernelPCA::new(KernelType::RBF { gamma: 0.1 }, 8, EigenSolver::Lanczos).unwrap();
            black_box(model.fit_transform(black_box(&x)).unwrap());
        })
    });
    group.finish();
}

/// t-SNE exact path: GEMM-form pairwise distances and gradient, few iterations
fn bench_tsne_exact(c: &mut Criterion) {
    let x = random_matrix(1000, 32, 8);
    let mut group = c.benchmark_group("tsne");
    group.sample_size(10);
    group.bench_function("tsne_exact_1000x32_50it", |b| {
        b.iter(|| {
            let model = TSNE::new(
                2,
                30.0,
                200.0,
                50,
                Some(42),
                Init::Random,
                TSNEMethod::Exact,
            )
            .unwrap();
            black_box(model.fit_transform(black_box(&x)).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans_fit,
    bench_knn_predict,
    bench_svc_fit,
    bench_logistic_fit,
    bench_pca_fit_transform,
    bench_kernel_pca,
    bench_tsne_exact
);
criterion_main!(benches);
