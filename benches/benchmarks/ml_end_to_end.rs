//! End-to-end criterion benchmarks over the classical-ML public API
//!
//! Tracks the wall-clock effect of the GEMM/parallelism work (and guards against regressions)
//! at the level a user sees: whole `fit`/`predict`/`transform` calls. Detailed reports and
//! saved baselines live under `target/criterion/`; compare across changes with
//! `cargo bench --bench ml_end_to_end -- --save-baseline <name>` and `-- --baseline <name>`
//!
//! The micro-level serial/parallel crossovers behind the gate constants are calibrated
//! separately by `cargo bench --bench parallel_gates` (see benches/RESULTS.md)

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use rustyml::machine_learning::decomposition::kernel_pca::{EigenSolver, KernelPCA};
use rustyml::machine_learning::decomposition::pca::{PCA, SVDSolver};
use rustyml::machine_learning::manifold::t_sne::{Init, TSNE, TSNEMethod};
use rustyml::machine_learning::{
    KMeans, KNN, KernelType, LDA, LogisticRegression, MeanShift, SVC, Solver, WeightingStrategy,
    generate_polynomial_features,
};
use rustyml::types::{DistanceCalculationMetric, Gamma};
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
            let mut model = KMeans::new(16, 50, 1e-4).unwrap().with_random_state(42);
            model.fit(black_box(&x)).unwrap();
            black_box(model.get_inertia());
        })
    });
}

/// KMeans fit at high cluster count (k = 256, 128 features, 4k samples): a small sample count
/// keeps the assignment GEMM modest so the per-iteration centroid-averaging step - which fires one
/// rayon job per centroid - is a visible fraction. Stresses the nested per-centroid parallelism in
/// the cluster-mean update
fn bench_kmeans_fit_high_k(c: &mut Criterion) {
    let x = random_matrix(4_096, 128, 11);
    let mut group = c.benchmark_group("kmeans_high_k");
    group.sample_size(20);
    group.bench_function("kmeans_fit_4096x128_k256_30it", |b| {
        b.iter(|| {
            let mut model = KMeans::new(256, 30, 1e-4).unwrap().with_random_state(42);
            model.fit(black_box(&x)).unwrap();
            black_box(model.get_inertia());
        })
    });
    group.finish();
}

/// LDA fit across two parallelism regimes. The fit parallelizes the per-class scatter statistics
/// over the *classes*, but the current parallel decision keys off the total data size; each class
/// task runs a `[d, n_class] x [n_class, d]` scatter GEMM that can itself fork. The two configs
/// separate the regimes:
/// - few classes + high-dim: only a 3-wide class fan (far below the core count), so the class axis
///   alone cannot fill the pool while each per-class GEMM is large - probes idle cores + the
///   parallel branch's whole-matrix clone
/// - many classes + moderate dim: a 64-wide class fan (above the core count) where each per-class
///   GEMM still clears the GEMM parallel gate - probes the "class axis fills the pool while the
///   inner GEMM also forks" nesting case
fn bench_lda_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("lda_fit");
    group.sample_size(20);

    // (label, n_classes, n_features, samples_per_class, n_components). Dimensions are kept modest
    // (d <= 128) so the per-class scatter GEMMs - the parallel section - dominate over the O(d^3)
    // eigendecomposition, which would otherwise mask the parallelism signal.
    let configs: &[(&str, usize, usize, usize, usize)] = &[
        ("lda_fit_few_classes_4c_128f_64000n", 4, 128, 16_000, 3),
        ("lda_fit_many_classes_64c_96f_25600n", 64, 96, 400, 16),
    ];
    for &(label, n_classes, n_features, per_class, n_components) in configs {
        let n_samples = n_classes * per_class;
        let x = random_matrix(n_samples, n_features, 9);
        let y: Array1<i32> = Array1::from_iter((0..n_samples).map(|i| (i % n_classes) as i32));
        group.bench_function(label, |b| {
            b.iter(|| {
                let mut model = LDA::new(n_components).unwrap().with_solver(Solver::SVD);
                model.fit(black_box(&x), black_box(&y)).unwrap();
                black_box(&model);
            })
        });
    }
    group.finish();
}

/// KNN predict on the brute-force Euclidean path (64 features > the kd-tree ceiling):
/// cache-resident training set, so the per-query GEMV swarm serves the projections
fn bench_knn_predict(c: &mut Criterion) {
    let x_train = random_matrix(20_000, 64, 2);
    let y_train: Array1<i64> = Array1::from_iter((0..20_000).map(|i| (i % 4) as i64));
    let queries = random_matrix(512, 64, 3);

    let mut model = KNN::new(8)
        .unwrap()
        .with_weighting_strategy(WeightingStrategy::Uniform)
        .with_metric(DistanceCalculationMetric::Euclidean)
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
            let mut model = SVC::new(
                KernelType::RBF {
                    gamma: Gamma::Value(0.5),
                },
                1.0,
                1e-3,
                100,
            )
            .unwrap()
            .with_random_state(42);
            model.fit(black_box(&x), black_box(&y)).unwrap();
            black_box(model.get_bias());
        })
    });
    group.finish();
}

/// SVC predict on a fitted RBF model: batched kernel matrix + the decision-value GEMV
/// (`[n_query, n_sv] x [n_sv]`). The model is fit once outside the timing loop so the measured
/// work is the prediction path - the kernel evaluation plus the batched matvec that A1 reroutes
/// from ndarray `.dot()` to the gemm-crate backend
fn bench_svc_predict(c: &mut Criterion) {
    let x = random_matrix(1500, 16, 4);
    let y: Array1<f64> = Array1::from_iter(
        x.rows()
            .into_iter()
            .map(|r| if r.sum() > 0.0 { 1.0 } else { -1.0 }),
    );
    let mut model = SVC::new(
        KernelType::RBF {
            gamma: Gamma::Value(0.5),
        },
        1.0,
        1e-3,
        100,
    )
    .unwrap()
    .with_random_state(42);
    model.fit(&x, &y).unwrap();

    let queries = random_matrix(2000, 16, 14);
    let mut group = c.benchmark_group("svc_predict");
    group.sample_size(20);
    group.bench_function("svc_predict_2000q_rbf_1500x16", |b| {
        b.iter(|| {
            black_box(model.predict(black_box(&queries)).unwrap());
        })
    });
    group.finish();
}

/// MeanShift fit: one task per seed (all samples are seeds by default), each running RBF-weighted
/// updates whose per-iteration cost is two matvecs over the full sample matrix. With seeds far
/// above the core count the seed axis fills the pool, so A3 forces those inner matvecs serial via
/// the gemm-crate backend - this bench tracks that nested-matvec path end to end
fn bench_mean_shift_fit(c: &mut Criterion) {
    let x = random_matrix(1500, 16, 12);
    let mut group = c.benchmark_group("mean_shift");
    group.sample_size(10);
    group.bench_function("mean_shift_fit_1500x16_bw3", |b| {
        b.iter(|| {
            let mut model = MeanShift::new(3.0).unwrap().with_max_iter(20).unwrap();
            model.fit(black_box(&x)).unwrap();
            black_box(&model);
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
            let mut model = LogisticRegression::new(true, 0.1, 100, 1e-9).unwrap();
            model.fit(black_box(&x), black_box(&y)).unwrap();
            black_box(model.get_actual_iterations());
        })
    });
}

/// generate_polynomial_features: the expansion forks one rayon job per output monomial column.
/// The small config (degree 3 over 12 features = hundreds of monomials, few samples) is dominated
/// by that repeated per-monomial fork-join overhead - B1 gates the maps to run serial below the
/// cheap-map threshold. The large degree-1 config keeps the first-order copy on the parallel path
/// (its `n_samples * n_features` work clears the gate) to guard that path against a regression
fn bench_poly_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("poly_features");
    group.sample_size(20);

    let x_small = random_matrix(2_000, 12, 21);
    group.bench_function("poly_features_2000x12_deg3", |b| {
        b.iter(|| black_box(generate_polynomial_features(black_box(&x_small), 3)))
    });

    let x_large = random_matrix(400_000, 12, 22);
    group.bench_function("poly_features_400000x12_deg1", |b| {
        b.iter(|| black_box(generate_polynomial_features(black_box(&x_large), 1)))
    });

    group.finish();
}

/// PCA fit + transform: covariance GEMM (power-iteration solver) and projection GEMM
fn bench_pca_fit_transform(c: &mut Criterion) {
    let x = random_matrix(10_000, 128, 6);
    c.bench_function("pca_fit_transform_10000x128_16c", |b| {
        b.iter(|| {
            let mut model = PCA::new(16)
                .unwrap()
                .with_svd_solver(SVDSolver::PowerIteration);
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
            let mut model = KernelPCA::new(
                KernelType::RBF {
                    gamma: Gamma::Value(0.1),
                },
                8,
            )
            .unwrap()
            .with_eigen_solver(EigenSolver::Lanczos);
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
            let model = TSNE::new(2, 30.0, 200.0, 50)
                .unwrap()
                .with_random_state(42)
                .with_init(Init::Random)
                .with_method(TSNEMethod::Exact)
                .unwrap();
            black_box(model.fit_transform(black_box(&x)).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans_fit,
    bench_kmeans_fit_high_k,
    bench_lda_fit,
    bench_knn_predict,
    bench_svc_fit,
    bench_svc_predict,
    bench_mean_shift_fit,
    bench_logistic_fit,
    bench_poly_features,
    bench_pca_fit_transform,
    bench_kernel_pca,
    bench_tsne_exact
);
criterion_main!(benches);
