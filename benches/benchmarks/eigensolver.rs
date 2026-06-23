//! Power-iteration eigensolver benchmarks for the `utils` transformers.
//!
//! Isolates `utils::linalg::top_eigenpairs_power_iteration` (power iteration with Hotelling
//! deflation) through the two public APIs that select it: the PCA `PowerIteration` SVD solver
//! and the KernelPCA `PowerIteration` eigen solver. Configs are chosen so the iterative inner
//! loop - not the one-off covariance/kernel GEMM - dominates the wall clock, so a change to the
//! per-iteration matvec count or the deflation step shows up cleanly.
//!
//! ```bash
//! cargo bench --bench eigensolver -- --save-baseline before   # before the change
//! cargo bench --bench eigensolver -- --baseline before        # after the change
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use rustyml::utils::kernel_pca::{EigenSolver, Gamma, KernelPCA, KernelType};
use rustyml::utils::pca::{PCA, SVDSolver};
use std::hint::black_box;

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
}

/// KernelPCA with the `PowerIteration` eigen solver: the centered kernel matrix is `n x n`, so
/// each power-iteration step is an `n x n` GEMV. With a small feature count the one-off kernel
/// GEMM (`O(n^2 d)`) is dwarfed by the iterative solve (`n_components` deflation rounds, each many
/// `O(n^2)` matvecs), so this is the cleanest isolation of the deflated power-iteration inner loop.
fn bench_kernel_pca_power_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("eig_kernel_pca_power_iteration");
    group.sample_size(10);

    // (label, n_samples, n_features, n_components)
    let configs: &[(&str, usize, usize, usize)] = &[
        ("kpca_poweriter_1000x16_8c", 1000, 16, 8),
        ("kpca_poweriter_1500x16_6c", 1500, 16, 6),
    ];
    for &(label, n, d, k) in configs {
        let x = random_matrix(n, d, 7);
        group.bench_function(label, |b| {
            b.iter(|| {
                let mut model = KernelPCA::new(
                    KernelType::RBF {
                        gamma: Gamma::Value(0.1),
                    },
                    k,
                )
                .unwrap()
                .with_eigen_solver(EigenSolver::PowerIteration);
                black_box(model.fit_transform(black_box(&x)).unwrap());
            })
        });
    }
    group.finish();
}

/// PCA with the `PowerIteration` SVD solver: the solver builds the `p x p` covariance once and then
/// runs deflated power iteration on it. A modest sample count keeps the covariance GEMM cheap so the
/// per-component iterative solve over the `p x p` matrix dominates.
fn bench_pca_power_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("eig_pca_power_iteration");
    group.sample_size(10);

    // (label, n_samples, n_features, n_components)
    let configs: &[(&str, usize, usize, usize)] = &[
        ("pca_poweriter_500x200_10c", 500, 200, 10),
        ("pca_poweriter_800x256_12c", 800, 256, 12),
    ];
    for &(label, n, p, k) in configs {
        let x = random_matrix(n, p, 6);
        group.bench_function(label, |b| {
            b.iter(|| {
                let mut model = PCA::new(k)
                    .unwrap()
                    .with_svd_solver(SVDSolver::PowerIteration);
                black_box(model.fit_transform(black_box(&x)).unwrap());
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_kernel_pca_power_iteration,
    bench_pca_power_iteration
);
criterion_main!(benches);
