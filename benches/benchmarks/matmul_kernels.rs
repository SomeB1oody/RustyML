//! Kernel-level GEMM benchmarks: `Dense::forward` across a shape sweep.
//!
//! Dense forward is one `gemm_internal` call (plus a cheap bias + activation), so this isolates
//! the matrix-multiply backend across the regimes that matter (small, medium, large, huge, wide,
//! thin-K). Stable across backend changes since it only uses the public neural-network API.
//!
//! ```bash
//! cargo bench --bench matmul_kernels
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array;
use rustyml::neural_network::layers::{Activation, Dense};
use rustyml::neural_network::traits::Layer;
use std::hint::black_box;

fn bench_gemm_shapes(c: &mut Criterion) {
    // (label, batch, in_features, out_features)
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("small_256x256x256", 256, 256, 256),
        ("medium_512x1024x1024", 512, 1024, 1024),
        ("big_1024x2048x2048", 1024, 2048, 2048),
        ("huge_2048x2048x2048", 2048, 2048, 2048),
        ("wide_256x256x8192", 256, 256, 8192),
        ("thin_256x8192x256", 256, 8192, 256),
    ];

    let mut group = c.benchmark_group("gemm_dense_forward");
    group.sample_size(20);
    for &(label, batch, fin, fout) in shapes {
        let mut layer = Dense::new(fin, fout, Activation::ReLU)
            .unwrap()
            .with_random_state(42);
        let x = Array::from_elem((batch, fin), 0.5f32).into_dyn();
        group.bench_function(label, |b| b.iter(|| black_box(layer.forward(&x).unwrap())));
    }
    group.finish();
}

criterion_group!(benches, bench_gemm_shapes);
criterion_main!(benches);
