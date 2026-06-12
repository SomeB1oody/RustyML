//! End-to-end criterion benchmarks over the public neural-network API
//!
//! These track the real, user-visible cost of the hot paths (layer forwards and a small training
//! loop) so performance regressions show up as criterion deltas. Run with:
//!
//! ```bash
//! cargo bench --bench nn_end_to_end
//! # compare against a saved baseline:
//! cargo bench --bench nn_end_to_end -- --save-baseline main
//! cargo bench --bench nn_end_to_end -- --baseline main
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array;
use rustyml::neural_network::layers::*;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use std::hint::black_box;

/// Dense forward on a batch large enough to engage the block-parallel GEMM
fn dense_forward(c: &mut Criterion) {
    let mut layer = Dense::new(784, 512, Activation::ReLU, Some(42)).unwrap();
    let x = Array::from_elem((256, 784), 0.5f32).into_dyn();
    c.bench_function("dense_forward_256x784x512", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// Conv2D forward at batch == 1 (the single-sample inference case the engine used to run serial)
fn conv2d_forward_batch1(c: &mut Criterion) {
    let mut layer = Conv2D::new(
        64,
        (3, 3),
        vec![1, 32, 96, 96],
        (1, 1),
        PaddingType::Valid,
        Activation::ReLU,
        Some(42),
    )
    .unwrap();
    let x = Array::from_elem((1, 32, 96, 96), 0.5f32).into_dyn();
    c.bench_function("conv2d_forward_1x32x96x96_64f", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// LSTM forward: fused-gate projections plus the sequential recurrence
fn lstm_forward(c: &mut Criterion) {
    let mut layer = LSTM::new(64, 128, Activation::Tanh, Some(42)).unwrap();
    let x = Array::from_elem((32, 64, 64), 0.5f32).into_dyn();
    c.bench_function("lstm_forward_32x64x64_128u", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// One epoch of MLP training: forward + loss + backward + optimizer across the whole stack
fn mlp_fit_epoch(c: &mut Criterion) {
    let x = Array::from_elem((512, 256), 0.5f32).into_dyn();
    let y = Array::from_elem((512, 10), 1.0f32).into_dyn();
    let mut group = c.benchmark_group("training");
    group.sample_size(20);
    group.bench_function("mlp_fit_epoch_512x256-128-10", |b| {
        b.iter(|| {
            let mut model = Sequential::new();
            model
                .add(Dense::new(256, 128, Activation::ReLU, Some(42)).unwrap())
                .add(Dense::new(128, 10, Activation::Linear, Some(43)).unwrap())
                .compile(
                    SGD::new(0.01, None, 0.0, false, 0.0).unwrap(),
                    MeanSquaredError::new(),
                );
            model.fit(&x, &y, 1).unwrap();
            black_box(model);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    dense_forward,
    conv2d_forward_batch1,
    lstm_forward,
    mlp_fit_epoch
);
criterion_main!(benches);
