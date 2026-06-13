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

/// Spatial BatchNorm forward at conv scale (training mode): per-channel plane folds plus the
/// per-plane center/normalize passes on the native [B, C, *spatial] layout
fn batchnorm_forward_spatial(c: &mut Criterion) {
    let mut layer = BatchNormalization::new(vec![32, 64, 64, 64], 0.99, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 7 + ch * 13 + h * 3 + w) as f32 * 0.137).sin()
    })
    .into_dyn();
    c.bench_function("batchnorm_forward_32x64x64x64", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// Spatial BatchNorm backward at the same scale: five per-channel plane folds plus two
/// per-plane elementwise passes over the cached forward tensors
fn batchnorm_backward_spatial(c: &mut Criterion) {
    let mut layer = BatchNormalization::new(vec![32, 64, 64, 64], 0.99, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 7 + ch * 13 + h * 3 + w) as f32 * 0.137).sin()
    })
    .into_dyn();
    layer.forward(&x).unwrap();
    let grad = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 11 + ch * 5 + h * 7 + w) as f32 * 0.293).sin()
    })
    .into_dyn();
    c.bench_function("batchnorm_backward_32x64x64x64", |b| {
        b.iter(|| black_box(layer.backward(&grad).unwrap()))
    });
}

/// LayerNorm forward at transformer scale (Default axis): per-row statistics over the
/// trailing feature axis
fn layernorm_forward_default(c: &mut Criterion) {
    let mut layer =
        LayerNormalization::new(vec![32, 512, 768], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 512, 768), |(b, t, d)| {
        ((b * 7 + t * 13 + d * 3) as f32 * 0.137).sin()
    })
    .into_dyn();
    c.bench_function("layernorm_forward_32x512x768", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// LayerNorm backward at the same scale: per-row gradient composition plus the gamma/beta
/// column reductions
fn layernorm_backward_default(c: &mut Criterion) {
    let mut layer =
        LayerNormalization::new(vec![32, 512, 768], LayerNormalizationAxis::Default, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 512, 768), |(b, t, d)| {
        ((b * 7 + t * 13 + d * 3) as f32 * 0.137).sin()
    })
    .into_dyn();
    layer.forward(&x).unwrap();
    let grad = Array::from_shape_fn((32, 512, 768), |(b, t, d)| {
        ((b * 11 + t * 5 + d * 7) as f32 * 0.293).sin()
    })
    .into_dyn();
    c.bench_function("layernorm_backward_32x512x768", |b| {
        b.iter(|| black_box(layer.backward(&grad).unwrap()))
    });
}

/// LayerNorm forward with a Multiple (merged trailing axes) configuration at conv scale: the
/// merged-axis layout transform is the interesting cost here
fn layernorm_forward_multi(c: &mut Criterion) {
    let mut layer = LayerNormalization::new(
        vec![32, 64, 64, 64],
        LayerNormalizationAxis::Multiple(vec![1, 2, 3]),
        1e-5,
    )
    .unwrap();
    let x = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 7 + ch * 13 + h * 3 + w) as f32 * 0.137).sin()
    })
    .into_dyn();
    c.bench_function("layernorm_forward_multi_32x64x64x64", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// GroupNorm forward at conv scale (channels-first, 8 groups): per-instance group statistics
/// over contiguous [channels/groups x spatial] blocks
fn groupnorm_forward(c: &mut Criterion) {
    let mut layer = GroupNormalization::new(vec![32, 64, 64, 64], 8, 1, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 7 + ch * 13 + h * 3 + w) as f32 * 0.137).sin()
    })
    .into_dyn();
    c.bench_function("groupnorm_forward_32x64x64x64_8g", |b| {
        b.iter(|| black_box(layer.forward(&x).unwrap()))
    });
}

/// GroupNorm backward at the same scale: per-channel parameter folds plus the per-instance
/// gradient composition
fn groupnorm_backward(c: &mut Criterion) {
    let mut layer = GroupNormalization::new(vec![32, 64, 64, 64], 8, 1, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 7 + ch * 13 + h * 3 + w) as f32 * 0.137).sin()
    })
    .into_dyn();
    layer.forward(&x).unwrap();
    let grad = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 11 + ch * 5 + h * 7 + w) as f32 * 0.293).sin()
    })
    .into_dyn();
    c.bench_function("groupnorm_backward_32x64x64x64_8g", |b| {
        b.iter(|| black_box(layer.backward(&grad).unwrap()))
    });
}

/// InstanceNorm forward at the same scale (one group per channel: many small instances)
fn instancenorm_forward(c: &mut Criterion) {
    let mut layer = InstanceNormalization::new(vec![32, 64, 64, 64], 1, 1e-5).unwrap();
    let x = Array::from_shape_fn((32, 64, 64, 64), |(n, ch, h, w)| {
        ((n * 7 + ch * 13 + h * 3 + w) as f32 * 0.137).sin()
    })
    .into_dyn();
    c.bench_function("instancenorm_forward_32x64x64x64", |b| {
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
    batchnorm_forward_spatial,
    batchnorm_backward_spatial,
    layernorm_forward_default,
    layernorm_backward_default,
    layernorm_forward_multi,
    groupnorm_forward,
    groupnorm_backward,
    instancenorm_forward,
    mlp_fit_epoch
);
criterion_main!(benches);
