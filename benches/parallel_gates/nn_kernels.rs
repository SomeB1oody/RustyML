//! Neural-network forward-kernel gates: the conv engine FLOP gate, the pooling-engine op gate,
//! and the f32 elementwise classes (cheap maps, exp maps, per-element rng, fused optimizer
//! updates) that the activation/dropout/optimizer thresholds govern.

use crate::harness::{Row, Section, time_per_call_ns};
use ndarray::{Array1, IxDyn};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustyml::bench_internals::{PoolKind, conv_forward_forced, windowed_pool_forward_impl};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::PaddingType;
use std::hint::black_box;

// ---- conv engine: CONV_PARALLEL_MIN_FLOPS ----

pub fn calibrate_conv_forward() -> Section {
    let mut rows = Vec::new();
    // (cin, filters, img, k) at batch == 1, the case the gate must serve hardest
    for &(cin, f, img, k) in &[
        (3usize, 8usize, 16usize, 3usize),
        (3, 16, 32, 3),
        (8, 16, 32, 3),
        (16, 32, 32, 3),
        (16, 32, 64, 3),
        (32, 64, 64, 3),
        (64, 64, 128, 3),
    ] {
        let input = Tensor::from_elem(IxDyn(&[1, cin, img, img]), 1.0f32);
        let weights = vec![0.5f32; f * cin * k * k];
        let bias = vec![0.0f32; f];
        let out = img - k + 1;
        let flops = 2 * f * out * out * cin * k * k;
        let run = |force: bool| {
            black_box(
                conv_forward_forced(
                    &input,
                    &weights,
                    &[f, cin, k, k],
                    &bias,
                    &[1, 1],
                    PaddingType::Valid,
                    Some(force),
                )
                .unwrap(),
            );
        };
        let s = time_per_call_ns(|| run(false));
        let p = time_per_call_ns(|| run(true));
        rows.push(Row {
            label: format!("conv {cin}c->{f}f {img}px k{k}"),
            work: flops,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "conv engine FLOPs gate (CONV_PARALLEL_MIN_FLOPS), batch == 1",
        work_unit: "FLOPs",
        pick_fastest: false,
        rows,
    }
}

// ---- pooling engine: POOL_PARALLEL_MIN_OPS ----

pub fn calibrate_pooling() -> Section {
    let mut rows = Vec::new();
    let run = |input: &Tensor, force: bool| {
        black_box(windowed_pool_forward_impl(
            input,
            &[2, 2],
            &[2, 2],
            PoolKind::Max,
            PaddingType::Valid,
            Some(force),
        ));
    };
    // Few large planes (batch 1, 3 channels - the case a plane-count gate starves)
    for &img in &[32usize, 64, 128, 256, 512, 1024] {
        let input = Tensor::from_elem(IxDyn(&[1, 3, img, img]), 1.0f32);
        let work = 3 * (img / 2) * (img / 2) * 4; // bc * plane_out * window taps
        let s = time_per_call_ns(|| run(&input, false));
        let p = time_per_call_ns(|| run(&input, true));
        rows.push(Row {
            label: format!("maxpool 1x3x{img}x{img}"),
            work,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    // Many tiny planes
    for &bc in &[16usize, 64, 256, 1024] {
        let input = Tensor::from_elem(IxDyn(&[bc, 1, 16, 16]), 1.0f32);
        let work = bc * 8 * 8 * 4;
        let s = time_per_call_ns(|| run(&input, false));
        let p = time_per_call_ns(|| run(&input, true));
        rows.push(Row {
            label: format!("maxpool {bc}x1x16x16"),
            work,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "pooling ops gate (POOL_PARALLEL_MIN_OPS)",
        work_unit: "window taps",
        pick_fastest: false,
        rows,
    }
}

// ---- elementwise kernel classes: activation / dropout / optimizer thresholds ----

pub fn calibrate_elementwise() -> Vec<Section> {
    let sizes = [
        512usize, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576,
    ];
    let mut sections = Vec::new();

    // Cheap op (ReLU-like): x.max(0) is idempotent, so in-place reapplication is stable
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut buf = Array1::from_elem(len, 0.5f32);
        let s = time_per_call_ns(|| {
            buf.mapv_inplace(|x| x.max(0.0));
            black_box(&buf);
        });
        let mut buf = Array1::from_elem(len, 0.5f32);
        let p = time_per_call_ns(|| {
            buf.par_mapv_inplace(|x| x.max(0.0));
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("relu-like max(0) {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "relu-like cheap map (CHEAP_MAP_PARALLEL_THRESHOLD class: ReLU, dropout masks)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Exp op (sigmoid-like): repeated sigmoid converges to a benign fixpoint in (0, 1)
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut buf = Array1::from_elem(len, 0.5f32);
        let s = time_per_call_ns(|| {
            buf.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            black_box(&buf);
        });
        let mut buf = Array1::from_elem(len, 0.5f32);
        let p = time_per_call_ns(|| {
            buf.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("sigmoid-like exp {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "sigmoid/tanh-like exp map (EXP_MAP_PARALLEL_THRESHOLD class: sigmoid, tanh, softmax)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Per-element RNG (dropout-like): Bernoulli draw per element; the parallel path mirrors the
    // per-chunk-rng pattern a parallel dropout needs
    let mut rows = Vec::new();
    for &len in &sizes {
        let mut rng = StdRng::seed_from_u64(7);
        let mut buf = vec![0.0f32; len];
        let s = time_per_call_ns(|| {
            for x in buf.iter_mut() {
                *x = if rng.random::<f32>() < 0.5 { 0.0 } else { 2.0 };
            }
            black_box(&buf);
        });
        let mut buf = vec![0.0f32; len];
        let p = time_per_call_ns(|| {
            buf.par_chunks_mut(4096).enumerate().for_each(|(i, chunk)| {
                let mut rng = StdRng::seed_from_u64(7 + i as u64);
                for x in chunk.iter_mut() {
                    *x = if rng.random::<f32>() < 0.5 { 0.0 } else { 2.0 };
                }
            });
            black_box(&buf);
        });
        rows.push(Row {
            label: format!("dropout-like rng {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "dropout-like per-element rng (reference only: the dropout layers keep one serial rng stream for seed reproducibility)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    // Adam-style fused multi-slice update (optimizer kernels' PARALLEL_THRESHOLD)
    let mut rows = Vec::new();
    for &len in &sizes {
        let grad = vec![0.01f32; len];
        let step = |p: &mut f32, g: f32, m: &mut f32, v: &mut f32| {
            *m = 0.9 * *m + 0.1 * g;
            *v = 0.999 * *v + 0.001 * g * g;
            *p -= 0.001 * *m / (v.sqrt() + 1e-8);
        };
        let mut param = vec![1.0f32; len];
        let mut m = vec![0.0f32; len];
        let mut v = vec![0.0f32; len];
        let s = time_per_call_ns(|| {
            for ((p, &g), (m, v)) in param
                .iter_mut()
                .zip(grad.iter())
                .zip(m.iter_mut().zip(v.iter_mut()))
            {
                step(p, g, m, v);
            }
            black_box(&param);
        });
        let mut param = vec![1.0f32; len];
        let mut m = vec![0.0f32; len];
        let mut v = vec![0.0f32; len];
        let p = time_per_call_ns(|| {
            param
                .par_iter_mut()
                .zip(grad.par_iter())
                .zip(m.par_iter_mut().zip(v.par_iter_mut()))
                .for_each(|((p, &g), (m, v))| step(p, g, m, v));
            black_box(&param);
        });
        rows.push(Row {
            label: format!("adam-like fused {len}"),
            work: len,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    sections.push(Section {
        title: "adam-like fused slice update (FUSED_SLICE_PARALLEL_THRESHOLD class: optimizer kernels)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    });

    sections
}

// ---- spatial-dropout per-channel scale: SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS ----

/// The spatial-dropout fused scale: multiply each `(batch, channel)` segment of a
/// `[batch, channels, *spatial]` tensor by its channel's inverted-dropout factor, in one pass
/// writing a fresh output (matching the production allocate-per-call). Forced serial vs forced
/// parallel of the same per-segment scale; each element is independent, so the flag never
/// changes the bits and the gate is a pure performance knob. The ladder varies the segment
/// count (`B * C`) and segment length (`spatial`) across the crossover.
pub fn calibrate_spatial_dropout_scale() -> Section {
    let mut rows = Vec::new();
    for &(n_seg, seg) in &[
        (64usize, 256usize),
        (128, 512),
        (256, 1024),
        (512, 2048),
        (256, 16_384),
        (2_048, 4_096),
    ] {
        let total = n_seg * seg;
        let src = vec![0.5f32; total];
        // The multiply cost is the same whether a channel is kept or dropped
        let mask = vec![1.0f32; n_seg];
        let scale = 1.0 / (1.0 - 0.2);
        let run = |parallel: bool| {
            let mut out = vec![0.0f32; total];
            let task = |((o, x), &m): ((&mut [f32], &[f32]), &f32)| {
                let factor = m * scale;
                for (o_elem, &x_elem) in o.iter_mut().zip(x) {
                    *o_elem = x_elem * factor;
                }
            };
            if parallel {
                out.par_chunks_mut(seg)
                    .zip(src.par_chunks(seg))
                    .zip(mask.par_iter())
                    .for_each(task);
            } else {
                out.chunks_mut(seg)
                    .zip(src.chunks(seg))
                    .zip(mask.iter())
                    .for_each(task);
            }
            black_box(out);
        };
        let s = time_per_call_ns(|| run(false));
        let p = time_per_call_ns(|| run(true));
        rows.push(Row {
            label: format!("{n_seg}seg x {seg}"),
            work: total,
            serial_ns: s,
            parallel_ns: p,
        });
    }
    Section {
        title: "spatial-dropout per-channel scale (SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS)",
        work_unit: "elements",
        pick_fastest: false,
        rows,
    }
}
