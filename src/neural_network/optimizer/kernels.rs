//! Flat-slice optimizer update kernels.
//!
//! Each optimizer's per-parameter math lives here as a single function operating on `&mut [f32]`
//! parameter data plus `&[f32]` gradients (and any optimizer state slices). Because every layer
//! exposes its parameters as flat slices via [`Layer::parameters`](crate::neural_network::neural_network_trait::Layer::parameters),
//! these kernels work for any parameter shape — replacing the previous per-shape, per-optimizer
//! state structs and update implementations.

use rayon::prelude::*;

/// Element-count threshold above which a kernel switches to parallel evaluation.
const PARALLEL_THRESHOLD: usize = 1024;

/// SGD update: `param -= lr * grad`.
pub fn sgd_step(param: &mut [f32], grad: &[f32], lr: f32) {
    if param.len() >= PARALLEL_THRESHOLD {
        param
            .par_iter_mut()
            .zip(grad.par_iter())
            .for_each(|(p, &g)| *p -= lr * g);
    } else {
        for (p, &g) in param.iter_mut().zip(grad) {
            *p -= lr * g;
        }
    }
}

/// Adam update with bias correction at timestep `t`.
///
/// `m`/`v` are the first/second moment buffers for this parameter (same length as `param`).
///
/// ```text
/// m = beta1*m + (1-beta1)*g
/// v = beta2*v + (1-beta2)*g^2
/// param -= lr * (m / (1-beta1^t)) / (sqrt(v / (1-beta2^t)) + epsilon)
/// ```
#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u64,
) {
    let bias_correction1 = 1.0 - beta1.powi(t as i32);
    let bias_correction2 = 1.0 - beta2.powi(t as i32);

    let step = |p: &mut f32, g: f32, m: &mut f32, v: &mut f32| {
        *m = beta1 * *m + (1.0 - beta1) * g;
        *v = beta2 * *v + (1.0 - beta2) * g * g;
        let m_hat = *m / bias_correction1;
        let v_hat = *v / bias_correction2;
        *p -= lr * m_hat / (v_hat.sqrt() + epsilon);
    };

    if param.len() >= PARALLEL_THRESHOLD {
        param
            .par_iter_mut()
            .zip(grad.par_iter())
            .zip(m.par_iter_mut())
            .zip(v.par_iter_mut())
            .for_each(|(((p, &g), m), v)| step(p, g, m, v));
    } else {
        for (((p, &g), m), v) in param
            .iter_mut()
            .zip(grad)
            .zip(m.iter_mut())
            .zip(v.iter_mut())
        {
            step(p, g, m, v);
        }
    }
}

/// RMSprop update.
///
/// `cache` is the running average of squared gradients for this parameter.
///
/// ```text
/// cache = rho*cache + (1-rho)*g^2
/// param -= lr * g / (sqrt(cache) + epsilon)
/// ```
pub fn rmsprop_step(
    param: &mut [f32],
    grad: &[f32],
    cache: &mut [f32],
    rho: f32,
    lr: f32,
    epsilon: f32,
) {
    let step = |p: &mut f32, g: f32, c: &mut f32| {
        *c = rho * *c + (1.0 - rho) * g * g;
        *p -= lr * g / (c.sqrt() + epsilon);
    };

    if param.len() >= PARALLEL_THRESHOLD {
        param
            .par_iter_mut()
            .zip(grad.par_iter())
            .zip(cache.par_iter_mut())
            .for_each(|((p, &g), c)| step(p, g, c));
    } else {
        for ((p, &g), c) in param.iter_mut().zip(grad).zip(cache.iter_mut()) {
            step(p, g, c);
        }
    }
}

/// AdaGrad update.
///
/// `accumulator` is the running sum of squared gradients for this parameter.
///
/// ```text
/// accumulator += g^2
/// param -= lr * g / (sqrt(accumulator) + epsilon)
/// ```
pub fn adagrad_step(
    param: &mut [f32],
    grad: &[f32],
    accumulator: &mut [f32],
    lr: f32,
    epsilon: f32,
) {
    let step = |p: &mut f32, g: f32, a: &mut f32| {
        *a += g * g;
        *p -= lr * g / (a.sqrt() + epsilon);
    };

    if param.len() >= PARALLEL_THRESHOLD {
        param
            .par_iter_mut()
            .zip(grad.par_iter())
            .zip(accumulator.par_iter_mut())
            .for_each(|((p, &g), a)| step(p, g, a));
    } else {
        for ((p, &g), a) in param.iter_mut().zip(grad).zip(accumulator.iter_mut()) {
            step(p, g, a);
        }
    }
}
