//! Flat-slice optimizer update kernels
//!
//! Each optimizer's per-parameter math lives here as a single function operating on `&mut [f32]`
//! parameter data plus `&[f32]` gradients (and any optimizer state slices). Because every layer
//! exposes its parameters as flat slices via [`Layer::parameters`](crate::neural_network::traits::Layer::parameters),
//! these kernels work for any parameter shape

use crate::parallel_gates::FUSED_SLICE_PARALLEL_THRESHOLD;
use rayon::prelude::*;
use std::borrow::Cow;

/// Scales a gradient by `grad_scale`, used to apply clip-by-global-norm before the optimizer step
///
/// Returns a borrow of the original gradient when `grad_scale == 1.0` (the no-clip case), so the
/// common path allocates nothing; only an active clip (`grad_scale < 1.0`) materializes a scaled copy
///
/// # Parameters
///
/// - `grad` - the gradient slice to scale
/// - `grad_scale` - the scaling factor; `1.0` means no clipping
///
/// # Returns
///
/// - `Cow<[f32]>` - a borrow of `grad` when `grad_scale == 1.0`, otherwise an owned scaled copy
pub fn scaled_grad(grad: &[f32], grad_scale: f32) -> Cow<'_, [f32]> {
    if grad_scale == 1.0 {
        Cow::Borrowed(grad)
    } else {
        Cow::Owned(grad.iter().map(|&g| g * grad_scale).collect())
    }
}

/// SGD update: `param -= lr * grad`
///
/// # Parameters
///
/// - `param` - the parameter slice to update in place
/// - `grad` - the gradient slice
/// - `lr` - the learning rate
pub fn sgd_step(param: &mut [f32], grad: &[f32], lr: f32) {
    if param.len() >= FUSED_SLICE_PARALLEL_THRESHOLD {
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

/// SGD with (optionally Nesterov) momentum
///
/// ```text
/// v = momentum*v + grad
/// step = if nesterov { grad + momentum*v } else { v }
/// param -= lr * step
/// ```
///
/// # Parameters
///
/// - `param` - the parameter slice to update in place
/// - `grad` - the gradient slice
/// - `velocity` - the per-parameter momentum buffer, updated in place
/// - `lr` - the learning rate
/// - `momentum` - the momentum factor
/// - `nesterov` - whether to use the Nesterov look-ahead step
pub fn sgd_momentum_step(
    param: &mut [f32],
    grad: &[f32],
    velocity: &mut [f32],
    lr: f32,
    momentum: f32,
    nesterov: bool,
) {
    let step = |p: &mut f32, g: f32, v: &mut f32| {
        *v = momentum * *v + g;
        let s = if nesterov { g + momentum * *v } else { *v };
        *p -= lr * s;
    };
    if param.len() >= FUSED_SLICE_PARALLEL_THRESHOLD {
        param
            .par_iter_mut()
            .zip(grad.par_iter())
            .zip(velocity.par_iter_mut())
            .for_each(|((p, &g), v)| step(p, g, v));
    } else {
        for ((p, &g), v) in param.iter_mut().zip(grad).zip(velocity.iter_mut()) {
            step(p, g, v);
        }
    }
}

/// Decoupled (AdamW/SGDW-style) weight decay
///
/// Shrinks each parameter toward zero by the factor `(1 - lr * weight_decay)`, independent of the
/// gradient. A no-op when `weight_decay == 0`. Apply this *before* the optimizer's gradient step so
/// the shrink uses the pre-step parameter, matching the AdamW formulation
/// `theta <- theta - lr*(update + wd*theta)`. "Decoupled" means the penalty acts directly on the
/// weights rather than being folded into the gradient (where an adaptive optimizer would rescale it
/// inconsistently)
///
/// # Parameters
///
/// - `param` - the parameter slice to shrink in place
/// - `lr` - the learning rate
/// - `weight_decay` - the decay coefficient; `0` skips the shrink
pub fn apply_weight_decay(param: &mut [f32], lr: f32, weight_decay: f32) {
    if weight_decay == 0.0 {
        return;
    }
    let factor = 1.0 - lr * weight_decay;
    if param.len() >= FUSED_SLICE_PARALLEL_THRESHOLD {
        param.par_iter_mut().for_each(|p| *p *= factor);
    } else {
        for p in param.iter_mut() {
            *p *= factor;
        }
    }
}

/// Coupled (classic-Adam) L2 weight decay: returns `grad + weight_decay * param`, evaluated
/// against the pre-step parameter
///
/// Unlike [`apply_weight_decay`]'s decoupled shrink, this folds the penalty into the gradient, so
/// it flows through Adam's moment estimates and is rescaled by the adaptive denominator (the
/// classic `Adam(weight_decay=...)` behavior, as opposed to `AdamW`). Only called when
/// `weight_decay != 0`, so it always allocates the combined buffer
///
/// # Parameters
///
/// - `grad` - the gradient slice
/// - `param` - the pre-step parameter slice
/// - `weight_decay` - the L2 coefficient
///
/// # Returns
///
/// - `Vec<f32>` - the combined gradient `grad + weight_decay * param`
pub fn l2_regularized_grad(grad: &[f32], param: &[f32], weight_decay: f32) -> Vec<f32> {
    grad.iter()
        .zip(param)
        .map(|(&g, &p)| g + weight_decay * p)
        .collect()
}

/// Adam update with bias correction at timestep `t`
///
/// ```text
/// m = beta1*m + (1-beta1)*g
/// v = beta2*v + (1-beta2)*g^2
/// param -= lr * (m / (1-beta1^t)) / (sqrt(v / (1-beta2^t)) + epsilon)
/// ```
///
/// # Parameters
///
/// - `param` - the parameter slice to update in place
/// - `grad` - the gradient slice
/// - `m` - the first-moment buffer (same length as `param`), updated in place
/// - `v` - the second-moment buffer (same length as `param`), updated in place
/// - `lr` - the learning rate
/// - `beta1` - the first-moment decay rate
/// - `beta2` - the second-moment decay rate
/// - `epsilon` - the denominator stabilizer
/// - `t` - the 1-based timestep used for bias correction
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

    if param.len() >= FUSED_SLICE_PARALLEL_THRESHOLD {
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

/// RMSprop update
///
/// ```text
/// cache = rho*cache + (1-rho)*g^2
/// param -= lr * g / (sqrt(cache) + epsilon)
/// ```
///
/// # Parameters
///
/// - `param` - the parameter slice to update in place
/// - `grad` - the gradient slice
/// - `cache` - the running average of squared gradients, updated in place
/// - `rho` - the decay rate for the squared-gradient average
/// - `lr` - the learning rate
/// - `epsilon` - the denominator stabilizer
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

    if param.len() >= FUSED_SLICE_PARALLEL_THRESHOLD {
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

/// AdaGrad update
///
/// ```text
/// accumulator += g^2
/// param -= lr * g / (sqrt(accumulator) + epsilon)
/// ```
///
/// # Parameters
///
/// - `param` - the parameter slice to update in place
/// - `grad` - the gradient slice
/// - `accumulator` - the running sum of squared gradients, updated in place
/// - `lr` - the learning rate
/// - `epsilon` - the denominator stabilizer
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

    if param.len() >= FUSED_SLICE_PARALLEL_THRESHOLD {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // scaled_grad

    /// A scale of exactly 1.0 borrows the original gradient without allocating or changing values
    #[test]
    fn scaled_grad_unit_scale_borrows_unchanged() {
        let grad = vec![1.0_f32, -2.0, 3.5];
        let out = scaled_grad(&grad, 1.0);
        assert!(matches!(out, Cow::Borrowed(_)), "unit scale must borrow");
        assert_eq!(out.as_ref(), grad.as_slice());
    }

    /// A non-unit scale multiplies every element (owned copy)
    #[test]
    fn scaled_grad_nonunit_scale_multiplies() {
        let grad = vec![2.0_f32, -4.0, 6.0];
        let out = scaled_grad(&grad, 0.5);
        assert!(matches!(out, Cow::Owned(_)), "non-unit scale must own");
        assert_abs_diff_eq!(out[0], 1.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[1], -2.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[2], 3.0_f32, epsilon = 1e-6);
    }

    // SGD momentum / weight decay

    /// Plain (non-Nesterov) momentum accumulates velocity across steps: v = mu*v + g, p -= lr*v
    #[test]
    fn sgd_momentum_step_accumulates() {
        let mut param = vec![1.0_f32];
        let grad = vec![2.0_f32];
        let mut v = vec![0.0_f32];
        // step 1: v = 2, p = 1 - 0.1*2 = 0.8
        sgd_momentum_step(&mut param, &grad, &mut v, 0.1, 0.9, false);
        assert_abs_diff_eq!(v[0], 2.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[0], 0.8_f32, epsilon = 1e-6);
        // step 2: v = 0.9*2 + 2 = 3.8, p = 0.8 - 0.1*3.8 = 0.42
        sgd_momentum_step(&mut param, &grad, &mut v, 0.1, 0.9, false);
        assert_abs_diff_eq!(v[0], 3.8_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[0], 0.42_f32, epsilon = 1e-6);
    }

    /// Nesterov momentum uses the look-ahead step `grad + momentum*v`
    #[test]
    fn sgd_momentum_step_nesterov() {
        let mut param = vec![1.0_f32];
        let grad = vec![2.0_f32];
        let mut v = vec![0.0_f32];
        // v = 2, step = 2 + 0.9*2 = 3.8, p = 1 - 0.1*3.8 = 0.62
        sgd_momentum_step(&mut param, &grad, &mut v, 0.1, 0.9, true);
        assert_abs_diff_eq!(param[0], 0.62_f32, epsilon = 1e-6);
    }

    /// Decoupled weight decay scales the parameter by (1 - lr*wd) and is a no-op when wd == 0
    #[test]
    fn apply_weight_decay_scales_param() {
        let mut param = vec![1.0_f32, 2.0];
        apply_weight_decay(&mut param, 0.1, 0.5); // factor 1 - 0.05 = 0.95
        assert_abs_diff_eq!(param[0], 0.95_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[1], 1.9_f32, epsilon = 1e-6);

        let mut p2 = vec![3.0_f32];
        apply_weight_decay(&mut p2, 0.1, 0.0); // no-op
        assert_abs_diff_eq!(p2[0], 3.0_f32, epsilon = 1e-9);
    }

    /// Coupled L2 decay folds `weight_decay * param` into the gradient (used by classic Adam)
    #[test]
    fn l2_regularized_grad_adds_weight_decay_times_param() {
        let grad = vec![1.0_f32, -2.0, 0.0];
        let param = vec![10.0_f32, 4.0, -6.0];
        let out = l2_regularized_grad(&grad, &param, 0.1);
        // g + 0.1 * p
        assert_abs_diff_eq!(out[0], 1.0 + 1.0, epsilon = 1e-6); // 1 + 0.1*10
        assert_abs_diff_eq!(out[1], -2.0 + 0.4, epsilon = 1e-6); // -2 + 0.1*4
        assert_abs_diff_eq!(out[2], 0.0 - 0.6, epsilon = 1e-6); // 0 + 0.1*(-6)
    }

    // SGD

    /// SGD small path applies `param -= lr * grad`
    #[test]
    fn sgd_step_small_path() {
        let mut param = vec![1.0_f32, 2.0_f32];
        let grad = vec![0.5_f32, -1.0_f32];
        sgd_step(&mut param, &grad, 0.1);
        assert_abs_diff_eq!(param[0], 0.95_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[1], 2.1_f32, epsilon = 1e-6);
    }

    /// SGD parallel path (>=1024 elements) applies the same update via the rayon branch
    #[test]
    fn sgd_step_parallel_path() {
        let n = 1024_usize;
        let mut param = vec![1.0_f32; n];
        let grad = vec![0.5_f32; n];
        sgd_step(&mut param, &grad, 0.1);
        for (i, &p) in param.iter().enumerate() {
            assert!(
                (p - 0.95_f32).abs() <= 1e-6,
                "element {i} mismatch in parallel sgd_step: got {p}"
            );
        }
    }

    /// SGD with lr=0.0 leaves params unchanged
    #[test]
    fn sgd_step_zero_lr() {
        let mut param = vec![3.0_f32, -1.5_f32];
        let grad = vec![100.0_f32, -50.0_f32];
        sgd_step(&mut param, &grad, 0.0);
        assert_abs_diff_eq!(param[0], 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[1], -1.5_f32, epsilon = 1e-6);
    }

    // Adam

    /// Adam step t=1 from zero moments produces the canonical first update
    #[test]
    fn adam_step_t1_from_zero() {
        let mut param = vec![0.0_f32];
        let grad = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];

        adam_step(&mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 1);

        // m and v are updated first
        assert_abs_diff_eq!(m[0], 0.1_f32, epsilon = 1e-7);
        assert_abs_diff_eq!(v[0], 0.001_f32, epsilon = 1e-7);

        // param update: 0.0 - 0.1*(1.0/(1.0 + 1e-8)) ~= -0.09999999
        let expected_param = -0.1_f32 / (1.0_f32 + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }

    /// Adam moments accumulate correctly across two sequential t=1, t=2 calls
    #[test]
    fn adam_step_t2_moment_accumulation() {
        let mut param = vec![0.0_f32];
        let grad = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];

        adam_step(&mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 1);
        adam_step(&mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 2);

        // After t=2: m = 0.9*0.1 + 0.1 = 0.19
        assert_abs_diff_eq!(m[0], 0.19_f32, epsilon = 1e-6);
        // After t=2: v = 0.999*0.001 + 0.001 = 0.001999
        assert_abs_diff_eq!(v[0], 0.001999_f32, epsilon = 1e-6);

        // Both steps apply ~ -0.09999999, total ~= -0.19999998
        let single_update = 0.1_f32 / (1.0_f32 + 1e-8_f32);
        let expected = -2.0_f32 * single_update;
        assert_abs_diff_eq!(param[0], expected, epsilon = 1e-5);
    }

    /// Adam step with zero gradient leaves moments at zero and param unchanged
    #[test]
    fn adam_step_zero_gradient() {
        let mut param = vec![5.0_f32];
        let grad = vec![0.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];

        adam_step(&mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 1);

        // m = 0.9*0 + 0.1*0 = 0.0, v = 0, m_hat = 0, update = 0
        assert_abs_diff_eq!(m[0], 0.0_f32, epsilon = 1e-9);
        assert_abs_diff_eq!(v[0], 0.0_f32, epsilon = 1e-9);
        assert_abs_diff_eq!(param[0], 5.0_f32, epsilon = 1e-9);
    }

    // RMSprop

    /// RMSprop one step from zero cache produces the expected single-element update
    #[test]
    fn rmsprop_step_single_element() {
        let mut param = vec![2.0_f32];
        let grad = vec![1.0_f32];
        let mut cache = vec![0.0_f32];

        rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.01, 1e-8);

        let expected_cache = 0.1_f32;
        assert_abs_diff_eq!(cache[0], expected_cache, epsilon = 1e-7);

        let expected_param = 2.0_f32 - 0.01_f32 / (0.1_f32.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }

    /// RMSprop keeps independent per-element cache state across two elements
    #[test]
    fn rmsprop_step_two_elements() {
        let mut param = vec![2.0_f32, 3.0_f32];
        let grad = vec![1.0_f32, -0.5_f32];
        let mut cache = vec![0.0_f32, 0.0_f32];

        rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.01, 1e-8);

        // elem 0
        let expected_cache0 = 0.1_f32;
        assert_abs_diff_eq!(cache[0], expected_cache0, epsilon = 1e-7);
        let expected_p0 = 2.0_f32 - 0.01_f32 / (0.1_f32.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_p0, epsilon = 1e-6);

        // elem 1: cache = 0.1 * 0.25 = 0.025
        let expected_cache1 = 0.025_f32;
        assert_abs_diff_eq!(cache[1], expected_cache1, epsilon = 1e-7);
        // grad is -0.5, so param -= lr * (-0.5) / (sqrt(0.025)+eps) => param increases
        let expected_p1 = 3.0_f32 - 0.01_f32 * (-0.5_f32) / (0.025_f32.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[1], expected_p1, epsilon = 1e-6);
    }

    /// RMSprop with a non-zero initial cache carries state forward
    #[test]
    fn rmsprop_step_nonzero_initial_cache() {
        let mut param = vec![1.0_f32];
        let grad = vec![2.0_f32];
        let mut cache = vec![0.5_f32];

        rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.01, 1e-8);

        let expected_cache = 0.9_f32 * 0.5 + 0.1_f32 * 4.0;
        assert_abs_diff_eq!(cache[0], expected_cache, epsilon = 1e-6);

        let expected_param = 1.0_f32 - 0.01_f32 * 2.0_f32 / (expected_cache.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }

    // AdaGrad

    /// AdaGrad one step from zero accumulator produces the expected single-element update
    #[test]
    fn adagrad_step_single_element() {
        let mut param = vec![3.0_f32];
        let grad = vec![2.0_f32];
        let mut acc = vec![0.0_f32];

        adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);

        let expected_acc = 4.0_f32;
        assert_abs_diff_eq!(acc[0], expected_acc, epsilon = 1e-7);

        let expected_param = 3.0_f32 - 0.01_f32 * 2.0_f32 / (4.0_f32.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }

    /// AdaGrad accumulator grows monotonically as squared gradients accumulate across steps
    #[test]
    fn adagrad_step_accumulation_across_steps() {
        let mut param = vec![3.0_f32];
        let grad = vec![2.0_f32];
        let mut acc = vec![0.0_f32];

        adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);
        let param_after_step1 = param[0];

        adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);

        // accumulator must grow: 0 -> 4 -> 8
        assert_abs_diff_eq!(acc[0], 8.0_f32, epsilon = 1e-6);

        // second update uses sqrt(8)
        let expected_param = param_after_step1 - 0.01_f32 * 2.0_f32 / (8.0_f32.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }

    /// AdaGrad with a negative gradient still grows the accumulator and moves param positive
    #[test]
    fn adagrad_step_negative_gradient() {
        let mut param = vec![1.0_f32];
        let grad = vec![-0.5_f32];
        let mut acc = vec![0.0_f32];

        adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);

        // accumulator += (-0.5)^2 = 0.25
        assert_abs_diff_eq!(acc[0], 0.25_f32, epsilon = 1e-7);

        // param -= 0.01*(-0.5)/(sqrt(0.25)+1e-8) => param increases
        let expected_param = 1.0_f32 - 0.01_f32 * (-0.5_f32) / (0.25_f32.sqrt() + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }
    // Parallel-path coverage (>=1024 elements) for adam/rmsprop/adagrad: length FUSED_SLICE_PARALLEL_THRESHOLD forces the rayon branch

    /// Adam parallel path (1024 elements) applies the canonical t=1-from-zero update everywhere
    #[test]
    fn adam_step_parallel_path() {
        let n = 1024_usize; // == FUSED_SLICE_PARALLEL_THRESHOLD, forces the rayon branch
        let mut param = vec![0.0_f32; n];
        let grad = vec![1.0_f32; n];
        let mut m = vec![0.0_f32; n];
        let mut v = vec![0.0_f32; n];

        adam_step(&mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 1);

        let expected_param = -0.1_f32 / (1.0_f32 + 1e-8_f32);
        for i in 0..n {
            assert!(
                (m[i] - 0.1_f32).abs() <= 1e-7,
                "element {i} m mismatch in parallel adam_step: got {}",
                m[i]
            );
            assert!(
                (v[i] - 0.001_f32).abs() <= 1e-7,
                "element {i} v mismatch in parallel adam_step: got {}",
                v[i]
            );
            assert!(
                (param[i] - expected_param).abs() <= 1e-6,
                "element {i} param mismatch in parallel adam_step: got {}",
                param[i]
            );
        }
    }

    /// RMSprop parallel path (1024 elements) applies the same single-step update everywhere
    #[test]
    fn rmsprop_step_parallel_path() {
        let n = 1024_usize;
        let mut param = vec![2.0_f32; n];
        let grad = vec![1.0_f32; n];
        let mut cache = vec![0.0_f32; n];

        rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.01, 1e-8);

        let expected_param = 2.0_f32 - 0.01_f32 / (0.1_f32.sqrt() + 1e-8_f32);
        for i in 0..n {
            assert!(
                (cache[i] - 0.1_f32).abs() <= 1e-7,
                "element {i} cache mismatch in parallel rmsprop_step: got {}",
                cache[i]
            );
            assert!(
                (param[i] - expected_param).abs() <= 1e-6,
                "element {i} param mismatch in parallel rmsprop_step: got {}",
                param[i]
            );
        }
    }

    /// AdaGrad parallel path (1024 elements) applies the same single-step update everywhere
    #[test]
    fn adagrad_step_parallel_path() {
        let n = 1024_usize;
        let mut param = vec![3.0_f32; n];
        let grad = vec![2.0_f32; n];
        let mut acc = vec![0.0_f32; n];

        adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);

        let expected_param = 3.0_f32 - 0.01_f32 * 2.0_f32 / (4.0_f32.sqrt() + 1e-8_f32);
        for i in 0..n {
            assert!(
                (acc[i] - 4.0_f32).abs() <= 1e-7,
                "element {i} accumulator mismatch in parallel adagrad_step: got {}",
                acc[i]
            );
            assert!(
                (param[i] - expected_param).abs() <= 1e-6,
                "element {i} param mismatch in parallel adagrad_step: got {}",
                param[i]
            );
        }
    }
}
