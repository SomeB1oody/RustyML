//! Flat-slice optimizer update kernels.
//!
//! Each optimizer's per-parameter math lives here as a single function operating on `&mut [f32]`
//! parameter data plus `&[f32]` gradients (and any optimizer state slices). Because every layer
//! exposes its parameters as flat slices via [`Layer::parameters`](crate::neural_network::traits::Layer::parameters),
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── SGD ─────────────────────────────────────────────────────────────────

    /// SGD small path: param -= lr * grad
    /// param=[1.0, 2.0], grad=[0.5, -1.0], lr=0.1
    /// elem 0: 1.0 - 0.1*0.5 = 0.95
    /// elem 1: 2.0 - 0.1*(-1.0) = 2.1
    #[test]
    fn sgd_step_small_path() {
        let mut param = vec![1.0_f32, 2.0_f32];
        let grad = vec![0.5_f32, -1.0_f32];
        sgd_step(&mut param, &grad, 0.1);
        assert_abs_diff_eq!(param[0], 0.95_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[1], 2.1_f32, epsilon = 1e-6);
    }

    /// SGD parallel path (>=1024 elements): same formula, exercises rayon branch.
    /// param=vec![1.0; 1024], grad=vec![0.5; 1024], lr=0.1
    /// every element: 1.0 - 0.1*0.5 = 0.95
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

    /// SGD with lr=0.0 leaves params unchanged.
    #[test]
    fn sgd_step_zero_lr() {
        let mut param = vec![3.0_f32, -1.5_f32];
        let grad = vec![100.0_f32, -50.0_f32];
        sgd_step(&mut param, &grad, 0.0);
        assert_abs_diff_eq!(param[0], 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(param[1], -1.5_f32, epsilon = 1e-6);
    }

    // ── Adam ─────────────────────────────────────────────────────────────────

    /// Adam step t=1 from zero moments (standard canonical example).
    ///
    /// Inputs: param=0.0, grad=1.0, m=0.0, v=0.0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8
    ///
    /// Step-by-step (from the docstring formula):
    ///   m  = 0.9*0 + 0.1*1     = 0.1
    ///   v  = 0.999*0 + 0.001*1 = 0.001
    ///   bc1 = 1 - 0.9^1        = 0.1
    ///   bc2 = 1 - 0.999^1      = 0.001
    ///   m_hat = 0.1 / 0.1      = 1.0
    ///   v_hat = 0.001 / 0.001  = 1.0
    ///   update = 0.1 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.09999999
    ///   param  = 0.0 - 0.09999999 ≈ -0.09999999
    #[test]
    fn adam_step_t1_from_zero() {
        let mut param = vec![0.0_f32];
        let grad = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];

        adam_step(&mut param, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 1);

        // m and v should be updated first
        assert_abs_diff_eq!(m[0], 0.1_f32, epsilon = 1e-7);
        assert_abs_diff_eq!(v[0], 0.001_f32, epsilon = 1e-7);

        // param update: 0.0 - 0.1*(1.0/(1.0 + 1e-8)) ≈ -0.09999999
        let expected_param = -0.1_f32 / (1.0_f32 + 1e-8_f32);
        assert_abs_diff_eq!(param[0], expected_param, epsilon = 1e-6);
    }

    /// Adam step t=2: moments accumulate correctly over two sequential calls.
    ///
    /// After t=1 (same as above), for t=2 with same grad=1.0:
    ///   m  = 0.9*0.1 + 0.1*1      = 0.19
    ///   v  = 0.999*0.001 + 0.001*1 = 0.001999
    ///   bc1 = 1 - 0.9^2            = 0.19
    ///   bc2 = 1 - 0.999^2          = 0.001999
    ///   m_hat = 0.19/0.19           = 1.0
    ///   v_hat = 0.001999/0.001999   = 1.0
    ///   update ≈ 0.1 * 1.0 / (1.0 + 1e-8) ≈ 0.09999999 again
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

        // Both steps apply ~ -0.09999999, total ≈ -0.19999998
        let single_update = 0.1_f32 / (1.0_f32 + 1e-8_f32);
        let expected = -2.0_f32 * single_update;
        assert_abs_diff_eq!(param[0], expected, epsilon = 1e-5);
    }

    /// Adam step with zero gradient leaves moments at zero and param unchanged.
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

    // ── RMSprop ───────────────────────────────────────────────────────────────

    /// RMSprop one step from zero cache.
    ///
    /// Inputs: param=2.0, grad=1.0, cache=0.0, rho=0.9, lr=0.01, epsilon=1e-8
    ///
    /// cache = 0.9*0 + (1-0.9)*1^2 = 0.1
    /// param -= 0.01*1.0 / (sqrt(0.1) + 1e-8)
    ///        = 0.01 / 0.31622776601... ≈ 0.031622776601
    /// param ≈ 2.0 - 0.031622776601 ≈ 1.968377224
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

    /// RMSprop with two elements to verify independent per-element state.
    ///
    /// param=[2.0, 3.0], grad=[1.0, -0.5], cache=[0.0, 0.0], rho=0.9, lr=0.01, eps=1e-8
    ///
    /// elem 0: cache = 0.1, param -= 0.01/(sqrt(0.1)+1e-8)
    /// elem 1: cache = 0.9*0 + 0.1*0.25 = 0.025,
    ///         param -= 0.01*(-0.5)/(sqrt(0.025)+1e-8) = +0.005/sqrt(0.025)
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

    /// RMSprop with non-zero initial cache carries state forward.
    ///
    /// cache_0 = 0.5, grad = 2.0, rho=0.9, lr=0.01, eps=1e-8
    /// cache = 0.9*0.5 + 0.1*4 = 0.45 + 0.4 = 0.85
    /// param -= 0.01*2 / (sqrt(0.85)+1e-8)
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

    // ── AdaGrad ───────────────────────────────────────────────────────────────

    /// AdaGrad one step from zero accumulator.
    ///
    /// Inputs: param=3.0, grad=2.0, accumulator=0.0, lr=0.01, epsilon=1e-8
    ///
    /// accumulator += 2^2 = 4.0
    /// param -= 0.01*2.0 / (sqrt(4.0) + 1e-8)
    ///        = 0.02 / (2.0 + 1e-8) ≈ 0.009999999950
    /// param ≈ 3.0 - 0.009999999950 ≈ 2.990000005
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

    /// AdaGrad accumulates squared gradients across two steps (grows monotonically).
    ///
    /// Step 1: acc = 0 + 2^2 = 4.0, param -= 0.01*2/(2+1e-8)
    /// Step 2: acc = 4 + 2^2 = 8.0, param -= 0.01*2/(sqrt(8)+1e-8)
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

    /// AdaGrad with negative gradient: accumulator still grows (g^2 is always positive),
    /// and param moves in the positive direction.
    ///
    /// param=1.0, grad=-0.5, acc=0.0, lr=0.01, eps=1e-8
    /// acc = 0.25
    /// param -= 0.01*(-0.5)/(sqrt(0.25)+1e-8) = +0.005/(0.5+1e-8)
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
    // ── Parallel-path coverage (>=1024 elements) for adam/rmsprop/adagrad ───────
    //
    // Each of these mirrors the existing `sgd_step_parallel_path` test: with uniform
    // inputs of length PARALLEL_THRESHOLD the rayon branch must produce exactly the
    // documented element-wise update. Expected values are derived from the docstring
    // formula (NOT by running the impl), reusing the per-element hand calculations
    // already verified in the small-path tests above.

    /// Adam parallel path (1024 elements): same canonical t=1-from-zero update on every
    /// element. param=0.0, grad=1.0, m=0.0, v=0.0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8.
    ///   m=0.1, v=0.001, bc1=0.1, bc2=0.001, m_hat=1.0, v_hat=1.0
    ///   param = 0.0 - 0.1*(1.0/(1.0+1e-8)) ≈ -0.09999999
    #[test]
    fn adam_step_parallel_path() {
        let n = 1024_usize; // == PARALLEL_THRESHOLD, forces the rayon branch
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

    /// RMSprop parallel path (1024 elements): same single-step update on every element.
    /// param=2.0, grad=1.0, cache=0.0, rho=0.9, lr=0.01, eps=1e-8.
    ///   cache = 0.9*0 + 0.1*1^2 = 0.1
    ///   param = 2.0 - 0.01/(sqrt(0.1)+1e-8) ≈ 1.96837722
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

    /// AdaGrad parallel path (1024 elements): same single-step update on every element.
    /// param=3.0, grad=2.0, accumulator=0.0, lr=0.01, eps=1e-8.
    ///   accumulator = 0 + 2^2 = 4.0
    ///   param = 3.0 - 0.01*2.0/(sqrt(4.0)+1e-8) ≈ 2.99000000
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
