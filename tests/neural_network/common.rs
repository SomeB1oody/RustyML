//! Shared helpers for the `neural_network` integration tests.

#![allow(dead_code)]

use ndarray::{ArrayBase, Data, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;

/// A deterministically seeded RNG, for reproducible tests.
///
/// Always seed test RNGs. Never seed the thread RNG. This keeps failures reproducible.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// RAII guard for the crate-wide thread-local global seed.
///
/// [`GlobalSeedGuard::set`] installs the global seed (see [`rustyml::set_global_seed`]). The
/// guard clears the seed on drop, even during a panic. This keeps a failing global-seed test
/// from leaking its seed into a sibling test on the same thread.
///
/// The default test harness runs each test on its own thread. Under `--test-threads=1`, every
/// test shares 1 thread. The panic-safe clear on drop is what keeps the tests isolated there.
///
/// Always bind the guard to a variable. An unbound guard drops immediately and clears the seed
/// before the test runs. This is why the type carries `#[must_use]`.
#[must_use = "bind the guard to a variable; an unbound guard clears the seed immediately"]
pub struct GlobalSeedGuard;

impl GlobalSeedGuard {
    /// Installs `seed` as the thread-local global seed. Clears it when the returned guard drops.
    pub fn set(seed: u64) -> Self {
        rustyml::set_global_seed(seed);
        GlobalSeedGuard
    }
}

impl Drop for GlobalSeedGuard {
    fn drop(&mut self) {
        rustyml::clear_global_seed();
    }
}

/// Asserts 2 arrays or tensors are element-wise equal within `eps` (absolute difference).
///
/// For single scalars, use the `assert_abs_diff_eq!` or `assert_relative_eq!` macro from
/// `approx` directly.
pub fn assert_allclose<A, S1, S2, D>(actual: &ArrayBase<S1, D>, expected: &ArrayBase<S2, D>, eps: A)
where
    A: approx::AbsDiffEq<Epsilon = A> + Copy + std::fmt::Debug,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch: actual {:?} vs expected {:?}",
        actual.shape(),
        expected.shape()
    );
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            a.abs_diff_eq(e, eps),
            "element mismatch: actual {a:?} vs expected {e:?} (eps {eps:?})"
        );
    }
}
