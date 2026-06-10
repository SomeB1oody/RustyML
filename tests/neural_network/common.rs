//! Shared helpers for the `neural_network` integration tests

#![allow(dead_code)]

use ndarray::{ArrayBase, Data, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;

/// A deterministically seeded RNG, for reproducible tests
///
/// Always seed test RNGs, never the thread RNG, so failures stay reproducible
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// RAII guard for the crate-wide thread-local global seed
///
/// [`GlobalSeedGuard::set`] installs the global seed (see [`rustyml::set_global_seed`]) and
/// **clears it on drop - including on panic/unwind**. This keeps a *failing* global-seed test
/// from leaking its seed into a sibling test that shares the thread. The default parallel
/// harness spawns a fresh thread per test, but under `--test-threads=1` every test runs on the
/// same thread, so panic-safe cleanup is what makes the isolation hold there
///
/// Always bind the guard to a variable; `GlobalSeedGuard::set(s);` (unbound) drops immediately
/// and clears the seed before use - hence `#[must_use]`
#[must_use = "bind the guard to a variable; an unbound guard clears the seed immediately"]
pub struct GlobalSeedGuard;

impl GlobalSeedGuard {
    /// Installs `seed` as the thread-local global seed; cleared when the returned guard drops
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

/// Asserts two arrays/tensors are element-wise equal within `eps` (absolute difference)
///
/// For single scalars, use approx's `assert_abs_diff_eq!` / `assert_relative_eq!` macros directly
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
