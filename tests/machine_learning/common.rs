//! Shared helpers for the `machine_learning` integration tests.

#![allow(dead_code)]

use ndarray::{ArrayBase, Data, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;

/// Builds a deterministically seeded RNG for reproducible tests.
///
/// Test RNGs always use a fixed seed instead of the thread RNG, so a failure reproduces.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Asserts that 2 arrays are element-wise equal within `eps` (absolute difference).
///
/// For a single scalar, use approx's `assert_abs_diff_eq!` or `assert_relative_eq!` macro
/// directly.
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
