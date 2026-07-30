//! Shared helpers for the `utils` integration tests

#![allow(dead_code)]

use ndarray::{ArrayBase, Data, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;

/// Creates a deterministically seeded RNG, so a failing test reproduces with the same seed
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Asserts 2 arrays are element-wise equal within `eps` (absolute difference)
///
/// For a single scalar, use approx's `assert_abs_diff_eq!` or `assert_relative_eq!` directly
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
