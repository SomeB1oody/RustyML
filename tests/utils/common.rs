//! Shared helpers for the `utils` integration tests.
//! Provides a seeded random number generator and array assertions with absolute tolerances.

#![allow(dead_code)]

use ndarray::{ArrayBase, Data, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;

/// Creates a random number generator with the supplied seed.
///
/// # Parameters
///
/// - `seed` - seed for repeatable random values
///
/// # Returns
///
/// - `StdRng` - generator initialized with `seed`
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Checks that arrays have equal shapes and absolute element differences within `eps`.
///
/// Use approx's scalar assertion macros for scalar values.
///
/// # Parameters
///
/// - `actual` - array under test
/// - `expected` - reference array
/// - `eps` - maximum absolute difference for each element
///
/// # Panics
///
/// Panics if shapes differ or any element exceeds the tolerance.
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
