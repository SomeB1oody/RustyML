//! Shared helpers for the `metrics` integration tests
//!
//! The `metrics` feature does not pull in `ndarray-rand`, so there is no seeded-RNG
//! helper here - metric tests are deterministic closed-form checks against ground truth

#![allow(dead_code)]

use ndarray::{ArrayBase, Data, Dimension};

/// Asserts two arrays are element-wise equal within `eps` (absolute difference)
///
/// Use for array comparisons. For single scalars, use approx's
/// `assert_abs_diff_eq!` / `assert_relative_eq!` macros directly
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
