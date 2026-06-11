//! Recurrent layers (SimpleRNN, GRU, LSTM) and their shared helpers
//!
//! Re-exports the layer types and provides utilities used across them, such as
//! stable sigmoid activation and orthogonal weight initialization

use ndarray::{Array, Array2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Applies the logistic sigmoid to an array. Used by both GRU and LSTM gates
///
/// `1/(1 + e^-x)` is correct and finite for any finite `x` (when `e^-x` overflows to `+inf` the
/// result is the exact limit `0`), and it saturates to `1`/`0` at `±inf`, so no input clamping is
/// needed; the only non-finite output comes from a `NaN` input, which propagates
#[inline]
fn apply_sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Generates a square orthogonal matrix via Gram-Schmidt orthonormalization of a random matrix
///
/// Unlike independently normalizing each column (which only yields unit-norm columns), this makes
/// the columns mutually orthonormal. Orthogonal recurrent weights keep the hidden-state transition
/// norm-preserving, mitigating vanishing/exploding gradients. Used to initialize the recurrent
/// kernels of SimpleRNN and the GRU/LSTM gates
fn orthogonal_init(size: usize, rng: &mut StdRng) -> Array2<f32> {
    // Random starting matrix
    let mut matrix = Array::random_using((size, size), Uniform::new(-1.0, 1.0).unwrap(), rng);

    const EPSILON: f32 = 1e-8;

    for i in 0..size {
        // Orthogonalize column i against all previously finalized (already normalized) columns
        for j in 0..i {
            let mut projection = 0.0;
            for k in 0..size {
                projection += matrix[[k, i]] * matrix[[k, j]];
            }
            for k in 0..size {
                matrix[[k, i]] -= projection * matrix[[k, j]];
            }
        }

        // Normalize column i; fall back to a standard basis vector if it collapsed
        let mut norm = 0.0f32;
        for k in 0..size {
            norm += matrix[[k, i]] * matrix[[k, i]];
        }
        norm = norm.sqrt();

        if norm > EPSILON {
            for k in 0..size {
                matrix[[k, i]] /= norm;
            }
        } else {
            for k in 0..size {
                matrix[[k, i]] = if k == i { 1.0 } else { 0.0 };
            }
        }
    }

    matrix
}

/// Gate structure for recurrent cell operations (GRU, LSTM)
pub mod gate;
/// A GRU (Gated Recurrent Unit) layer implementation
pub mod gru;
/// A LSTM (Long Short-Term Memory) neural network layer implementation
pub mod lstm;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation
pub mod simple_rnn;
/// Input validation functions for Recurrent layers
mod validation;

pub use gru::GRU;
pub use lstm::LSTM;
pub use simple_rnn::SimpleRNN;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;

    // orthogonal_init

    /// For size=3, M^T M equals the 3x3 identity within 1e-5
    #[test]
    fn orthogonal_init_size3_columns_are_orthonormal() {
        let m = orthogonal_init(3, &mut StdRng::seed_from_u64(0));

        // Compute M^T M (should equal I_3)
        let mt_m = m.t().dot(&m);

        // Seeded draw is deterministic, so seed 0 keeps the classical-GS f32 round-off within 1e-5
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0_f32 } else { 0.0_f32 };
                assert_abs_diff_eq!(mt_m[[row, col]], expected, epsilon = 1e-5);
            }
        }
    }

    /// For size=1 the single entry has absolute value 1.0 after normalization
    #[test]
    fn orthogonal_init_size1_abs_is_one() {
        let m = orthogonal_init(1, &mut StdRng::seed_from_u64(0));
        assert_eq!(m.shape(), &[1, 1]);
        assert_abs_diff_eq!(m[[0, 0]].abs(), 1.0_f32, epsilon = 1e-6);
    }

    // apply_sigmoid

    /// sigmoid(0) equals 0.5
    #[test]
    fn apply_sigmoid_zero_gives_half() {
        let input = array![[0.0_f32]];
        let output = apply_sigmoid(input);
        assert_abs_diff_eq!(output[[0, 0]], 0.5_f32, epsilon = 1e-6);
    }

    /// sigmoid saturates to ~= 1.0 for large positive input with no overflow
    #[test]
    fn apply_sigmoid_large_positive_approaches_one() {
        let input = array![[500.0_f32]];
        let output = apply_sigmoid(input);
        // exp(-500) underflows to 0 in f32, so the result is exactly 1.0
        assert_abs_diff_eq!(output[[0, 0]], 1.0_f32, epsilon = 1e-6);
    }

    /// Large positive inputs all saturate to the same value (1.0); no clamping is involved
    #[test]
    fn apply_sigmoid_large_positive_inputs_saturate_equally() {
        let out_500 = apply_sigmoid(array![[500.0_f32]]);
        let out_1000 = apply_sigmoid(array![[1000.0_f32]]);
        // Both saturate to exactly 1.0 (exp(-x) underflows to 0)
        assert_abs_diff_eq!(out_500[[0, 0]], out_1000[[0, 0]], epsilon = 1e-9);
        assert_abs_diff_eq!(out_1000[[0, 0]], 1.0_f32, epsilon = 1e-6);
    }

    /// sigmoid(-1000) saturates to ~= 0.0
    #[test]
    fn apply_sigmoid_large_negative_approaches_zero() {
        let input = array![[-1000.0_f32]];
        let output = apply_sigmoid(input);
        assert_abs_diff_eq!(output[[0, 0]], 0.0_f32, epsilon = 1e-6);
    }

    /// Output stays finite even for ±inf and huge-magnitude inputs (no clamp needed)
    #[test]
    fn apply_sigmoid_no_nan_or_inf() {
        let input = array![[
            f32::NEG_INFINITY,
            -1e10,
            -1.0,
            0.0,
            1.0,
            1e10,
            f32::INFINITY
        ]];
        let output = apply_sigmoid(input);
        for &v in output.iter() {
            assert!(
                v.is_finite(),
                "apply_sigmoid produced non-finite value: {v}"
            );
        }
    }
}
