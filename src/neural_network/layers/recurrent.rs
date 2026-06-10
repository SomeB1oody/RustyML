//! Recurrent layers (SimpleRNN, GRU, LSTM) and their shared helpers
//!
//! Re-exports the layer types and provides utilities used across them, such as
//! stable sigmoid activation and orthogonal weight initialization

use ndarray::{Array, Array2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Gradient clipping bound shared by the recurrent layers (SimpleRNN, GRU, LSTM) to curb
/// exploding gradients during backpropagation through time
const GRADIENT_CLIP_VALUE: f32 = 5.0;

/// Applies stable sigmoid activation to an array
///
/// Clamps the input before computing sigmoid to prevent numerical overflow,
/// then computes sigmoid. Used by both GRU and LSTM gates
#[inline]
fn apply_sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|x| {
        let clipped_x = x.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-clipped_x).exp())
    })
}

/// Generates a square orthogonal matrix via Gram-Schmidt orthonormalization of a random matrix
///
/// Unlike independently normalizing each column (which only yields unit-norm columns), this makes
/// the columns mutually orthonormal. Orthogonal recurrent weights keep the hidden-state transition
/// norm-preserving, mitigating vanishing/exploding gradients. Used to initialize the recurrent
/// kernels of SimpleRNN and the GRU/LSTM gates
fn orthogonal_init(size: usize, rng: &mut StdRng) -> Array2<f32> {
    // Random starting matrix; Gram-Schmidt below orthonormalizes its columns in place
    let mut matrix = Array::random_using((size, size), Uniform::new(-1.0, 1.0).unwrap(), rng);

    const EPSILON: f32 = 1e-8;

    for i in 0..size {
        // Orthogonalize column i against all previously finalized (already normalized) columns,
        // so each projection denominator is 1
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

    /// sigmoid at the clamp boundary 500 is ~= 1.0 with no overflow
    #[test]
    fn apply_sigmoid_large_positive_approaches_one() {
        let input = array![[500.0_f32]];
        let output = apply_sigmoid(input);
        // exp(-500) is ~7e-218, so the result is indistinguishable from 1.0
        assert_abs_diff_eq!(output[[0, 0]], 1.0_f32, epsilon = 1e-6);
    }

    /// Input 1000.0 clamps to 500.0 and yields the same finite value as sigmoid(500)
    #[test]
    fn apply_sigmoid_clamped_input_1000_same_as_500() {
        let out_500 = apply_sigmoid(array![[500.0_f32]]);
        let out_1000 = apply_sigmoid(array![[1000.0_f32]]);
        // Both must produce the same float (clamp to 500) and be ~= 1.0
        assert_abs_diff_eq!(out_500[[0, 0]], out_1000[[0, 0]], epsilon = 1e-9);
        assert_abs_diff_eq!(out_1000[[0, 0]], 1.0_f32, epsilon = 1e-6);
    }

    /// sigmoid(-1000) clamps to -500 and yields ~= 0.0
    #[test]
    fn apply_sigmoid_large_negative_approaches_zero() {
        let input = array![[-1000.0_f32]];
        let output = apply_sigmoid(input);
        assert_abs_diff_eq!(output[[0, 0]], 0.0_f32, epsilon = 1e-6);
    }

    /// Output stays finite for inputs far beyond the clamp boundaries
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
