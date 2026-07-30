//! Recurrent layers (SimpleRNN, GRU, LSTM) and their shared helpers
//!
//! Re-exports the 3 layer types and provides 2 shared helpers. A numerically stable sigmoid
//! serves the GRU and LSTM gates, and Gram-Schmidt orthogonal initialization serves all 3 layers'
//! recurrent kernels.

use ndarray::{Array, Array2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Applies the logistic sigmoid to an array
///
/// The GRU and LSTM cells call this on their raw gate pre-activations, after their fused
/// recurrent GEMM. gemmkit's fused epilogue does not support a sigmoid activation.
///
/// `1/(1 + e^-x)` stays finite for any finite `x`. For a very negative `x`, `e^-x` overflows to
/// positive infinity and the result rounds to the exact limit `0`. The function saturates to `1`
/// or `0` at positive or negative infinity, so it needs no input clamping. A `NaN` input is the
/// only source of a non-finite output, and it propagates through unchanged.
#[inline]
fn apply_sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Generates a square orthogonal matrix via Gram-Schmidt orthonormalization of a random matrix
///
/// Independently normalizing each column only yields unit-norm columns. This process instead
/// makes the columns mutually orthonormal, which keeps the hidden state transition
/// norm-preserving and reduces vanishing or exploding gradients. SimpleRNN and the GRU and LSTM
/// gates use this function to initialize their recurrent kernels.
fn orthogonal_init(size: usize, rng: &mut StdRng) -> Array2<f32> {
    // Random starting matrix
    let mut matrix = Array::random_using((size, size), Uniform::new(-1.0, 1.0).unwrap(), rng);

    const EPSILON: f32 = 1e-8;

    for i in 0..size {
        // Orthogonalize column i against every already-normalized column before it
        for j in 0..i {
            let mut projection = 0.0;
            for k in 0..size {
                projection += matrix[[k, i]] * matrix[[k, j]];
            }
            for k in 0..size {
                matrix[[k, i]] -= projection * matrix[[k, j]];
            }
        }

        // Normalize column i. Fall back to a standard basis vector if it collapsed
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

/// Shared gate parameters and helpers for the GRU and LSTM cells
pub mod gate;
/// The GRU (Gated Recurrent Unit) layer
pub mod gru;
/// The LSTM (Long Short-Term Memory) layer
pub mod lstm;
/// The SimpleRNN layer
pub mod simple_rnn;
/// Dimension and shape validators for the recurrent layers
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

    /// For size 3, M^T M equals the 3x3 identity within 1e-5
    #[test]
    fn orthogonal_init_size3_columns_are_orthonormal() {
        let m = orthogonal_init(3, &mut StdRng::seed_from_u64(0));

        // Compute M^T M (should equal I_3)
        let mt_m = m.t().dot(&m);

        // Seed 0 is deterministic, so Gram-Schmidt keeps the f32 round-off within 1e-5
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0_f32 } else { 0.0_f32 };
                assert_abs_diff_eq!(mt_m[[row, col]], expected, epsilon = 1e-5);
            }
        }
    }

    /// For size 1, the single entry has absolute value 1.0 after normalization
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

    /// sigmoid saturates to about 1.0 for a large positive input, with no overflow
    #[test]
    fn apply_sigmoid_large_positive_approaches_one() {
        let input = array![[500.0_f32]];
        let output = apply_sigmoid(input);
        // exp(-500) underflows to 0 in f32, so the result is exactly 1.0
        assert_abs_diff_eq!(output[[0, 0]], 1.0_f32, epsilon = 1e-6);
    }

    /// Large positive inputs all saturate to the same value, 1.0. No clamping is involved
    #[test]
    fn apply_sigmoid_large_positive_inputs_saturate_equally() {
        let out_500 = apply_sigmoid(array![[500.0_f32]]);
        let out_1000 = apply_sigmoid(array![[1000.0_f32]]);
        // Both saturate to exactly 1.0 (exp(-x) underflows to 0)
        assert_abs_diff_eq!(out_500[[0, 0]], out_1000[[0, 0]], epsilon = 1e-9);
        assert_abs_diff_eq!(out_1000[[0, 0]], 1.0_f32, epsilon = 1e-6);
    }

    /// sigmoid(-1000) saturates to about 0.0
    #[test]
    fn apply_sigmoid_large_negative_approaches_zero() {
        let input = array![[-1000.0_f32]];
        let output = apply_sigmoid(input);
        assert_abs_diff_eq!(output[[0, 0]], 0.0_f32, epsilon = 1e-6);
    }

    /// Output stays finite even for +/-inf and huge-magnitude inputs (no clamp needed)
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
