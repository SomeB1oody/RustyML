//! Recurrent layers (SimpleRNN, GRU, LSTM) and their shared helpers
//!
//! Re-exports the 3 layer types and provides 2 shared helpers. A numerically stable sigmoid
//! serves the GRU and LSTM gates, and 1 index map serves the `go_backwards` option of all 3
//! layers. All 3 layers draw their recurrent kernels from
//! [`Initializer::Orthogonal`](crate::neural_network::Initializer::Orthogonal).

use ndarray::Array2;

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

/// Maps a processing step to the input timestep that the step consumes
///
/// All 3 recurrent layers process the timesteps in order when `go_backwards` is false, and from
/// last to first when it is true. The forward pass reads the input at this index, and the
/// backward pass writes the per-step gate gradient back to this index. The backward scatter must
/// use this index because the batched reductions multiply the gate gradients against the input
/// rows, which stay in input order.
///
/// # Parameters
///
/// - `step` - Processing step, from 0 to `timesteps` - 1
/// - `timesteps` - Length of the time axis
/// - `go_backwards` - Reverses the processing order when true
///
/// # Returns
///
/// - `usize` - The input timestep that `step` consumes
#[inline]
fn input_step(step: usize, timesteps: usize, go_backwards: bool) -> usize {
    if go_backwards {
        timesteps - 1 - step
    } else {
        step
    }
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

    // input_step

    /// Without `go_backwards`, a processing step consumes the input timestep of the same index
    #[test]
    fn input_step_forward_is_the_identity() {
        for step in 0..5 {
            assert_eq!(input_step(step, 5, false), step);
        }
    }

    /// With `go_backwards`, processing step 0 consumes the last input timestep
    #[test]
    fn input_step_backward_reverses_the_time_axis() {
        let got: Vec<usize> = (0..5).map(|step| input_step(step, 5, true)).collect();
        assert_eq!(got, vec![4, 3, 2, 1, 0]);
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
