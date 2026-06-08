use ndarray::{Array, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Gradient clipping bound shared by the recurrent layers (SimpleRNN, GRU, LSTM) to curb
/// exploding gradients during backpropagation through time.
const GRADIENT_CLIP_VALUE: f32 = 5.0;

/// Applies stable sigmoid activation to an array
///
/// Uses clipping to prevent numerical overflow before computing sigmoid.
/// This is used by both GRU and LSTM gates.
#[inline]
fn apply_sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|x| {
        let clipped_x = x.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-clipped_x).exp())
    })
}

/// Generates a square orthogonal matrix via Gram-Schmidt orthonormalization of a random matrix.
///
/// Unlike independently normalizing each column (which only yields unit-norm columns), this makes
/// the columns mutually orthonormal. Orthogonal recurrent weights keep the hidden-state transition
/// norm-preserving, mitigating vanishing/exploding gradients. Used to initialize the recurrent
/// kernels of SimpleRNN and the GRU/LSTM gates.
fn orthogonal_init(size: usize) -> Array2<f32> {
    // Random starting matrix; Gram-Schmidt below orthonormalizes its columns in place.
    let mut matrix = Array::random((size, size), Uniform::new(-1.0, 1.0).unwrap());

    const EPSILON: f32 = 1e-8;

    for i in 0..size {
        // Orthogonalize column i against all previously finalized (already normalized) columns,
        // so each projection denominator is 1.
        for j in 0..i {
            let mut projection = 0.0;
            for k in 0..size {
                projection += matrix[[k, i]] * matrix[[k, j]];
            }
            for k in 0..size {
                matrix[[k, i]] -= projection * matrix[[k, j]];
            }
        }

        // Normalize column i; fall back to a standard basis vector if it collapsed.
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
/// Input validation functions for Recurrent layers
mod validation;
/// A LSTM (Long Short-Term Memory) neural network layer implementation
pub mod lstm;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation
pub mod simple_rnn;

pub use gru::GRU;
pub use lstm::LSTM;
pub use simple_rnn::SimpleRNN;
