use ndarray::Array2;

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

/// Gate structure for recurrent cell operations (GRU, LSTM)
pub mod gate;
/// A GRU (Gated Recurrent Unit) layer implementation
pub mod gru;
/// Input validation functions for Recurrent layers
mod input_validation_function;
/// A LSTM (Long Short-Term Memory) neural network layer implementation
pub mod lstm;
/// A Simple Recurrent Neural Network (SimpleRNN) layer implementation
pub mod simple_rnn;

pub use gru::GRU;
pub use lstm::LSTM;
pub use simple_rnn::SimpleRNN;
