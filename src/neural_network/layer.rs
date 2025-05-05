pub mod dense;
pub mod lstm;
pub mod simple_rnn;

pub use dense::*;

pub use simple_rnn::*;

pub use lstm::*;

pub enum LayerWeight<'a> {
    Dense(DenseLayerWeight<'a>),
    SimpleRNN(SimpleRNNLayerWeight<'a>),
    LSTM(LSTMLayerWeight<'a>),
    Empty,
}

pub struct DenseLayerWeight<'a> {
    pub weight: &'a ndarray::Array2<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

pub struct SimpleRNNLayerWeight<'a> {
    pub kernel: &'a ndarray::Array2<f32>,
    pub recurrent_kernel: &'a ndarray::Array2<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

pub struct LSTMGateWeight<'a> {
    pub kernel: &'a ndarray::Array2<f32>,
    pub recurrent_kernel: &'a ndarray::Array2<f32>,
    pub bias: &'a ndarray::Array2<f32>,
}

pub struct LSTMLayerWeight<'a> {
    pub input: LSTMGateWeight<'a>,
    pub forget: LSTMGateWeight<'a>,
    pub cell: LSTMGateWeight<'a>,
    pub output: LSTMGateWeight<'a>,
}
