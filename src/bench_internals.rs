//! Internal hooks for the `benches/` targets
//!
//! **Not part of the public API.** This module is hidden from the documentation and carries no
//! stability guarantee; it exists only so the calibration bench (`benches/parallel_gates.rs`)
//! can drive crate-internal kernels with the parallel/serial gate forced to either side.
//! Production call paths never go through here.

pub use crate::neural_network::layers::convolution::convolution_engine::conv_forward_impl as conv_forward_forced;
pub use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_forward_impl,
};
pub use crate::neural_network::matmul::split_matmul;
