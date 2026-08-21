//! Internal hooks for the `benches/` targets
//!
//! Not part of the public API. This module is hidden from the documentation and carries no
//! stability guarantee. It exists only so the calibration bench in
//! `benches/calibrations/parallel_gates/` can drive crate-internal kernels with the parallel and
//! serial gate forced to either side. Production call paths never go through here

#[cfg(feature = "neural_network")]
pub use crate::neural_network::layers::convolution::convolution_engine::conv_forward_impl as conv_forward_forced;
#[cfg(feature = "neural_network")]
pub use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_forward_impl,
};
