//! Internal hooks for benchmarks and integration tests.
//!
//! The module exposes task-size caps and forced execution paths for selected neural network kernels.
//! These hooks have no stability guarantee.

/// Reads 1 task-size cap
#[cfg(feature = "neural_network")]
pub type SplitCapGetter = fn() -> usize;

/// Writes 1 task-size cap
#[cfg(feature = "neural_network")]
pub type SplitCapSetter = fn(usize);

/// Task-size cap names, getters, and setters for selected neural network drivers.
///
/// A cap of 0 keeps the driver's calibrated task size.
/// A positive cap limits each task to that many work units.
/// Small caps let small test inputs reach paths that use multiple tasks.
///
/// Pooling, resize, and embedding caps change task boundaries without a partial floating-point reduction across tasks.
/// The convolution cap changes the row count of each matrix product.
/// Matrix-product kernel choices can then change rounding, so leave that cap at 0 when exact result bits must agree.
///
/// Drivers with partial floating-point reductions have no cap in this list.
/// Their block sizes define the accumulation order.
/// Other drivers already split small test inputs across multiple tasks and need no cap.
#[cfg(feature = "neural_network")]
pub const SPLIT_CAPS: &[(&str, SplitCapGetter, SplitCapSetter)] = &[
    (
        "conv.forced_chunk_positions",
        crate::neural_network::layers::convolution::convolution_engine::conv_forced_chunk_positions,
        crate::neural_network::layers::convolution::convolution_engine::set_conv_forced_chunk_positions,
    ),
    (
        "pool.forced_chunk_out",
        crate::neural_network::layers::pooling::pooling_engine::pool_forced_chunk_out,
        crate::neural_network::layers::pooling::pooling_engine::set_pool_forced_chunk_out,
    ),
    (
        "pool.forced_chunk_channels",
        crate::neural_network::layers::pooling::pooling_engine::pool_forced_chunk_channels,
        crate::neural_network::layers::pooling::pooling_engine::set_pool_forced_chunk_channels,
    ),
    (
        "upsampling.forced_task_rows",
        crate::neural_network::layers::upsampling::resize_engine::upsample_forced_task_rows,
        crate::neural_network::layers::upsampling::resize_engine::set_upsample_forced_task_rows,
    ),
    (
        "embedding.forced_task_rows",
        crate::neural_network::layers::embedding::embedding_forced_task_rows,
        crate::neural_network::layers::embedding::set_embedding_forced_task_rows,
    ),
];

#[cfg(feature = "neural_network")]
pub use crate::neural_network::layers::convolution::convolution_engine::conv_forward_impl as conv_forward_forced;
#[cfg(feature = "neural_network")]
pub use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_forward_impl,
};
