//! Internal hooks for the `benches/` targets and for the golden-fixture test net
//!
//! Not part of the public API. This module is hidden from the documentation and carries no
//! stability guarantee. It holds 2 hooks, and a production call path goes through neither
//!
//! - The calibration bench in `benches/calibrations/parallel_gates/` drives crate-internal
//!   kernels with the parallel and serial gate forced to either side
//! - The golden-fixture test net in `tests/neural_network/golden` caps the task size of each
//!   parallel driver, so a small fixture input still builds more than 1 task

/// Reads 1 task-size cap.
#[cfg(feature = "neural_network")]
pub type SplitCapGetter = fn() -> usize;

/// Writes 1 task-size cap.
#[cfg(feature = "neural_network")]
pub type SplitCapSetter = fn(usize);

/// Every task-size cap of the neural-network layers, as a name, a getter, and a setter
///
/// # What a cap does
///
/// Each of these drivers splits its work into tasks, and each takes the task size from a
/// calibrated rule with a floor of 64 positions or more. Every tensor of the golden test net
/// holds 256 elements or fewer, so each rule gives a size at or above the whole input, and the
/// driver builds exactly 1 task. The parallel branch then runs, and the chunk arithmetic, the
/// block indexing, and the partial reassembly stay unread
///
/// A cap holds the task at that size or below. The production value of every cap is 0, which
/// keeps the calibrated size. See the `split_cap` helper of `crate::parallel_gates`
///
/// # Why a cap changes no value
///
/// Each driver below runs the same serial kernel over each task, and joins the task results in
/// task order. A task reads no other task's output, and no task holds part of a floating-point
/// reduction. The task size therefore decides where the boundaries fall and nothing else
///
/// `conv.forced_chunk_positions` is the 1 entry for which this does not hold today. Its task is
/// a GEMM, and the backend picks its accumulation order from the row count of the block, so the
/// same rows give different result bits in a short block than in a long one. The golden test
/// net therefore installs every cap but that one. See `CONV_FORCED_CHUNK_POSITIONS`
///
/// # The drivers that take no cap
///
/// A driver whose tasks each hold 1 partial of a floating-point reduction is absent, and must
/// stay absent. Its block size fixes the summation order, so a cap there would change result
/// bits. Those are `global_pool_forward` of the pooling engine, `par_col_sum` and `par_col_dot`
/// of the normalization folds, and the per-group statistic fold of the normalization layers
///
/// A driver whose tasks are already more than 1 at fixture size is also absent, because it
/// needs no cap. Those are the batch fans of the convolution backward pass, of the transposed
/// convolution, and of the normalization row passes, and the per-output-row split of the
/// depthwise convolution
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
