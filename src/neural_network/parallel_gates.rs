//! Shared parallel/serial gate thresholds for the elementwise kernel classes
//!
//! Every gated pass in the crate belongs to one of a few **cost classes**, and the calibration
//! bench (`cargo bench --bench parallel_gates`, results in `benches/RESULTS.md`) measures the
//! serial/parallel crossover **per class**, not per layer. Declaring one constant per class here
//! keeps each calibration result in exactly one place; the layers import the constant matching
//! their kernel's class instead of restating the value.
//!
//! The engine-specific gates stay with their engines, because their work metrics are
//! engine-specific rather than class-shared: `PAR_GEMM_MIN_FLOPS`/`PAR_GEMM_MIN_BLOCK`
//! (block-parallel GEMM), `CONV_PARALLEL_MIN_FLOPS`/`CONV_MIN_CHUNK_COLS` (im2col+GEMM engine),
//! `POOL_PARALLEL_MIN_OPS`/`POOL_MIN_CHUNK_OUT` (pooling engine), and
//! `BATCH_NORM_PARALLEL_THRESHOLD` (a per-layer analogy mapping).
//!
//! All values: calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11.

/// Cheap memory-bound maps: ReLU's `max(0, x)`, the dropout layers' compare-into-mask
/// thresholding, and similar one-stream copy-speed loops.
///
/// In calibration the parallel path **never beat serial up to 1M elements** - these ops run at
/// memory bandwidth on a single core, so rayon only adds fork/join overhead. The gate sits far
/// out; at every practical tensor size this class runs serial
pub(crate) const CHEAP_MAP_PARALLEL_THRESHOLD: usize = 4_000_000;

/// Exp-dominated maps: sigmoid, tanh, and softmax (whose per-element cost is dominated by the
/// shifted `exp`).
///
/// Measured crossover bracket: 64K-128K elements
pub(crate) const EXP_MAP_PARALLEL_THRESHOLD: usize = 131_072;

/// Fused multi-slice updates: the optimizer kernels' parameter/gradient/moment loops, which
/// stream several arrays at once.
///
/// Measured crossover bracket: 256K-1M elements
pub(crate) const FUSED_SLICE_PARALLEL_THRESHOLD: usize = 1_000_000;

/// Naive (non-im2col) convolution loop nests: the DepthwiseConv2D forward/backward and the
/// SeparableConv2D depthwise stage, gated on estimated FLOPs
/// (`2 * batch * channels [* depth_multiplier] * out_h * out_w * kh * kw`).
///
/// Estimated by analogy, not directly calibrated: these loops cost more per FLOP than the
/// im2col+GEMM engine (whose measured crossover is ~4M FLOPs), so the crossover sits
/// proportionally lower
pub(crate) const NAIVE_CONV_PARALLEL_MIN_FLOPS: usize = 1_000_000;
