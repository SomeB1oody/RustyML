//! Shared parallel/serial gate thresholds for the elementwise kernel classes
//!
//! Every gated pass in the crate belongs to one of a few **cost classes**. The calibration
//! bench measures the serial/parallel crossover per class, not per call site. Declaring one
//! gate per class here keeps each calibration result in one place; the call sites import the
//! getter matching their kernel's class instead of restating the value
//!
//! The classes come in two element widths: the `f32` gates serve the neural-network layers,
//! the `f64` gates serve the classical-ML and utils modules. The crossovers differ (an f64
//! stream moves twice the bytes per element; an f64 `exp` costs more than the f32 one), so the
//! widths are calibrated separately
//!
//! Each gate is a runtime-tunable atomic (see `tunable_gate!`): the default is the calibrated
//! value, overridable at runtime through [`crate::tuning`]. These gates only pick serial vs. rayon;
//! because the gated reductions use the deterministic blocked fold of
//! [`crate::math::reduction`], moving a gate never changes a result (the parallel path matches the
//! serial result)
//!
//! The engine-specific gates stay with their engines, because their work metrics are
//! engine-specific rather than class-shared: `MatmulElem::{gemm_rayon_min_flops,
//! gemv_rayon_min_flops}` and the tiling constants (`crate::math::matmul`),
//! `CONV_PARALLEL_MIN_FLOPS`/`CONV_MIN_CHUNK_COLS` (im2col+GEMM engine),
//! `POOL_PARALLEL_MIN_OPS`/`POOL_MIN_CHUNK_OUT` (pooling engine), and
//! `BATCH_NORM_PARALLEL_THRESHOLD` (a per-layer analogy mapping). `metrics` keeps its
//! silhouette gate module-local on purpose - a lightweight leaf module that does not import
//! crate internals - but documents its value against the same calibration tables

// f32 classes (neural-network layers)

tunable_gate! {
    /// Cheap memory-bound `f32` maps: ReLU's `max(0, x)`, the dropout layers' compare-into-mask
    /// thresholding, and similar one-stream copy-speed loops
    ///
    /// In calibration the parallel path never beat serial up to 1M elements: these ops run at
    /// memory bandwidth on a single core, so rayon only adds fork/join overhead. The gate sits far
    /// out; at every practical tensor size this class runs serial
    #[cfg(feature = "neural_network")]
    pub(crate) CHEAP_MAP_PARALLEL_THRESHOLD
        => cheap_map_parallel_threshold / set_cheap_map_parallel_threshold = 4_000_000
}

tunable_gate! {
    /// Exp-dominated `f32` maps: sigmoid, tanh, and softmax (whose per-element cost is dominated by
    /// the shifted `exp`)
    ///
    /// Measured crossover bracket: 64K-128K elements
    #[cfg(feature = "neural_network")]
    pub(crate) EXP_MAP_PARALLEL_THRESHOLD
        => exp_map_parallel_threshold / set_exp_map_parallel_threshold = 131_072
}

tunable_gate! {
    /// The spatial-dropout per-channel scale: a copy-with-scale that multiplies each
    /// `(batch, channel)` segment of a `[batch, channels, *spatial]` tensor by its channel's
    /// inverted-dropout factor. Each element is independent (no reduction), so the gate is a pure
    /// performance knob that never changes the result bits
    ///
    /// This is the cheap-map class: a single multiply per element makes it nearly a pure memory
    /// copy, which one core almost saturates up to ~1M elements, so the parallel path's fork/join
    /// and allocation only pay off well past 1M - the same crossover as
    /// `CHEAP_MAP_PARALLEL_THRESHOLD`. Measured crossover bracket 1M-4M elements (0.60x at 1M,
    /// 1.13x at 4M, 2.05x at 8.4M)
    #[cfg(feature = "neural_network")]
    pub(crate) SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS
        => spatial_dropout_scale_parallel_min_elems / set_spatial_dropout_scale_parallel_min_elems
        = 4_194_304
}

tunable_gate! {
    /// Fused multi-slice `f32` updates: the optimizer kernels' parameter/gradient/moment loops,
    /// which stream several arrays at once
    ///
    /// Measured crossover bracket: 256K-1M elements
    #[cfg(feature = "neural_network")]
    pub(crate) FUSED_SLICE_PARALLEL_THRESHOLD
        => fused_slice_parallel_threshold / set_fused_slice_parallel_threshold = 1_000_000
}

tunable_gate! {
    /// `f32`-elements, `f64`-accumulator square-sum reductions: the clip-by-global-norm gradient
    /// scan, gated per parameter tensor
    ///
    /// Above the gate, callers must use [`crate::math::reduction::det_reduce`] (or its
    /// index-range twin) rather than a bare rayon `sum`/`reduce`, which is not reproducible across
    /// runs on the same machine
    ///
    /// Measured crossover bracket: 32K-64K elements (0.88x at 32K, 1.13x at 64K, 12.7x at 1M)
    #[cfg(feature = "neural_network")]
    pub(crate) SQ_SUM_F32_PARALLEL_MIN_ELEMS
        => sq_sum_f32_parallel_min_elems / set_sq_sum_f32_parallel_min_elems = 65_536
}

tunable_gate! {
    /// Naive (non-im2col) convolution loop nests: the DepthwiseConv2D forward/backward and the
    /// SeparableConv2D depthwise stage, gated on estimated FLOPs
    /// (`2 * batch * channels [* depth_multiplier] * out_h * out_w * kh * kw`)
    ///
    /// Estimated by analogy, not directly calibrated: these loops cost more per FLOP than the
    /// im2col+GEMM engine (whose measured crossover is ~4M FLOPs), so the crossover sits
    /// proportionally lower
    #[cfg(feature = "neural_network")]
    pub(crate) NAIVE_CONV_PARALLEL_MIN_FLOPS
        => naive_conv_parallel_min_flops / set_naive_conv_parallel_min_flops = 1_000_000
}

// f64 classes (classical ML / utils)

tunable_gate! {
    /// Cheap memory-bound `f64` maps: centering, scaling, normalization, kernel-matrix centering,
    /// and similar one-or-two-stream copy-speed loops, gated on the total element count
    ///
    /// Measured crossover bracket: 1M-4.2M elements (1.95x at 4.2M) - the same far-out gate as the
    /// f32 class; at typical preprocessing sizes this class runs serial
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) CHEAP_MAP_F64_PARALLEL_THRESHOLD
        => cheap_map_f64_parallel_threshold / set_cheap_map_f64_parallel_threshold = 4_000_000
}

tunable_gate! {
    /// Exp-dominated `f64` maps: the logistic sigmoid and the RBF/Sigmoid kernel transforms, gated
    /// on the total element count
    ///
    /// Measured crossover bracket: 16K-32K elements, but the win at 32K is a thin 1.09x; the gate
    /// sits at 65K where the win is a 1.82x. Lower than the f32 class (131K), as expected from the
    /// costlier f64 `exp`
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) EXP_MAP_F64_PARALLEL_THRESHOLD
        => exp_map_f64_parallel_threshold / set_exp_map_f64_parallel_threshold = 65_536
}

tunable_gate! {
    /// Short `f64` row scans: KMeans' per-sample arg-min over centroid projections, LDA's per-row
    /// best-class pick, per-sample distance scans (DBSCAN region queries, MeanShift label
    /// assignment), and similar `O(row)` per-task loops, gated on the **total elements scanned**
    /// (tasks x per-task row length, including any per-element dimension multiplier)
    ///
    /// Measured crossover bracket: 65K-262K scanned elements (1.61x at 262K)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) SCAN_F64_PARALLEL_MIN_ELEMS
        => scan_f64_parallel_min_elems / set_scan_f64_parallel_min_elems = 262_144
}

tunable_gate! {
    /// Tree-traversal tasks: per-sample root-to-leaf walks (DecisionTree and IsolationForest
    /// prediction), gated on the **total node visits** (samples x walk length; for a forest,
    /// samples x trees x average path length)
    ///
    /// Measured on a synthetic depth-16 heap-layout tree (the same compare-and-jump shape):
    /// crossover bracket 65K-262K node visits (6.3x at 262K)
    #[cfg(feature = "machine_learning")]
    pub(crate) TREE_TRAVERSAL_MIN_VISITS
        => tree_traversal_min_visits / set_tree_traversal_min_visits = 262_144
}

tunable_gate! {
    /// Sort-dominated split-search tasks: DecisionTree's per-feature copy + sort + scan in
    /// `find_best_split`, gated on the **total sorted elements** (node samples x features)
    ///
    /// Measured on the same copy/sort/prefix-scan shape (8 feature tasks): crossover bracket
    /// 2K-8K sorted elements (1.8x at 8K)
    #[cfg(feature = "machine_learning")]
    pub(crate) SORT_SCAN_MIN_ELEMS
        => sort_scan_min_elems / set_sort_scan_min_elems = 8_192
}

tunable_gate! {
    /// `f64` sum-style reductions (sum of squares, Welford moments), gated on the element count
    /// (or an equivalent work metric, e.g. samples x features for k-means' per-sample
    /// centroid accumulation)
    ///
    /// Below the gate a parallel reduction cannot win; above it, callers must use
    /// [`crate::math::reduction::det_reduce`] (or its index-range twin) rather than a bare rayon
    /// `sum`/`reduce`, which is not reproducible across runs on the same machine
    ///
    /// Measured crossover bracket: 131K-262K elements (1.24x at 262K, 3.5x at 1M)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) SUM_F64_PARALLEL_MIN_ELEMS
        => sum_f64_parallel_min_elems / set_sum_f64_parallel_min_elems = 262_144
}
