//! Shared parallel/serial gate thresholds for the elementwise kernel classes
//!
//! Every gated pass in the crate belongs to one of a few cost classes. Declaring one gate per
//! class here keeps its value in one place. Call sites import the getter for their kernel's
//! class instead of restating the value.
//!
//! The classes come in 2 element widths. The `f32` gates serve the neural-network layers.
//! The `f64` gates serve the classical-ML and utils modules. An f64 stream moves twice the
//! bytes per element of an f32 stream. An f64 `exp` also costs more than an f32 `exp`. The
//! 2 widths use separate gates.
//!
//! Each gate is a runtime-tunable atomic (see `tunable_gate!`). Override the default at
//! runtime through [`crate::tuning`]. A gate only picks serial versus rayon. Because the gated
//! reductions use the deterministic blocked fold of [`crate::math::reduction`], moving a gate
//! never changes a result. The parallel path matches the serial result.
//!
//! The engine-specific gates stay with their engines, because their work metrics are
//! engine-specific rather than shared by a class:
//!
//! - the caller-side tiling constants in `crate::math::matmul` (the matrix product's own
//!   scheduling belongs to the `gemmkit` backend, whose knobs live there instead of here)
//! - `CONV_PARALLEL_MIN_FLOPS` and `CONV_MIN_CHUNK_POSITIONS` (the im2col+GEMM engine)
//! - `POOL_PARALLEL_MIN_OPS`, `POOL_MIN_CHUNK_OUT`, and `POOL_MIN_CHUNK_CHANNELS` (the pooling
//!   engine)
//! - `BATCH_NORM_PARALLEL_THRESHOLD` (a per-layer mapping)
//!
//! `metrics` keeps its silhouette gate module-local, because it is a lightweight leaf module
//! that does not import crate internals.

// f32 classes (neural-network layers)

tunable_gate! {
    /// Cheap memory-bound `f32` maps: the activations with no transcendental call, such as
    /// ReLU's `max(0, x)`, the dropout layers' compare-into-mask thresholding, and similar
    /// one-stream copy-speed loops. Gated on the total element count.
    #[cfg(feature = "neural_network")]
    pub(crate) CHEAP_MAP_PARALLEL_THRESHOLD
        => cheap_map_parallel_threshold / set_cheap_map_parallel_threshold = 4_000_000
}

tunable_gate! {
    /// Exp-dominated `f32` maps: sigmoid, tanh, softmax, ELU, SELU, softplus, and exponential,
    /// where 1 `exp` call dominates the per-element cost. Gated on the total element count.
    #[cfg(feature = "neural_network")]
    pub(crate) EXP_MAP_PARALLEL_THRESHOLD
        => exp_map_parallel_threshold / set_exp_map_parallel_threshold = 131_072
}

tunable_gate! {
    /// The spatial-dropout per-channel scale: a copy-with-scale that multiplies each
    /// `(batch, channel)` segment of a `[batch, *spatial, channels]` tensor by its channel's
    /// inverted-dropout factor. Each element is independent, so the gate is a pure performance
    /// knob that never changes the result bits. Gated on the total element count, the same
    /// cheap-map class as `CHEAP_MAP_PARALLEL_THRESHOLD`.
    #[cfg(feature = "neural_network")]
    pub(crate) SPATIAL_DROPOUT_SCALE_PARALLEL_MIN_ELEMS
        => spatial_dropout_scale_parallel_min_elems / set_spatial_dropout_scale_parallel_min_elems
        = 4_194_304
}

tunable_gate! {
    /// Fused multi-slice `f32` updates: the optimizer kernels' parameter, gradient, and moment
    /// loops, which stream several arrays at once. Gated on the total element count.
    #[cfg(feature = "neural_network")]
    pub(crate) FUSED_SLICE_PARALLEL_THRESHOLD
        => fused_slice_parallel_threshold / set_fused_slice_parallel_threshold = 1_000_000
}

tunable_gate! {
    /// `f32`-elements, `f64`-accumulator square-sum reductions: the clip-by-global-norm gradient
    /// scan, gated per parameter tensor on the element count.
    ///
    /// Above the gate, callers must use [`crate::math::reduction::det_reduce`] (or its
    /// index-range twin). A bare rayon `sum`/`reduce` does not reproduce across runs on the
    /// same machine.
    #[cfg(feature = "neural_network")]
    pub(crate) SQ_SUM_F32_PARALLEL_MIN_ELEMS
        => sq_sum_f32_parallel_min_elems / set_sq_sum_f32_parallel_min_elems = 65_536
}

tunable_gate! {
    /// Naive (non-im2col) convolution loop nests: the DepthwiseConv2D forward/backward and the
    /// SeparableConv2D depthwise stage. Gated on estimated FLOPs
    /// (`2 * batch * channels [* depth_multiplier] * out_h * out_w * kh * kw`).
    #[cfg(feature = "neural_network")]
    pub(crate) NAIVE_CONV_PARALLEL_MIN_FLOPS
        => naive_conv_parallel_min_flops / set_naive_conv_parallel_min_flops = 1_000_000
}

// f64 classes (classical ML / utils)

tunable_gate! {
    /// Cheap memory-bound `f64` maps: centering, scaling, normalization, kernel-matrix
    /// centering, and similar one- or two-stream copy-speed loops. Gated on the total element
    /// count.
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) CHEAP_MAP_F64_PARALLEL_THRESHOLD
        => cheap_map_f64_parallel_threshold / set_cheap_map_f64_parallel_threshold = 4_000_000
}

tunable_gate! {
    /// Exp-dominated `f64` maps: the logistic sigmoid and the RBF/Sigmoid kernel transforms.
    /// Gated on the total element count.
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) EXP_MAP_F64_PARALLEL_THRESHOLD
        => exp_map_f64_parallel_threshold / set_exp_map_f64_parallel_threshold = 65_536
}

tunable_gate! {
    /// Short `f64` row scans: KMeans' per-sample arg-min over centroid projections, LDA's
    /// per-row best-class pick, and per-sample distance scans (DBSCAN region queries,
    /// MeanShift label assignment). Also covers similar `O(row)` per-task loops. Gated on the
    /// total elements scanned (tasks times per-task row length, including any per-element
    /// dimension multiplier).
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) SCAN_F64_PARALLEL_MIN_ELEMS
        => scan_f64_parallel_min_elems / set_scan_f64_parallel_min_elems = 262_144
}

tunable_gate! {
    /// Tree-traversal tasks: per-sample root-to-leaf walks (DecisionTree and IsolationForest
    /// prediction). Gated on the total node visits (samples times walk length, or for a
    /// forest, samples times trees times average path length).
    #[cfg(feature = "machine_learning")]
    pub(crate) TREE_TRAVERSAL_MIN_VISITS
        => tree_traversal_min_visits / set_tree_traversal_min_visits = 262_144
}

tunable_gate! {
    /// Sort-dominated split-search tasks: DecisionTree's per-feature copy, sort, and scan in
    /// `find_best_split`. Gated on the total sorted elements (node samples times features).
    #[cfg(feature = "machine_learning")]
    pub(crate) SORT_SCAN_MIN_ELEMS
        => sort_scan_min_elems / set_sort_scan_min_elems = 8_192
}

tunable_gate! {
    /// `f64` sum-style reductions: sum of squares and Welford moments. Gated on the element
    /// count or an equivalent work metric, for example samples times features for k-means'
    /// per-sample centroid accumulation.
    ///
    /// Above the gate, callers must use [`crate::math::reduction::det_reduce`] (or its
    /// index-range twin). A bare rayon `sum`/`reduce` does not reproduce across runs on the
    /// same machine.
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) SUM_F64_PARALLEL_MIN_ELEMS
        => sum_f64_parallel_min_elems / set_sum_f64_parallel_min_elems = 262_144
}
