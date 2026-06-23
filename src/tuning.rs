//! Runtime overrides for the crate's parallel/serial gate thresholds
//!
//! Every parallelized kernel decides serial vs. rayon (and, for GEMM, which parallel strategy)
//! by comparing a work estimate against a calibrated threshold. The defaults are tuned on the
//! maintainer's machine; this module overrides them at runtime, without a recompile, to match a
//! different machine's core count, cache sizes, or memory bandwidth
//!
//! # What a gate does and does not change
//!
//! A gate only selects an execution strategy; it never changes what is computed. The elementwise
//! and reduction gates give the same result serial or parallel, and the GEMM strategy gates (the
//! `matmul` submodule) reproduce their result across runs on the same machine (not necessarily
//! bit-for-bit). Retuning the gates does not change this
//!
//! # Storage vs. API
//!
//! The default and atomic backing store for each gate lives next to the code (and calibration
//! comment) it governs; this module is the discoverable public entry point that forwards to those
//! per-site setters. Each setter is a single relaxed atomic store and each getter a single relaxed
//! load that sits on the kernels' hot paths
//!
//! # Example
//!
//! ```ignore
//! // Retune the f32 GEMM serial/parallel crossover for a machine with fewer, faster cores
//! rustyml::tuning::matmul::set_gemm_min_flops_f32(4_000_000);
//! let current = rustyml::tuning::matmul::get_gemm_min_flops_f32();
//! ```

/// Generates a `set_*` / `get_*` forwarding pair that delegates to a per-site gate's
/// `pub(crate)` setter/getter
///
/// The `$what` string is spliced into the generated docs ("Sets {what}" / "Returns the current
/// {what}"), so it should read as a noun phrase
///
/// # Usage and expansion
///
/// This invocation:
///
/// ```ignore
/// fwd!(
///     set_chunk_elems => b::set_gemm_chunk_elems,
///     get_chunk_elems => b::gemm_chunk_elems,
///     "the element budget for one row-chunk of a tiled product"
/// );
/// ```
///
/// expands to:
///
/// ```ignore
/// /// Sets the element budget for one row-chunk of a tiled product
/// pub fn set_chunk_elems(value: usize) { b::set_gemm_chunk_elems(value); }
///
/// /// Returns the current element budget for one row-chunk of a tiled product
/// pub fn get_chunk_elems() -> usize { b::gemm_chunk_elems() }
/// ```
///
/// Prefix the invocation with `#[cfg(...)]` to gate both generated functions together
macro_rules! fwd {
    ($set:ident => $bset:path, $get:ident => $bget:path, $what:expr) => {
        #[doc = concat!("Sets ", $what)]
        pub fn $set(value: usize) {
            $bset(value);
        }
        #[doc = concat!("Returns the current ", $what)]
        pub fn $get() -> usize {
            $bget()
        }
    };
}

/// GEMM / GEMV parallelism gates for the matrix-product backend (see [`crate::math::matmul`])
///
/// `gemm_min_flops` / `gemv_min_flops` are the estimated-FLOPs crossovers at or above which a
/// product is parallelized, calibrated per dtype (`f32` and `f64` have different optimal values
/// that the single threshold the `gemm` crate exposes cannot capture). `colpar_min_cols_per_thread`
/// is the columns-per-thread floor below which `gemm_par_auto` splits rows itself instead of using
/// the backend's column parallelism. `chunk_elems` and `cache_resident_max_bytes` size the
/// tiled-product path; the latter matches a machine's actual L3 cache
#[cfg(feature = "math")]
pub mod matmul {
    use crate::math::matmul as b;

    #[cfg(any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    ))]
    fwd!(
        set_gemm_min_flops_f32 => b::set_gemm_rayon_min_flops_f32,
        get_gemm_min_flops_f32 => b::gemm_rayon_min_flops_f32,
        "the f32 GEMM serial-vs-rayon crossover, in estimated FLOPs (`2*m*k*n`)"
    );
    #[cfg(any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    ))]
    fwd!(
        set_gemm_min_flops_f64 => b::set_gemm_rayon_min_flops_f64,
        get_gemm_min_flops_f64 => b::gemm_rayon_min_flops_f64,
        "the f64 GEMM serial-vs-rayon crossover, in estimated FLOPs (`2*m*k*n`)"
    );
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fwd!(
        set_gemv_min_flops_f32 => b::set_gemv_rayon_min_flops_f32,
        get_gemv_min_flops_f32 => b::gemv_rayon_min_flops_f32,
        "the f32 GEMV row-split crossover, in estimated FLOPs (`2*m*k`)"
    );
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fwd!(
        set_gemv_min_flops_f64 => b::set_gemv_rayon_min_flops_f64,
        get_gemv_min_flops_f64 => b::gemv_rayon_min_flops_f64,
        "the f64 GEMV row-split crossover, in estimated FLOPs (`2*m*k`)"
    );
    #[cfg(any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    ))]
    fwd!(
        set_colpar_min_cols_per_thread => b::set_gemm_colpar_min_cols_per_thread,
        get_colpar_min_cols_per_thread => b::gemm_colpar_min_cols_per_thread,
        "the columns-per-thread floor below which a `m >= n` GEMM splits rows instead of using \
         the backend's column parallelism"
    );
    fwd!(
        set_chunk_elems => b::set_gemm_chunk_elems,
        get_chunk_elems => b::gemm_chunk_elems,
        "the element budget for one row-chunk of a tiled product"
    );
    fwd!(
        set_cache_resident_max_bytes => b::set_cache_resident_max_bytes,
        get_cache_resident_max_bytes => b::cache_resident_max_bytes,
        "the cache-resident size threshold (bytes) for the per-row-GEMV-swarm vs. tiled-GEMM \
         decision - set this to the machine's actual shared-L3 size"
    );
}

/// Elementwise-map parallelism gates (memory-bound and exp-dominated maps); see
/// `crate::parallel_gates`. Moving these never changes a result bit
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub mod elementwise {
    use crate::parallel_gates as b;

    #[cfg(feature = "neural_network")]
    fwd!(
        set_cheap_map_f32 => b::set_cheap_map_parallel_threshold,
        get_cheap_map_f32 => b::cheap_map_parallel_threshold,
        "the f32 cheap-map (ReLU, dropout mask) serial-vs-rayon element-count gate"
    );
    #[cfg(feature = "neural_network")]
    fwd!(
        set_exp_map_f32 => b::set_exp_map_parallel_threshold,
        get_exp_map_f32 => b::exp_map_parallel_threshold,
        "the f32 exp-dominated map (sigmoid, tanh, softmax) element-count gate"
    );
    #[cfg(feature = "neural_network")]
    fwd!(
        set_spatial_dropout_scale => b::set_spatial_dropout_scale_parallel_min_elems,
        get_spatial_dropout_scale => b::spatial_dropout_scale_parallel_min_elems,
        "the spatial-dropout per-channel scale element-count gate"
    );
    #[cfg(feature = "neural_network")]
    fwd!(
        set_fused_slice => b::set_fused_slice_parallel_threshold,
        get_fused_slice => b::fused_slice_parallel_threshold,
        "the fused multi-slice optimizer-update element-count gate"
    );
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fwd!(
        set_cheap_map_f64 => b::set_cheap_map_f64_parallel_threshold,
        get_cheap_map_f64 => b::cheap_map_f64_parallel_threshold,
        "the f64 cheap-map (centering, scaling, normalization) element-count gate"
    );
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fwd!(
        set_exp_map_f64 => b::set_exp_map_f64_parallel_threshold,
        get_exp_map_f64 => b::exp_map_f64_parallel_threshold,
        "the f64 exp-dominated map (logistic sigmoid, RBF/Sigmoid kernels) element-count gate"
    );
}

/// Deterministic-reduction parallelism gates; see `crate::parallel_gates` and
/// [`crate::math::reduction`]. The blocked fold gives the same result serial or parallel, so
/// moving these never changes a result
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils",
    feature = "math"
))]
pub mod reduction {
    #[cfg(feature = "neural_network")]
    fwd!(
        set_sq_sum_f32 => crate::parallel_gates::set_sq_sum_f32_parallel_min_elems,
        get_sq_sum_f32 => crate::parallel_gates::sq_sum_f32_parallel_min_elems,
        "the f32 square-sum (clip-by-global-norm) reduction element-count gate"
    );
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fwd!(
        set_sum_f64 => crate::parallel_gates::set_sum_f64_parallel_min_elems,
        get_sum_f64 => crate::parallel_gates::sum_f64_parallel_min_elems,
        "the f64 sum-style reduction element-count gate"
    );
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fwd!(
        set_scan_f64 => crate::parallel_gates::set_scan_f64_parallel_min_elems,
        get_scan_f64 => crate::parallel_gates::scan_f64_parallel_min_elems,
        "the f64 short-row-scan (arg-min, distance-scan) scanned-element gate"
    );
    #[cfg(feature = "math")]
    fwd!(
        set_exp_reduce => crate::math::set_exp_reduce_min_elems,
        get_exp_reduce => crate::math::exp_reduce_min_elems,
        "the logistic-loss exp-reduction element-count gate"
    );
}

/// Tree-walk and split-search parallelism gates for the tree models; see
/// `crate::parallel_gates`
#[cfg(feature = "machine_learning")]
pub mod tree {
    fwd!(
        set_traversal_min_visits => crate::parallel_gates::set_tree_traversal_min_visits,
        get_traversal_min_visits => crate::parallel_gates::tree_traversal_min_visits,
        "the tree-traversal (predict) total-node-visits gate"
    );
    fwd!(
        set_sort_scan_min_elems => crate::parallel_gates::set_sort_scan_min_elems,
        get_sort_scan_min_elems => crate::parallel_gates::sort_scan_min_elems,
        "the DecisionTree split-search total-sorted-elements gate"
    );
}

/// Convolution-engine parallelism gates; see `crate::neural_network::layers::convolution`
#[cfg(feature = "neural_network")]
pub mod conv {
    fwd!(
        set_parallel_min_flops => crate::neural_network::layers::convolution::convolution_engine::set_conv_parallel_min_flops,
        get_parallel_min_flops => crate::neural_network::layers::convolution::convolution_engine::conv_parallel_min_flops,
        "the im2col+GEMM convolution-engine estimated-FLOPs gate"
    );
    fwd!(
        set_naive_parallel_min_flops => crate::parallel_gates::set_naive_conv_parallel_min_flops,
        get_naive_parallel_min_flops => crate::parallel_gates::naive_conv_parallel_min_flops,
        "the naive (depthwise/separable) convolution estimated-FLOPs gate"
    );
}

/// Pooling-engine parallelism gate; see `crate::neural_network::layers::pooling`
#[cfg(feature = "neural_network")]
pub mod pool {
    fwd!(
        set_parallel_min_ops => crate::neural_network::layers::pooling::pooling_engine::set_pool_parallel_min_ops,
        get_parallel_min_ops => crate::neural_network::layers::pooling::pooling_engine::pool_parallel_min_ops,
        "the pooling-engine estimated-element-ops gate"
    );
}

/// Normalization-layer parallelism gates (BatchNorm, LayerNorm, GroupNorm)
#[cfg(feature = "neural_network")]
pub mod norm {
    use crate::neural_network::layers::regularization::normalization as gn;
    use crate::neural_network::layers::regularization::normalization::batch_normalization as bn;
    use crate::neural_network::layers::regularization::normalization::layer_normalization as ln;

    fwd!(
        set_batch_norm => bn::set_batch_norm_parallel_threshold,
        get_batch_norm => bn::batch_norm_parallel_threshold,
        "the BatchNorm forward/backward total-element gate"
    );
    fwd!(
        set_bn_col_stats => bn::set_bn_col_stats_parallel_min_elems,
        get_bn_col_stats => bn::bn_col_stats_parallel_min_elems,
        "the BatchNorm 2-D per-channel statistics reduction gate"
    );
    fwd!(
        set_bn_plane_stats => bn::set_bn_plane_stats_parallel_min_elems,
        get_bn_plane_stats => bn::bn_plane_stats_parallel_min_elems,
        "the BatchNorm rank>=3 per-channel statistics reduction gate"
    );
    fwd!(
        set_ln_row => ln::set_ln_row_parallel_min_elems,
        get_ln_row => ln::ln_row_parallel_min_elems,
        "the LayerNorm per-row fold gate"
    );
    fwd!(
        set_ln_col_stats => ln::set_ln_col_stats_parallel_min_elems,
        get_ln_col_stats => ln::ln_col_stats_parallel_min_elems,
        "the LayerNorm gamma/beta gradient column-fold gate"
    );
    fwd!(
        set_gn_row => gn::set_gn_row_parallel_min_elems,
        get_gn_row => gn::gn_row_parallel_min_elems,
        "the GroupNorm per-row fold gate"
    );
    fwd!(
        set_gn_plane_stats => gn::set_gn_plane_stats_parallel_min_elems,
        get_gn_plane_stats => gn::gn_plane_stats_parallel_min_elems,
        "the GroupNorm per-channel gradient plane-fold gate"
    );
}

/// Metrics parallelism gate (silhouette score); see [`crate::metrics`]
#[cfg(feature = "metrics")]
pub mod metrics {
    fwd!(
        set_silhouette => crate::metrics::clustering::set_silhouette_parallel_min_elems,
        get_silhouette => crate::metrics::clustering::silhouette_parallel_min_elems,
        "the silhouette-score pairwise-distance-fill scanned-element gate"
    );
}

#[cfg(all(test, feature = "math"))]
mod tests {
    /// A public setter reaches the per-site atomic and the getter reads it back (the full
    /// facade -> `pub(crate)` backing -> atomic path)
    #[test]
    fn matmul_chunk_elems_gate_roundtrips() {
        let orig = super::matmul::get_chunk_elems();
        super::matmul::set_chunk_elems(orig + 7);
        assert_eq!(super::matmul::get_chunk_elems(), orig + 7);
        super::matmul::set_chunk_elems(orig);
        assert_eq!(super::matmul::get_chunk_elems(), orig);
    }
}
