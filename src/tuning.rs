//! Runtime overrides for the crate's parallel and serial gate thresholds
//!
//! Every parallelized kernel picks serial execution or rayon, and for GEMM picks a parallel
//! strategy, by comparing a work estimate against a calibrated threshold. This module overrides
//! the defaults at runtime, without a recompile, to match a different machine's core count,
//! cache size, or memory bandwidth. The defaults come from calibration on an AMD Ryzen 9 9950X
//! with 16 cores, 32 threads, and 64 MiB of L3 in 2 instances, which is 32 MiB per die.
//!
//! # What a gate does and does not change
//!
//! A gate only picks an execution strategy. It never changes what the code computes. The
//! elementwise and reduction gates give the same result serial or parallel. The matrix-product
//! scheduling lives in the `gemmkit` backend (see the `matmul` submodule). Its results reproduce
//! on the same machine for a fixed configuration regardless of worker count. The `matmul` gates
//! kept here only shape caller-side tiling. Retuning a gate does not change any result.
//!
//! # Storage vs. API
//!
//! The default and the atomic backing store for each gate live next to the code they govern.
//! This module is the discoverable public entry point that forwards to those per-site setters.
//! Each setter makes a single relaxed atomic store. Each getter makes a single relaxed load on
//! the kernel's hot path.
//!
//! # Examples
//!
//! ```ignore
//! // Retune the GEMM serial/parallel work gate for a machine with fewer, faster cores
//! // (a gemmkit knob, forwarded through the `backend` alias)
//! rustyml::tuning::matmul::backend::set_parallel_threshold(2_000_000);
//! // Match the tiled-product policy to the L3 1 core of this machine reaches
//! rustyml::tuning::matmul::set_cache_resident_max_bytes(16 * 1024 * 1024);
//! ```

/// Generates a `set_*` / `get_*` forwarding pair that calls a per-site gate's `pub(crate)`
/// setter and getter.
///
/// The `$what` string fills the generated docs ("Sets {what}" / "Returns the current {what}").
/// Write it as a noun phrase.
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
/// Prefix the invocation with `#[cfg(...)]` to gate both generated functions together.
macro_rules! fwd {
    ($set:ident => $bset:path, $get:ident => $bget:path, $what:expr) => {
        #[doc = concat!("Sets ", $what)]
        pub fn $set(value: usize) {
            $bset(value);
        }
        #[doc = concat!("Returns the current ", $what)]
        #[inline]
        pub fn $get() -> usize {
            $bget()
        }
    };
}

/// Tiling policy for the matrix-product callers, plus the `backend` alias to gemmkit's own
/// tuning surface (see [`crate::math::matmul`]).
///
/// The GEMM/GEMV serial-vs-parallel crossovers, worker ramps, and kernel blocking live in the
/// [`gemmkit`](https://docs.rs/gemmkit) backend. Retune them one of 3 ways.
///
/// - **Env profile** (no recompile): every backend knob is a `GEMMKIT_*` environment variable,
///   read once per process. Run `cargo install gemmkit-tune`, then `gemmkit-tune` on the target
///   machine, to sweep them and emit a ready-to-source profile.
/// - **Programmatically**: through the `backend` alias, for example
///   `tuning::matmul::backend::set_parallel_threshold(..)`. An in-code setter beats the env var.
/// - **Per call**: not exposed publicly. The estimators pass the backend's automatic
///   parallelism, or force serial inside an already-parallel region, at each call site.
///
/// What remains here is the caller-side tiling policy. `chunk_elems` sizes the row-chunks of a
/// tiled product. `cache_resident_max_bytes` picks a GEMV-swarm or a tiled GEMM. Set it to the
/// L3 that 1 core of the machine reaches directly.
#[cfg(feature = "math")]
pub mod matmul {
    use crate::math::matmul as b;

    /// gemmkit's own tuning module, forwarded through the adapter so backend knobs are reachable
    /// without a direct `gemmkit` dependency. Every `GEMMKIT_*` env var has a `set_*`/getter pair
    /// here.
    ///
    /// The path goes through `gemmkit_ndarray` on purpose. The knobs are process-global atomics,
    /// so a setter called on a separately resolved second `gemmkit` would write a copy that the
    /// adapter in use never reads.
    pub use gemmkit_ndarray::tuning as backend;

    fwd!(
        set_chunk_elems => b::set_gemm_chunk_elems,
        get_chunk_elems => b::gemm_chunk_elems,
        "the element budget for one row-chunk of a tiled product"
    );
    fwd!(
        set_cache_resident_max_bytes => b::set_cache_resident_max_bytes,
        get_cache_resident_max_bytes => b::cache_resident_max_bytes,
        "the cache-resident size threshold (bytes) for the per-row-GEMV-swarm vs. tiled-GEMM \
         decision - set this to the L3 that 1 core reaches directly, and never to a package total"
    );
}

/// Elementwise-map parallelism gates (memory-bound and exp-dominated maps). See
/// `crate::parallel_gates`. Moving a gate never changes a result bit.
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

/// Deterministic-reduction parallelism gates. See `crate::parallel_gates` and
/// [`crate::math::reduction`]. The blocked fold gives the same result serial or parallel, so
/// moving a gate never changes a result.
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
}

/// Tree-walk and split-search parallelism gates for the tree models. See
/// `crate::parallel_gates`.
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

/// Convolution-engine parallelism gates. See `crate::neural_network::layers::convolution`.
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

/// Pooling-engine parallelism gate. See `crate::neural_network::layers::pooling`.
#[cfg(feature = "neural_network")]
pub mod pool {
    fwd!(
        set_parallel_min_ops => crate::neural_network::layers::pooling::pooling_engine::set_pool_parallel_min_ops,
        get_parallel_min_ops => crate::neural_network::layers::pooling::pooling_engine::pool_parallel_min_ops,
        "the pooling-engine estimated-element-ops gate"
    );
}

/// Upsampling-engine parallelism gate (`UpSampling1D/2D/3D`).
#[cfg(feature = "neural_network")]
pub mod upsampling {
    fwd!(
        set_parallel_min_ops => crate::neural_network::layers::upsampling::resize_engine::set_upsample_parallel_min_ops,
        get_parallel_min_ops => crate::neural_network::layers::upsampling::resize_engine::upsample_parallel_min_ops,
        "the upsampling-engine element-ops gate (destination elements times taps)"
    );
}

/// Normalization-layer parallelism gates. The 3 layers share 2 kernel shapes, a per-channel
/// column fold and a row pass, so 1 gate each covers all of them.
#[cfg(feature = "neural_network")]
pub mod norm {
    use crate::neural_network::layers::regularization::normalization as n;
    use crate::neural_network::layers::regularization::normalization::batch_normalization as bn;

    fwd!(
        set_batch_norm => bn::set_batch_norm_parallel_threshold,
        get_batch_norm => bn::batch_norm_parallel_threshold,
        "the BatchNorm forward/backward total-element gate"
    );
    fwd!(
        set_col_fold => n::set_col_fold_parallel_min_elems,
        get_col_fold => n::col_fold_parallel_min_elems,
        "the shared per-channel column-fold gate (BatchNorm statistics, LayerNorm and GroupNorm \
         gamma/beta gradients)"
    );
    fwd!(
        set_row_pass => n::set_row_pass_parallel_min_elems,
        get_row_pass => n::row_pass_parallel_min_elems,
        "the shared normalization row-pass gate (LayerNorm rows, GroupNorm per-item sweeps)"
    );
}

/// Round-trip test for the gate forwarding macro.
#[cfg(all(test, feature = "math"))]
mod tests {
    /// A public setter reaches the per-site atomic, and the getter reads it back (the full
    /// facade -> `pub(crate)` backing -> atomic path).
    #[test]
    fn matmul_chunk_elems_gate_roundtrips() {
        let orig = super::matmul::get_chunk_elems();
        super::matmul::set_chunk_elems(orig + 7);
        assert_eq!(super::matmul::get_chunk_elems(), orig + 7);
        super::matmul::set_chunk_elems(orig);
        assert_eq!(super::matmul::get_chunk_elems(), orig);
    }
}
