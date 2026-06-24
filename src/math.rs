//! Shared numeric primitives used across estimators and metrics
//!
//! - [`matmul`] provides `gemm`-crate-backed matrix products, parallelized on the rayon pool
//! - [`reduction`] provides deterministic blocked parallel reductions
//! - [`distance`] holds the pairwise distance primitives and the [`DistanceCalculationMetric`]
//!   dispatcher
//!
//! It also hosts the tunable exp-reduction parallel gate used by the logistic-regression log-loss

/// Matrix products backed by the `gemm` crate (crate-internal `gemm_par_auto` / `gemv_par_auto`),
/// parallelized on the rayon pool for large products
pub mod matmul;

/// Deterministic blocked parallel reductions (`det_reduce`, `det_reduce_range`) that
/// reproduce the same result across runs on the same machine
pub mod reduction;

/// Pairwise distance primitives (`squared_euclidean_distance_row`, `manhattan_distance_row`,
/// `minkowski_distance_row`) and the [`DistanceCalculationMetric`](distance::DistanceCalculationMetric)
/// dispatcher layered on top of them
pub mod distance;

pub use distance::{
    DistanceCalculationMetric, manhattan_distance_row, minkowski_distance_row,
    squared_euclidean_distance_row,
};

tunable_gate! {
    /// Parallel gate for exp-heavy `f64` reductions (the logistic-regression log-loss): below this
    /// element count the deterministic blocked fold cannot beat the serial sum
    ///
    /// Crossover bracket is 16K-32K elements (0.96x at 16K, 1.85x at 32K, 14.3x at 1M). Sits below
    /// the cheap-sum gate because each element costs an `exp` plus an `ln`
    ///
    /// Overridable via [`crate::tuning`]
    pub(crate) EXP_REDUCE_MIN_ELEMS => exp_reduce_min_elems / set_exp_reduce_min_elems = 32_768
}
