//! Shared low-level numeric primitives used across estimators and metrics
//!
//! - [`matmul`](crate::math::matmul) provides `gemm`-crate-backed matrix products, parallelized on the rayon pool
//! - [`reduction`](crate::math::reduction) provides deterministic blocked parallel reductions
//! - [`distance`](crate::math::distance) holds the pairwise distance primitives and the
//!   [`DistanceCalculationMetric`](crate::math::DistanceCalculationMetric) dispatcher
//!
//! It also hosts the tunable exp-reduction parallel gate used by the logistic-regression log-loss
//!
//! # What belongs here
//!
//! A function lives in `math` only if it is **(1)** pure and stateless, **(2)** model-agnostic (it
//! encodes no single algorithm's policy), and **(3)** is - or plausibly could be - shared by more
//! than one caller. Per-algorithm solvers live next to their model; post-hoc evaluation metrics
//! live in [`crate::metrics`] and call these primitives; trainable, gradient-aware losses live in
//! `neural_network::losses`
//!
//! # Example
//!
//! ```rust
//! use rustyml::math::{DistanceCalculationMetric, squared_euclidean_distance_row};
//! use ndarray::array;
//!
//! // Distance primitive plus the configurable metric dispatcher
//! let v1 = array![1.0, 2.0];
//! let v2 = array![4.0, 6.0];
//! let sq = squared_euclidean_distance_row(&v1, &v2);
//! let d = DistanceCalculationMetric::Euclidean.distance(v1.view(), v2.view());
//! ```

/// `gemm`-backed matrix products, parallelized on the rayon pool for large products
pub mod matmul;

/// Deterministic blocked parallel reductions
pub mod reduction;

/// Pairwise distance primitives and the [`DistanceCalculationMetric`] dispatcher
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
