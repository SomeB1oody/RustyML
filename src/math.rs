//! Shared low-level numeric primitives used across estimators and metrics
//!
//! - [`matmul`](crate::math::matmul) provides `gemmkit`-backed matrix products with automatic
//!   parallelism
//! - [`reduction`](crate::math::reduction) provides deterministic blocked parallel reductions
//! - [`distance`](crate::math::distance) holds the pairwise distance primitives and the
//!   [`DistanceCalculationMetric`](crate::math::DistanceCalculationMetric) dispatcher
//!
//! It also hosts the tunable exp-reduction parallel gate used by the logistic-regression
//! log-loss.
//!
//! # What belongs here
//!
//! A function lives in `math` only if it meets 3 conditions. It is pure and stateless. It is
//! model-agnostic, so it encodes no single algorithm's policy. It is, or plausibly could be,
//! shared by more than one caller. Per-algorithm solvers live next to their model. Post-hoc
//! evaluation metrics live in [`crate::metrics`] and call these primitives. Trainable,
//! gradient-aware losses live in `neural_network::losses`.
//!
//! # Examples
//!
//! ```rust
//! use rustyml::math::{DistanceCalculationMetric, squared_euclidean_distance_row};
//! use ndarray::array;
//!
//! // Distance primitive plus the configurable metric dispatcher.
//! let v1 = array![1.0, 2.0];
//! let v2 = array![4.0, 6.0];
//! let sq = squared_euclidean_distance_row(&v1, &v2);
//! let d = DistanceCalculationMetric::Euclidean.distance(v1.view(), v2.view());
//! ```

/// `gemmkit`-backed matrix products with automatic parallelism.
pub mod matmul;

/// Deterministic blocked parallel reductions.
pub mod reduction;

/// Pairwise distance primitives and the [`DistanceCalculationMetric`] dispatcher.
pub mod distance;

pub use distance::{
    DistanceCalculationMetric, manhattan_distance_row, minkowski_distance_row,
    squared_euclidean_distance_row,
};

tunable_gate! {
    /// Parallel gate for exp-heavy `f64` reductions (the logistic-regression log-loss). Below
    /// this element count, the deterministic blocked fold cannot beat the serial sum.
    ///
    /// Sits below the cheap-sum gate because each element costs an `exp` plus an `ln`.
    ///
    /// Overridable through [`crate::tuning`].
    pub(crate) EXP_REDUCE_MIN_ELEMS => exp_reduce_min_elems / set_exp_reduce_min_elems = 32_768
}
