//! Numeric primitives shared by estimators, metrics, and neural networks.
//!
//! The submodules provide matrix products, fixed-block reductions, and vector distances.
//! This module also defines the logistic loss reduction threshold.
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
//!
//! [`matmul`]: crate::math::matmul
//! [`reduction`]: crate::math::reduction
//! [`distance`]: crate::math::distance
//! [`DistanceCalculationMetric`]: crate::math::DistanceCalculationMetric

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

/// Element threshold for parallel logistic loss reductions with `exp` and `ln` calls.
///
/// The value equals 2 fixed reduction blocks, the smallest count that yields multiple tasks.
/// This threshold has no runtime setter.
#[cfg(feature = "machine_learning")]
pub(crate) const EXP_REDUCE_MIN_ELEMS: usize = 32_768;
