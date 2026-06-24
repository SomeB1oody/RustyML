//! Pairwise distance primitives and the [`DistanceCalculationMetric`] dispatcher
//!
//! The three `*_distance_row` functions are the allocation-free per-vector kernels;
//! [`DistanceCalculationMetric`] is the configurable-metric abstraction layered on top of
//! them, used by both the machine-learning estimators (KNN, DBSCAN, the spatial index) and
//! the clustering metrics (e.g. `silhouette_score`)

#[cfg(any(feature = "machine_learning", feature = "utils"))]
use crate::{Deserialize, Serialize};
use ndarray::{ArrayBase, ArrayView1, Data, Ix1, Zip};

/// Calculates the squared Euclidean distance between two vectors
///
/// # Parameters
///
/// - `x1` - First vector
/// - `x2` - Second vector
///
/// # Returns
///
/// - `f64` - Squared Euclidean distance between the two vectors
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::squared_euclidean_distance_row;
///
/// let v1 = array![1.0, 2.0, 3.0];
/// let v2 = array![4.0, 5.0, 6.0];
/// let dist = squared_euclidean_distance_row(&v1, &v2);
/// // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
/// assert!((dist - 27.0).abs() < 1e-10);
/// ```
#[inline]
pub fn squared_euclidean_distance_row<S1, S2>(
    x1: &ArrayBase<S1, Ix1>,
    x2: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    // Accumulate in a single pass with no intermediate allocation
    let mut sum = 0.0;
    Zip::from(x1).and(x2).for_each(|&a, &b| {
        let d = a - b;
        sum += d * d;
    });
    sum
}

/// Calculates the Manhattan (L1) distance between two vectors
///
/// # Parameters
///
/// - `x1` - First vector
/// - `x2` - Second vector
///
/// # Returns
///
/// - `f64` - Manhattan distance between the two vectors
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::manhattan_distance_row;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let distance = manhattan_distance_row(&v1, &v2);
/// // |1-4| + |2-6| = 3 + 4 = 7
/// assert!((distance - 7.0).abs() < 1e-6);
/// ```
#[inline]
pub fn manhattan_distance_row<S1, S2>(x1: &ArrayBase<S1, Ix1>, x2: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    // Single-pass, allocation-free L1 norm of the difference
    let mut sum = 0.0;
    Zip::from(x1)
        .and(x2)
        .for_each(|&a, &b| sum += (a - b).abs());
    sum
}

/// Calculates the Minkowski distance between two vectors
///
/// Computes the p-norm of the difference between two 1D arrays
///
/// # Parameters
///
/// - `x1` - First vector
/// - `x2` - Second vector
/// - `p` - Order of the norm (must be at least 1.0)
///
/// # Returns
///
/// - `f64` - Minkowski distance between the two vectors
///
/// # Panics
///
/// - Panics if `p < 1.0` (or `p` is `NaN`): for such orders the result is not a valid metric
///   (the triangle inequality fails), and `p <= 0` additionally yields a meaningless `sum^inf`
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::minkowski_distance_row;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let distance = minkowski_distance_row(&v1, &v2, 3.0);
/// // Expected distance is approximately 4.497
/// assert!((distance - 4.497).abs() < 1e-3);
/// ```
#[inline]
pub fn minkowski_distance_row<S1, S2>(
    x1: &ArrayBase<S1, Ix1>,
    x2: &ArrayBase<S2, Ix1>,
    p: f64,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    // `p.is_nan()` is rejected alongside `p < 1.0`; orders below 1 break the triangle inequality
    if p < 1.0 || p.is_nan() {
        panic!("invalid parameter `p`: Minkowski order must be at least 1.0, got {p}");
    }

    let mut sum = 0.0;
    Zip::from(x1)
        .and(x2)
        .for_each(|&a, &b| sum += (a - b).abs().powf(p));
    sum.powf(1.0 / p)
}

/// Distance calculation methods used across machine learning algorithms
///
/// Defines common distance metrics for clustering algorithms, nearest neighbor
/// searches, and other applications where distance between points is relevant
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(
    any(feature = "machine_learning", feature = "utils"),
    derive(Deserialize, Serialize)
)]
pub enum DistanceCalculationMetric {
    /// Euclidean distance (L2 norm): the square root of the sum of squared
    /// differences between corresponding coordinates
    #[default]
    Euclidean,
    /// Manhattan distance (L1 norm): the sum of absolute differences between
    /// corresponding coordinates
    Manhattan,
    /// Generalized metric with Euclidean and Manhattan as special cases; the
    /// `f64` is the order parameter `p`
    Minkowski(f64),
}

impl DistanceCalculationMetric {
    /// Computes the distance between two vectors under this metric
    ///
    /// Single source of truth for metric dispatch; models such as KNN and DBSCAN
    /// call it instead of re-implementing the `match` over variants
    ///
    /// # Parameters
    ///
    /// - `a` - First vector
    /// - `b` - Second vector
    ///
    /// # Returns
    ///
    /// - `f64` - The distance between `a` and `b` under this metric
    #[inline]
    pub fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        match *self {
            DistanceCalculationMetric::Euclidean => squared_euclidean_distance_row(&a, &b).sqrt(),
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(&a, &b),
            DistanceCalculationMetric::Minkowski(p) => minkowski_distance_row(&a, &b, p),
        }
    }

    /// Returns whether `distance(a, b) <= threshold` under this metric
    #[inline]
    pub fn within(&self, a: ArrayView1<f64>, b: ArrayView1<f64>, threshold: f64) -> bool {
        self.comparable_distance(a, b) <= self.comparable_scalar(threshold)
    }

    /// Maps a non-negative scalar (a true distance or a per-axis coordinate gap) into this
    /// metric's order-preserving "comparable" space, where the final root is skipped:
    /// `Euclidean -> t^2`, `Manhattan -> t`, `Minkowski(p) -> t^p`
    ///
    /// Used by spatial indexes so radius thresholds and per-axis pruning bounds can be
    /// compared against [`comparable_distance`](Self::comparable_distance) without repeated
    /// roots. The mapping is monotonic on `t >= 0`, so all ordering decisions are preserved
    pub(crate) fn comparable_scalar(&self, t: f64) -> f64 {
        match *self {
            DistanceCalculationMetric::Euclidean => t * t,
            DistanceCalculationMetric::Manhattan => t,
            DistanceCalculationMetric::Minkowski(p) => t.powf(p),
        }
    }

    /// Distance between two vectors in this metric's comparable space (see
    /// [`comparable_scalar`](Self::comparable_scalar)): the monotonic, root-free form of
    /// [`distance`](Self::distance). Equals `distance(a, b)` raised to the metric's power
    /// (squared for Euclidean, `^p` for Minkowski, unchanged for Manhattan)
    pub(crate) fn comparable_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        match *self {
            DistanceCalculationMetric::Euclidean => squared_euclidean_distance_row(&a, &b),
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(&a, &b),
            DistanceCalculationMetric::Minkowski(p) => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs().powf(p))
                .sum(),
        }
    }

    /// Converts a comparable-space distance back to a true distance (inverse of
    /// [`comparable_distance`](Self::comparable_distance)): `Euclidean -> sqrt`,
    /// `Manhattan -> identity`, `Minkowski(p) -> ^(1/p)`
    #[cfg(feature = "machine_learning")]
    pub(crate) fn distance_from_comparable(&self, c: f64) -> f64 {
        match *self {
            DistanceCalculationMetric::Euclidean => c.sqrt(),
            DistanceCalculationMetric::Manhattan => c,
            DistanceCalculationMetric::Minkowski(p) => c.powf(1.0 / p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // DistanceCalculationMetric::distance

    // Euclidean: sqrt((3-0)^2 + (4-0)^2) = sqrt(9+16) = 5
    #[test]
    fn distance_euclidean_345_triangle() {
        let metric = DistanceCalculationMetric::Euclidean;
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 5.0, epsilon = 1e-6);
    }

    // Manhattan: |3-0| + |4-0| = 7
    #[test]
    fn distance_manhattan_345() {
        let metric = DistanceCalculationMetric::Manhattan;
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 7.0, epsilon = 1e-6);
    }

    // Minkowski(3): (|3|^3 + |4|^3)^(1/3) = (27+64)^(1/3) = 91^(1/3) ~= 4.497941
    #[test]
    fn distance_minkowski_p3() {
        let metric = DistanceCalculationMetric::Minkowski(3.0);
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        let expected = 91.0_f64.powf(1.0 / 3.0); // ~= 4.497941445275415
        assert_abs_diff_eq!(
            metric.distance(a.view(), b.view()),
            expected,
            epsilon = 1e-6
        );
    }

    // Euclidean is symmetric: distance(a,b) == distance(b,a)
    #[test]
    fn distance_euclidean_symmetry() {
        let metric = DistanceCalculationMetric::Euclidean;
        let a = array![1.0_f64, 2.0, 3.0];
        let b = array![4.0_f64, 6.0, 8.0];
        assert_abs_diff_eq!(
            metric.distance(a.view(), b.view()),
            metric.distance(b.view(), a.view()),
            epsilon = 1e-10
        );
    }

    // Zero distance: identical vectors -> 0
    #[test]
    fn distance_euclidean_identical_vectors() {
        let metric = DistanceCalculationMetric::Euclidean;
        let a = array![1.0_f64, 2.0];
        assert_abs_diff_eq!(metric.distance(a.view(), a.view()), 0.0, epsilon = 1e-6);
    }

    // Manhattan is symmetric
    #[test]
    fn distance_manhattan_symmetry() {
        let metric = DistanceCalculationMetric::Manhattan;
        let a = array![1.0_f64, 5.0];
        let b = array![3.0_f64, 2.0];
        assert_abs_diff_eq!(
            metric.distance(a.view(), b.view()),
            metric.distance(b.view(), a.view()),
            epsilon = 1e-10
        );
    }
}
