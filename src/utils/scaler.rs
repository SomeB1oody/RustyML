//! Stateful feature scalers that remember their training statistics
//!
//! Houses the `fit` and `transform` counterparts to the stateless [`standardize`] and
//! [`normalize`] functions, mirroring scikit-learn's `sklearn.preprocessing` scalers. Each
//! one learns its per-feature statistics once, on the training matrix. It reuses those
//! frozen numbers for every later batch: the test split, a validation fold, or a single
//! sample arriving at inference time. This keeps a train-test boundary honest. Rescaling a
//! test set by its own column statistics applies a different linear map than the one the
//! model was trained under.
//!
//! | Scaler | Maps each feature by | Reach for it when |
//! |---|---|---|
//! | [`StandardScaler`] | `(x - mean) / std` | the default for distance- and gradient-based models |
//! | [`MinMaxScaler`] | `(x - min) / (max - min)`, rescaled into a target range | you need values bounded to `[0, 1]` (or any range) |
//! | [`MaxAbsScaler`] | `x / max(\|x\|)` | zeros and signs must survive, for example sparse data |
//! | [`RobustScaler`] | `(x - median) / IQR` | outliers you do not want to remove |
//! | [`Normalizer`] | each *sample* divided by its own norm | only a sample's direction carries signal |
//!
//! [`Normalizer`] is the odd one out. It rescales samples (rows), not features (columns), so
//! it learns nothing beyond the feature count. It exists so row normalization uses the same
//! [`Fit`](crate::traits::Fit) and [`Transform`](crate::traits::Transform) contract as the
//! rest. The other 4 are per-feature and stateful in the full sense. All but
//! [`RobustScaler`] also support `partial_fit`. Their statistics merge exactly across batches,
//! but quantiles do not.
//!
//! Every scaler here is 2-D and per-feature, exactly like scikit-learn's. The free functions
//! handle row-wise or whole-array standardization, and N-D arrays.
//!
//! [`standardize`]: crate::utils::standardize::standardize
//! [`normalize`]: crate::utils::normalize::normalize

use crate::error::Error;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, scan_f64_parallel_min_elems};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Data, Ix2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Scale features by their maximum absolute value, preserving zeros and signs
pub mod max_abs_scaler;
/// Scale features to a `[0, 1]` (or custom) range
pub mod min_max_scaler;
/// Scale each sample to unit norm
pub mod normalizer;
/// Scale features by their median and interquartile range, resisting outliers
pub mod robust_scaler;
/// Standardize features to zero mean and unit variance
pub mod standard_scaler;

pub use max_abs_scaler::MaxAbsScaler;
pub use min_max_scaler::MinMaxScaler;
pub use normalizer::Normalizer;
pub use robust_scaler::RobustScaler;
pub use standard_scaler::StandardScaler;

/// A divisor at or below this magnitude is treated as degenerate and replaced by `1.0`
///
/// [`normalize`](crate::utils::normalize::normalize) uses the same threshold for a near-zero
/// lane. It also matches scikit-learn's `_handle_zeros_in_scale` rule
const DEGENERATE_SCALE_THRESHOLD: f64 = 10.0 * f64::EPSILON;

/// Replaces a degenerate divisor with `1.0`
///
/// A feature with no spread otherwise divides by about zero and blows up to `NaN` or `Inf`.
/// This happens for a constant column in [`MinMaxScaler`] or [`RobustScaler`], and for an
/// all-zero column in [`MaxAbsScaler`]. A divisor of `1.0` instead leaves the value unchanged
/// after centering
///
/// [`StandardScaler`] does not use this rule. It detects a constant feature from the variance
/// itself, a magnitude-relative bound. scikit-learn reserves this finer test for the standard
/// scaler too
#[inline]
fn handle_zero_scale(scale: f64) -> f64 {
    if scale.abs() < DEGENERATE_SCALE_THRESHOLD {
        1.0
    } else {
        scale
    }
}

/// Computes the per-feature `(min, max)` over the rows of `x`
///
/// Columns are folded independently, so the serial and parallel paths produce identical
/// results. The gate only decides who does the work
fn column_min_max(x: &ArrayView2<f64>) -> Vec<(f64, f64)> {
    let fold_lane = |lane: ArrayView1<f64>| {
        lane.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &value| {
                (lo.min(value), hi.max(value))
            })
    };

    let lanes: Vec<ArrayView1<f64>> = x.lanes(Axis(0)).into_iter().collect();

    // Scan-class gate: one O(n_samples) pass per feature, so the work is the element count
    if x.len() >= scan_f64_parallel_min_elems() {
        lanes.into_par_iter().map(fold_lane).collect()
    } else {
        lanes.into_iter().map(fold_lane).collect()
    }
}

/// Computes the requested quantiles of every feature
///
/// `quantiles` holds fractions in `[0, 1]`. The returned vector has one entry per feature,
/// each listing that feature's quantiles in the order requested. The function copies each
/// column out, sorts it once, and reads it at every requested position. Asking for 3
/// quantiles this way costs one sort rather than 3 passes
///
/// Interpolates linearly between the 2 order statistics that bracket a fractional position.
/// This is NumPy's default `linear` method, the same one scikit-learn's `RobustScaler` uses,
/// so its quantiles line up with a ported pipeline
///
/// Each column is handled independently, so the serial and parallel paths produce identical
/// results. The gate only decides who does the work. Each worker copies only one column at a
/// time, so the extra memory is `threads * n_samples`, not a second copy of the matrix
fn column_quantiles(x: &ArrayView2<f64>, quantiles: &[f64]) -> Vec<Vec<f64>> {
    let lane_quantiles = |lane: ArrayView1<f64>| {
        // A borrowed lane is strided and cannot be sorted in place, so sort an owned copy
        let mut sorted: Vec<f64> = lane.iter().copied().collect();
        sorted.sort_unstable_by(f64::total_cmp);
        quantiles
            .iter()
            .map(|&q| quantile_of_sorted(&sorted, q))
            .collect()
    };

    let lanes: Vec<ArrayView1<f64>> = x.lanes(Axis(0)).into_iter().collect();

    // Scan-class gate: one sort per feature, so the work scales with the element count
    if x.len() >= scan_f64_parallel_min_elems() {
        lanes.into_par_iter().map(lane_quantiles).collect()
    } else {
        lanes.into_iter().map(lane_quantiles).collect()
    }
}

/// Reads the `q`-quantile (a fraction in `[0, 1]`) out of an ascending slice
///
/// The position is `(n - 1) * q`. A fractional position interpolates linearly between its 2
/// neighbors. `sorted` must be non-empty
#[inline]
fn quantile_of_sorted(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q;
    let lower = position.floor() as usize;
    let fraction = position - lower as f64;

    // An exact hit, or the top of the range, has no upper neighbor to interpolate toward
    if fraction == 0.0 || lower + 1 >= sorted.len() {
        sorted[lower]
    } else {
        sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower])
    }
}

/// Applies `row_op` to every row of `x` in place, on rayon above the cheap-map gate
fn for_each_row<F>(x: &mut Array2<f64>, row_op: F)
where
    F: Fn(ArrayViewMut1<f64>) + Send + Sync,
{
    if x.len() >= cheap_map_f64_parallel_threshold() {
        x.axis_iter_mut(Axis(0)).into_par_iter().for_each(row_op);
    } else {
        x.axis_iter_mut(Axis(0)).for_each(row_op);
    }
}

/// Rejects a matrix that is empty, featureless, or non-finite
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` has no rows or no columns
/// - [`Error::NonFinite`] - If any element is NaN or infinite
fn validate_matrix<S>(x: &ArrayBase<S, Ix2>) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    if x.nrows() == 0 {
        return Err(Error::empty_input("input data"));
    }
    if x.ncols() == 0 {
        return Err(Error::empty_input("features"));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(Error::non_finite("input data"));
    }
    Ok(())
}

/// Validates a matrix handed to a `transform`-style method against the fitted feature count
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` has no rows or no columns
/// - [`Error::DimensionMismatch`] - If `x`'s feature count differs from `n_features`
/// - [`Error::NonFinite`] - If any element is NaN or infinite
fn validate_transform_matrix<S>(x: &ArrayBase<S, Ix2>, n_features: usize) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    validate_matrix(x)?;
    if x.ncols() != n_features {
        return Err(Error::dimension_mismatch(n_features, x.ncols()));
    }
    Ok(())
}

/// Borrows a fitted statistic, or reports that the scaler has not been fitted
///
/// # Errors
///
/// - [`Error::NotFitted`] - If `stat` is `None`
#[inline]
fn fitted<'a>(
    stat: &'a Option<Array1<f64>>,
    model: &'static str,
) -> Result<&'a Array1<f64>, Error> {
    stat.as_ref().ok_or_else(|| Error::not_fitted(model))
}
