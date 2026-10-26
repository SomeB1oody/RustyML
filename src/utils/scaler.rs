//! Stateful scalers for 2D feature matrices.
//!
//! Feature scalers store training statistics and reuse them for later transforms.
//! `Normalizer` instead scales each row by its own norm and stores only the feature count.
//! Shared helpers check matrices, compute column statistics, and apply row transforms.

use crate::error::Error;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, scan_f64_parallel_min_elems};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Data, Ix2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Scale features by their maximum absolute value, preserving 0 values and signs
pub mod max_abs_scaler;
/// Scale features to a `[0, 1]` (or custom) range
pub mod min_max_scaler;
/// Scale each sample to unit norm
pub mod normalizer;
/// Scale features by their median and interquartile range, resisting outliers
pub mod robust_scaler;
/// Standardize features to mean 0 and variance 1
pub mod standard_scaler;

pub use max_abs_scaler::MaxAbsScaler;
pub use min_max_scaler::MinMaxScaler;
pub use normalizer::Normalizer;
pub use robust_scaler::RobustScaler;
pub use standard_scaler::StandardScaler;

/// Absolute divisor magnitude below which the scale becomes `1.0`.
const DEGENERATE_SCALE_THRESHOLD: f64 = 10.0 * f64::EPSILON;

/// Replaces a degenerate divisor with `1.0`
///
/// A feature with no spread otherwise divides by about 0 and blows up to `NaN` or `Inf`.
/// This happens for a constant column in [`MinMaxScaler`] or [`RobustScaler`], and for an
/// column with all values 0 in [`MaxAbsScaler`]. A divisor of `1.0` instead leaves the value unchanged
/// after centering
///
/// [`StandardScaler`] uses a variance-based bound to detect constant features instead.
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

    // Scan-class gate: 1 O(n_samples) pass per feature, so the work is the element count
    if x.len() >= scan_f64_parallel_min_elems() {
        lanes.into_par_iter().map(fold_lane).collect()
    } else {
        lanes.into_iter().map(fold_lane).collect()
    }
}

/// Computes the requested quantiles of every feature
///
/// `quantiles` holds fractions in `[0, 1]`. The returned vector has 1 entry per feature,
/// each listing that feature's quantiles in the order requested. The function copies each
/// column out, sorts it once, and reads it at every requested position. Asking for 3
/// quantiles this way costs 1 sort rather than 3 passes
///
/// Interpolates linearly between the 2 order statistics that bracket a fractional position.
/// Fractional positions use the adjacent sorted values.
///
/// Each column is handled independently, so the serial and parallel paths produce identical
/// results. The gate only decides who does the work. Each worker copies only 1 column at a
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

    // Scan-class gate: 1 sort per feature, so the work scales with the element count
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
